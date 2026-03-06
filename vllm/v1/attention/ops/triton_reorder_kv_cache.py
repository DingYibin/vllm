# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Triton kernels for reordering KV cache after tree speculation acceptance.

After the rejection sampler determines which branch of the tree was accepted,
the KV cache needs to be updated:
1. Accepted tokens should be at their final positions (according to seq lens)
2. Rejected tokens (final_slot = -1) should be ignored/removed

Key observations:
1. For the same request: slot_mapping and final_slot_mapping may overlap
    - Accepted tokens stay in their sequence positions
    - Source and destination can be the same, in which case no copy is needed
2. For different requests: No overlap between slot_mappings
    - Safe to process all requests in parallel

Solution: Use two kernels with a temporary buffer
- Kernel 1 (extract): Copy source data from original cache to temp buffer
                    based on slot_mapping, only for accepted tokens (final_slot >= 0)
- Kernel 2 (store): Copy data from temp buffer to final destinations
                    based on final_slot_mapping
"""
import torch
import logging

from packaging import version

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

is_hip_ = current_platform.is_rocm()

logger = logging.getLogger(__name__)

# Only print the following warnings when triton version < 3.2.0.
# The issue won't affect performance or accuracy.
if version.parse(triton.__version__) < version.parse("3.2.0"):
    logger.warning(
        "The following error message 'operation scheduled before its operands' "
        "can be ignored."
    )

@triton.jit
def _extract_cache_kernel(
    cache_ptr,
    temp_cache_ptr,
    slot_mapping_ptr,
    final_slot_mapping_ptr,
    head_size,
    block_size,
    stride_cache_block,
    stride_cache_block_size,
    stride_cache_head,
    stride_temp_tokens,
    stride_temp_head,
    TILE_SIZE: tl.constexpr = 64,
):
    """
    Extract source data from cache to temp buffer.
    """
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    tile_idx = tl.program_id(2)

    src_slot = tl.load(slot_mapping_ptr + token_idx)
    dst_slot = tl.load(final_slot_mapping_ptr + token_idx)

    if dst_slot < 0:
        return

    src_block = src_slot // block_size
    src_offset = src_slot % block_size
    head_offset = tile_idx * TILE_SIZE
    mask = head_offset + tl.arange(0, TILE_SIZE) < head_size

    src_base = (src_block * stride_cache_block +
                src_offset * stride_cache_block_size +
                head_idx * stride_cache_head +
                head_offset)
    
    dst_base = (token_idx * stride_temp_tokens +
                head_idx * stride_temp_head +
                head_offset)

    src_data = tl.load(cache_ptr + src_base + tl.arange(0, TILE_SIZE), mask=mask)
    tl.store(temp_cache_ptr + dst_base + tl.arange(0, TILE_SIZE), src_data, mask=mask)

@triton.jit
def _store_cache_kernel(
    cache_ptr,
    temp_cache_ptr,
    final_slot_mapping_ptr,
    head_size,
    block_size,
    stride_cache_block,
    stride_cache_block_size,
    stride_cache_head,
    stride_temp_tokens,
    stride_temp_head,
    TILE_SIZE: tl.constexpr = 64,
):
    """
    Store data from temp buffer to final destinations in cache.
    """
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    tile_idx = tl.program_id(2)

    dst_slot = tl.load(final_slot_mapping_ptr + token_idx)

    if dst_slot < 0:
        return

    dst_block = dst_slot // block_size
    dst_offset = dst_slot % block_size
    
    head_offset = tile_idx * TILE_SIZE
    mask = head_offset + tl.arange(0, TILE_SIZE) < head_size

    src_base = (token_idx * stride_temp_tokens +
                head_idx * stride_temp_head +
                head_offset)

    dst_base = (dst_block * stride_cache_block +
                dst_offset * stride_cache_block_size +
                head_idx * stride_cache_head +
                head_offset)

    temp_data = tl.load(temp_cache_ptr + src_base + tl.arange(0, TILE_SIZE), mask=mask)
    tl.store(cache_ptr + dst_base + tl.arange(0, TILE_SIZE), temp_data, mask=mask)

def _get_cache_stride(cache: torch.Tensor):
    num_blocks, block_size, num_heads, head_size = cache.shape
    stride_cache_block = block_size * num_heads * head_size
    stride_cache_block_size = num_heads * head_size
    stride_cache_head = head_size

    return (
        block_size,
        num_heads,
        head_size,
        stride_cache_block,
        stride_cache_block_size,
        stride_cache_head,
    )

def _reorder_cache(
    cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    final_slot_mapping: torch.Tensor,
    tile_size: int = 64,
) -> None:
    """
    Reorder a single cache (key or value) using temp buffer.
    """
    num_tokens = slot_mapping.shape[0]
    device = cache.device
    dtype = cache.dtype
    (
        block_size,
        num_heads,
        head_size,
        stride_cache_block,
        stride_cache_block_size,
        stride_cache_head,
    ) = _get_cache_stride(cache)

    temp_cache = torch.zeros(num_tokens, num_heads, head_size,
                             device=device, dtype=dtype)
    stride_temp_tokens = num_heads * head_size
    stride_temp_head = head_size
    num_tiles = (head_size + tile_size - 1) // tile_size
    grid = (num_tokens, num_heads, num_tiles)

    _extract_cache_kernel[grid](
        cache_ptr=cache,
        temp_cache_ptr=temp_cache,
        slot_mapping_ptr=slot_mapping,
        final_slot_mapping_ptr=final_slot_mapping,
        head_size=head_size,
        block_size=block_size,
        stride_cache_block=stride_cache_block,
        stride_cache_block_size=stride_cache_block_size,
        stride_cache_head=stride_cache_head,
        stride_temp_tokens=stride_temp_tokens,
        stride_temp_head=stride_temp_head,
        TILE_SIZE=tile_size,
    )

    _store_cache_kernel[grid](
        cache_ptr=cache,
        temp_cache_ptr=temp_cache,
        final_slot_mapping_ptr=final_slot_mapping,
        head_size=head_size,
        block_size=block_size,
        stride_cache_block=stride_cache_block,
        stride_cache_block_size=stride_cache_block_size,
        stride_cache_head=stride_cache_head,
        stride_temp_tokens=stride_temp_tokens,
        stride_temp_head=stride_temp_head,
        TILE_SIZE=tile_size,
    )

def reorder_kv_cache(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    final_slot_mapping: torch.Tensor,
    tile_size: int = 64,
) -> None:
    """
    Reorder KV cache after tree speculation acceptance.
    """
    
    _reorder_cache(
        cache=key_cache,
        slot_mapping=slot_mapping,
        final_slot_mapping=final_slot_mapping,
        tile_size=tile_size,
    )

    _reorder_cache(
        cache=value_cache,
        slot_mapping=slot_mapping,
        final_slot_mapping=final_slot_mapping,
        tile_size=tile_size,
    )

def test_reorder_kv_cache():
    num_blocks = 100
    block_size = 16
    num_kv_heads = 8
    key_head_size = 128
    val_head_size = 64
    key_cache = torch.randn(
        num_blocks, block_size, num_kv_heads, key_head_size,
        device='cuda', dtype=torch.bfloat16,
    )
    val_cache = torch.randn(
        num_blocks, block_size, num_kv_heads, val_head_size,
        device='cuda', dtype=torch.bfloat16,
    )
    key_cache_ori = key_cache.clone()
    val_cache_ori = val_cache.clone()
    slot_mapping = torch.arange(15, dtype=torch.int32, device='cuda') + 116
    final_slot_mapping = torch.tensor(
        [116, 117, -1, 118, -1, -1, -1, -1, 119, -1, -1, -1, -1, -1, -1],
    dtype=torch.int32, device='cuda')
    reorder_kv_cache(
        key_cache, val_cache, slot_mapping, final_slot_mapping,
    )
    mask = (slot_mapping != -1) & (final_slot_mapping != -1)
    masked_slot_mapping = slot_mapping[mask]
    masked_final_slot_mapping = final_slot_mapping[mask]
    
    final_key = key_cache.view(-1, num_kv_heads, key_head_size)[masked_final_slot_mapping]
    ori_key = key_cache_ori.view(-1, num_kv_heads, key_head_size)[masked_slot_mapping]

    final_val = val_cache.view(-1, num_kv_heads, val_head_size)[masked_final_slot_mapping]
    ori_val = val_cache_ori.view(-1, num_kv_heads, val_head_size)[masked_slot_mapping]

    print(f"{(final_key - ori_key).abs().max()=}")
    print(f"{(final_val - ori_val).abs().max()=}")
    

if __name__ == "__main__":
    test_reorder_kv_cache()

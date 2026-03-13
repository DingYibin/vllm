# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Triton kernels for reordering KV cache after tree speculation acceptance.

================================================================================
Background
================================================================================

【Tree Speculation】
A variant of speculative decoding where the model simultaneously predicts multiple
candidate tokens forming a tree structure, rather than a single linear sequence.
By expanding the speculation space, the speculation hit rate can be improved,
thus accelerating the decoding process.

Example tree structure:
    prompt: "The"
    speculation tree:
        The → cat → sat → on
            → dog → ran
            → bird
    Each branch is a possible continuation path.

【Rejection Sampler】
After speculation is complete, the rejection sampler validates whether each
candidate token conforms to the target model's probability distribution.
Accepted tokens are retained, rejected tokens are discarded.

For example, in the above tree, if "cat → sat → on" is accepted and other
branches are rejected:
    Final sequence: "The cat sat on"
    KV cache to retain: cache for ["The", "cat", "sat", "on"]

【Reordering Problem】
Cache positions allocated during speculation may not match final sequence positions:
    - During speculation: tokens are allocated slots according to tree structure,
      possibly scattered
    - After acceptance: tokens need to be arranged consecutively in cache,
      conforming to final sequence order

Example:
    slot_mapping = [10, 11, 12, 13, 14, 15]  # Slots during speculation
    final_slot_mapping = [20, 21, -1, 22, -1, -1]  # Final slots after acceptance (-1 = rejected)

    Required:
    - slot 10 → slot 20 (token 0 accepted)
    - slot 11 → slot 21 (token 1 accepted)
    - slot 12 → discard (token 2 rejected)
    - slot 13 → slot 22 (token 3 accepted)
    - slot 14, 15 → discard

================================================================================
Design Points
================================================================================

【Why can't we reorder in-place?】
Direct in-place copying causes data races:

    Assume slot_mapping = [0, 1], final_slot_mapping = [1, 0]

    Step 1: Copy slot 0 → position 1
            Position 1's original data is now overwritten!

    Step 2: Copy slot 1 → position 0
            But position 1's original data is lost, cannot copy correctly

【Solution: Two-stage kernel + temporary buffer】

    Stage 1 (Extract): Original cache → Temporary buffer
        - Read source data according to slot_mapping
        - Write to temporary buffer (indexed by token_idx)
        - Only process accepted tokens (final_slot >= 0)

    Stage 2 (Store): Temporary buffer → Final cache positions
        - Read from temporary buffer
        - Write to final positions according to final_slot_mapping

    This completely separates reads from writes, avoiding data races.

【Parallel Strategy】
Use 3D grid parallelism: (num_tokens, num_heads, num_tiles)
    - Token dimension: different tokens processed independently
    - Head dimension: different attention heads processed independently
    - Tile dimension: head_size processed in chunks, optimizing memory access

【Memory Layout】
KV cache uses PagedAttention's paged layout:
    cache[num_blocks, block_size, num_heads, head_size]

    A block is a contiguous memory chunk containing block_size slots.
    Each slot stores KV data for all heads.

    Address calculation:
        slot = slot_mapping[token_idx]
        block_id = slot // block_size
        block_offset = slot % block_size
        address = block_id * stride_block + block_offset * stride_slot + head_idx * stride_head + offset
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
    Triton kernel: Extract data from original KV cache to temporary buffer (Stage 1).

    This kernel is the first step of the two-stage reordering, responsible for
    reading source data from original cache positions and writing to a temporary
    buffer. The temporary buffer acts as an intermediary to avoid data races
    caused by in-place operations.

    Parallel strategy:
        - program_id(0): token_idx - token index
        - program_id(1): head_idx - attention head index
        - program_id(2): tile_idx - tile index for head dimension

    Cache memory layout:
        cache[num_blocks, block_size, num_heads, head_size]
        - Each block contains block_size slots
        - Each slot stores KV data for all heads
        - Each head stores head_size dimensional features

    Args:
        cache_ptr: Original KV cache pointer, shape [num_blocks, block_size, num_heads, head_size]
        temp_cache_ptr: Temporary buffer pointer, shape [num_tokens, num_heads, head_size]
        slot_mapping_ptr: Source slot mapping, shape [num_tokens], original cache position for each token
        final_slot_mapping_ptr: Destination slot mapping, shape [num_tokens], final cache position for each token
            - final_slot >= 0: accepted token, needs reordering
            - final_slot == -1: rejected token, skip processing
        head_size: Feature dimension for each attention head
        block_size: Number of slots in each block
        stride_cache_block: Stride for block dimension
        stride_cache_block_size: Stride for slot dimension within a block
        stride_cache_head: Stride for head dimension
        stride_temp_tokens: Stride for token dimension in temporary buffer
        stride_temp_head: Stride for head dimension in temporary buffer
        TILE_SIZE: Tile size (compile-time constant), used for processing head dimension

    Note:
        This kernel skips tokens with final_slot < 0 (rejected candidates).
        Data is written to the token_idx position in the temporary buffer,
        to be read by the stage 2 kernel.
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
    Triton kernel: Store data from temporary buffer to final cache positions (Stage 2).

    This kernel is the second step of the two-stage reordering, responsible for
    reading data from the temporary buffer and writing to final cache positions.
    At this point, source data is safely preserved in the temporary buffer,
    allowing safe writes to any destination without data loss.

    Parallel strategy:
        - program_id(0): token_idx - token index
        - program_id(1): head_idx - attention head index
        - program_id(2): tile_idx - tile index for head dimension

    Args:
        cache_ptr: Target KV cache pointer, shape [num_blocks, block_size, num_heads, head_size]
        temp_cache_ptr: Temporary buffer pointer, shape [num_tokens, num_heads, head_size]
        final_slot_mapping_ptr: Destination slot mapping, shape [num_tokens]
            - final_slot >= 0: final cache position for accepted token
            - final_slot == -1: rejected token, skip processing
        head_size: Feature dimension for each attention head
        block_size: Number of slots in each block
        stride_cache_block: Stride for block dimension
        stride_cache_block_size: Stride for slot dimension within a block
        stride_cache_head: Stride for head dimension
        stride_temp_tokens: Stride for token dimension in temporary buffer
        stride_temp_head: Stride for head dimension in temporary buffer
        TILE_SIZE: Tile size (compile-time constant), used for processing head dimension

    Note:
        This kernel reads the output of stage 1 from the temporary buffer and
        writes to final cache positions. The temporary buffer is indexed by
        token_idx, and the final position is determined by final_slot_mapping.
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
    """
    Compute memory strides for KV cache tensor.

    KV cache uses PagedAttention's paged storage layout. Understanding strides
    is crucial for correctly computing memory addresses.

    Memory layout illustration:
        cache[num_blocks, block_size, num_heads, head_size]

        Assuming num_blocks=2, block_size=4, num_heads=2, head_size=3:

        Block 0:                              Block 1:
        [slot0: head0:[x,x,x] head1:[x,x,x]]  [slot0: ...]
         [slot1: head0:[x,x,x] head1:[x,x,x]]  [slot1: ...]
         [slot2: head0:[x,x,x] head1:[x,x,x]]  [slot2: ...]
         [slot3: head0:[x,x,x] head1:[x,x,x]]  [slot3: ...]

    Stride calculation:
        - stride_cache_block: stride to skip one block = block_size * num_heads * head_size
        - stride_cache_block_size: stride to skip one slot within a block = num_heads * head_size
        - stride_cache_head: stride to skip one head = head_size

    Address calculation formula:
        address = block_id * stride_cache_block
                + block_offset * stride_cache_block_size
                + head_id * stride_cache_head
                + head_offset

        Where block_id = slot // block_size, block_offset = slot % block_size

    Args:
        cache: KV cache tensor, shape [num_blocks, block_size, num_heads, head_size]

    Returns:
        tuple: (block_size, num_heads, head_size,
                stride_cache_block, stride_cache_block_size, stride_cache_head)
    """
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
    Reorder a single KV cache (Key or Value).

    This function coordinates the execution of two-stage kernels, implementing
    safe data reordering through a temporary buffer.

    Execution flow:
        1. Parse cache shape and compute strides
        2. Allocate temporary buffer temp_cache[num_tokens, num_heads, head_size]
        3. Launch kernel 1 (extract): original cache → temporary buffer
        4. Launch kernel 2 (store): temporary buffer → final cache positions

    Why two stages?
        Direct in-place reordering causes data races. For example:
        - slot_mapping = [0, 1], final_slot_mapping = [1, 0]
        - If we first write slot 0 → position 1, position 1's original data is overwritten
        - Then we cannot correctly read position 1's original data to write to position 0
        Using a temporary buffer decouples reads from writes, avoiding this problem.

    Args:
        cache: KV cache tensor, shape [num_blocks, block_size, num_heads, head_size],
               this tensor is modified in-place
        slot_mapping: Source slot mapping, shape [num_tokens], dtype=torch.int32
            - slot_mapping[i] represents the original cache position of the i-th token
            - Used by kernel 1 to read source data
        final_slot_mapping: Destination slot mapping, shape [num_tokens], dtype=torch.int32
            - final_slot_mapping[i] represents the final cache position of the i-th token
            - final_slot >= 0: accepted token
            - final_slot == -1: rejected token, skip processing
        tile_size: Tile size for head dimension, default 64. Used to optimize GPU memory
                   access by processing head_size dimension in tiles to leverage shared
                   memory and cache locality.

    Returns:
        None (modifies cache tensor in-place)

    Note:
        The temporary buffer temp_cache is allocated within the function and
        automatically released after reordering completes.
        Memory overhead is num_tokens * num_heads * head_size * sizeof(dtype).
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
    slot_mapping_map: torch.Tensor,
    tile_size: int = 64,
) -> None:
    """
    Reorder KV cache after tree speculation acceptance.

    This is the main entry point for this module, called after the rejection
    sampling in tree speculation decoding completes. Based on the rejection
    sampling results, it moves KV cache of accepted tokens to correct final
    positions.

    Tree speculation background:
        In tree speculation decoding, the model simultaneously predicts multiple
        candidate tokens forming a tree structure.
        For example: predicting "The cat sat on the mat" might have branches:
              The → cat → sat    (branch 1: accepted)
                  → dog → walked  (branch 2: rejected)
        After the rejection sampler determines which branches are accepted:
        1. Retain KV cache for accepted tokens
        2. Discard KV cache for rejected tokens
        3. Move retained cache to consecutive positions (conforming to final sequence order)

    Usage scenarios:
        1. Called during the accept phase of tree speculation decoding
        2. slot_mapping comes from cache allocation during speculation phase
        3. final_slot_mapping is computed by rejection sampler based on acceptance results

    Args:
        key_cache: Key cache tensor, shape [num_blocks, block_size, num_kv_heads, head_size]
        value_cache: Value cache tensor, shape [num_blocks, block_size, num_kv_heads, head_size]
        slot_mapping: Source slot mapping, shape [num_tokens], dtype=torch.int32
            - slot_mapping[i] = original cache slot of the i-th speculated token
        final_slot_mapping: Destination slot mapping, shape [num_tokens], dtype=torch.int32
            - final_slot_mapping[i] >= 0: final slot for accepted token
            - final_slot_mapping[i] == -1: rejected token
        tile_size: Tile size for head dimension, used for Triton kernel optimization, default 64

    Returns:
        None (modifies key_cache and value_cache in-place)

    Example:
        >>> # Assume 4 speculated tokens, at slots [10, 11, 12, 13]
        >>> slot_mapping = torch.tensor([10, 11, 12, 13], device='cuda')
        >>> # Only tokens 0, 2 are accepted, moved to consecutive positions [20, 21]
        >>> final_slot_mapping = torch.tensor([20, -1, 21, -1], device='cuda')
        >>> reorder_kv_cache(key_cache, value_cache, slot_mapping, final_slot_mapping)
        >>> # Now slot 20 contains data from original slot 10
        >>> # Now slot 21 contains data from original slot 12

    Performance:
        - Uses Triton kernels for efficient parallel execution on GPU
        - Two-stage operation requires 2x memory reads/writes + temporary buffer
        - Temporary buffer size = num_tokens * num_kv_heads * head_size * sizeof(dtype)
    """
    
    final_slot_mapping = torch.where(
        slot_mapping_map > 0,
        slot_mapping[torch.clamp(slot_mapping_map, 0)], -1)

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
    """
    Test the correctness of KV cache reordering.

    This test verifies whether the reordering operation correctly moves
    accepted token data to target positions while maintaining data integrity.

    Test scenario:
        - Create random Key/Value cache data
        - Set slot_mapping representing original positions of 15 tokens
        - Set final_slot_mapping simulating acceptance/rejection results:
            - tokens 0, 1, 3, 8 are accepted (final_slot >= 0)
            - other tokens are rejected (final_slot == -1)
        - Verify after reordering that accepted token data is correctly migrated

    Verification method:
        Filter accepted tokens via mask, then compare:
        - Data at slot_mapping positions in original cache
        - Data at final_slot_mapping positions in reordered cache
        These should be identical (allowing for floating point errors)

    How to run:
        python -m vllm.v1.attention.ops.triton_reorder_kv_cache

    Or in code:
        from vllm.v1.attention.ops.triton_reorder_kv_cache import test_reorder_kv_cache
        test_reorder_kv_cache()
    """
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
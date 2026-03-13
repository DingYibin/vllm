# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from dataclasses import replace

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1.outputs import LogprobsLists, LogprobsTensors, SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.ops.bad_words import apply_bad_words_with_drafts
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
from vllm.v1.sample.rejection_sampler import (
    RejectionSampler,
    expand_batch_to_tokens,
    PLACEHOLDER_TOKEN_ID,
    GREEDY_TEMPERATURE,
    MAX_SPEC_LEN,
)
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

logger = init_logger(__name__)

GREEDY_RANDOM_TEMPERATURE: tl.constexpr = 1e-4


class TreeSimpleValidator(RejectionSampler):
    """Rejection sampler for tree-based speculative decoding.

    This class extends RejectionSampler to handle tree-structured draft tokens
    where multiple branches are explored simultaneously. The key differences
    from linear rejection sampling are:

    1. Draft tokens form a tree structure with parent-child relationships.
    2. If a parent node is rejected, all its children are also rejected.
    3. We find the longest accepted path among all branches.

    The tree structure is defined by:
    - tree_choices: list of paths from root to each node
    - tree_father: parent index for each node (-1 for root)
    """

    def forward(
        self,
        metadata: SpecDecodeMetadata,
        # [num_tokens, vocab_size]
        draft_probs: torch.Tensor | None,
        # [num_tokens + batch_size, vocab_size]
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput:
        """
        Args:
            metadata: Metadata for tree-based spec decoding.
            draft_probs: Probability distribution for draft tokens.
                Shape is [num_tokens, vocab_size]. Can be None for ngram spec.
            logits: Target model's logits distribution.
                Shape is [num_tokens + batch_size, vocab_size].
            sampling_metadata: Sampling parameters.

        Returns:
            SamplerOutput with sampled token IDs and optional logprobs.
        """
        assert metadata.max_spec_len <= MAX_SPEC_LEN
        assert metadata.is_tree_mode, "TreeSimpleValidator requires tree mode metadata"

        batch_size = metadata.cu_num_draft_tokens.shape[0]
        device = logits.device

        # Compute cumulative indices for all tokens (draft + bonus)
        # Layout: [draft_req0..., bonus_req0, draft_req1..., bonus_req1, ...]
        # Each request has num_draft_tokens[i] draft tokens + 1 bonus token
        cu_num_sampled_tokens = metadata.cu_num_draft_tokens + torch.arange(
            batch_size, dtype=torch.int32, device=device
        ) + 1

        # Process all logits at once (draft + bonus) with sampling constraints
        raw_logits = logits.to(torch.float32)
        processed_logits = raw_logits.clone() if not self.is_processed_logprobs_mode else raw_logits
        processed_logits = apply_sampling_constraints(
            processed_logits,
            cu_num_sampled_tokens,
            sampling_metadata,
        )

        probs = processed_logits.exp()
        sampled_token_ids = sample_all_tokens(
            metadata.num_draft_tokens,
            probs,
            sampling_metadata.generators,
            device,
        )

        # Perform tree-based rejection sampling
        # Returns:
        #   - output_token_ids: accepted token IDs from the longest accepted path
        #   - slot_mapping_map: mapping from output positions to original slot indices
        #     (used for KV cache reordering when speculation tree doesn't match final sequence)
        output_token_ids, slot_mapping_map = tree_simple_validate(
            metadata.key_token_ids,
            sampled_token_ids,
            cu_num_sampled_tokens,
            metadata.tree_father,
            metadata.max_spec_len + 1,
        )

        logprobs_tensors = None
        # TODO: Add logprobs computation for tree-based rejection sampling
        # The implementation should:
        # 1. Gather logits for accepted tokens based on final_slots_mapping
        # 2. Compute logprobs using sampler.compute_logprobs()
        # 3. Use sampler.gather_logprobs() to get top-k logprobs for each token
        # 4. Filter out rejected tokens (marked with -1) in the output
        # Note: This requires handling tree structure where rejected branches
        # need to be properly filtered from the logprobs output
        if sampling_metadata.max_num_logprobs is not None:
            # TODO: Implement _get_logprobs_tensors for tree mode
            # logprobs_tensors = self._get_logprobs_tensors(
            #     sampling_metadata.max_num_logprobs,
            #     metadata,
            #     logits,
            #     processed_logits,
            #     output_token_ids,
            #     final_slots_mapping,
            # )
            pass

        return SamplerOutput(
            sampled_token_ids=output_token_ids,
            logprobs_tensors=logprobs_tensors,
            slot_mapping_map=slot_mapping_map,
        )

def sample_all_tokens(
    num_draft_tokens: list[int],
    probs: torch.Tensor,
    generators: dict[int, torch.Generator],
    device: torch.device,
) -> torch.Tensor:
    """Sample recovered tokens for tree-based rejection sampling.

    Uses the same algorithm as linear mode but processes all tree nodes.
    """
    num_tokens, vocab_size = probs.shape
    num_tokens_list = [i + 1 for i in num_draft_tokens]
    cu_num_tokens = torch.cumsum(torch.tensor([0] + num_tokens_list), 0)

    # Create one distribution per request (not per token)
    q = torch.empty(
        (num_tokens, vocab_size),
        dtype=torch.float32,
        device=device,
    )
    q.exponential_()
    for i, generator in generators.items():
        if generator is not None:
            q[cu_num_tokens[i]:cu_num_tokens[i + 1]].exponential_(generator=generator)

    return probs.div_(q).argmax(dim=-1).view(-1)


def apply_sampling_constraints(
    logits: torch.Tensor,  # [num_tokens, vocab_size]
    cu_num_tokens: torch.Tensor,  # [batch_size]
    sampling_metadata: SamplingMetadata,
) -> torch.Tensor:
    """Process logits based on sampling metadata.

    This function applies temperature scaling to the logits,
    as well as top-k and top-p. For greedy decoding, it returns
    the original logits.

    Args:
        logits: Input logits tensor to be processed.
        cu_num_draft_tokens: Cumulative number of draft tokens.
        sampling_metadata: Metadata containing sampling parameters such as
            temperature and whether greedy sampling is used.

    Returns:
        torch.Tensor: Processed logits if non-greedy sampling is used,
        otherwise returns the original logits.
    """
    assert logits.ndim == 2
    assert cu_num_tokens.ndim == 1
    if sampling_metadata.all_greedy:
        return logits

    num_tokens = logits.shape[0]
    temperature = expand_batch_to_tokens(
        sampling_metadata.temperature,
        cu_num_tokens,
        num_tokens,
        replace_from=GREEDY_TEMPERATURE,
        replace_to=GREEDY_RANDOM_TEMPERATURE,
    )
    # NOTE(woosuk): Update `logits` in place to avoid allocating a new tensor.
    logits.div_(temperature.unsqueeze(-1))

    # Get expanded top_k and top_p tensors.
    top_k = None
    if sampling_metadata.top_k is not None:
        top_k = expand_batch_to_tokens(
            sampling_metadata.top_k,
            cu_num_tokens,
            num_tokens,
        )
    top_p = None
    if sampling_metadata.top_p is not None:
        top_p = expand_batch_to_tokens(
            sampling_metadata.top_p,
            cu_num_tokens,
            num_tokens,
        )

    # NOTE(woosuk): `apply_top_k_top_p` uses sorting to calculate the mask,
    # which is slow for large vocab sizes. This may cause performance issues.
    return apply_top_k_top_p(logits, top_k, top_p)

def tree_simple_validate(
    key_token_ids,
    sampled_token_ids,
    cu_num_sampled_tokens,
    tree_father,
    max_sampled_len,
) -> torch.Tensor:
    """Perform tree-based rejection sampling via Triton kernel.

    This function prepares input tensors and launches the kernel for each
    request in the batch.

    Args:
        input_ids: [num_tokens] Input token IDs (original draft tokens)
        sampled_token_ids: [num_tokens] Sampled token IDs from target model
        cu_num_sampled_tokens: [batch_size] Cumulative token counts per request
        tree_father: [num_tokens] Parent index for each token
        max_sampled_len: Maximum number of tokens per request

    Returns:
        Tuple of (output_ids, slot_mapping_map):
        - output_ids: [batch_size, max_sampled_len] Accepted token IDs
        - slot_mapping_map: [num_tokens] Slot mapping for accepted tokens
    """
    device = key_token_ids.device
    batch_size = cu_num_sampled_tokens.shape[0]

    # Build token range array: [start_idx for each request]
    # num_tokens_range[i] = starting index of tokens for request i
    num_tokens_range = torch.zeros(batch_size + 1,
                                   dtype=cu_num_sampled_tokens.dtype,
                                   device=device)
    num_tokens_range[1:] = cu_num_sampled_tokens

    # Launch one kernel instance per request
    grid = (batch_size,)

    # Allocate output buffer for accepted token IDs
    output_ids = torch.full(
        (batch_size, max_sampled_len), -1,
        dtype=key_token_ids.dtype, device=device,
    )

    slot_mapping_map = torch.full_like(key_token_ids, -1)

    # Launch kernel
    tree_simple_validate_kernel[grid](
        output_ids,
        slot_mapping_map,
        key_token_ids,
        sampled_token_ids,
        num_tokens_range,
        tree_father,
        triton.next_power_of_2(max_sampled_len),
    )

    return output_ids, slot_mapping_map

@triton.jit(do_not_specialize=["max_sampled_len"])
def tree_simple_validate_kernel(
    output_ids_ptr,  # [batch_size, max_sampled_len] Output buffer for accepted token IDs
    slot_mapping_map_ptr,  # [num_tokens] Slot mapping for KV cache reordering
    key_token_ids_ptr,  # [num_tokens] Input token IDs (original draft tokens to validate)
    sampled_token_ids_ptr,  # [num_tokens] Sampled token IDs from target model
    num_tokens_range_ptr,  # [batch_size + 1] Cumulative token counts, start index per request
    tree_father_ptr,  # [num_tokens] Parent index for each token (-1 for root)
    max_sampled_len: tl.constexpr,  # Max tokens per request (compile-time constant)
):
    """Triton kernel for tree-based rejection sampling.

    This kernel validates draft tokens following a tree structure:
    - Each node represents a draft token with a parent-child relationship
    - A token is accepted only if its parent is accepted AND key_token_id matches
      the parent's sampled token
    - The kernel finds the longest accepted path and outputs those tokens

    Each program instance processes one request independently.

    Args:
        output_ids_ptr: Output buffer [batch_size, max_sampled_len] for accepted token IDs
        slot_mapping_map_ptr: Output buffer [num_tokens] for slot mapping
        key_token_ids_ptr: Input token IDs that were supposed to be sampled
        sampled_token_ids_ptr: Actually sampled token IDs from target model
        num_tokens_range_ptr: Cumulative token counts, num_tokens_range_ptr[i] = start index
        tree_father_ptr: Parent index for each token defining tree structure
        max_sampled_len: Maximum tokens per request (compile-time constant)
    """
    # Get request index this program instance handles
    req_idx = tl.program_id(0)

    # Load token range [start_idx, end_idx) for this request
    start_idx = tl.load(num_tokens_range_ptr + req_idx)
    end_idx = tl.load(num_tokens_range_ptr + req_idx + 1)
    num_tokens = end_idx - start_idx

    # Early exit for empty requests
    if num_tokens == 0:
        return

    # Create offset range for vectorized loads
    # Use max_sampled_len as the vectorization width (compile-time constant)
    offsets = tl.arange(0, max_sampled_len)

    # Mask for valid token positions within this request
    valid_mask = offsets < num_tokens

    # Load tree structure: parent index for each token
    # parent[i] = index of parent node relative to start_idx, -1 for root
    parents = tl.load(tree_father_ptr + start_idx + offsets,
                      mask=valid_mask, other=0)

    # Load input token IDs (original draft tokens that were expected)
    key_token_ids = tl.load(key_token_ids_ptr + start_idx + offsets,
                            mask=valid_mask, other=-1)
    accepted_len = tl.zeros([max_sampled_len], dtype=tl.int32)
    accepted_len = tl.where(offsets == 0, 1, accepted_len)
    for pi in range(max_sampled_len):
        if pi < num_tokens:
            sampled_token = tl.load(sampled_token_ids_ptr + start_idx + pi)
            now_len = tl.sum(tl.where(offsets == pi, accepted_len, 0))
            accepted = ((key_token_ids == sampled_token)
                        & (now_len > 0)
                        & (parents == pi))
            new_len = now_len + 1
            accepted_len = tl.where(accepted, new_len, accepted)


    # Track the longest accepted path
    max_len, max_end_idx = tl.max(accepted_len, return_indices=True)
    # Backtrack from the longest path end to collect accepted tokens
    # Initialize output buffers with -1 (placeholder for rejected/unused)
    output_ids = tl.full([max_sampled_len], -1, dtype=tl.int32)
    slot_mapping_map = tl.full([max_sampled_len], -1, dtype=tl.int32)

    # Fill output buffers by backtracking from max_end_idx to root
    curr_idx = max_end_idx
    for i in range(max_sampled_len):  # Iterate from max depth down to 0
        if curr_idx != -1:
            # Store the sampled token at this position
            token = tl.load(sampled_token_ids_ptr + start_idx + curr_idx)
            output_ids = tl.where(offsets == curr_idx, token, output_ids)

            # Slot mapping: position i in output comes from slot start_idx + i
            slot_mapping_map = tl.where(offsets == curr_idx, start_idx + max_len - 1 - i, slot_mapping_map)

            # Move to parent for next iteration
            curr_idx = tl.load(tree_father_ptr + start_idx + curr_idx)

    # Write results to global memory
    # Output IDs are written per-request row
    tl.store(output_ids_ptr + req_idx * max_sampled_len + offsets,
             output_ids, mask=offsets < max_sampled_len)

    # Slot mapping is written to the global token position
    tl.store(slot_mapping_map_ptr + start_idx + offsets,
             slot_mapping_map, mask=valid_mask)


def test_tree_simple_validate():
    """Test the tree_simple_validate kernel.

    This test verifies the correctness of tree-based rejection sampling.

    Test scenario:
        We create a simple tree structure for 2 requests:
        - Request 0: 4 tokens forming a tree with 2 branches
            Tree structure:
                0 (root)
               / \\
              1   2
             /
            3
            tree_father: [0, 0, 0, 1] (relative indices within request)

        - Request 1: 3 tokens forming a linear chain
            Tree structure:
                0 (root)
               /
              1
             /
            2
            tree_father: [0, 0, 1]

    How to run:
        python -m vllm.v1.sample.tree_rejection_sampler

    Or in code:
        from vllm.v1.sample.tree_rejection_sampler import test_tree_simple_validate
        test_tree_simple_validate()
    """
    import torch

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("Warning: Running on CPU, Triton kernels may not work correctly")

    # Test case 1: Simple tree with branching
    # Request 0: 4 tokens
    # Request 1: 3 tokens
    batch_size = 2
    max_sampled_len = 4

    # Key token IDs: what the draft model predicted
    # Request 0: [100, 101, 102, 103] (4 tokens)
    # Request 1: [200, 201, 202] (3 tokens)
    key_token_ids = torch.tensor([100, 101, 102, 103, 200, 201, 202],
                                  dtype=torch.int32, device=device)

    # Sampled token IDs: what the target model sampled
    # For Request 0: match at position 0, 1, 3 -> accepted path is 0->1->3
    # For Request 1: all match -> accepted path is 0->1->2
    sampled_token_ids = torch.tensor([101, 103, 104, 105, 202, 203, 203],
                                      dtype=torch.int32, device=device)

    # Tree father: parent index for each token (relative to request start)
    # Request 0: [0, 0, 0, 1] -> token 0 is root, token 1 & 2 are children of 0,
    #                            token 3 is child of 1
    # Request 1: [0, 0, 1] -> linear chain
    # Note: tree_father values are global indices
    tree_father = torch.tensor([-1, 0, 0, 1, -1, 0, 0],
                               dtype=torch.int32, device=device)

    # Cumulative token counts per request
    # Request 0: tokens 0-3 (4 tokens)
    # Request 1: tokens 4-6 (3 tokens)
    cu_num_sampled_tokens = torch.tensor([4, 7], dtype=torch.int32, device=device)

    # Build num_tokens_range (cumulative start indices)
    num_tokens_range = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    num_tokens_range[1:] = cu_num_sampled_tokens


    # Call the kernel
    output_token_ids, result_slot_mapping = tree_simple_validate(
        key_token_ids,
        sampled_token_ids,
        cu_num_sampled_tokens,
        tree_father,
        max_sampled_len,
    )

    print("Test Results:")
    print(f"key_token_ids: {key_token_ids}")
    print(f"sampled_token_ids: {sampled_token_ids}")
    print(f"tree_father: {tree_father}")
    print(f"output_token_ids: {output_token_ids}")
    print(f"slot_mapping_map: {result_slot_mapping}")

    # Expected results:
    # Request 0: tokens 0, 1, 3 are accepted (path: 0->1->3, token 2 rejected)
    #   output: [100, 101, 103, -1]
    # Request 1: all tokens accepted
    #   output: [200, 201, 202, -1]


    return output_token_ids, result_slot_mapping


if __name__ == "__main__":
    test_tree_simple_validate()

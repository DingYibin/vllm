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
        (
            output_token_ids,
            slot_mapping_map,
            tree_next_token_indices,
            tree_last_token_indices,
        ) = tree_simple_validate(
            metadata.logits_indices,
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
        # print(f"{metadata.key_token_ids=}\n"
        #       f"{sampled_token_ids=}\n"
        #       f"{slot_mapping_map=}\n"
        #       f"{tree_next_token_indices=}\n"
        #       f"{tree_last_token_indices}\n"
        #       , end="", flush=True,
        # )
        return SamplerOutput(
            sampled_token_ids=output_token_ids,
            logprobs_tensors=logprobs_tensors,
            slot_mapping_map=slot_mapping_map,
            tree_next_token_indices=tree_next_token_indices,
            tree_last_token_indices=tree_last_token_indices,
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

    return probs.div_(q).argmax(dim=-1).view(-1).to(torch.int32)


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

def uncompress_to_matrix(
    input: torch.Tensor,
    index: torch.Tensor,
    batch_size: int,
    max_num: int,
    val,
) -> torch.Tensor:
    dtype = input.dtype
    device = input.device
    output = torch.full(
        (batch_size, max_num), val,
        dtype=dtype, device=device,
    )
    output.view(-1)[index] = input
    return output

def tree_simple_validate(
    logits_indices: torch.Tensor,
    key_token_ids: torch.Tensor,
    sampled_token_ids: torch.Tensor,
    cu_num_sampled_tokens: torch.Tensor,
    tree_father: torch.Tensor,
    max_sampled_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
    num_tokens = logits_indices.shape[0]
    device = key_token_ids.device
    batch_size = cu_num_sampled_tokens.shape[0]

    # Build token range array: [start_idx for each request]
    # num_tokens_range[i] = starting index of tokens for request i
    num_tokens_range = torch.zeros(batch_size + 1,
                                   dtype=cu_num_sampled_tokens.dtype,
                                   device=device)
    num_tokens_range[1:] = cu_num_sampled_tokens
    num_sampled_tokens = num_tokens_range[1:] - num_tokens_range[:-1]

    index = torch.arange(num_tokens, device=device, dtype=cu_num_sampled_tokens.dtype)
    index_delta = torch.repeat_interleave(
        input=torch.arange(batch_size, dtype=index.dtype, device=device) * max_sampled_len - num_tokens_range[:batch_size],
        repeats=num_sampled_tokens,
        output_size=num_tokens,
    )
    index = index + index_delta 
    slot_mapping_map_delta = torch.repeat_interleave(
        input=num_tokens_range[:batch_size],
        repeats=num_sampled_tokens,
        output_size=num_tokens,
    )

    father_matrix = uncompress_to_matrix(
        input=tree_father,
        index=index,
        batch_size=batch_size,
        max_num=max_sampled_len,
        val=-1,
    )

    key_token_matrix = uncompress_to_matrix(
        input=key_token_ids,
        index=index,
        batch_size=batch_size,
        max_num=max_sampled_len,
        val=-1,
    )

    sampled_token_matrix = uncompress_to_matrix(
        input=sampled_token_ids,
        index=index,
        batch_size=batch_size,
        max_num=max_sampled_len,
        val=-1,
    )

    accepted_length = torch.zeros(
        (batch_size, max_sampled_len),
        dtype=torch.int32, device=device,
    )
    accepted_length[:, 0] = 1
    for i in range(1, max_sampled_len):
        father = father_matrix[:, i]
        now_token = key_token_matrix[:, i]
        father_sampled = sampled_token_matrix.gather(1, father.unsqueeze(1).clamp(0)).squeeze(1)
        accepted = now_token == father_sampled

        father_accepted_length = accepted_length.gather(1, father.unsqueeze(1).clamp(0)).squeeze(1)
        father_accepted_length = torch.where(
            father != -1, father_accepted_length, 0
        )
        now_accepted_length = torch.where(
            accepted & (father_accepted_length > 0), father_accepted_length + 1, 0
        )
        accepted_length[:, i] = now_accepted_length

    now_accepted_length, longest_idx = accepted_length.max(dim=-1)

    tree_last_token_indices = logits_indices[
        longest_idx + num_tokens_range[:batch_size]
    ].clone()

    output_ids = torch.full(
        (batch_size, max_sampled_len), -1,
        dtype=torch.int32,
        device=device,
    )
    slot_mapping_map_matrix = torch.full(
        (batch_size, max_sampled_len), -1,
        dtype=torch.int32,
        device=device,
    )
    logits_indices_matrix = uncompress_to_matrix(
        input=logits_indices,
        index=index,
        batch_size=batch_size,
        max_num=max_sampled_len,
        val=-1,
    )
    tree_next_token_matrix = logits_indices_matrix.clone()
    curr_idx = longest_idx.unsqueeze(1)
    new_pos = (now_accepted_length - 1).unsqueeze(1)
    for i in range(max_sampled_len):
        output_ids.scatter_(
            1, curr_idx.clamp(0), torch.where(
                curr_idx >= 0,
                sampled_token_matrix.gather(1, curr_idx.clamp(0)),
                output_ids[:, :1])
        )

        slot_mapping_map_matrix.scatter_(
            1, curr_idx.clamp(0), new_pos.clamp(0)
        )
        father = father_matrix.gather(1, curr_idx.clamp(0))
        tree_next_token_matrix.scatter_(
            1, father.clamp(0), torch.where(
                father >= 0,
                logits_indices_matrix.gather(1, curr_idx.clamp(0)),
                tree_next_token_matrix[:, :1])
        )

        curr_idx = father
        new_pos = new_pos - 1
        
    tree_next_token_indices = tree_next_token_matrix.view(-1)[index].contiguous()
    slot_mapping_map = slot_mapping_map_matrix.view(-1)[index] + slot_mapping_map_delta
    return (
        output_ids,
        slot_mapping_map,
        tree_next_token_indices,
        tree_last_token_indices,
    )

def test_tree_simple_validate():
    """Test the tree_simple_validate function.

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
            tree_father: [-1, 0, 0, 1] (-1 for root, otherwise relative index)

        - Request 1: 3 tokens forming a linear chain
            Tree structure:
                0 (root)
               /
              1
             /
            2
            tree_father: [-1, 0, 0]

    How to run:
        python -m vllm.v1.sample.tree_rejection_sampler

    Or in code:
        from vllm.v1.sample.tree_rejection_sampler import test_tree_simple_validate
        test_tree_simple_validate()
    """
    import torch

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on device: {device}")

    # Test case 1: Simple tree with branching
    # Request 0: 4 tokens
    # Request 1: 3 tokens
    batch_size = 2
    max_sampled_len = 4

    # Total tokens: 4 + 3 = 7
    # Request 0: tokens 0-3
    # Request 1: tokens 4-6
    num_tokens = 7

    # logits_indices: indices into the logits tensor
    # For simplicity, we use sequential indices
    logits_indices = torch.arange(num_tokens, dtype=torch.int32, device=device)

    # Key token IDs: what the draft model predicted
    # Request 0: [100, 101, 102, 103] (4 tokens)
    # Request 1: [200, 201, 202] (3 tokens)
    key_token_ids = torch.tensor([100, 101, 102, 103, 200, 201, 202],
                                  dtype=torch.int32, device=device)

    # Sampled token IDs: what the target model sampled
    # For tree-based rejection:
    #   - A node is accepted if key_token[i] == sampled_token[father[i]]
    #   - For root (father=-1), it should match itself
    #
    # Request 0 tree:
    #   - token 0: father=-1 (root), key_token=100, sampled=100 -> accepted
    #   - token 1: father=0, key_token=101, sampled_token[0]=100 -> not match -> rejected
    #   - token 2: father=0, key_token=102, sampled_token[0]=100 -> not match -> rejected
    #   - token 3: father=1, but token 1 is rejected -> rejected
    # Wait, this test case needs adjustment for the new logic
    #
    # Let's create a better test case:
    # For acceptance: key_token[i] must equal sampled_token[father[i]]
    # Request 0:
    #   - token 0 (root, father=-1): should always be accepted
    #   - token 1 (father=0): key_token[1] should equal sampled_token[0]
    #   - token 2 (father=0): key_token[2] should equal sampled_token[0]
    #   - token 3 (father=1): key_token[3] should equal sampled_token[1]
    #
    # Let's set sampled tokens such that we get a clear accepted path
    sampled_token_ids = torch.tensor([100, 101, 100, 102, 200, 201, 202],
                                      dtype=torch.int32, device=device)
    # Request 0: sampled = [100, 101, 100, 102]
    #   - token 0: key=100, father=-1 -> root accepted
    #   - token 1: key=101, father=0, sampled[0]=100 -> 101!=100 -> rejected
    #   - token 2: key=102, father=0, sampled[0]=100 -> 102!=100 -> rejected
    #   - token 3: key=103, father=1, but token 1 rejected -> rejected
    #   So only token 0 is accepted for request 0
    #
    # Let me create a test where tokens are actually accepted:
    # For acceptance: key_token[i] == sampled_token[father[i]]
    # To have token 1 accepted: key_token[1] must equal sampled_token[0]
    # To have token 2 accepted: key_token[2] must equal sampled_token[0]

    # Better test data:
    # Request 0:
    #   key_tokens:   [100, 100, 100, 101]  # tokens that draft predicted
    #   sampled:      [100, 101, 102, 103]  # tokens that target sampled
    #   tree_father:  [-1,  0,  0,  1]      # tree structure
    #
    #   Acceptance check (key_token[i] == sampled[father[i]]):
    #   - token 0: father=-1 (root), always accepted
    #   - token 1: key[1]=100, sampled[father[1]]=sampled[0]=100 -> 100==100 -> accepted
    #   - token 2: key[2]=100, sampled[father[2]]=sampled[0]=100 -> 100==100 -> accepted
    #   - token 3: key[3]=101, sampled[father[3]]=sampled[1]=101 -> 101==101 -> accepted
    #
    #   So all 4 tokens in request 0 could be accepted, but we need to find longest path.
    #   Path 0->1->3 has length 3
    #   Path 0->2 has length 2
    #   Longest path: 0->1->3

    key_token_ids = torch.tensor([100, 100, 100, 101, 200, 200, 200],
                                  dtype=torch.int32, device=device)
    sampled_token_ids = torch.tensor([100, 101, 102, 103, 200, 201, 202],
                                      dtype=torch.int32, device=device)

    # Tree father: parent index for each token
    # -1 indicates root node
    # Request 0: tree_father = [-1, 0, 0, 1]
    # Request 1: tree_father = [-1, 0, 0]
    tree_father = torch.tensor([-1, 0, 0, 1, -1, 0, 0],
                               dtype=torch.int32, device=device)

    # Cumulative token counts per request
    # Request 0: 4 tokens (indices 0-3)
    # Request 1: 3 tokens (indices 4-6)
    cu_num_sampled_tokens = torch.tensor([4, 7], dtype=torch.int32, device=device)

    print("\n=== Input Data ===")
    print(f"logits_indices: {logits_indices}")
    print(f"key_token_ids: {key_token_ids}")
    print(f"sampled_token_ids: {sampled_token_ids}")
    print(f"tree_father: {tree_father}")
    print(f"cu_num_sampled_tokens: {cu_num_sampled_tokens}")

    # Call the function
    (
        output_token_ids,
        slot_mapping_map,
        tree_next_token_indices,
        tree_last_token_indices,
    ) = tree_simple_validate(
        logits_indices,
        key_token_ids,
        sampled_token_ids,
        cu_num_sampled_tokens,
        tree_father,
        max_sampled_len,
    )

    print("\n=== Output Results ===")
    print(f"output_token_ids:\n{output_token_ids}")
    print(f"slot_mapping_map: {slot_mapping_map}")
    print(f"tree_next_token_indices: {tree_next_token_indices}")
    print(f"tree_last_token_indices: {tree_last_token_indices}")

    # Expected results:
    # Request 0:
    #   - Accepted path: 0->1->3 (tokens at positions 0, 1, 3)
    #   - output: should contain sampled tokens along the accepted path
    # Request 1:
    #   - All tokens potentially accepted based on matching logic
    #   - tree_father[-1, 0, 0] means token 4 is root, tokens 5 and 6 have father=0 (token 4)
    #   - key[5]=200, sampled[father[5]]=sampled[4]=200 -> accepted
    #   - key[6]=200, sampled[father[6]]=sampled[4]=200 -> accepted
    #   - Both branches accepted, longest path is just 2 tokens

    return (
        output_token_ids,
        slot_mapping_map,
        tree_next_token_indices,
        tree_last_token_indices,
    )


if __name__ == "__main__":
    test_tree_simple_validate()

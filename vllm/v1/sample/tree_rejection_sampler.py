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

        batch_size = metadata.bonus_logits_indices.shape[0]
        device = logits.device

        # Compute cumulative indices for all tokens (draft + bonus)
        # Layout: [draft_req0..., bonus_req0, draft_req1..., bonus_req1, ...]
        # Each request has num_draft_tokens[i] draft tokens + 1 bonus token
        cu_num_all_tokens = metadata.cu_num_draft_tokens + torch.arange(
            batch_size, dtype=torch.int32, device=device
        ) + 1

        # Process all logits at once (draft + bonus) with sampling constraints
        raw_logits = logits.to(torch.float32)
        processed_logits = raw_logits.clone() if not self.is_processed_logprobs_mode else raw_logits
        processed_logits = apply_sampling_constraints(
            processed_logits,
            cu_num_all_tokens,
            sampling_metadata,
        )

        # Extract target and bonus logits from processed logits
        target_logits = processed_logits[metadata.target_logits_indices]
        bonus_logits = processed_logits[metadata.bonus_logits_indices]

        # Apply logits processors to target logits only
        # (bad words, penalties, etc. require draft tokens to be contiguous)
        target_logits = self.apply_logits_processors(
            target_logits, sampling_metadata, metadata
        )

        # Sample bonus tokens
        bonus_sampler_output = self.sampler(
            logits=bonus_logits,
            sampling_metadata=replace(
                sampling_metadata,
                max_num_logprobs=-1,
            ),
            predict_bonus_token=True,
            logprobs_mode_override="processed_logits"
            if self.is_processed_logprobs_mode
            else "raw_logits",
        )
        bonus_token_ids = bonus_sampler_output.sampled_token_ids

        # Perform tree-based rejection sampling
        output_token_ids = tree_rejection_sample(
            metadata.draft_token_ids,
            metadata.num_draft_tokens,
            metadata.max_spec_len,
            metadata.cu_num_draft_tokens,
            draft_probs,
            target_logits,
            bonus_token_ids,
            sampling_metadata,
            metadata.tree_father,
            metadata.tree_choices,
            metadata.num_tree_nodes,
        )

        logprobs_tensors = None
        if sampling_metadata.max_num_logprobs is not None:
            # For logprobs, use raw logits for target if not processed mode
            raw_target_logits = raw_logits[metadata.target_logits_indices]
            logprobs_tensors = self._get_logprobs_tensors(
                sampling_metadata.max_num_logprobs,
                metadata,
                target_logits if self.is_processed_logprobs_mode else raw_target_logits,
                bonus_sampler_output.logprobs_tensors.logprobs,
                output_token_ids,
            )

        return SamplerOutput(
            sampled_token_ids=output_token_ids,
            logprobs_tensors=logprobs_tensors,
        )

    def _get_logprobs_tensors(
        self,
        max_num_logprobs: int,
        metadata: SpecDecodeMetadata,
        target_logits: torch.Tensor,
        bonus_logits: torch.Tensor,
        sampled_token_ids: torch.Tensor,
    ) -> LogprobsTensors:
        """Compute logprobs for tree-based sampling.

        For tree mode, we compute logprobs for the longest accepted path.
        """
        cu_num_sampled_tokens = torch.zeros_like(metadata.cu_num_sampled_tokens)
        cu_num_sampled_tokens[1:] = metadata.cu_num_sampled_tokens[:-1]

        # Collect target and bonus logits
        num_tokens = metadata.draft_token_ids.shape[0]
        batch_size = metadata.bonus_logits_indices.shape[0]
        vocab_size = target_logits.shape[-1]

        final_logits = torch.zeros(
            (num_tokens + batch_size, vocab_size),
            dtype=torch.float32,
            device=target_logits.device,
        )
        final_logits[metadata.target_logits_indices] = target_logits.to(torch.float32)
        final_logits[metadata.bonus_logits_indices] = bonus_logits.to(torch.float32)

        logit_start_indices = cu_num_sampled_tokens
        offsets = torch.arange(
            sampled_token_ids.shape[-1],
            device=logit_start_indices.device,
            dtype=logit_start_indices.dtype,
        )
        accepted_logit_indices = (
            logit_start_indices.unsqueeze(1) + offsets.unsqueeze(0)
        ).flatten()
        accepted_logit_indices.clamp_(max=final_logits.shape[0] - 1)
        accepted_tokens = sampled_token_ids.clone().flatten()
        accepted_tokens[accepted_tokens == PLACEHOLDER_TOKEN_ID] = 0

        accepted_logits = final_logits[accepted_logit_indices]
        accepted_logprobs = (
            accepted_logits
            if self.is_logits_logprobs_mode
            else self.sampler.compute_logprobs(accepted_logits)
        )
        return self.sampler.gather_logprobs(
            accepted_logprobs,
            max_num_logprobs,
            accepted_tokens.to(torch.int64),
        )

    @staticmethod
    def parse_output(
        output_token_ids: torch.Tensor,
        vocab_size: int,
        discard_req_indices: Sequence[int] = (),
        logprobs_tensors: LogprobsTensors | None = None,
    ) -> tuple[list[list[int]], LogprobsLists | None]:
        """Parse the output of tree rejection sampler.

        This is identical to the linear rejection sampler's parse_output,
        as the output format is the same.
        """
        output_token_ids_np = output_token_ids.cpu().numpy()
        valid_mask = (output_token_ids_np != PLACEHOLDER_TOKEN_ID) & (
            output_token_ids_np < vocab_size
        )
        output_logprobs = None
        if logprobs_tensors is not None:
            cu_num_tokens = [0] + valid_mask.sum(axis=1).cumsum().tolist()
            filtered_tensors = logprobs_tensors.filter(valid_mask.flatten())
            output_logprobs = filtered_tensors.tolists(cu_num_tokens)

        if len(discard_req_indices) > 0:
            valid_mask[discard_req_indices] = False
        outputs = [
            row[valid_mask[i]].tolist() for i, row in enumerate(output_token_ids_np)
        ]
        return outputs, output_logprobs

    def apply_logits_processors(
        self,
        logits: torch.Tensor,  # [num_tokens, vocab_size] - target logits only
        sampling_metadata: SamplingMetadata,
        metadata: SpecDecodeMetadata,
    ) -> torch.Tensor:
        """Apply logits processors to target logits.

        This handles penalties, allowed token ids, and bad words filtering
        for draft tokens only.
        """
        has_penalties = not sampling_metadata.no_penalties
        any_penalties_or_bad_words = (
            sampling_metadata.bad_words_token_ids or has_penalties
        )

        output_token_ids = sampling_metadata.output_token_ids
        if any_penalties_or_bad_words:
            output_token_ids = self._combine_outputs_with_spec_tokens(
                output_token_ids,
                sampling_metadata.spec_token_ids,
            )

        if sampling_metadata.allowed_token_ids_mask is not None or has_penalties:
            num_requests = len(sampling_metadata.output_token_ids)
            num_draft_tokens = torch.tensor(metadata.num_draft_tokens, device="cpu")
            original_indices = torch.arange(num_requests, device="cpu")
            repeat_indices_cpu = original_indices.repeat_interleave(num_draft_tokens)
            repeat_indices = repeat_indices_cpu.to(
                device=logits.device, non_blocking=True
            )
            logits = self.apply_penalties(
                logits, sampling_metadata, metadata, repeat_indices, output_token_ids
            )

            if sampling_metadata.allowed_token_ids_mask is not None:
                token_mask = sampling_metadata.allowed_token_ids_mask[repeat_indices]
                logits.masked_fill_(token_mask, float("-inf"))

        if bad_words_token_ids := sampling_metadata.bad_words_token_ids:
            apply_bad_words_with_drafts(
                logits, bad_words_token_ids, output_token_ids, metadata.num_draft_tokens
            )

        return logits


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


def tree_rejection_sample(
    # [num_tokens]
    draft_token_ids: torch.Tensor,
    # [batch_size]
    num_draft_tokens: list[int],
    max_spec_len: int,
    # [batch_size]
    cu_num_draft_tokens: torch.Tensor,
    # [num_tokens, vocab_size]
    draft_probs: torch.Tensor | None,
    # [num_tokens, vocab_size]
    target_logits: torch.Tensor,
    # [batch_size, 1]
    bonus_token_ids: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    # [num_tree_nodes] - parent index for each node, -1 for root
    tree_father: torch.Tensor | None,
    # list of paths from root to each node
    tree_choices: list[tuple[int, ...]] | None,
    # number of tree nodes per request
    num_tree_nodes: int | None,
) -> torch.Tensor:
    """Perform tree-based rejection sampling.

    This function validates all branches in the tree simultaneously and
    selects the longest accepted path. The key insight is that if a parent
    node is rejected, all its children are implicitly rejected.

    Args:
        draft_token_ids: Draft token IDs from the proposer [num_tokens]
        num_draft_tokens: Number of draft tokens per request [batch_size]
        max_spec_len: Maximum speculative length
        cu_num_draft_tokens: Cumulative number of draft tokens [batch_size]
        draft_probs: Draft probabilities [num_tokens, vocab_size] or None
        target_logits: Target logits [num_tokens, vocab_size]
        bonus_token_ids: Bonus token IDs [batch_size, 1]
        sampling_metadata: Sampling metadata
        tree_father: Parent indices for tree nodes [num_tree_nodes]
        tree_choices: List of paths from root to each node
        num_tree_nodes: Number of tree nodes

    Returns:
        Output token IDs [batch_size, max_spec_len + 1]
    """
    assert draft_token_ids.ndim == 1
    assert draft_probs is None or draft_probs.ndim == 2
    assert cu_num_draft_tokens.ndim == 1
    assert target_logits.ndim == 2
    assert tree_father is not None, "tree_father is required for tree mode"
    assert tree_choices is not None, "tree_choices is required for tree mode"

    batch_size = len(num_draft_tokens)
    num_tokens = draft_token_ids.shape[0]
    vocab_size = target_logits.shape[-1]
    device = target_logits.device
    assert draft_token_ids.is_contiguous()
    assert draft_probs is None or draft_probs.is_contiguous()
    assert bonus_token_ids.is_contiguous()
    assert target_logits.shape == (num_tokens, vocab_size)

    # Create output buffer
    output_token_ids = torch.full(
        (batch_size, max_spec_len + 1),
        PLACEHOLDER_TOKEN_ID,
        dtype=torch.int32,
        device=device,
    )

    # For simplicity in tree mode, we assume batch_size == 1
    # (one request with its tree structure)
    # This is a common pattern in tree speculative decoding
    if batch_size != 1:
        # For batched tree decoding, we need to process each request's tree
        # This requires more complex indexing
        raise NotImplementedError(
            "Batched tree speculative decoding is not yet supported. "
            "Use batch_size=1 for tree mode."
        )

    # Get tree structure for this request
    tree_father_req = tree_father
    num_nodes = num_tree_nodes or len(tree_choices)

    if sampling_metadata.all_greedy:
        is_greedy = None
    else:
        is_greedy = sampling_metadata.temperature == GREEDY_TEMPERATURE

    if not sampling_metadata.all_random:
        # Greedy rejection sampling for tree
        target_argmax = target_logits.argmax(dim=-1)
        tree_rejection_greedy_sample_kernel[(batch_size,)](
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            target_argmax,
            bonus_token_ids,
            is_greedy,
            tree_father_req,
            max_spec_len,
            num_nodes,
        )
        if sampling_metadata.all_greedy:
            return output_token_ids

    # Random sampling path
    target_probs = target_logits.softmax(dim=-1, dtype=torch.float32)
    assert target_probs.is_contiguous()

    # Generate uniform probabilities for rejection sampling
    uniform_probs = generate_uniform_probs_tree(
        num_tokens,
        num_draft_tokens,
        sampling_metadata.generators,
        device,
    )

    # Sample recovered tokens
    recovered_token_ids = sample_recovered_tokens_tree(
        max_spec_len,
        num_draft_tokens,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        sampling_metadata,
        device,
    )

    # Random rejection sampling for tree
    tree_rejection_random_sample_kernel[(batch_size,)](
        output_token_ids,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        bonus_token_ids,
        recovered_token_ids,
        uniform_probs,
        is_greedy,
        tree_father_req,
        max_spec_len,
        vocab_size,
        num_nodes,
        NO_DRAFT_PROBS=draft_probs is None,
    )

    return output_token_ids


def generate_uniform_probs_tree(
    num_tokens: int,
    num_draft_tokens: list[int],
    generators: dict[int, torch.Generator],
    device: torch.device,
) -> torch.Tensor:
    """Generate uniform probabilities for tree-based rejection sampling."""
    uniform_probs = torch.rand(
        (num_tokens,),
        dtype=torch.float64,
        device=device,
    )
    start_idx = 0
    for req_idx, n in enumerate(num_draft_tokens):
        if n == 0:
            continue
        end_idx = start_idx + n
        generator = generators.get(req_idx)
        if generator is not None:
            uniform_probs[start_idx:end_idx].uniform_(generator=generator)
        start_idx = end_idx
    return uniform_probs


def sample_recovered_tokens_tree(
    max_spec_len: int,
    num_draft_tokens: list[int],
    cu_num_draft_tokens: torch.Tensor,
    draft_token_ids: torch.Tensor,
    draft_probs: torch.Tensor | None,
    target_probs: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    device: torch.device,
) -> torch.Tensor:
    """Sample recovered tokens for tree-based rejection sampling.

    Uses the same algorithm as linear mode but processes all tree nodes.
    """
    batch_size = len(num_draft_tokens)
    vocab_size = target_probs.shape[-1]

    # Create one distribution per request (not per token)
    q = torch.empty(
        (batch_size, vocab_size),
        dtype=torch.float32,
        device=device,
    )
    q.exponential_()
    for i, generator in sampling_metadata.generators.items():
        if num_draft_tokens[i] > 0:
            q[i].exponential_(generator=generator)

    recovered_token_ids = torch.empty_like(draft_token_ids)

    # For tree mode, we use a modified kernel that handles tree structure
    sample_recovered_tokens_tree_kernel[(batch_size, max_spec_len)](
        recovered_token_ids,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        q,
        vocab_size,
        triton.next_power_of_2(vocab_size),
        NO_DRAFT_PROBS=draft_probs is None,
    )
    return recovered_token_ids


# =============================================================================
# Triton Kernels for Tree-based Rejection Sampling
# =============================================================================

@triton.jit(do_not_specialize=["max_spec_len", "num_nodes"])
def tree_rejection_greedy_sample_kernel(
    output_token_ids_ptr,  # [batch_size, max_spec_len + 1]
    cu_num_draft_tokens_ptr,  # [batch_size]
    draft_token_ids_ptr,  # [num_tokens]
    target_argmax_ptr,  # [num_tokens]
    bonus_token_ids_ptr,  # [batch_size]
    is_greedy_ptr,  # [batch_size] or None
    tree_father_ptr,  # [num_tree_nodes] - parent indices
    max_spec_len,
    num_nodes,
    MAX_SPEC_LEN_CONST: tl.constexpr = 128,  # Max spec length constant
):
    """Greedy rejection sampling kernel for tree structure.

    For each node in the tree:
    1. Check if parent was accepted (root always starts as accepted)
    2. If parent accepted, compare draft_token with target_argmax
    3. Accept if equal, reject if not equal
    4. Find the longest accepted path and output those tokens
    """
    req_idx = tl.program_id(0)

    is_greedy = True if is_greedy_ptr is None else tl.load(is_greedy_ptr + req_idx)
    if not is_greedy:
        return

    start_idx = 0 if req_idx == 0 else tl.load(cu_num_draft_tokens_ptr + req_idx - 1)

    # Track the longest accepted path
    # For tree mode: accept[node] = (draft == target) AND accept[parent]
    longest_accepted_depth = -1
    longest_accepted_node = -1

    # Process nodes in order (tree_choices are typically ordered by depth)
    for node_idx in range(num_nodes):
        node_token_idx = start_idx + node_idx

        # Get parent
        parent_idx = tl.load(tree_father_ptr + node_idx)

        # Check if all ancestors are accepted
        # A node can only be accepted if all its ancestors are accepted
        all_ancestors_accepted = True
        if parent_idx >= 0:
            # Verify parent is on the current longest accepted path
            current = longest_accepted_node
            found_ancestor = False
            while current >= 0:
                if current == parent_idx:
                    found_ancestor = True
                    break
                current = tl.load(tree_father_ptr + current)
            if not found_ancestor and longest_accepted_node >= 0:
                all_ancestors_accepted = False

        if not all_ancestors_accepted:
            continue

        draft_token = tl.load(draft_token_ids_ptr + node_token_idx)
        target_token = tl.load(target_argmax_ptr + node_token_idx)

        # Check acceptance condition
        if draft_token == target_token:
            # This node is accepted
            node_depth = 0
            current = node_idx
            while current >= 0:
                node_depth += 1
                current = tl.load(tree_father_ptr + current)

            if node_depth > longest_accepted_depth:
                longest_accepted_depth = node_depth
                longest_accepted_node = node_idx

    # Output the longest accepted path
    if longest_accepted_node >= 0:
        # Walk from longest_accepted_node back to root, then reverse
        temp_path = tl.zeros([MAX_SPEC_LEN_CONST], dtype=tl.int32)
        temp_idx = 0

        current = longest_accepted_node
        while current >= 0 and temp_idx < MAX_SPEC_LEN_CONST:
            node_token_idx = start_idx + current
            token = tl.load(target_argmax_ptr + node_token_idx)
            temp_path = tl.where(tl.arange(0, MAX_SPEC_LEN_CONST) == temp_idx, token, temp_path)
            temp_idx += 1
            current = tl.load(tree_father_ptr + current)

        # Reverse and store
        for i in range(temp_idx):
            reversed_idx = temp_idx - 1 - i
            token = tl.load(temp_path + reversed_idx)
            tl.store(
                output_token_ids_ptr + req_idx * (max_spec_len + 1) + i,
                token,
            )

        # Add bonus token
        bonus_token = tl.load(bonus_token_ids_ptr + req_idx)
        tl.store(
            output_token_ids_ptr + req_idx * (max_spec_len + 1) + temp_idx,
            bonus_token,
        )
    else:
        # No nodes accepted, use bonus token
        bonus_token = tl.load(bonus_token_ids_ptr + req_idx)
        tl.store(
            output_token_ids_ptr + req_idx * (max_spec_len + 1),
            bonus_token,
        )


@triton.jit(do_not_specialize=["max_spec_len", "num_nodes", "vocab_size"])
def tree_rejection_random_sample_kernel(
    output_token_ids_ptr,  # [batch_size, max_spec_len + 1]
    cu_num_draft_tokens_ptr,  # [batch_size]
    draft_token_ids_ptr,  # [num_tokens]
    draft_probs_ptr,  # [num_tokens, vocab_size] or None
    target_probs_ptr,  # [num_tokens, vocab_size]
    bonus_token_ids_ptr,  # [batch_size]
    recovered_token_ids_ptr,  # [num_tokens]
    uniform_probs_ptr,  # [num_tokens]
    is_greedy_ptr,  # [batch_size]
    tree_father_ptr,  # [num_tree_nodes]
    max_spec_len,
    vocab_size,
    num_nodes,
    NO_DRAFT_PROBS: tl.constexpr,
    MAX_SPEC_LEN_CONST: tl.constexpr = 128,  # Max spec length constant
):
    """Random rejection sampling kernel for tree structure.

    For each node, check acceptance based on probability ratio.
    Track the longest accepted path through the tree.
    """
    req_idx = tl.program_id(0)

    is_greedy = tl.load(is_greedy_ptr + req_idx)
    if is_greedy:
        return

    start_idx = 0 if req_idx == 0 else tl.load(cu_num_draft_tokens_ptr + req_idx - 1)

    # Track the longest accepted path
    longest_accepted_depth = -1
    longest_accepted_node = -1
    # Track which node first rejected (for recovered token)
    first_rejected_node = -1

    for node_idx in range(num_nodes):
        node_token_idx = start_idx + node_idx

        # Get parent
        parent_idx = tl.load(tree_father_ptr + node_idx)

        # Check if all ancestors are accepted
        all_ancestors_accepted = True
        current = parent_idx
        while current >= 0 and all_ancestors_accepted:
            # Check if this ancestor is on the longest path
            if longest_accepted_node >= 0:
                # Check if current is ancestor of longest_accepted_node
                check = longest_accepted_node
                found = False
                while check >= 0:
                    if check == current:
                        found = True
                        break
                    check = tl.load(tree_father_ptr + check)
                if not found:
                    all_ancestors_accepted = False
                    break

            current = tl.load(tree_father_ptr + current)

        if not all_ancestors_accepted:
            continue

        # Now check this node
        draft_token = tl.load(draft_token_ids_ptr + node_token_idx)
        if NO_DRAFT_PROBS:
            draft_prob = 1.0
        else:
            draft_prob = tl.load(
                draft_probs_ptr + node_token_idx * vocab_size + draft_token
            )
        target_prob = tl.load(
            target_probs_ptr + node_token_idx * vocab_size + draft_token
        )
        uniform_prob = tl.load(uniform_probs_ptr + node_token_idx)

        # Compute depth
        node_depth = 0
        current = node_idx
        while current >= 0:
            node_depth += 1
            current = tl.load(tree_father_ptr + current)

        if draft_prob > 0 and target_prob / draft_prob >= uniform_prob:
            # Accept this node
            if node_depth > longest_accepted_depth:
                longest_accepted_depth = node_depth
                longest_accepted_node = node_idx
                first_rejected_node = -1  # Reset rejection
        else:
            # Reject this node
            # The path up to parent is valid, and we use recovered token here
            if node_depth - 1 >= longest_accepted_depth:
                longest_accepted_depth = node_depth - 1
                longest_accepted_node = parent_idx if parent_idx >= 0 else -1
                first_rejected_node = node_idx

    # Output the longest accepted path with tokens
    if longest_accepted_node >= 0 or first_rejected_node >= 0:
        # Collect path tokens
        temp_tokens = tl.zeros([MAX_SPEC_LEN_CONST], dtype=tl.int32)
        temp_idx = 0

        # Build the path from root to longest_accepted_node
        # Plus potentially the rejected node with recovered token
        path_nodes = []
        if longest_accepted_node >= 0:
            current = longest_accepted_node
            while current >= 0:
                path_nodes.append(current)
                current = tl.load(tree_father_ptr + current)
            path_nodes = list(reversed(path_nodes))

        # Add rejected node if exists
        if first_rejected_node >= 0:
            # Check if rejected node's parent is on the path
            rejected_parent = tl.load(tree_father_ptr + first_rejected_node)
            if rejected_parent == longest_accepted_node or longest_accepted_node == -1:
                path_nodes.append(first_rejected_node)

        # Collect tokens for the path
        for _, node_idx in enumerate(path_nodes):
            if temp_idx >= MAX_SPEC_LEN_CONST:
                break
            node_token_idx = start_idx + node_idx

            draft_token = tl.load(draft_token_ids_ptr + node_token_idx)
            if NO_DRAFT_PROBS:
                draft_prob = 1.0
            else:
                draft_prob = tl.load(
                    draft_probs_ptr + node_token_idx * vocab_size + draft_token
                )
            target_prob = tl.load(
                target_probs_ptr + node_token_idx * vocab_size + draft_token
            )
            uniform_prob = tl.load(uniform_probs_ptr + node_token_idx)

            if draft_prob > 0 and target_prob / draft_prob >= uniform_prob:
                token = draft_token
            else:
                token = tl.load(recovered_token_ids_ptr + node_token_idx)

            temp_tokens = tl.where(tl.arange(0, MAX_SPEC_LEN_CONST) == temp_idx, token, temp_tokens)
            temp_idx += 1

        # Store tokens
        for i in range(temp_idx):
            token = tl.load(temp_tokens + i)
            tl.store(
                output_token_ids_ptr + req_idx * (max_spec_len + 1) + i,
                token,
            )

        # Add bonus token only if no rejection occurred
        if first_rejected_node < 0:
            bonus_token = tl.load(bonus_token_ids_ptr + req_idx)
            tl.store(
                output_token_ids_ptr + req_idx * (max_spec_len + 1) + temp_idx,
                bonus_token,
            )
    else:
        # No path accepted, just use bonus token
        bonus_token = tl.load(bonus_token_ids_ptr + req_idx)
        tl.store(
            output_token_ids_ptr + req_idx * (max_spec_len + 1),
            bonus_token,
        )


@triton.jit
def sample_recovered_tokens_tree_kernel(
    output_token_ids_ptr,  # [num_tokens]
    cu_num_draft_tokens_ptr,  # [batch_size]
    draft_token_ids_ptr,  # [num_tokens]
    draft_probs_ptr,  # [num_tokens, vocab_size] or None
    target_probs_ptr,  # [num_tokens, vocab_size]
    q_ptr,  # [batch_size, vocab_size]
    vocab_size,
    PADDED_VOCAB_SIZE: tl.constexpr,
    NO_DRAFT_PROBS: tl.constexpr,
):
    """Sample recovered tokens for tree-based rejection sampling.

    This is similar to the linear version but handles tree nodes.
    """
    req_idx = tl.program_id(0)
    start_idx = 0 if req_idx == 0 else tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
    num_draft_tokens = end_idx - start_idx

    pos = tl.program_id(1)
    if pos >= num_draft_tokens:
        return

    vocab_offset = tl.arange(0, PADDED_VOCAB_SIZE)

    if NO_DRAFT_PROBS:
        draft_token_id = tl.load(draft_token_ids_ptr + start_idx + pos)
        prob = tl.load(
            target_probs_ptr + (start_idx + pos) * vocab_size + vocab_offset,
            mask=((vocab_offset < vocab_size) & (vocab_offset != draft_token_id)),
            other=0,
        )
    else:
        draft_prob = tl.load(
            draft_probs_ptr + (start_idx + pos) * vocab_size + vocab_offset,
            mask=vocab_offset < vocab_size,
            other=0,
        )
        target_prob = tl.load(
            target_probs_ptr + (start_idx + pos) * vocab_size + vocab_offset,
            mask=vocab_offset < vocab_size,
            other=0,
        )
        prob = tl.maximum(target_prob - draft_prob, 0)

    q = tl.load(
        q_ptr + req_idx * vocab_size + vocab_offset,
        mask=vocab_offset < vocab_size,
        other=float("-inf"),
    )
    recovered_id = tl.argmax(prob / q, axis=-1)
    tl.store(output_token_ids_ptr + start_idx + pos, recovered_id)
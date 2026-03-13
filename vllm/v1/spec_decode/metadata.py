# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
import torch


@dataclass
class SpecDecodeMetadata:
    """Metadata for speculative decoding.

    Supports both linear chain and tree-based speculative decoding.

    Linear mode (default):
        Draft tokens form a single chain per request.
        Each request has `num_draft_tokens[i]` sequential draft tokens.
        Rejection sampling proceeds linearly until rejection.

    Tree mode (when tree_choices is not None):
        Draft tokens form a tree structure per request.
        Multiple branches are explored simultaneously.
        Rejection sampling validates all branches and selects the longest
        accepted path.
    """

    # === Common fields (used by both linear and tree mode) ===

    # [num_tokens] Draft token IDs from the proposer
    draft_token_ids: torch.Tensor

    # [batch_size] Number of draft tokens per request
    num_draft_tokens: list[int]

    # [batch_size] Cumulative number of draft tokens
    # cu_num_draft_tokens[i] = sum(num_draft_tokens[0:i+1])
    cu_num_draft_tokens: torch.Tensor

    # [batch_size] Cumulative number of sampled tokens (draft + bonus)
    cu_num_sampled_tokens: torch.Tensor

    # [num_tokens] Indices into target logits for draft tokens
    target_logits_indices: torch.Tensor

    # [batch_size] Indices into target logits for bonus tokens
    bonus_logits_indices: torch.Tensor | None

    # [num_tokens + batch_size] Combined indices for all logits
    logits_indices: torch.Tensor

    # === Tree-mode specific fields ===

    key_token_ids: torch.Tensor | None = None

    # [num_tree_nodes] Parent index for each tree node.
    # -1 indicates the root node (no parent).
    # Used to propagate rejection from parent to children.
    tree_father: torch.Tensor | None = None

    def __post_init__(self):
        self.max_spec_len = max(self.num_draft_tokens)

    @property
    def is_tree_mode(self) -> bool:
        """Check if this metadata represents tree-based speculative decoding."""
        return self.tree_father is not None

    @classmethod
    def make_dummy(
        cls,
        draft_token_ids: list[list[int]],
        device: torch.device,
    ) -> "SpecDecodeMetadata":
        batch_size = len(draft_token_ids)
        num_draft_tokens = [len(ids) for ids in draft_token_ids]
        num_sampled_tokens = [len(ids) + 1 for ids in draft_token_ids]
        flattened_draft_token_ids = sum(draft_token_ids, [])
        num_tokens = len(flattened_draft_token_ids)

        draft_token_ids_tensor = torch.tensor(
            flattened_draft_token_ids, dtype=torch.int32, device=device
        )
        cu_num_draft_tokens = np.cumsum(num_draft_tokens, dtype=np.int32)
        cu_num_draft_tokens_tensor = torch.from_numpy(cu_num_draft_tokens).to(device)
        cu_num_sampled_tokens = np.cumsum(num_sampled_tokens, dtype=np.int32)
        cu_num_sampled_tokens_tensor = torch.from_numpy(cu_num_sampled_tokens).to(
            device
        )

        target_logits_indices = torch.zeros(
            num_tokens, dtype=torch.int32, device=device
        )
        bonus_logits_indices = torch.zeros(batch_size, dtype=torch.int32, device=device)
        logits_indices = torch.zeros(
            num_tokens + batch_size, dtype=torch.int32, device=device
        )
        return cls(
            draft_token_ids=draft_token_ids_tensor,
            num_draft_tokens=num_draft_tokens,
            cu_num_draft_tokens=cu_num_draft_tokens_tensor,
            cu_num_sampled_tokens=cu_num_sampled_tokens_tensor,
            target_logits_indices=target_logits_indices,
            bonus_logits_indices=bonus_logits_indices,
            logits_indices=logits_indices,
        )

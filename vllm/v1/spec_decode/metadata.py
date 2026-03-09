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
    bonus_logits_indices: torch.Tensor

    # [num_tokens + batch_size] Combined indices for all logits
    logits_indices: torch.Tensor

    # === Tree-mode specific fields ===

    input_ids: torch.Tensor | None = None
    # Tree structure definition: list of paths from root to each node.
    # Example: [(0,), (1,), (0, 0), (0, 1), (1, 0), (1, 1)]
    # represents a 2-branch, 2-level tree:
    #         root
    #        /    \
    #      (0)    (1)
    #     /  \    /  \
    #   (0,0)(0,1)(1,0)(1,1)
    tree_choices: list[tuple[int, ...]] | None = None

    # [num_tree_nodes] Parent index for each tree node.
    # -1 indicates the root node (no parent).
    # Used to propagate rejection from parent to children.
    tree_father: torch.Tensor | None = None

    # Cumulative number of draft tokens at each tree level.
    # Used to index into draft_token_ids by level.
    # Example: for a 2-level tree with 2 branches each:
    #   cu_drafts_per_level = [2, 6] (level 0: 2 nodes, total 2; level 1: 4 nodes, total 6)
    cu_drafts_per_level: list[int] | None = None

    # Number of child nodes per parent at each level.
    # Example: [2, 2] means each node at level 0 and 1 has 2 children.
    child_drafts_per_level: list[int] | None = None

    # Number of tree nodes per request (for batched tree decoding).
    # In tree mode, this is len(tree_choices).
    # None for linear mode or when batch_size > 1 with different trees.
    num_tree_nodes: int | None = None

    # Maximum tree depth (number of levels).
    tree_depth: int | None = None

    def __post_init__(self):
        self.max_spec_len = max(self.num_draft_tokens)

        # Compute tree-specific properties if in tree mode
        if self.tree_choices is not None:
            self.num_tree_nodes = len(self.tree_choices)
            self.tree_depth = len(self.tree_choices[-1]) if self.tree_choices else 0

    @property
    def is_tree_mode(self) -> bool:
        """Check if this metadata represents tree-based speculative decoding."""
        return self.tree_choices is not None

    def get_parent_indices(self, node_indices: torch.Tensor) -> torch.Tensor:
        """Get parent indices for given node indices in tree mode.

        Args:
            node_indices: [N] tensor of node indices

        Returns:
            [N] tensor of parent indices, -1 for root nodes
        """
        if self.tree_father is None:
            raise ValueError("tree_father is not set, not in tree mode")

        return self.tree_father[node_indices]

    def get_children_indices(self, parent_idx: int) -> list[int]:
        """Get all children indices for a given parent node.

        Args:
            parent_idx: Index of the parent node

        Returns:
            List of children node indices
        """
        if self.tree_father is None:
            raise ValueError("tree_father is not set, not in tree mode")

        return (self.tree_father == parent_idx).nonzero(as_tuple=True)[0].tolist()

    def get_nodes_at_level(self, level: int) -> list[int]:
        """Get all node indices at a specific tree level.

        Args:
            level: Tree level (0-indexed)

        Returns:
            List of node indices at the given level
        """
        if self.tree_choices is None:
            raise ValueError("tree_choices is not set, not in tree mode")

        return [
            i for i, path in enumerate(self.tree_choices)
            if len(path) - 1 == level
        ]

    def get_path_to_node(self, node_idx: int) -> list[int]:
        """Get the path from root to a node (inclusive).

        Args:
            node_idx: Index of the target node

        Returns:
            List of node indices from root to the target node
        """
        if self.tree_father is None or self.tree_choices is None:
            raise ValueError("Not in tree mode")

        path = [node_idx]
        current = node_idx
        while current >= 0:
            parent = self.tree_father[current].item()
            if parent >= 0:
                path.append(parent)
            current = parent

        return list(reversed(path))

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

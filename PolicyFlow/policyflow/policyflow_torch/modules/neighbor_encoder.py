"""neighbor_encoder.py — Cross-attention encoder for variable-size neighbor sets.

Each drone's own state acts as a query; the N-1 neighbor observations are keys
and values.  Output is always a fixed ``emb_dim`` vector regardless of N, so
the downstream policy network is fully agnostic to the number of agents.

Neighbor observations carry an explicit ``r_min`` dimension (last element) so
that heterogeneous agents — including static or dynamic obstacles modelled as
virtual drones with a large collision radius — can be handled without any
architectural change.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class NeighborEncoder(nn.Module):
    """Cross-attention neighbor encoder: own-state query over variable neighbors.

    Args:
        own_dim:       Dimension of the per-drone own-state slice.
        neighbor_dim:  Dimension of each neighbor's observation (including r_min).
        emb_dim:       Output embedding dimension (fixed regardless of N).
        num_heads:     Number of attention heads.  Must divide ``emb_dim``.
    """

    def __init__(
        self,
        own_dim: int,
        neighbor_dim: int,
        emb_dim: int,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        assert emb_dim % num_heads == 0, "emb_dim must be divisible by num_heads"
        self.own_dim = own_dim
        self.neighbor_dim = neighbor_dim
        self.emb_dim = emb_dim

        self.q_proj = nn.Linear(own_dim, emb_dim)
        self.k_proj = nn.Linear(neighbor_dim, emb_dim)
        self.v_proj = nn.Linear(neighbor_dim, emb_dim)
        self.attn = nn.MultiheadAttention(emb_dim, num_heads, batch_first=True)

        # Learned null key/value used when there are no neighbors (N=1).
        self.null_kv = nn.Parameter(torch.zeros(1, 1, emb_dim))
        nn.init.normal_(self.null_kv, std=0.02)

    def forward(
        self,
        own: torch.Tensor,
        neighbors: torch.Tensor,
    ) -> torch.Tensor:
        """Attend from own state over neighbor observations.

        Args:
            own:       ``(B, own_dim)``
            neighbors: ``(B, N_neigh, neighbor_dim)``, ``N_neigh`` may be 0.

        Returns:
            ``(B, emb_dim)``
        """
        B = own.shape[0]
        q = self.q_proj(own).unsqueeze(1)          # (B, 1, emb_dim)

        if neighbors.shape[1] == 0:
            k = v = self.null_kv.expand(B, 1, self.emb_dim)
        else:
            k = self.k_proj(neighbors)             # (B, N_neigh, emb_dim)
            v = self.v_proj(neighbors)

        attn_out, _ = self.attn(q, k, v)           # (B, 1, emb_dim)
        return attn_out.squeeze(1)                  # (B, emb_dim)

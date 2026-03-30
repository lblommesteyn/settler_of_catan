"""
StaticBoardGCN — fast GCN exploiting that all Catan boards share the same topology.

All 169k training graphs have the same 54-node hex grid. Instead of running PyG
message passing independently per graph (slow on CPU), we:

  1. Precompute the normalised adjacency A_hat = D^{-1/2} A D^{-1/2}  (54×54, float32)
  2. Stack all node feature matrices → tensor shape  (N, 54, node_dim)
  3. Run GCN layers as batched matrix multiplications:
       H' = ReLU( A_hat @ H @ W )
     using torch.einsum or bmm — no PyG overhead at all.

This is ~20-50x faster on CPU vs PyG message passing for large batches.

The tradeoff: no per-edge attention (GAT-style). But for a 54-node fixed graph
with binary adjacency, simple spectral convolution is nearly as expressive.

Architecture
────────────
Input:  (N, 54, node_dim=16)    + seat (N,)
Layer 1: graph conv (node_dim → hidden)  + ReLU + dropout
Layer 2: graph conv (hidden → hidden)    + ReLU + dropout
Layer 3: graph conv (hidden → hidden)    + ReLU + dropout
Readout: v1_emb ⊕ v2_emb ⊕ mean_pool ⊕ seat_emb  → MLP → P(win)
Multi-task: also predicts normalised rank.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

NODE_DIM = 16
SEAT_DIM = 4
N_VERTS  = 54


def build_norm_adj(edge_index: torch.Tensor, n: int = N_VERTS) -> torch.Tensor:
    """
    Build the symmetric normalised adjacency A_hat = D^{-1/2}(A+I)D^{-1/2}
    as a dense (n, n) float32 tensor.
    """
    # A + self-loops
    A = torch.zeros(n, n)
    src, dst = edge_index
    A[src, dst] = 1.0
    A[dst, src] = 1.0
    A.fill_diagonal_(1.0)   # self-loop

    D_inv_sqrt = torch.diag(A.sum(dim=1).pow(-0.5))
    A_hat = D_inv_sqrt @ A @ D_inv_sqrt
    return A_hat.float()


class GraphConvLayer(nn.Module):
    """Single GCN layer: H' = ReLU(A_hat H W)"""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.W    = nn.Linear(in_dim, out_dim, bias=True)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, H: torch.Tensor, A_hat: torch.Tensor) -> torch.Tensor:
        """
        H:     (N, 54, in_dim)
        A_hat: (54, 54)
        returns (N, 54, out_dim)
        """
        # Step 1: linear transform  (N, 54, in_dim) → (N, 54, out_dim)
        HW = self.W(H)
        # Step 2: graph aggregation  A_hat @ HW  ≡ einsum 'ij,njk->nik'
        out = torch.einsum('ij,njk->nik', A_hat, HW)
        return self.norm(out)


class StaticBoardGCN(nn.Module):
    """
    Fast GCN using precomputed normalised adjacency.

    node_feat_matrix: (N, 54, NODE_DIM) tensor — call dataset_to_tensor() to build.
    """

    def __init__(
        self,
        hidden:    int   = 64,
        n_layers:  int   = 3,
        dropout:   float = 0.2,
        multitask: bool  = True,
    ):
        super().__init__()
        self.dropout   = dropout
        self.multitask = multitask

        # GCN layers
        layers = [GraphConvLayer(NODE_DIM, hidden)]
        for _ in range(n_layers - 1):
            layers.append(GraphConvLayer(hidden, hidden))
        self.conv_layers = nn.ModuleList(layers)

        # Seat embedding
        self.seat_emb = nn.Embedding(SEAT_DIM + 1, 8, padding_idx=SEAT_DIM)

        # MLP: v1_emb + v2_emb + global_mean + seat_emb
        mlp_in = hidden * 3 + 8
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64),     nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        if multitask:
            self.rank_head = nn.Sequential(
                nn.Linear(mlp_in, 64), nn.ReLU(),
                nn.Linear(64, 1),      nn.Sigmoid(),
            )

        # Will be set once from data
        self.register_buffer("A_hat", torch.eye(N_VERTS))

    def set_adj(self, A_hat: torch.Tensor):
        self.A_hat = A_hat.to(next(self.parameters()).device)

    def forward(
        self,
        X:        torch.Tensor,   # (N, 54, NODE_DIM)
        v1_idx:   torch.Tensor,   # (N,)   vertex index of first settlement
        v2_idx:   torch.Tensor,   # (N,)   vertex index of second settlement
        seat:     torch.Tensor,   # (N,)   seat 0-3
    ):
        N = X.shape[0]
        H = X

        # GCN layers with residual
        for i, conv in enumerate(self.conv_layers):
            H_new = conv(H, self.A_hat)
            H_new = F.relu(H_new)
            H_new = F.dropout(H_new, p=self.dropout, training=self.training)
            if i > 0:
                H = H + H_new   # residual (skip first layer — dim mismatch possible)
            else:
                H = H_new

        # Readout
        v1_emb  = H[torch.arange(N), v1_idx]   # (N, hidden)
        v2_emb  = H[torch.arange(N), v2_idx]   # (N, hidden)
        g       = H.mean(dim=1)                 # (N, hidden) — global mean pool
        s_emb   = self.seat_emb(seat.clamp(0, SEAT_DIM - 1))  # (N, 8)

        h = torch.cat([v1_emb, v2_emb, g, s_emb], dim=-1)  # (N, mlp_in)

        win_logit = self.mlp(h)
        rank_pred = self.rank_head(h) if self.multitask else None
        return win_logit, rank_pred

    @torch.no_grad()
    def predict_proba(self, X, v1_idx, v2_idx, seat) -> torch.Tensor:
        self.eval()
        logit, _ = self.forward(X, v1_idx, v2_idx, seat)
        return torch.sigmoid(logit).squeeze(-1)


# ── dataset helpers ───────────────────────────────────────────────────────────

def dataset_to_tensors(dataset: list, device="cpu"):
    """
    Convert list of PyG Data objects → flat tensors for StaticBoardGCN.

    Returns:
        X       (N, 54, 16)  float32
        v1_idx  (N,)         long
        v2_idx  (N,)         long
        seat    (N,)         long
        y_win   (N,)         float32
        y_rank  (N,)         float32  (normalised rank, may be all-zero if absent)
    """
    X      = torch.stack([d.x for d in dataset])                                # (N,54,16)
    v1_idx = torch.tensor([d.v1_local.item() for d in dataset], dtype=torch.long)
    v2_idx = torch.tensor([d.v2_local.item() for d in dataset], dtype=torch.long)
    seat   = torch.tensor([d.seat.item()     for d in dataset], dtype=torch.long)
    y_win  = torch.tensor([d.y.item()        for d in dataset], dtype=torch.float32)
    y_rank = torch.tensor([d.y_rank.item() if (hasattr(d, "y_rank") and d.y_rank is not None) else 0.0
                           for d in dataset], dtype=torch.float32)

    return (X.to(device), v1_idx.to(device), v2_idx.to(device),
            seat.to(device), y_win.to(device), y_rank.to(device))

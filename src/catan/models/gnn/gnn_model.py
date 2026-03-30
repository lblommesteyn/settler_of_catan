"""
Board-GAT: Graph Attention Network for Catan opening evaluation.

Architecture
────────────
Input:  54-node board graph, node features (16 dims), global seat embedding (4 dims)
Body:   3 × GAT layers (4 heads each) with residual connections + LayerNorm
        After each layer, v1 and v2 node embeddings are also computed via cross-attention
        to capture their interaction
Readout: concatenate
          - v1 embedding (hidden_dim)
          - v2 embedding (hidden_dim)
          - global mean-pooled board embedding (hidden_dim)
          - seat embedding (8)
          → MLP → sigmoid P(win)

Multi-task: optionally also predicts final rank (regression head).

Why GAT over GCN:
  Attention weights let the model learn which neighbours matter
  (e.g., the vertex adjacent to a 6-pip neighbour matters more than a 2-pip one).
  Attention weights are also interpretable — we can visualise which vertices
  each settlement "looks at" for its value estimate.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

from .board_to_graph import NODE_DIM, SEAT_DIM


class BoardGAT(nn.Module):
    """
    Graph Attention Network operating on the Catan board graph.

    Args:
        node_dim:   input node feature dimension (default 16)
        hidden:     hidden dimension per attention head (default 32, so 4 heads = 128 total)
        heads:      number of attention heads
        n_layers:   number of GAT message-passing rounds
        dropout:    dropout rate on attention + MLP
        multitask:  if True, also predicts final rank via a second head
    """

    def __init__(
        self,
        node_dim: int   = NODE_DIM,
        hidden: int     = 32,
        heads: int      = 4,
        n_layers: int   = 3,
        dropout: float  = 0.2,
        multitask: bool = True,
    ):
        super().__init__()
        self.dropout   = dropout
        self.multitask = multitask
        d = hidden * heads   # full width after concatenating heads

        # Input projection
        self.input_proj = nn.Linear(node_dim, d)

        # GAT layers with residuals
        self.gat_layers = nn.ModuleList()
        self.norms      = nn.ModuleList()
        for _ in range(n_layers):
            self.gat_layers.append(
                GATConv(d, hidden, heads=heads, dropout=dropout, concat=True)
            )
            self.norms.append(nn.LayerNorm(d))

        # Seat embedding
        self.seat_emb = nn.Embedding(SEAT_DIM, 8)

        # MLP head for win probability
        mlp_in = d * 3 + 8   # v1 + v2 + global + seat
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        # Optional rank regression head
        if multitask:
            self.rank_head = nn.Sequential(
                nn.Linear(mlp_in, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid(),
            )

    def forward(self, data):
        """
        Args:
            data: PyG Batch/Data with fields:
                  x            (N_total, node_dim)
                  edge_index   (2, E_total)
                  u            (batch_size, 4)   — seat one-hot
                  batch        (N_total,)        — PyG batch assignment
                  v1_idx       (batch_size,)     — node index of v1 within each graph
                  v2_idx       (batch_size,)     — node index of v2 within each graph
                  v1_batch_idx (batch_size,)     — global node index of v1
                  v2_batch_idx (batch_size,)     — global node index of v2

        Returns:
            win_logit (batch_size, 1) — raw logit (apply sigmoid for probability)
            rank_pred (batch_size, 1) or None — normalised rank [0,1]
        """
        x, ei, batch = data.x, data.edge_index, data.batch

        # Input projection
        x = F.relu(self.input_proj(x))

        # GAT rounds with residuals
        for gat, norm in zip(self.gat_layers, self.norms):
            x_new = gat(x, ei)
            x_new = F.elu(x_new)
            x_new = F.dropout(x_new, p=self.dropout, training=self.training)
            x = norm(x + x_new)          # residual + LayerNorm

        # Global board embedding (mean pool over all 54 nodes)
        g = global_mean_pool(x, batch)   # (B, d)

        # v1 and v2 embeddings.
        # PyG auto-concatenates v1_local/v2_local from each graph → shape (B,).
        # All graphs have exactly 54 nodes, so graph i starts at offset i*54.
        B = g.shape[0]
        offsets = torch.arange(B, device=x.device) * 54
        v1_batch_idx = data.v1_local.view(-1) + offsets
        v2_batch_idx = data.v2_local.view(-1) + offsets
        v1_emb = x[v1_batch_idx]   # (B, d)
        v2_emb = x[v2_batch_idx]   # (B, d)

        # Seat embedding
        seat_idx = data.seat.view(-1)        # (B,) stored directly as int
        s_emb = self.seat_emb(seat_idx)      # (B, 8)

        h = torch.cat([v1_emb, v2_emb, g, s_emb], dim=-1)  # (B, d*3+8)

        win_logit = self.mlp(h)              # (B, 1)
        rank_pred = self.rank_head(h) if self.multitask else None

        return win_logit, rank_pred

    def predict_proba(self, data) -> torch.Tensor:
        """Returns win probabilities in [0,1]."""
        with torch.no_grad():
            logit, _ = self.forward(data)
            return torch.sigmoid(logit).squeeze(-1)


# ── attention visualisation helper ────────────────────────────────────────────

def get_attention_weights(model: BoardGAT, data, layer: int = -1):
    """
    Returns (edge_index, alpha) for the specified GAT layer.
    Useful for visualising which board edges the model attends to.
    """
    hooks = []
    alphas = {}

    def hook(name):
        def _hook(module, input, output):
            # GATConv returns (x, alpha) when return_attention_weights=True
            # We manually save from the module's last call
            pass
        return _hook

    # Forward pass with return_attention_weights
    model.eval()
    gat = model.gat_layers[layer]
    with torch.no_grad():
        # project input to d
        x = F.relu(model.input_proj(data.x))
        for i, (g, norm) in enumerate(zip(model.gat_layers, model.norms)):
            if i == (layer % len(model.gat_layers)):
                out, (ei, alpha) = g(x, data.edge_index,
                                     return_attention_weights=True)
                return ei, alpha
            x_new = g(x, data.edge_index)
            x_new = F.elu(x_new)
            x = norm(x + x_new)
    return None, None

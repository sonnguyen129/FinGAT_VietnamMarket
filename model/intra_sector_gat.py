"""Intra-sector GAT layer (Paper Eq. 7-8).

Applies GAT on fully-connected intra-sector stock graph.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv


class IntraSectorGAT(nn.Module):
    """GAT over fully-connected intra-sector stock graph.

    Input:  stock embeddings [num_stocks, hidden_dim]
    Output: graph-enriched embeddings [num_stocks, hidden_dim]
    """

    def __init__(self, hidden_dim: int, heads: int = 1, dropout: float = 0.0):
        super().__init__()
        self.gat = GATConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            heads=heads,
            concat=False,
            dropout=dropout,
            add_self_loops=True,
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                return_attention: bool = False):
        if return_attention:
            out, (edge_idx, alpha) = self.gat(
                x, edge_index, return_attention_weights=True)
            return torch.relu(out), alpha
        return torch.relu(self.gat(x, edge_index))

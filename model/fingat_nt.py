"""FinGAT-NT — No sector information variant (Paper Eq. 19, for EQ2).

Differences from FinGAT:
  1. No sector-level modeling (no graph pooling, no inter-sector GAT)
  2. Uses fully-connected graph of ALL stocks (not sector-based)
  3. Fusion from 2 vectors: [tau_G || tau_A] instead of 3
"""

import torch
import torch.nn as nn

from model.attentive_gru import AttentiveGRU
from model.intra_sector_gat import IntraSectorGAT
from data.graph_builder import build_fully_connected_edges


class FinGAT_NT(nn.Module):
    """FinGAT without sector information."""

    def __init__(self, input_dim: int, hidden_dim: int,
                 num_stocks: int,
                 week_num: int = 3,
                 gat_heads: int = 1,
                 gat_dropout: float = 0.0,
                 use_mtl: bool = True,
                 device: str = 'cpu'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.week_num = week_num
        self.use_mtl = use_mtl
        self.device = device

        # Fully-connected graph of ALL stocks
        fc_edges = build_fully_connected_edges(num_stocks)
        self.register_buffer('edge_index', fc_edges)

        # Short-term GRU
        self.short_term_gru = AttentiveGRU(input_dim, hidden_dim)

        # All-stock GAT (replaces intra-sector GAT)
        self.all_stock_gat = IntraSectorGAT(hidden_dim, gat_heads, gat_dropout)

        # Long-term GRUs
        self.long_term_gru_A = AttentiveGRU(hidden_dim, hidden_dim)
        self.long_term_gru_G = AttentiveGRU(hidden_dim, hidden_dim)

        # Fusion: only 2 vectors (no sector embedding)
        self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)

        # Prediction heads
        self.reg_head = nn.Linear(hidden_dim, 1)
        if use_mtl:
            self.cls_head = nn.Sequential(
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )

    def forward(self, weekly_inputs: list) -> tuple:
        """
        Args:
            weekly_inputs: list of week_num tensors,
                          each [num_stocks, days_per_week, input_dim]
        Returns:
            reg_out: [num_stocks, 1]
            cls_out: [num_stocks, 1] or None
        """
        weekly_A = []
        weekly_G = []

        for week_data in weekly_inputs:
            a = self.short_term_gru(week_data)
            weekly_A.append(a)
            g = self.all_stock_gat(a, self.edge_index)
            weekly_G.append(g)

        stacked_A = torch.stack(weekly_A, dim=1)
        stacked_G = torch.stack(weekly_G, dim=1)

        tau_A = self.long_term_gru_A(stacked_A)
        tau_G = self.long_term_gru_G(stacked_G)

        # Eq. 19: fusion from 2 vectors
        fused = torch.cat([tau_G, tau_A], dim=-1)
        fused = torch.relu(self.fusion(fused))

        reg_out = self.reg_head(fused)
        cls_out = None
        if self.use_mtl:
            cls_out = self.cls_head(fused)

        return reg_out, cls_out

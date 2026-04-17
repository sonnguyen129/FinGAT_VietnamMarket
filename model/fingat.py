"""Full FinGAT model — CategoricalGraphAtt (Paper Section 4).

Architecture:
  1. Short-term Attentive GRU: daily features → weekly embedding (per week)
  2. Intra-sector GAT: stock relationships within sectors (per week)
  3. Long-term Attentive GRU: weekly → multi-week (two streams: raw + GAT)
  4. Graph Pooling + Inter-sector GAT: sector-level modeling
  5. Fusion + Multi-task prediction heads

Constructor flags for ablation (EQ3):
  - use_intra: enable/disable intra-sector GAT
  - use_inter: enable/disable inter-sector GAT + graph pooling
  - use_mtl: enable/disable multi-task learning (BCE loss)
  - use_mse: replace BCE with MSE for movement loss
"""

import torch
import torch.nn as nn

from model.attentive_gru import AttentiveGRU
from model.intra_sector_gat import IntraSectorGAT
from model.inter_sector_gat import InterSectorGAT, GraphPooling


class FinGAT(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int,
                 intra_edge_index: torch.Tensor,
                 inter_edge_index: torch.Tensor,
                 sector_assignments: torch.Tensor,
                 num_sectors: int,
                 week_num: int = 3,
                 days_per_week: int = 5,
                 gat_heads: int = 1,
                 gat_dropout: float = 0.0,
                 use_intra: bool = True,
                 use_inter: bool = True,
                 use_mtl: bool = True,
                 use_mse: bool = False,
                 device: str = 'cpu'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.week_num = week_num
        self.use_intra = use_intra
        self.use_inter = use_inter
        self.use_mtl = use_mtl
        self.use_mse = use_mse
        self.num_sectors = num_sectors
        self.device = device

        # Register graph structure as buffers
        self.register_buffer('intra_edge_index', intra_edge_index)
        self.register_buffer('inter_edge_index', inter_edge_index)
        self.register_buffer('sector_assignments', sector_assignments)

        # 1a. Short-term Attentive GRU (daily → weekly)
        self.short_term_gru = AttentiveGRU(input_dim, hidden_dim)

        # 1b. Intra-sector GAT
        if use_intra:
            self.intra_gat = IntraSectorGAT(hidden_dim, gat_heads, gat_dropout)

        # 1c. Long-term Attentive GRUs (weekly → multi-week)
        self.long_term_gru_A = AttentiveGRU(hidden_dim, hidden_dim)
        if use_intra:
            self.long_term_gru_G = AttentiveGRU(hidden_dim, hidden_dim)

        # 2. Inter-sector components
        if use_inter:
            self.graph_pooling = GraphPooling()
            self.inter_gat = InterSectorGAT(hidden_dim, gat_heads, gat_dropout)

        # 3. Fusion
        # Determine fusion input dimension based on ablation flags
        fusion_inputs = 1  # tau_A always present
        if use_intra:
            fusion_inputs += 1  # tau_G
        if use_inter:
            fusion_inputs += 1  # tau_sector
        self.fusion = nn.Linear(hidden_dim * fusion_inputs, hidden_dim)

        # 4. Prediction heads
        self.reg_head = nn.Linear(hidden_dim, 1)
        if use_mtl:
            if use_mse:
                self.cls_head = nn.Linear(hidden_dim, 1)
            else:
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
            reg_out: [num_stocks, 1] predicted return ratios
            cls_out: [num_stocks, 1] predicted up probability (or None if not use_mtl)
        """
        weekly_A = []
        weekly_G = []

        for week_data in weekly_inputs:
            # Short-term: daily → weekly embedding
            a = self.short_term_gru(week_data)  # [num_stocks, hidden_dim]
            weekly_A.append(a)

            # Intra-sector GAT
            if self.use_intra:
                g = self.intra_gat(a, self.intra_edge_index)
                weekly_G.append(g)

        # Long-term: aggregate across weeks
        stacked_A = torch.stack(weekly_A, dim=1)  # [num_stocks, weeks, hidden_dim]
        tau_A = self.long_term_gru_A(stacked_A)   # [num_stocks, hidden_dim]

        fusion_parts = [tau_A]

        if self.use_intra:
            stacked_G = torch.stack(weekly_G, dim=1)
            tau_G = self.long_term_gru_G(stacked_G)
            fusion_parts.append(tau_G)

            # Source for sector pooling
            pool_source = tau_G
        else:
            pool_source = tau_A

        if self.use_inter:
            # Sector-level: graph pooling
            sector_emb = self.graph_pooling(
                pool_source, self.sector_assignments, self.num_sectors)
            # Inter-sector GAT
            sector_out = self.inter_gat(sector_emb, self.inter_edge_index)
            # Map back to stock level
            tau_sector = sector_out[self.sector_assignments]
            fusion_parts.append(tau_sector)

        # Fusion
        fused = torch.cat(fusion_parts, dim=-1)
        fused = torch.relu(self.fusion(fused))

        # Prediction
        reg_out = self.reg_head(fused)
        cls_out = None
        if self.use_mtl:
            cls_out = self.cls_head(fused)

        return reg_out, cls_out

    def extract_attention(self, weekly_inputs: list) -> dict:
        """Extract attention weights for visualization (EQ5).

        Returns dict with 'intra_attention' and 'inter_attention'.
        """
        self.eval()
        with torch.no_grad():
            weekly_A = []
            weekly_G = []
            intra_attns = []

            for week_data in weekly_inputs:
                a = self.short_term_gru(week_data)
                weekly_A.append(a)
                if self.use_intra:
                    g, attn = self.intra_gat(
                        a, self.intra_edge_index, return_attention=True)
                    weekly_G.append(g)
                    intra_attns.append(attn.cpu())

            stacked_A = torch.stack(weekly_A, dim=1)
            tau_A = self.long_term_gru_A(stacked_A)

            inter_attn = None
            if self.use_intra and self.use_inter:
                stacked_G = torch.stack(weekly_G, dim=1)
                tau_G = self.long_term_gru_G(stacked_G)
                sector_emb = self.graph_pooling(
                    tau_G, self.sector_assignments, self.num_sectors)
                _, inter_attn = self.inter_gat(
                    sector_emb, self.inter_edge_index, return_attention=True)
                inter_attn = inter_attn.cpu()

        return {
            'intra_attention': intra_attns,
            'inter_attention': inter_attn,
        }

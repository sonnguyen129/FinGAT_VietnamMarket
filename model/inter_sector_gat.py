"""Inter-sector GAT layer + Graph Pooling (Paper Eq. 11-12).

Graph pooling: element-wise max pooling over stocks in each sector.
Inter-sector GAT: same architecture as intra-sector, on sector graph.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv


class GraphPooling(nn.Module):
    """Element-wise max pooling from stock-level to sector-level.

    Eq. 11: z_πc = MaxPool({τ_G(sq) | ∀sq ∈ M_πc})
    """

    def __init__(self):
        super().__init__()

    def forward(self, stock_embeddings: torch.Tensor,
                sector_assignments: torch.Tensor,
                num_sectors: int) -> torch.Tensor:
        """
        Args:
            stock_embeddings: [num_stocks, hidden_dim]
            sector_assignments: [num_stocks] — sector index per stock
            num_sectors: total number of sectors

        Returns:
            sector_embeddings: [num_sectors, hidden_dim]
        """
        hidden_dim = stock_embeddings.shape[1]
        device = stock_embeddings.device
        sector_embeddings = torch.full(
            (num_sectors, hidden_dim), float('-inf'), device=device)

        for s in range(num_sectors):
            mask = sector_assignments == s
            if mask.any():
                sector_embeddings[s] = stock_embeddings[mask].max(dim=0).values

        # Replace any remaining -inf (empty sectors) with zeros
        sector_embeddings = torch.where(
            sector_embeddings == float('-inf'),
            torch.zeros_like(sector_embeddings),
            sector_embeddings
        )
        return sector_embeddings


class InterSectorGAT(nn.Module):
    """GAT over fully-connected inter-sector graph.

    Input:  sector embeddings [num_sectors, hidden_dim]
    Output: enriched sector embeddings [num_sectors, hidden_dim]
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

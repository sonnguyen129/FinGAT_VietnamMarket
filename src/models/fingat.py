"""
FinGAT Model Implementation for Vietnam Market

This module implements the FinGAT (Financial Graph Attention Networks) model
as described in the paper, specifically adapted for Vietnamese stock market data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GATv2Conv
from torch_geometric.data import Data, Batch
from typing import Dict, List, Tuple, Optional
import numpy as np
import logging

from utils.logging_utils import get_logger

logger = get_logger(__name__)


class AttentiveGRU(nn.Module):
    """
    Attentive GRU module for temporal feature extraction.

    This component processes sequential stock price data and applies
    attention mechanism to focus on important temporal patterns.
    """

    def __init__(self, input_size: int, hidden_size: int = 128, 
                 num_layers: int = 2, dropout: float = 0.2,
                 attention_size: int = 64):
        """
        Initialize AttentiveGRU.

        Args:
            input_size: Input feature dimension
            hidden_size: GRU hidden dimension
            num_layers: Number of GRU layers
            dropout: Dropout rate
            attention_size: Attention mechanism dimension
        """
        super(AttentiveGRU, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.attention_size = attention_size

        # GRU layers
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # Attention mechanism
        self.attention_linear1 = nn.Linear(hidden_size, attention_size)
        self.attention_linear2 = nn.Linear(attention_size, 1)
        self.dropout = nn.Dropout(dropout)

        logger.debug(f"AttentiveGRU initialized: input_size={input_size}, hidden_size={hidden_size}")

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through AttentiveGRU.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            lengths: Actual sequence lengths for padding handling

        Returns:
            Attended output tensor of shape (batch_size, hidden_size)
        """
        batch_size, seq_len, _ = x.shape

        # GRU forward pass
        gru_output, _ = self.gru(x)  # (batch_size, seq_len, hidden_size)

        # Apply attention mechanism
        # Transform GRU outputs through linear layers
        attention_weights = self.attention_linear1(gru_output)  # (batch_size, seq_len, attention_size)
        attention_weights = torch.tanh(attention_weights)
        attention_weights = self.dropout(attention_weights)

        # Calculate attention scores
        attention_scores = self.attention_linear2(attention_weights)  # (batch_size, seq_len, 1)
        attention_scores = attention_scores.squeeze(-1)  # (batch_size, seq_len)

        # Handle variable length sequences if lengths provided
        if lengths is not None:
            mask = torch.arange(seq_len, device=x.device)[None, :] >= lengths[:, None]
            attention_scores.masked_fill_(mask, float('-inf'))

        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch_size, seq_len)

        # Apply attention weights to GRU outputs
        attended_output = torch.sum(
            gru_output * attention_weights.unsqueeze(-1), 
            dim=1
        )  # (batch_size, hidden_size)

        return attended_output


class IntraSectorGAT(nn.Module):
    """
    Intra-sector Graph Attention Network.

    Processes relationships between stocks within the same sector.
    """

    def __init__(self, input_size: int, hidden_size: int = 128, 
                 num_heads: int = 8, dropout: float = 0.3, alpha: float = 0.2):
        """
        Initialize IntraSectorGAT.

        Args:
            input_size: Input feature dimension
            hidden_size: Hidden dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
            alpha: LeakyReLU negative slope
        """
        super(IntraSectorGAT, self).__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads

        # GAT layer with multiple heads
        self.gat_conv = GATv2Conv(
            in_channels=input_size,
            out_channels=hidden_size // num_heads,
            heads=num_heads,
            dropout=dropout,
            negative_slope=alpha,
            concat=True
        )

        self.dropout = nn.Dropout(dropout)

        logger.debug(f"IntraSectorGAT initialized: input_size={input_size}, hidden_size={hidden_size}, heads={num_heads}")

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through IntraSectorGAT.

        Args:
            x: Node features of shape (num_nodes, input_size)
            edge_index: Graph connectivity of shape (2, num_edges)
            edge_attr: Edge attributes (optional)

        Returns:
            Updated node features of shape (num_nodes, hidden_size)
        """
        # Apply GAT convolution
        x = self.gat_conv(x, edge_index)
        x = self.dropout(x)

        return x


class InterSectorGAT(nn.Module):
    """
    Inter-sector Graph Attention Network.

    Processes relationships between stocks from different sectors.
    """

    def __init__(self, input_size: int, hidden_size: int = 128,
                 num_heads: int = 8, dropout: float = 0.3, alpha: float = 0.2):
        """
        Initialize InterSectorGAT.

        Args:
            input_size: Input feature dimension  
            hidden_size: Hidden dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
            alpha: LeakyReLU negative slope
        """
        super(InterSectorGAT, self).__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads

        # GAT layer for inter-sector relationships
        self.gat_conv = GATv2Conv(
            in_channels=input_size,
            out_channels=hidden_size // num_heads,
            heads=num_heads,
            dropout=dropout,
            negative_slope=alpha,
            concat=True
        )

        self.dropout = nn.Dropout(dropout)

        logger.debug(f"InterSectorGAT initialized: input_size={input_size}, hidden_size={hidden_size}, heads={num_heads}")

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through InterSectorGAT.

        Args:
            x: Node features of shape (num_nodes, input_size)
            edge_index: Graph connectivity of shape (2, num_edges)
            edge_attr: Edge attributes (optional)

        Returns:
            Updated node features of shape (num_nodes, hidden_size)
        """
        # Apply GAT convolution
        x = self.gat_conv(x, edge_index)
        x = self.dropout(x)

        return x


class FinGAT(nn.Module):
    """
    Complete FinGAT model for stock movement prediction.

    Combines temporal modeling (AttentiveGRU) with graph-based modeling
    (IntraSectorGAT + InterSectorGAT) for comprehensive stock analysis.
    """

    def __init__(self, config: Dict):
        """
        Initialize FinGAT model.

        Args:
            config: Configuration dictionary containing model parameters
        """
        super(FinGAT, self).__init__()

        self.config = config
        model_config = config.get('model', {})

        # Extract configuration
        temporal_config = model_config.get('temporal', {})
        intra_config = model_config.get('graph', {}).get('intra_sector', {})
        inter_config = model_config.get('graph', {}).get('inter_sector', {})
        output_config = model_config.get('output', {})

        # Model dimensions
        self.input_size = temporal_config.get('input_size', 20)  # Number of features per timestep
        self.temporal_hidden = temporal_config.get('hidden_size', 128)
        self.graph_hidden = intra_config.get('hidden_size', 128)
        self.output_hidden = output_config.get('hidden_size', 64)
        self.num_classes = output_config.get('num_classes', 2)

        # Temporal component
        self.attentive_gru = AttentiveGRU(
            input_size=self.input_size,
            hidden_size=self.temporal_hidden,
            num_layers=temporal_config.get('num_layers', 2),
            dropout=temporal_config.get('dropout', 0.2),
            attention_size=temporal_config.get('attention_size', 64)
        )

        # Graph components
        self.intra_sector_gat = IntraSectorGAT(
            input_size=self.temporal_hidden,
            hidden_size=self.graph_hidden,
            num_heads=intra_config.get('num_heads', 8),
            dropout=intra_config.get('dropout', 0.3),
            alpha=intra_config.get('alpha', 0.2)
        )

        self.inter_sector_gat = InterSectorGAT(
            input_size=self.graph_hidden,
            hidden_size=self.graph_hidden,
            num_heads=inter_config.get('num_heads', 8),
            dropout=inter_config.get('dropout', 0.3),
            alpha=inter_config.get('alpha', 0.2)
        )

        # Feature fusion
        self.feature_fusion = nn.Linear(
            self.temporal_hidden + self.graph_hidden * 2, 
            self.output_hidden
        )

        # Output layers
        self.dropout = nn.Dropout(output_config.get('dropout', 0.5))

        # Movement prediction head
        self.movement_classifier = nn.Linear(self.output_hidden, self.num_classes)

        # Ranking prediction head (for pairwise ranking loss)
        self.ranking_predictor = nn.Linear(self.output_hidden, 1)

        logger.info(f"FinGAT model initialized with config: {model_config}")

    def forward(self, temporal_data: torch.Tensor, 
                intra_graph: Data, inter_graph: Data,
                return_embeddings: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass through FinGAT model.

        Args:
            temporal_data: Sequential data of shape (batch_size, seq_len, input_size)
            intra_graph: Intra-sector graph data
            inter_graph: Inter-sector graph data
            return_embeddings: Whether to return intermediate embeddings

        Returns:
            Dictionary containing predictions and optionally embeddings
        """
        batch_size = temporal_data.shape[0]

        # 1. Temporal feature extraction
        temporal_features = self.attentive_gru(temporal_data)  # (batch_size, temporal_hidden)

        # 2. Graph feature extraction
        # For graph processing, we need to handle the batch dimension differently
        # Assuming temporal_features correspond to graph node features

        # Intra-sector GAT
        intra_features = self.intra_sector_gat(
            temporal_features, 
            intra_graph.edge_index,
            intra_graph.edge_attr if hasattr(intra_graph, 'edge_attr') else None
        )  # (batch_size, graph_hidden)

        # Inter-sector GAT
        inter_features = self.inter_sector_gat(
            intra_features,
            inter_graph.edge_index, 
            inter_graph.edge_attr if hasattr(inter_graph, 'edge_attr') else None
        )  # (batch_size, graph_hidden)

        # 3. Feature fusion
        # Concatenate temporal and graph features
        fused_features = torch.cat([
            temporal_features,  # (batch_size, temporal_hidden)
            intra_features,     # (batch_size, graph_hidden)
            inter_features      # (batch_size, graph_hidden)
        ], dim=1)

        # Apply fusion layer
        fused_features = self.feature_fusion(fused_features)  # (batch_size, output_hidden)
        fused_features = F.relu(fused_features)
        fused_features = self.dropout(fused_features)

        # 4. Predictions
        # Movement classification (up/down)
        movement_logits = self.movement_classifier(fused_features)  # (batch_size, num_classes)
        movement_probs = F.softmax(movement_logits, dim=1)

        # Ranking score
        ranking_scores = self.ranking_predictor(fused_features)  # (batch_size, 1)

        outputs = {
            'movement_logits': movement_logits,
            'movement_probs': movement_probs,
            'ranking_scores': ranking_scores
        }

        if return_embeddings:
            outputs.update({
                'temporal_features': temporal_features,
                'intra_features': intra_features, 
                'inter_features': inter_features,
                'fused_features': fused_features
            })

        return outputs

    def predict_movement(self, temporal_data: torch.Tensor,
                        intra_graph: Data, inter_graph: Data) -> torch.Tensor:
        """
        Predict stock movement direction.

        Args:
            temporal_data: Sequential data
            intra_graph: Intra-sector graph
            inter_graph: Inter-sector graph

        Returns:
            Movement predictions (0=down, 1=up)
        """
        outputs = self.forward(temporal_data, intra_graph, inter_graph)
        predictions = torch.argmax(outputs['movement_probs'], dim=1)
        return predictions

    def get_ranking_scores(self, temporal_data: torch.Tensor,
                          intra_graph: Data, inter_graph: Data) -> torch.Tensor:
        """
        Get ranking scores for stocks.

        Args:
            temporal_data: Sequential data
            intra_graph: Intra-sector graph  
            inter_graph: Inter-sector graph

        Returns:
            Ranking scores for each stock
        """
        outputs = self.forward(temporal_data, intra_graph, inter_graph)
        return outputs['ranking_scores'].squeeze(-1)


def create_fingat_model(config: Dict) -> FinGAT:
    """
    Factory function to create FinGAT model from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Initialized FinGAT model
    """
    model = FinGAT(config)

    # Initialize weights
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.GRU):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    torch.nn.init.orthogonal_(param)
                elif 'bias' in name:
                    torch.nn.init.zeros_(param)

    model.apply(init_weights)

    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"FinGAT model created:")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")

    return model


# Example usage
if __name__ == "__main__":
    # Example configuration
    config = {
        'model': {
            'temporal': {
                'input_size': 20,
                'hidden_size': 128,
                'num_layers': 2,
                'dropout': 0.2,
                'attention_size': 64
            },
            'graph': {
                'intra_sector': {
                    'hidden_size': 128,
                    'num_heads': 8,
                    'dropout': 0.3,
                    'alpha': 0.2
                },
                'inter_sector': {
                    'hidden_size': 128,
                    'num_heads': 8, 
                    'dropout': 0.3,
                    'alpha': 0.2
                }
            },
            'output': {
                'hidden_size': 64,
                'num_classes': 2,
                'dropout': 0.5
            }
        }
    }

    # Create model
    model = create_fingat_model(config)
    print(f"FinGAT model created successfully: {model}")
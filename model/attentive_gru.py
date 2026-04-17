"""Attentive GRU — GRU with feed-forward attention (Paper Eq. 4-6).

Used at two levels:
  1. Short-term: daily features within a week → weekly embedding
  2. Long-term: weekly embeddings across weeks → multi-week embedding
"""

import torch
import torch.nn as nn


class AttentiveGRU(nn.Module):
    """GRU with feed-forward attention aggregation over time steps.

    Input:  [batch, seq_len, input_dim]
    Output: [batch, hidden_dim]
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.attention_W = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, input_dim]
        h, _ = self.gru(x)  # h: [batch, seq_len, hidden_dim]

        # Attention: alpha_j = softmax(tanh(W_0 * h_j))
        attn_scores = torch.tanh(self.attention_W(h))  # [batch, seq_len, 1]
        attn_weights = torch.softmax(attn_scores, dim=1)  # [batch, seq_len, 1]

        # Weighted sum: output = sum(alpha_j * h_j)
        output = torch.sum(attn_weights * h, dim=1)  # [batch, hidden_dim]
        return output

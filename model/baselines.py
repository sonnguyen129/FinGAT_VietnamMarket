"""Baseline models for comparison (Paper Section 5.2).

All baselines:
  - Take input [num_stocks, seq_len, features] (concatenated weeks)
  - Output (reg_out [N,1], cls_out [N,1] or None)
  - Use same loss function and evaluation as FinGAT

1. MLP: 2 hidden layers (32 -> 8)
2. GRU: 1 layer, hidden_dim=32
3. GRU+Att: GRU 32-dim + attention
4. FineNet: dilated CNN (32+16) for short+long term
5. RankLSTM: LSTM 16-dim + pairwise ranking (uses graph)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StockMLP(nn.Module):
    """MLP baseline (Tang et al. 2015).

    2 hidden layers: input -> 32 -> 8 -> output.
    Flatten all features across time.
    """

    def __init__(self, input_dim: int = 15, seq_len: int = 15,
                 hidden_dims: tuple = (32, 8), use_mtl: bool = True):
        super().__init__()
        flat_dim = input_dim * seq_len
        self.fc1 = nn.Linear(flat_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.reg_head = nn.Linear(hidden_dims[1], 1)
        self.use_mtl = use_mtl
        if use_mtl:
            self.cls_head = nn.Sequential(
                nn.Linear(hidden_dims[1], 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> tuple:
        # x: [num_stocks, seq_len, features]
        x = x.view(x.size(0), -1)  # [N, seq_len * features]
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        reg_out = self.reg_head(x)
        cls_out = self.cls_head(x) if self.use_mtl else None
        return reg_out, cls_out


class StockGRU(nn.Module):
    """GRU baseline (Cho et al. 2014).

    1 GRU layer, hidden_dim=32, use last hidden state.
    """

    def __init__(self, input_dim: int = 15, hidden_dim: int = 32,
                 use_mtl: bool = True):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.reg_head = nn.Linear(hidden_dim, 1)
        self.use_mtl = use_mtl
        if use_mtl:
            self.cls_head = nn.Sequential(
                nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> tuple:
        # x: [num_stocks, seq_len, features]
        _, h_n = self.gru(x)  # h_n: [1, N, hidden_dim]
        h = h_n.squeeze(0)    # [N, hidden_dim]
        reg_out = self.reg_head(h)
        cls_out = self.cls_head(h) if self.use_mtl else None
        return reg_out, cls_out


class StockGRUAtt(nn.Module):
    """GRU + Attention baseline (Dhingra et al. 2017).

    GRU 32-dim + feed-forward attention.
    """

    def __init__(self, input_dim: int = 15, hidden_dim: int = 32,
                 use_mtl: bool = True):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.attn_W = nn.Linear(hidden_dim, 1, bias=False)
        self.reg_head = nn.Linear(hidden_dim, 1)
        self.use_mtl = use_mtl
        if use_mtl:
            self.cls_head = nn.Sequential(
                nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> tuple:
        h, _ = self.gru(x)  # [N, seq_len, hidden_dim]
        attn = torch.softmax(torch.tanh(self.attn_W(h)), dim=1)
        out = torch.sum(attn * h, dim=1)  # [N, hidden_dim]
        reg_out = self.reg_head(out)
        cls_out = self.cls_head(out) if self.use_mtl else None
        return reg_out, cls_out


class FineNet(nn.Module):
    """FineNet baseline (Tsai et al. 2019).

    Joint short-term + long-term via dilated 1D CNN.
    Short-term: 1D dilated CNN (32-dim conv + 16-dim conv)
    Long-term: 1D dilated CNN
    Fusion -> prediction.
    """

    def __init__(self, input_dim: int = 15, short_dim: int = 32,
                 long_dim: int = 16, use_mtl: bool = True):
        super().__init__()
        # Short-term: dilated conv
        self.short_conv1 = nn.Conv1d(input_dim, short_dim, kernel_size=3,
                                     padding=1, dilation=1)
        self.short_conv2 = nn.Conv1d(short_dim, long_dim, kernel_size=3,
                                     padding=2, dilation=2)
        # Long-term: dilated conv
        self.long_conv1 = nn.Conv1d(input_dim, short_dim, kernel_size=5,
                                    padding=4, dilation=2)
        self.long_conv2 = nn.Conv1d(short_dim, long_dim, kernel_size=5,
                                    padding=8, dilation=4)
        # Fusion
        self.fc = nn.Linear(long_dim * 2, long_dim)
        self.reg_head = nn.Linear(long_dim, 1)
        self.use_mtl = use_mtl
        if use_mtl:
            self.cls_head = nn.Sequential(
                nn.Linear(long_dim, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> tuple:
        # x: [N, seq_len, features] -> transpose to [N, features, seq_len]
        x_t = x.transpose(1, 2)

        # Short-term path
        s = torch.relu(self.short_conv1(x_t))
        s = torch.relu(self.short_conv2(s))
        s = s.mean(dim=2)  # Global average pooling [N, long_dim]

        # Long-term path
        l = torch.relu(self.long_conv1(x_t))
        l = torch.relu(self.long_conv2(l))
        l = l.mean(dim=2)  # [N, long_dim]

        # Fusion
        fused = torch.relu(self.fc(torch.cat([s, l], dim=1)))
        reg_out = self.reg_head(fused)
        cls_out = self.cls_head(fused) if self.use_mtl else None
        return reg_out, cls_out


class RankLSTM(nn.Module):
    """RankLSTM baseline (Feng et al. 2019).

    LSTM 16-dim + pairwise ranking.
    Uses pre-defined graph relations (sector-based for HOSE).
    """

    def __init__(self, input_dim: int = 15, hidden_dim: int = 16,
                 use_mtl: bool = True):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.reg_head = nn.Linear(hidden_dim, 1)
        self.use_mtl = use_mtl
        if use_mtl:
            self.cls_head = nn.Sequential(
                nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> tuple:
        # x: [N, seq_len, features]
        _, (h_n, _) = self.lstm(x)  # h_n: [1, N, hidden_dim]
        h = h_n.squeeze(0)
        reg_out = self.reg_head(h)
        cls_out = self.cls_head(h) if self.use_mtl else None
        return reg_out, cls_out


def get_baseline_factory(name: str):
    """Return (model_factory, is_baseline_flag) for a baseline name."""
    factories = {
        'MLP': lambda config: StockMLP(
            input_dim=config.num_features,
            seq_len=config.days_per_week * config.week_num,
            hidden_dims=config.baseline_mlp_dims),
        'GRU': lambda config: StockGRU(
            input_dim=config.num_features,
            hidden_dim=config.baseline_gru_dim),
        'GRU+Att': lambda config: StockGRUAtt(
            input_dim=config.num_features,
            hidden_dim=config.baseline_gru_dim),
        'FineNet': lambda config: FineNet(
            input_dim=config.num_features,
            short_dim=config.baseline_cnn_dims[0],
            long_dim=config.baseline_cnn_dims[1]),
        'RankLSTM': lambda config: RankLSTM(
            input_dim=config.num_features,
            hidden_dim=config.baseline_tgc_dim),
    }
    return factories[name]

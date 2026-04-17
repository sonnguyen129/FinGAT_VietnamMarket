"""Centralized configuration for FinGAT experiments on HOSE data."""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class Config:
    # === Paths ===
    data_dir: str = "datasets/VN_datasets"
    companies_file: str = "datasets/company_industries.csv"
    cache_dir: str = "data/cache"
    output_dir: str = "results"
    model_dir: str = "checkpoints"

    # === Data ===
    start_date: str = "2020-01-01"
    end_date: str = "2025-12-31"
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2
    min_trading_days: int = 800
    min_avg_volume: int = 10000

    # === Features ===
    num_features: int = 15
    days_per_week: int = 5

    # === Model ===
    hidden_dim: int = 16
    week_num: int = 3
    gat_heads: int = 1
    gat_dropout: float = 0.0

    # === Training ===
    epochs: int = 30
    batch_size: int = 128
    lr: float = 0.001
    weight_decay: float = 1e-4
    delta: float = 0.01  # Movement loss weight

    # === Loss weights (code-level) ===
    alpha_reg: float = 1.0   # Regression (MAE) loss weight
    beta_cls: float = 0.01   # Classification (BCE) loss weight
    gamma_rank: float = 1.0  # Pairwise ranking loss weight

    # === Evaluation ===
    top_k_values: Tuple = (5, 10, 20)
    num_runs: int = 10
    seed_base: int = 42

    # === Device ===
    device: str = "cuda"

    # === RankLSTM-specific ===
    ranklstm_alpha_search: List[float] = field(
        default_factory=lambda: [0.01, 0.1, 1.0, 10.0]
    )

    # === Baseline dims ===
    baseline_gru_dim: int = 32
    baseline_cnn_dims: Tuple = (32, 16)
    baseline_mlp_dims: Tuple = (32, 8)
    baseline_tgc_dim: int = 16

    # === Pair sampling for ranking loss ===
    max_pairs: int = 1000

    def __post_init__(self):
        import torch
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"

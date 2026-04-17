# PLAN.md — FinGAT Full Reproduction on HOSE (Vietnamese Stock Exchange)

> **Mục tiêu**: Reproduce toàn bộ thí nghiệm trong paper "FinGAT: Financial Graph Attention Networks for Recommending Top-K Profitable Stocks" (arXiv:2106.10159) trên dữ liệu sàn HOSE.
> **Paper reference**: Hsu, Tsai & Li (2021), IEEE TKDE.
> **Original repo**: github.com/Roytsai27/Financial-GraphAttention

---

## MỤC LỤC THÍ NGHIỆM (theo paper)

Paper có **5 evaluation questions (EQ1–EQ5)**, tương ứng với các nhóm thí nghiệm:

| EQ  | Thí nghiệm | Table/Figure | Mô tả |
|-----|-------------|-------------|--------|
| EQ1 | Main Results | Table 2 | So sánh FinGAT vs 5 baselines trên 3 datasets, K∈{5,10,20} |
| EQ2 | Without Sector Info | Figure 3 | FinGAT-NT trên 5 stock subsets, MRR@3 |
| EQ3 | Ablation Study | Table 3 | 5 variants: Full, w/o intra, w/o inter, w/o MTL, w/ MSE |
| EQ4 | Hyperparameter Analysis | Figure 4(a-f) | Số weeks, embedding dim, balancing δ |
| EQ5 | Attention Visualization | Figure 5,6,7 | Intra-sector heatmap, Inter-sector heatmap, Distribution plots |

**Thêm cho HOSE**: Thí nghiệm bổ sung so sánh với market-specific baselines.

---

## PHASE 0: PROJECT SETUP

### Step 0.1: Khởi tạo project structure

```
fingat-hose/
├── PLAN.md                         # File này
├── README.md
├── requirements.txt
├── config.py                       # Dataclass chứa ALL hyperparameters
│
├── data/
│   ├── collector.py                # Download HOSE OHLCV via vnstock
│   ├── sector_map.py               # ICB sector mapping
│   ├── features.py                 # 15 features per stock per day
│   ├── dataset.py                  # Sliding window, train/val/test split, pickle I/O
│   ├── graph_builder.py            # Intra-sector + inter-sector edge_index
│   ├── subset_builder.py           # Tạo 5 stock subsets cho EQ2
│   └── cache/                      # Raw + processed data
│
├── model/
│   ├── attentive_gru.py            # Attentive GRU (short-term + long-term)
│   ├── intra_sector_gat.py         # Intra-sector GAT layer
│   ├── inter_sector_gat.py         # Inter-sector GAT + graph pooling
│   ├── fingat.py                   # Full FinGAT model
│   ├── fingat_nt.py                # FinGAT-NT (no sector info, EQ2)
│   ├── loss.py                     # Multi-task loss (ranking + BCE + L2)
│   └── baselines.py               # MLP, GRU, GRU+Att, FineNet, RankLSTM
│
├── experiments/
│   ├── eq1_main_results.py         # EQ1: Main comparison, Table 2
│   ├── eq2_no_sector.py            # EQ2: Without sector info, Figure 3
│   ├── eq3_ablation.py             # EQ3: Ablation study, Table 3
│   ├── eq4_hyperparam.py           # EQ4: Hyperparameter sweep, Figure 4
│   ├── eq5_attention_viz.py        # EQ5: Attention weight visualization, Figure 5-7
│   └── runner.py                   # Unified experiment runner
│
├── train.py                        # Training loop (shared by all experiments)
├── evaluate.py                     # MRR@K, Precision@K, ACC, IRR
├── predict.py                      # Inference + top-K recommendation
├── visualize.py                    # Plotting: heatmaps, distributions, bar charts
└── utils.py                        # Seed, logging, device, timing
```

### Step 0.2: Dependencies (requirements.txt)

```
torch>=2.0.0
torch-geometric>=2.5.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
vnstock>=3.0.9
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
tabulate>=0.9.0
```

### Step 0.3: Config (config.py)

Tạo dataclass `Config` chứa TẤT CẢ hyperparameters. Mỗi experiment sẽ override các giá trị cần thiết.

```python
from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class Config:
    # === Paths ===
    data_dir: str = "data/cache"
    output_dir: str = "results"
    model_dir: str = "checkpoints"

    # === Data ===
    start_date: str = "2020-01-01"
    end_date: str = "2025-12-31"
    train_ratio: float = 0.6        # 60% train
    val_ratio: float = 0.2          # 20% val
    test_ratio: float = 0.2         # 20% test
    min_trading_days: int = 1200
    min_avg_volume: int = 10000

    # === Features ===
    num_features: int = 15          # Paper: 15 features per stock per day
    days_per_week: int = 5          # Trading days per week

    # === Model ===
    hidden_dim: int = 16            # Paper default, tested {8, 16, 32, 64}
    week_num: int = 3               # Paper default, tested {1, 2, 3, 4}
    gat_heads: int = 1              # Number of GAT attention heads
    gat_dropout: float = 0.0

    # === Training ===
    epochs: int = 30
    batch_size: int = 128           # Paper: 128
    lr: float = 0.001               # Paper searches {0.0005, 0.001, 0.005}
    weight_decay: float = 1e-4      # Paper λ = 0.0001
    delta: float = 0.01             # Paper optimal δ = 0.01

    # === Loss weights (code-level) ===
    alpha_reg: float = 1.0          # Regression (MAE) loss weight
    beta_cls: float = 0.01          # Classification (BCE) loss weight
    gamma_rank: float = 1.0         # Pairwise ranking loss weight

    # === Evaluation ===
    top_k_values: Tuple = (5, 10, 20)
    num_runs: int = 10              # Paper: average of 10 runs
    seed_base: int = 42

    # === Device ===
    device: str = "cuda"

    # === RankLSTM-specific ===
    ranklstm_alpha_search: List[float] = field(
        default_factory=lambda: [0.01, 0.1, 1.0, 10.0]
    )
```

### Step 0.4: Utility functions (utils.py)

```python
# Cần implement:
# - set_seed(seed): torch, numpy, random, cudnn deterministic
# - get_device(): auto-detect cuda/mps/cpu
# - Timer context manager
# - Logger setup
# - save_results(results_dict, path): save experiment results as JSON + CSV
```

---

## PHASE 1: DATA PIPELINE

### Step 1.1: Thu thập dữ liệu HOSE (data/collector.py)

**Input**: Không có.
**Output**: `data/cache/raw/{ticker}.csv` cho mỗi cổ phiếu.

```python
# Cần implement:
# 1. get_hose_tickers() → list of ~400-500 tickers
#    - Dùng Vnstock().listing_companies(), filter comGroupCode == 'HOSE'
#
# 2. download_ohlcv(ticker, start, end, source='VCI') → DataFrame
#    - Columns: date, open, high, low, close, volume
#    - Rate limiting: time.sleep(0.3) giữa các requests
#
# 3. download_all(tickers, start, end) → dict {ticker: DataFrame}
#    - Save từng ticker ra CSV: data/cache/raw/{ticker}.csv
#    - Retry logic cho failed downloads
#
# 4. filter_stocks(all_data, min_days, min_avg_volume) → filtered dict
#    - Loại stocks thiếu data, volume thấp, IPO giữa chừng
#    - Target: 150-300 stocks sau filter
```

**Lưu ý HOSE**:
- vnstock v3 cần register free API key tại vnstocks.com/login
- HOSE không có adjusted close riêng → dùng `close` thay `adjclose`
- Giới hạn giá ±7%/ngày ảnh hưởng phân phối return

### Step 1.2: Sector mapping (data/sector_map.py)

**Input**: Danh sách tickers đã filter.
**Output**: `data/cache/sector_mapping.json`

```python
# Cần implement:
# 1. build_sector_mapping(tickers) → dict {ticker: {industry, industryID}}
#    - Dùng Vnstock().stock(symbol, source='TCBS').overview()
#    - Fallback: dùng cột VN-sector trong listing_companies()
#
# 2. clean_sectors(mapping) → cleaned mapping
#    - Gộp sectors < 5 stocks vào "Other"
#    - Map tên tiếng Việt → English
#    - Target: ~8-12 sectors (paper Taiwan có 5, S&P có 9)
#
# 3. get_sector_list(mapping) → sorted list of unique sector names
# 4. get_sector_stock_counts(mapping) → dict {sector: count}
#
# Output format:
# {
#   "VNM": {"industry": "Consumer Staples", "industryID": 3577, "sector_idx": 0},
#   "VCB": {"industry": "Financials", "industryID": 8355, "sector_idx": 1},
#   ...
# }
```

### Step 1.3: Feature engineering (data/features.py)

**Input**: Raw OHLCV DataFrames.
**Output**: Feature matrices per stock.

Paper định nghĩa chính xác **15 features**:

| # | Feature | Công thức |
|---|---------|-----------|
| 1 | open | raw open price |
| 2 | close | raw close price |
| 3 | high | raw high price |
| 4 | low | raw low price |
| 5 | adjclose | adjusted close (= close cho HOSE) |
| 6 | return_ratio | (close_j - close_{j-1}) / close_{j-1} |
| 7 | pr_open | open/close - 1 |
| 8 | pr_high | high/close - 1 |
| 9 | pr_low | low/close - 1 |
| 10 | ma_5 | MA(5)/adjclose_{j-1} |
| 11 | ma_10 | MA(10)/adjclose_{j-1} |
| 12 | ma_15 | MA(15)/adjclose_{j-1} |
| 13 | ma_20 | MA(20)/adjclose_{j-1} |
| 14 | ma_25 | MA(25)/adjclose_{j-1} |
| 15 | ma_30 | MA(30)/adjclose_{j-1} |

```python
# Cần implement:
# 1. compute_features(df) → df with 15 feature columns
#    - Price-ratio: F_μ = μ/close - 1, μ ∈ {open, high, low}
#    - Moving-average: F_φ = (Σ adjclose_j / φ) / adjclose_{j-1}
#      φ ∈ {5, 10, 15, 20, 25, 30}
#    - Return ratio: R = (p_j - p_{j-1}) / p_{j-1}
#
# 2. normalize_features(df) → normalized features
#    - Option A (recommended): raw prices → price-relative (divide by prev close)
#    - Option B: per-stock z-score rolling window
#
# 3. align_dates(all_features) → aligned features with common date range
#    - Đảm bảo mọi stock có cùng date index
#    - Forward-fill gaps ≤ 3 days, drop stocks with larger gaps
```

### Step 1.4: Dataset construction (data/dataset.py)

**Input**: Feature matrices, sector mapping.
**Output**: `data/cache/hose_fingat_data.pickle`

Paper sử dụng **sliding window**: 3 tuần (15 ngày giao dịch) dự đoán return ngày thứ 16.

```python
# Cần implement:
# 1. create_weekly_windows(stock_features, num_weeks=3, days_per_week=5)
#    Output shapes (cho ~200 stocks):
#    - x_weeks: list of num_weeks arrays, each [num_samples, num_stocks, 5, 15]
#    - y_return: [num_samples, num_stocks]
#    - y_binary: [num_samples, num_stocks] (1 if return > 0)
#
# 2. split_data(x_weeks, y_return, y_binary, ratios=(0.6, 0.2, 0.2))
#    - Chronological split — KHÔNG shuffle (time series!)
#    - Paper: 60% train / 20% val / 20% test
#
# 3. save_dataset(splits, path) → pickle file
#    Format: {
#      'train': {'x1': array, 'x2': array, 'x3': array,
#                'y_return_ratio': array, 'y_up_or_down': array},
#      'val': {...},
#      'test': {...}
#    }
#
# 4. print_data_statistics(splits) → in ra table tương tự Table 1 trong paper:
#    | Market | # Stocks | # Sectors | # Train Days | # Val Days | # Test Days |
```

### Step 1.5: Graph construction (data/graph_builder.py)

**Input**: Tickers list, sector mapping.
**Output**: Edge index files (.npy)

```python
# Cần implement:
# 1. build_intra_sector_edges(tickers, sector_mapping)
#    - Fully-connected graph TRONG mỗi sector
#    - Output: edge_index [2, E_intra] (PyG format)
#    - Validation: E_intra = Σ_c n_c*(n_c - 1) cho mỗi sector c
#
# 2. build_inter_sector_edges(sector_list)
#    - Fully-connected graph GIỮA các sectors
#    - Output: edge_index [2, E_inter]
#    - Validation: E_inter = num_sectors * (num_sectors - 1)
#
# 3. build_sector_assignments(tickers, sector_mapping, sector_list)
#    - Output: tensor [num_stocks] mapping stock → sector index
#
# 4. Save tất cả:
#    - data/cache/hose_inner_edge.npy
#    - data/cache/hose_outer_edge.npy
#    - data/cache/hose_sector_assignments.npy
```

---

## PHASE 2: MODEL IMPLEMENTATIONS

### Step 2.1: Attentive GRU (model/attentive_gru.py)

```python
# class AttentiveGRU(nn.Module):
#   - __init__(input_dim, hidden_dim)
#   - forward(x: [batch, seq_len, input_dim]) → [batch, hidden_dim]
#   - GRU + feed-forward attention (Luong-style)
#   - α_j = softmax(tanh(W_0 · h_j))
#   - output = Σ α_j · h_j
#
# Unit test: input [50, 5, 15] → output [50, 16]
```

### Step 2.2: Intra-sector GAT (model/intra_sector_gat.py)

```python
# class IntraSectorGAT(nn.Module):
#   - Dùng torch_geometric.nn.GATConv
#   - in_channels=hidden_dim, out_channels=hidden_dim
#   - heads=1, concat=False, add_self_loops=True
#   - forward(x, edge_index) → ReLU(GAT(x, edge_index))
#
# Eq. 7: GAT(G_πc; sq) = ReLU(Σ β_qn · W1 · a_sn)
# Eq. 8: β_qn = softmax(LeakyReLU(r^T[W2·a_sq || W2·a_sn]))
```

### Step 2.3: Inter-sector GAT + Graph Pooling (model/inter_sector_gat.py)

```python
# class IntraSectorGraphPooling(nn.Module):
#   - Element-wise max pooling over stocks in each sector
#   - Eq. 11: z_πc = MaxPool({τ_G(sq) | ∀sq ∈ M_πc})
#   - Input: τ_G [num_stocks, hidden_dim], sector_assignments [num_stocks]
#   - Output: sector_embeddings [num_sectors, hidden_dim]
#
# class InterSectorGAT(nn.Module):
#   - Same architecture as IntraSectorGAT nhưng on sector graph
#   - Eq. 12: τ_i(πc) = GAT(G_π, πc)
```

### Step 2.4: Full FinGAT model (model/fingat.py)

```python
# class FinGAT(nn.Module):
#   Components:
#   1. short_term_gru: AttentiveGRU(input_dim=15, hidden_dim=16)
#      → a_sq per week
#   2. intra_gat: IntraSectorGAT(hidden_dim=16)
#      → g_sq per week
#   3. long_term_gru_A: AttentiveGRU(hidden_dim=16, hidden_dim=16)
#      → τ_A(sq) over all weeks
#   4. long_term_gru_G: AttentiveGRU(hidden_dim=16, hidden_dim=16)
#      → τ_G(sq) over all weeks
#   5. graph_pooling: IntraSectorGraphPooling()
#      → z_πc per sector
#   6. inter_gat: InterSectorGAT(hidden_dim=16)
#      → τ(πc) per sector
#   7. fusion: Linear(hidden_dim * 3, hidden_dim)
#      → Eq. 13: τ_F(sq) = ReLU([τ_G || τ_A || τ(πc)] · Wf)
#   8. reg_head: Linear(hidden_dim, 1) → ŷ_return
#   9. cls_head: Linear(hidden_dim, 1) + Sigmoid → ŷ_move
#
#   CONSTRUCTOR FLAGS cho ablation (EQ3):
#   - use_intra: bool = True    → bật/tắt intra-sector GAT
#   - use_inter: bool = True    → bật/tắt inter-sector GAT
#   - use_mtl: bool = True      → bật/tắt multi-task (BCE loss)
#   - use_mse: bool = False     → thay BCE bằng MSE
#
# forward(weekly_inputs: list of [num_stocks, 5, 15]) → (reg_out, cls_out)
#
# QUAN TRỌNG: Xavier uniform initialization cho tất cả parameters
```

### Step 2.5: FinGAT-NT — No sector info variant (model/fingat_nt.py)

**Cho EQ2**. Khác FinGAT ở:
1. KHÔNG có sector-level modeling (bỏ graph pooling + inter-sector GAT)
2. KHÔNG có intra-sector graph riêng biệt → tạo 1 fully-connected graph GT của TẤT CẢ stocks
3. Eq. 19: π_F(sq) = ReLU([π_G(sq) || π_A(sq)] · Wf) — chỉ concat 2 vectors thay vì 3

```python
# class FinGAT_NT(nn.Module):
#   1. short_term_gru: AttentiveGRU(15, 16)
#   2. all_stock_gat: GATConv(16, 16) — fully-connected graph of ALL stocks
#   3. long_term_gru_G: AttentiveGRU(16, 16)
#   4. long_term_gru_A: AttentiveGRU(16, 16)
#   5. fusion: Linear(16 * 2, 16)  ← CHỈ 2 embeddings (không có sector)
#   6. reg_head + cls_head
#
# NOTE: Fully-connected graph of N stocks = N*(N-1) edges
# Cho 10 stocks (EQ2): 90 edges → OK
# Cho 200 stocks: 39,800 edges → tốn memory, nhưng vẫn feasible
```

### Step 2.6: Baseline models (model/baselines.py)

Paper so sánh với **5 baselines**. Phải implement TẤT CẢ:

```python
# 1. MLP (Tang et al. 2015, [26])
# class StockMLP(nn.Module):
#   - 2 hidden layers: 32-dim → 8-dim → output
#   - Input: flatten toàn bộ features (15 * days_per_week * num_weeks)
#   - Output: predicted return ratio
#   - Loss: cùng multi-task loss như FinGAT

# 2. GRU (Cho et al. 2014, [7])
# class StockGRU(nn.Module):
#   - 1 GRU layer, hidden_dim=32
#   - Input: sequence of daily features [seq_len, 15]
#   - Output: last hidden state → linear → return prediction

# 3. GRU+Att (Dhingra et al. 2017, [9])
# class StockGRUAtt(nn.Module):
#   - 1 GRU layer 32-dim + attention layer
#   - Giống AttentiveGRU nhưng hidden_dim=32, standalone (no graph)

# 4. FineNet (Tsai et al. 2019, [27])
# class FineNet(nn.Module):
#   - Joint CNN + RNN:
#   - Short-term: 1D dilated CNN (32-dim conv + 16-dim conv)
#   - Long-term: 1D dilated CNN
#   - Fusion → prediction
#   - KEY: 2 dilated convolution neural networks for joint
#          long-term and short-term sequential patterns

# 5. RankLSTM (Feng et al. 2019, [13])
# class RankLSTM(nn.Module):
#   - Temporal graph convolution + LSTM
#   - 16-dim embeddings
#   - Pairwise ranking loss + pointwise regression loss
#   - Uses PRE-DEFINED graph relations
#   - Hyperparameter α searched in {0.01, 0.1, 1, 10}
#   - NOTE: Cho HOSE, dùng sector-based connections làm proxy
#     (2 stocks linked nếu cùng sector)
```

**QUAN TRỌNG về baselines**: Mỗi baseline dùng CÙNG loss function (multi-task) và CÙNG evaluation procedure như FinGAT. Chỉ khác ở model architecture. Predicted return ratio từ mỗi model → rank stocks → compute MRR, Precision.

### Step 2.7: Multi-task loss (model/loss.py)

```python
# class FinGATLoss(nn.Module):
#   Paper Eq. 15: L = (1-δ)·L_rank + δ·L_move + λ·||Θ||²
#
#   L_rank (Eq. 16 top): Pairwise ranking loss
#     = Σ_i Σ_sq Σ_sk max(0, -Δ̂ × Δ)
#     where Δ̂ = ŷ_return(sq) - ŷ_return(sk)
#           Δ = y_return(sq) - y_return(sk)
#     → Penalize khi predicted ranking order ≠ ground-truth order
#
#   L_move (Eq. 16 bottom): Binary cross-entropy
#     = -Σ [y_move·log(ŷ_move) + (1-y_move)·log(1-ŷ_move)]
#
#   λ·||Θ||²: L2 regularization (handled by optimizer weight_decay)
#
#   IMPORTANT IMPLEMENTATION DETAIL:
#   Pairwise ranking loss là O(N²) → cần sampling cho N>100 stocks
#   Recommendation: random sample 500-1000 pairs per time step
#
#   Variant cho ablation:
#   - w/o MTL: loss = L_rank only (set delta=0)
#   - w/ MSE: replace L_move (BCE) with L_MSE, remove sigmoid
```

---

## PHASE 3: TRAINING INFRASTRUCTURE

### Step 3.1: Training loop (train.py)

```python
# def train_model(model, data, config, experiment_name):
#   """Generic training loop used by ALL experiments."""
#
#   Key implementation points:
#   1. Xavier uniform init cho tất cả parameters có dim > 1
#   2. Adam optimizer, lr=config.lr, weight_decay=config.weight_decay
#   3. Training: iterate over time steps (not batch of stocks)
#      - Mỗi time step t: feed weekly_inputs cho TẤT CẢ stocks cùng lúc
#      - Compute loss trên toàn bộ stocks tại time step t
#   4. Validation: evaluate MRR@5 trên val set mỗi epoch
#   5. Early stopping: save best model based on val MRR@5
#   6. Return: best model state_dict, training history
#
#   Per-epoch flow:
#   for t in range(num_train_samples):
#       weekly = [data['train'][f'x{w+1}'][t] for w in range(week_num)]
#       y_ret = data['train']['y_return_ratio'][t]
#       y_bin = data['train']['y_up_or_down'][t]
#       reg_out, cls_out = model(weekly)
#       loss = criterion(reg_out, cls_out, y_ret, y_bin)
#       loss.backward()
#       optimizer.step()
#
# def train_multi_run(model_factory, data, config, num_runs=10):
#   """Train num_runs times with different seeds, return all results."""
#   all_results = []
#   for run in range(num_runs):
#       set_seed(config.seed_base + run)
#       model = model_factory(config)
#       results = train_model(model, data, config, f"run_{run}")
#       all_results.append(results)
#   return aggregate_results(all_results)  # mean ± std
```

### Step 3.2: Evaluation metrics (evaluate.py)

Paper dùng **3 metrics chính** + accuracy. Implement theo đúng Eq. 17-18.

```python
# 1. MRR@K (Eq. 17):
#    MRR@K = (1/K) Σ_{sq ∈ top-K predicted} 1/rank_true(sq)
#    where rank_true(sq) = ground-truth rank of sq based on actual return
#
# 2. Precision@K (Eq. 18):
#    P@K = |predicted_top_K ∩ actual_top_K| / K
#
# 3. ACC:
#    = correct binary movement predictions / total predictions
#
# 4. (Thêm) IRR@K: Investment Return Ratio
#    = Σ actual_return[predicted_top_K]  (total return nếu invest vào top-K)
#
# QUAN TRỌNG:
# - Tính trên TỪNG test day → average across all test days
# - Report average of 10 runs (10 random seeds)
# - Stocks ranked by predicted return ratio (higher = better)
#
# def evaluate_model(model, data, config):
#     """Evaluate trên test set, return dict of metrics."""
#     model.eval()
#     all_mrr = {k: [] for k in config.top_k_values}
#     all_prec = {k: [] for k in config.top_k_values}
#     all_acc = []
#
#     for t in range(num_test_samples):
#         weekly = [data['test'][f'x{w+1}'][t] for w in range(week_num)]
#         reg_out, cls_out = model(weekly)
#         y_true = data['test']['y_return_ratio'][t]
#         y_binary = data['test']['y_up_or_down'][t]
#
#         for k in config.top_k_values:
#             all_mrr[k].append(mrr_at_k(y_true, reg_out, k))
#             all_prec[k].append(precision_at_k(y_true, reg_out, k))
#         all_acc.append(accuracy(y_binary, reg_out))
#
#     return {
#         f'MRR@{k}': np.mean(all_mrr[k]) for k in config.top_k_values,
#         f'Precision@{k}': np.mean(all_prec[k]) for k in config.top_k_values,
#         'ACC': np.mean(all_acc),
#     }
```

---

## PHASE 4: THÍ NGHIỆM EQ1 — MAIN RESULTS (Table 2)

### Step 4.1: experiments/eq1_main_results.py

**Mục tiêu**: Reproduce Table 2 — so sánh FinGAT vs 5 baselines trên HOSE data.

```python
# EXPERIMENT SPECIFICATION:
#
# Models to compare (6 total + 2 HOSE-specific):
# 1. MLP: 2 hidden layers (32, 8)
# 2. GRU: 1 layer, hidden_dim=32
# 3. GRU+Att: 1 layer 32-dim + attention
# 4. FineNet: dilated CNN (32-dim + 16-dim) + RNN
# 5. RankLSTM: temporal graph conv 16-dim + pairwise ranking
#    - Search α ∈ {0.01, 0.1, 1, 10}, pick best on val
# 6. FinGAT: full model, hidden_dim=16
# 7. (bonus) Momentum: rank by past-week return (no learning)
# 8. (bonus) Random: random ranking (sanity check)
#
# Dataset: HOSE full (tất cả stocks đã filter)
#
# Evaluation:
# - K ∈ {5, 10, 20}
# - Metrics: MRR@K, Precision@K, ACC
# - Average over 10 runs (except Momentum/Random: 1 run)
#
# Output format (giống Table 2):
# | Model     | K=5 MRR | K=5 Prec | K=10 MRR | K=10 Prec | K=20 MRR | K=20 Prec | ACC   |
# |-----------|---------|----------|----------|-----------|----------|-----------|-------|
# | MLP       |         |          |          |           |          |           |       |
# | GRU       |         |          |          |           |          |           |       |
# | GRU+Att   |         |          |          |           |          |           |       |
# | FineNet   |         |          |          |           |          |           |       |
# | RankLSTM  |         |          |          |           |          |           |       |
# | FinGAT    |         |          |          |           |          |           |       |
# | Improv.   |  XX%    |   XX%    |   XX%    |    XX%    |   XX%    |    XX%    |  XX%  |
#
# Improv. = (FinGAT - best_baseline) / best_baseline * 100
#
# PROCEDURE per run:
# 1. Set seed
# 2. Load data (shared across models)
# 3. For each model:
#    a. Initialize model
#    b. Train on train set
#    c. Select best epoch on val set (early stopping by MRR@5)
#    d. Evaluate on test set
#    e. Record metrics
# 4. After 10 runs: compute mean ± std for each metric
```

---

## PHASE 5: THÍ NGHIỆM EQ2 — WITHOUT SECTOR INFO (Figure 3)

### Step 5.1: Tạo 5 stock subsets (data/subset_builder.py)

Paper tạo **5 subsets từ Taiwan Stock** dựa trên **market value**. Adapt cho HOSE:

```python
# Cần implement:
# 1. Lấy market cap cho mỗi stock
#    - vnstock: Vnstock().stock(symbol, source='TCBS').overview()['mktCap']
#    - Hoặc dùng listing_companies()['marketCap'] nếu có
# 2. Sort stocks by market cap (descending)
# 3. Tạo 5 subsets:
#
#    a. "Best 10": 10 stocks market cap cao nhất
#       → HOSE example: VIC, VHM, VCB, BID, GAS, HPG, MSN, VNM, TCB, MBB
#
#    b. "Worst 10": 10 stocks market cap thấp nhất (trong filtered set)
#
#    c. "Best 5 Worst 5": 5 highest + 5 lowest market cap
#
#    d. "Random 10": random sample 10 stocks (seed=42 cho reproducibility)
#
#    e. "Uniform 10": chia danh sách sorted thành 10 zones,
#       random 1 stock mỗi zone
#
# 4. Cho mỗi subset: rebuild dataset with only those 10 stocks
# 5. Build fully-connected graph cho 10 stocks (FinGAT-NT: NO sector info)
```

### Step 5.2: experiments/eq2_no_sector.py

```python
# EXPERIMENT SPECIFICATION:
#
# Model: FinGAT-NT (no sector info, Eq. 19)
# Baselines: MLP, GRU, GRU+Att, FineNet, RankLSTM
# (ALL trained WITHOUT sector info on small subsets)
#
# Datasets: 5 subsets × 10 stocks each
# Metric: MRR@3 (paper dùng K=3 cho experiment này, KHÔNG phải K=5!)
#
# Output format (giống Figure 3 — grouped bar chart):
# X-axis: 5 subsets (Best 10, Worst 10, Random 10, Best5 Worst5, Uniform 10)
# Y-axis: MRR@3 (range ~0.8 to 1.0 in paper)
# Bars: 6 models side-by-side per subset
# Colors: MLP=gray, GRU=blue, GRU+Att=green, FineNet=orange, RankLSTM=red, FinGAT-NT=purple
#
# PROCEDURE:
# For each of 5 subsets:
#   1. Extract 10-stock data from full dataset
#   2. Build fully-connected graph for FinGAT-NT (10*9=90 edges)
#   3. Build sector-based graph for RankLSTM (subset stocks may cross sectors)
#   4. For each of 6 models:
#      a. Train on subset train data
#      b. Evaluate MRR@3 on subset test data
#      c. Average over 10 runs
#
# Expected findings to verify:
# - FinGAT-NT outperforms all baselines on all subsets
# - "Best 10", "Worst 10", "Best 5 Worst 5" show LARGER gap
#   (homogeneous stocks → latent relations are stronger)
# - "Random 10", "Uniform 10" show smaller but still positive gap
```

---

## PHASE 6: THÍ NGHIỆM EQ3 — ABLATION STUDY (Table 3)

### Step 6.1: experiments/eq3_ablation.py

**Mục tiêu**: Reproduce Table 3 — kiểm chứng đóng góp của từng component.

```python
# EXPERIMENT SPECIFICATION:
#
# 5 model variants (implement via constructor flags in fingat.py):
#
# 1. "Full model": FinGAT(use_intra=True, use_inter=True, use_mtl=True, use_mse=False)
#    - All components active
#
# 2. "w/o intra": FinGAT(use_intra=False, use_inter=True, use_mtl=True)
#    - Remove: intra-sector GAT + long_term_gru_G
#    - Keep: τ_A(sq) and τ(πc)
#    - Fusion input dim: hidden_dim * 2 (not 3)
#    - How to compute sector pooling without τ_G? → pool from τ_A instead
#
# 3. "w/o inter": FinGAT(use_intra=True, use_inter=False, use_mtl=True)
#    - Remove: graph_pooling + inter-sector GAT
#    - Keep: τ_G(sq) and τ_A(sq)
#    - Fusion input dim: hidden_dim * 2 (not 3)
#
# 4. "w/o MTL": FinGAT(use_intra=True, use_inter=True, use_mtl=False)
#    - Loss = L_rank ONLY (delta=0, no L_move)
#    - Still produce reg_out, but no cls_out or BCE loss
#
# 5. "w/ MSE": FinGAT(use_intra=True, use_inter=True, use_mtl=True, use_mse=True)
#    - Replace BCE movement loss with MSE regression loss
#    - Remove sigmoid in cls_head (Eq. 14)
#    - L = (1-δ)·L_rank + δ·L_MSE
#
# Evaluation: Same as EQ1 (K∈{5,10,20}, MRR, Precision, ACC)
# Dataset: HOSE full (same as EQ1)
# Runs: 10 per variant
#
# Output format (giống Table 3):
# | Model      | K=5 MRR | K=5 Prec | K=10 MRR | K=10 Prec | K=20 MRR | K=20 Prec | ACC   |
# |------------|---------|----------|----------|-----------|----------|-----------|-------|
# | Full model |         |          |          |           |          |           |       |
# | w/o intra  |         |          |          |           |          |           |       |
# | w/o inter  |         |          |          |           |          |           |       |
# | w/o MTL    |         |          |          |           |          |           |       |
# | w/ MSE     |         |          |          |           |          |           |       |
#
# Expected findings:
# - Full model BEST across all metrics
# - w/o intra → LARGEST drop (intra-sector relations most important)
# - w/o MTL → SMALLEST drop (but MTL still helps)
# - w/ MSE → DRASTIC drop (MSE loss harder to optimize than BCE)
```

---

## PHASE 7: THÍ NGHIỆM EQ4 — HYPERPARAMETER ANALYSIS (Figure 4)

### Step 7.1: experiments/eq4_hyperparam.py

**Mục tiêu**: Reproduce Figure 4(a-f) — phân tích 3 hyperparameters.

```python
# EXPERIMENT SPECIFICATION:
# Default fixed values when not varying: week_num=3, hidden_dim=16, delta=0.01
# All experiments on HOSE full dataset, FinGAT full model
#
# === Experiment 4a+4b: Number of training weeks ===
# Vary: week_num ∈ {1, 2, 3, 4}
# For each value: train FinGAT, evaluate MRR@K and Precision@K, K∈{5,10,20}
# Average over 10 runs
#
# Output: 2 bar/line charts (Figure 4a = MRR, Figure 4b = Precision)
#   X-axis: Number of weeks (1, 2, 3, 4)
#   Y-axis: metric value
#   3 grouped bars/lines: K=5, K=10, K=20
#
# Expected: 3+ weeks → better; 1-2 weeks → worse (missing long-term info)

# === Experiment 4c+4d: Embedding dimension ===
# Vary: hidden_dim ∈ {8, 16, 32, 64}
# Same eval procedure
#
# Output: 2 bar/line charts (Figure 4c = MRR, Figure 4d = Precision)
#   X-axis: Embedding Dimension (8, 16, 32, 64)
#
# Expected: 16 best; 8 underfitting; 32/64 overfitting

# === Experiment 4e+4f: Balancing parameter δ ===
# Vary: delta ∈ {0, 1e-4, 1e-3, 1e-2, 1e-1, 1}
# NOTE:
#   delta=0 → L_rank only (no movement prediction)
#   delta=1 → L_move only (no ranking)
#
# Output: 2 bar/line charts (Figure 4e = MRR, Figure 4f = Precision)
#   X-axis: Loss Weight (0, 10^-4, 10^-3, 10^-2, 10^-1, 1) — quasi-log scale
#
# Expected: delta=0.01 best; extremes (0 or 1) much worse

# TOTAL TRAINING RUNS:
# (4 + 4 + 6) values × 10 runs = 140 runs
# Optimization: reduce num_runs=3 for quick sweep, 10 for final
```

---

## PHASE 8: THÍ NGHIỆM EQ5 — ATTENTION VISUALIZATION (Figure 5, 6, 7)

### Step 8.1: Extract attention weights from trained model

```python
# Modify GAT layers to optionally return attention weights:
#
# In IntraSectorGAT and InterSectorGAT:
#   forward(x, edge_index, return_attention=False):
#       if return_attention:
#           out, (edge_idx, alpha) = self.gat(x, edge_index,
#                                             return_attention_weights=True)
#           return out, alpha  # alpha: [num_edges, heads]
#       else:
#           return self.gat(x, edge_index)
#
# In FinGAT: add method extract_attention(weekly_inputs)
#   → returns dict with 'intra_attention' and 'inter_attention' matrices
```

### Step 8.2: experiments/eq5_attention_viz.py

```python
# EXPERIMENT SPECIFICATION:
# Use the BEST trained model from EQ1 (or train a fresh one)
# Select random test instances for visualization
#
# === Figure 5: Intra-sector attention heatmaps ===
#
# 5(a): Pick 1 large sector (e.g., "Financials" for HOSE — ~30 stocks)
#   - From 1 random test instance:
#   - Extract intra-sector attention weights β_qn for that sector
#   - Reshape to [n_stocks_in_sector, n_stocks_in_sector]
#   - Plot heatmap (seaborn.heatmap)
#   - Axes: stock indices (0, 1, ..., n-1)
#   - Colorbar: attention weight intensity
#
# 5(b): Pick a different sector (e.g., "Real Estate" or "Industrials")
#   - Same procedure as 5(a)
#   - Compare patterns: subgroups? uniform? dominant stocks?
#
# HOSE-specific sectors to visualize:
# - "Ngân hàng" (Banking): VCB, BID, CTG, TCB, MBB, VPB, ACB, STB, HDB, LPB, ...
# - "Bất động sản" (Real Estate): VIC, VHM, NVL, DXG, KDH, ...

# === Figure 6: Inter-sector attention heatmap ===
#
# 6(a): From 1 random test instance:
#   - Extract inter-sector attention weights
#   - Matrix [num_sectors, num_sectors]
#   - Plot heatmap with sector names as labels
#
# Expected for HOSE:
# - "Financials" ↔ "Real Estate" high attention (known strong coupling)
# - "Energy" may influence multiple sectors

# === Figure 7: Distributions of attention weights ===
#
# 7(a): Collect ALL attention weights across ALL test instances
#   - Separate into intra-sector and inter-sector sets
#   - Plot overlaid histograms (density plot)
#   - Blue: inter-sector, Orange: intra-sector
#   - X: Attention Score, Y: Frequency
#
# Expected:
# - Inter-sector: right-skewed (few dominant sectors)
# - Intra-sector: concentrated around 1/n_stocks_in_sector
#
# 7(b): For each attention edge:
#   - Compute variance across all test instances
#   - Plot overlaid histograms of variances
#   - Blue: inter-sector variance, Orange: intra-sector variance
#
# Expected:
# - Inter-sector: higher variance (sector dominance changes)
# - Intra-sector: lower variance (stable within-sector relations)
```

### Step 8.3: Visualization code (visualize.py)

```python
# Cần implement 7 plotting functions:
#
# 1. plot_intra_sector_heatmap(attn_matrix, sector_name, stock_ids, save_path)
#    → seaborn heatmap, diverging colormap (Blues)
#
# 2. plot_inter_sector_heatmap(attn_matrix, sector_names, save_path)
#    → seaborn heatmap, annotated with sector names
#
# 3. plot_attention_distribution(intra_weights, inter_weights, save_path)
#    → overlaid density histograms
#
# 4. plot_attention_variance(intra_vars, inter_vars, save_path)
#    → overlaid density histograms
#
# 5. generate_table(results_dict, caption, save_path)
#    → formatted LaTeX/markdown table (for Table 2, Table 3)
#
# 6. plot_eq2_barchart(results_dict, save_path)
#    → grouped bar chart (Figure 3)
#
# 7. plot_eq4_line_charts(results_dict, param_name, metric_name, save_path)
#    → line chart with 3 lines for K=5,10,20 (Figure 4)
#
# Style settings:
# - matplotlib rcParams: font.size=12, figure.dpi=300
# - Save as both PNG and PDF
# - Consistent color palette across all figures
```

---

## PHASE 9: EXPERIMENT RUNNER & REPORTING

### Step 9.1: Unified experiment runner (experiments/runner.py)

```python
# def run_all_experiments(config):
#     """Run all 5 experiment groups sequentially."""
#
#     # Phase 0: Prepare data (only once)
#     data = prepare_data(config)  # returns loaded pickle data
#     edges = load_edges(config)   # returns intra/inter edge_index
#
#     # EQ1: Main Results (~60 training runs)
#     print("=" * 60)
#     print("EQ1: Main Results (Table 2)")
#     eq1 = run_eq1(data, edges, config)
#     save_results(eq1, "results/eq1_main_results.json")
#     generate_table(eq1, "results/eq1_table2.csv")
#     print(format_table(eq1))
#
#     # EQ2: No Sector Info (~300 runs but small data)
#     print("=" * 60)
#     print("EQ2: Without Sector Info (Figure 3)")
#     eq2 = run_eq2(data, config)
#     save_results(eq2, "results/eq2_no_sector.json")
#     plot_eq2_barchart(eq2, "results/eq2_figure3.png")
#
#     # EQ3: Ablation Study (~50 runs)
#     print("=" * 60)
#     print("EQ3: Ablation Study (Table 3)")
#     eq3 = run_eq3(data, edges, config)
#     save_results(eq3, "results/eq3_ablation.json")
#     generate_table(eq3, "results/eq3_table3.csv")
#
#     # EQ4: Hyperparameter Analysis (~140 runs)
#     print("=" * 60)
#     print("EQ4: Hyperparameter Analysis (Figure 4)")
#     eq4 = run_eq4(data, edges, config)
#     save_results(eq4, "results/eq4_hyperparam.json")
#     plot_eq4_all(eq4, "results/")
#
#     # EQ5: Attention Visualization (1 model + extract)
#     print("=" * 60)
#     print("EQ5: Attention Visualization (Figures 5-7)")
#     eq5 = run_eq5(data, edges, config)
#     plot_eq5_all(eq5, "results/")
#
#     # Generate full report
#     generate_full_report(eq1, eq2, eq3, eq4, eq5, "results/full_report.md")
#     print("All experiments complete! Results in results/")
```

### Step 9.2: Output file structure

```
results/
├── eq1_main_results.json          # Raw numbers
├── eq1_table2.csv                 # Formatted Table 2
├── eq2_no_sector.json
├── eq2_figure3.png                # Bar chart Figure 3
├── eq3_ablation.json
├── eq3_table3.csv                 # Formatted Table 3
├── eq4_hyperparam.json
├── eq4_fig4a_weeks_mrr.png
├── eq4_fig4b_weeks_precision.png
├── eq4_fig4c_embed_mrr.png
├── eq4_fig4d_embed_precision.png
├── eq4_fig4e_delta_mrr.png
├── eq4_fig4f_delta_precision.png
├── eq5_fig5a_intra_sector1.png    # Heatmap
├── eq5_fig5b_intra_sector2.png    # Heatmap
├── eq5_fig6_inter_sector.png      # Heatmap
├── eq5_fig7a_attn_dist.png        # Distribution
├── eq5_fig7b_attn_var.png         # Variance distribution
└── full_report.md                 # Summary with key findings
```

---

## PHASE 10: STEP-BY-STEP IMPLEMENTATION ORDER CHO CLAUDE CODE

Thực hiện tuần tự. Mỗi step PHẢI test xong trước khi qua step sau.

### Batch 1: Foundation (Steps 1-5)

| Step | File(s) | Task | Test criteria |
|------|---------|------|---------------|
| 1 | `requirements.txt`, `config.py`, `utils.py` | Setup project | `import torch; import torch_geometric` passes |
| 2 | `data/collector.py` | Download VN30 (30 stocks) để test nhanh | 30 CSV files in `data/cache/raw/` |
| 3 | `data/sector_map.py` | Build sector mapping | JSON with all 30 tickers mapped to sectors |
| 4 | `data/features.py` | Compute 15 features | No NaN after warmup (30 days), shapes correct |
| 5 | `data/dataset.py` | Sliding windows + splits | Shapes match: `[N, 30, 5, 15]`, dates contiguous |

### Batch 2: Graphs + Core Model (Steps 6-10)

| Step | File(s) | Task | Test criteria |
|------|---------|------|---------------|
| 6 | `data/graph_builder.py` | Build edge indices | Edge counts match formula |
| 7 | `model/attentive_gru.py` | Attentive GRU | `[50, 5, 15]` → `[50, 16]` |
| 8 | `model/intra_sector_gat.py` | Intra-sector GAT | `[30, 16]` + edges → `[30, 16]` |
| 9 | `model/inter_sector_gat.py` | Inter-sector GAT + pooling | `[30, 16]` → `[5, 16]` → `[5, 16]` |
| 10 | `model/fingat.py` | Full FinGAT | Forward pass: random data → `[30, 1]` × 2 |

### Batch 3: Loss + Training + First Results (Steps 11-14)

| Step | File(s) | Task | Test criteria |
|------|---------|------|---------------|
| 11 | `model/loss.py` | Multi-task loss | Gradients flow, loss ≠ NaN |
| 12 | `train.py`, `evaluate.py` | Training loop + metrics | Loss decreases on VN30, MRR > random |
| 13 | `model/baselines.py` | Implement ALL 5 baselines | Each trains and predicts without error |
| 14 | `experiments/eq1_main_results.py` | **RUN EQ1** | Table 2 generated with 6+ models |

### Batch 4: Remaining Experiments (Steps 15-19)

| Step | File(s) | Task | Test criteria |
|------|---------|------|---------------|
| 15 | `model/fingat_nt.py`, `data/subset_builder.py` | FinGAT-NT + 5 subsets | Forward pass works on 10-stock subset |
| 16 | `experiments/eq2_no_sector.py` | **RUN EQ2** | Figure 3 bar chart generated |
| 17 | `experiments/eq3_ablation.py` | **RUN EQ3** | Table 3 with 5 variants generated |
| 18 | `experiments/eq4_hyperparam.py` | **RUN EQ4** | 6 line/bar charts (Fig 4a-f) generated |
| 19 | `experiments/eq5_attention_viz.py`, `visualize.py` | **RUN EQ5** | 5 attention plots (Fig 5-7) generated |

### Batch 5: Scale to Full HOSE + Final Report (Steps 20-22)

| Step | File(s) | Task | Test criteria |
|------|---------|------|---------------|
| 20 | `data/collector.py` | Scale to full HOSE (200+ stocks) | All CSVs downloaded, features computed |
| 21 | ALL `experiments/eq*.py` | Re-run ALL EQ1-EQ5 on full data, 10 runs each | Complete results in `results/` |
| 22 | `experiments/runner.py` | Full pipeline automation + report | `results/full_report.md` generated |

---

## APPENDIX A: CHI TIẾT KỸ THUẬT QUAN TRỌNG

### A.1: Exact training settings (Paper Section 5.1)

```
GRU hidden dim:     16 (all GRU layers in FinGAT)
GAT hidden dim:     16 (all GAT layers in FinGAT)
Baseline GRU dim:   32 (for standalone GRU, GRU+Att)
Baseline CNN dims:  32 + 16 (for FineNet)
Baseline TGC dim:   16 (for RankLSTM)
MLP layers:         32 → 8

Learning rate:      search {0.0005, 0.001, 0.005}
Batch size:         128
δ (loss balance):   0.01
λ (L2 reg):         0.0001
Optimizer:          Adam
Init:               Xavier uniform

RankLSTM α search:  {0.01, 0.1, 1, 10}
```

### A.2: Data splitting protocol

```
Taiwan Stock & S&P 500:
  Total: 965 days → 579 train / 193 val / 193 test (60/20/20)
NASDAQ:
  Total: 1245 days → 756 train / 252 val / 237 test

Each data instance:
  Input: 3 consecutive weeks (15 trading days) of features
  Target: return ratio on day 16 (first day of week 4)
  Sliding window: shift by 1 day

Evaluation:
  "# Testing Days × # Stocks" results per dataset
  Average of 10 runs with different random seeds
```

### A.3: HOSE-specific adaptations

| Aspect | Taiwan/US (paper) | HOSE (adaptation) |
|--------|-------------------|-------------------|
| Adjusted close | Available | Use raw close |
| Daily price limit | None / varies | ±7% |
| Settlement | T+1 | T+2 (consider predicting T+2) |
| Sector system | GICS-like | ICB (11 Level-1 industries) |
| Dominant sectors | Semiconductors | Financials + Real Estate (~47%) |
| Pre-defined relations | Investment facts (RankLSTM) | Sector-based proxy |
| Trading holidays | ~10 days/year | ~15+ days/year (incl. Tết ~7-10 days) |
| Data source | Yahoo Finance | vnstock (TCBS/VCI/KBS APIs) |

### A.4: Computational budget estimate

```
Per single training run (200 stocks, 30 epochs, GPU):
  FinGAT:     ~15-30 min
  Baselines:  ~5-15 min each

Total experiments (10 runs each):
  EQ1: 6 models × 10 runs                    = 60 runs   → ~15-20 GPU hrs
  EQ2: 6 models × 5 subsets × 10 runs        = 300 runs  → ~5 GPU hrs (small data)
  EQ3: 5 variants × 10 runs                  = 50 runs   → ~10-15 GPU hrs
  EQ4: (4+4+6) configs × 10 runs             = 140 runs  → ~30-45 GPU hrs
  EQ5: 1 trained model + visualization        = 1 run     → ~1 GPU hr

  TOTAL ESTIMATE: ~60-90 GPU hours

  Quick pass (3 runs instead of 10):          ~20-30 GPU hours
```

### A.5: Key formulas quick reference

```
Eq. 1:  R_ij = (p_j - p_{j-1}) / p_{j-1}                    # Return ratio
Eq. 2:  F_μ = μ_j / close_j - 1                              # Price-ratio features
Eq. 3:  F_φ = (Σ adjclose / φ) / adjclose_{j-1}              # Moving-avg features
Eq. 4:  h_j = GRU(v_j, h_{j-1})                              # GRU hidden state
Eq. 5:  a = Σ α_j · h_j                                      # Attention aggregation
Eq. 6:  α_j = softmax(tanh(W_0 · h_j))                       # Attention weights
Eq. 7:  GAT(G; sq) = ReLU(Σ β_qn · W1 · a_sn)               # Graph attention
Eq. 8:  β_qn = softmax(LeakyReLU(r^T[W2·a_sq||W2·a_sn]))    # GAT attn weights
Eq. 9:  U_G = {g_{i-t}, ..., g_{i-1}}                        # Long-term input seq
Eq. 10: τ_G = Attention(U_G)                                  # Long-term embedding
Eq. 11: z_πc = MaxPool({τ_G(sq) | ∀sq ∈ M_πc})               # Sector pooling
Eq. 12: τ(πc) = GAT(G_π, πc)                                  # Inter-sector GAT
Eq. 13: τ_F = ReLU([τ_G || τ_A || τ(πc)] · Wf)              # Fusion (3 vectors)
Eq. 14: ŷ_return = e1^T · τ_F + b1                           # Return prediction
        ŷ_move = σ(e2^T · τ_F + b2)                          # Movement prediction
Eq. 15: L = (1-δ)·L_rank + δ·L_move + λ·||Θ||²               # Total loss
Eq. 16: L_rank = Σ max(0, -Δ̂·Δ), L_move = BCE               # Component losses
Eq. 17: MRR@K = (1/K) Σ 1/rank(sq)                           # Eval metric
Eq. 18: P@K = |pred_topK ∩ true_topK| / K                    # Precision
Eq. 19: π_F = ReLU([π_G || π_A] · Wf)                        # FinGAT-NT (2 vectors)
```

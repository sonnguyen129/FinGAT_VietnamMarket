# PLAN.md — Reimplementing FinGAT on Vietnamese Stock Exchange (HOSE) Data

**FinGAT (Financial Graph Attention Networks) recommends top-K profitable stocks by learning latent relationships between stocks and sectors through graph attention, without requiring pre-defined corporate relations.** This plan provides a step-by-step blueprint for reimplementing the model from arXiv:2106.10159 on HOSE data using modern PyTorch and PyTorch Geometric. The original codebase (github.com/Roytsai27/Financial-GraphAttention) targets Taiwan/US markets with PyTorch 1.0 — this plan modernizes the stack and adapts all data pipelines for Vietnam's market structure. Each section below maps directly to an implementation module that Claude Code can build sequentially.

---

## 1. Project structure and dependencies

Create the following project layout. Each module has a single responsibility, making the codebase testable and debuggable in isolation.

```
fingat-hose/
├── PLAN.md
├── README.md
├── requirements.txt
├── config.py                  # All hyperparameters and paths
├── data/
│   ├── collector.py           # Download HOSE data via vnstock
│   ├── sector_map.py          # Build sector/industry mapping
│   ├── features.py            # Feature engineering (15 features)
│   ├── graph_builder.py       # Construct intra-sector + inter-sector edge_index
│   ├── dataset.py             # PyTorch Dataset class, pickle I/O
│   └── cache/                 # Downloaded raw data cache
├── model/
│   ├── attentive_gru.py       # Attentive GRU module
│   ├── intra_sector_gat.py    # Intra-sector GAT layer
│   ├── inter_sector_gat.py    # Inter-sector GAT layer
│   ├── fingat.py              # Full FinGAT model (CategoricalGraphAtt)
│   └── loss.py                # Multi-task loss (ranking + BCE + L2)
├── train.py                   # Training loop
├── evaluate.py                # MRR, Precision@K, Accuracy, IRR
├── predict.py                 # Inference and top-K recommendation
└── utils.py                   # Helpers, logging, reproducibility
```

### Dependencies (requirements.txt)

```
torch>=2.0.0
torch-geometric>=2.5.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
vnstock>=3.0.9
matplotlib>=3.7.0
tqdm>=4.65.0
```

Install PyTorch Geometric following official instructions for your CUDA version: `pip install torch-geometric` (plus `pyg-lib`, `torch-scatter`, `torch-sparse`, `torch-cluster`, `torch-spline-conv` if needed).

---

## 2. Data collection from HOSE via vnstock

The **vnstock** library (v3.x, github.com/thinh-vu/vnstock, 1.2k+ stars) is the primary data source. It wraps TCBS, VCI, SSI, and KBS APIs to provide OHLCV prices, financial statements, and company metadata for all HOSE-listed stocks.

### 2.1 Get the full list of HOSE stocks

```python
# data/collector.py
from vnstock import Vnstock

def get_hose_tickers():
    """Return list of all HOSE-listed ticker symbols."""
    stock = Vnstock(show_log=False)
    listing = stock.listing_companies()  # Wifeed source
    hose = listing[listing['comGroupCode'] == 'HOSE']
    return sorted(hose['ticker'].tolist())
    # Expected: ~400-500 tickers
```

**Important notes on vnstock:**
- Register a free API key at vnstocks.com/login and call `register_user()` for better rate limits.
- The library uses unofficial brokerage APIs. Add `time.sleep(0.3)` between requests to avoid throttling.
- For historical OHLCV, use the `VCI` or `KBS` source (more reliable than TCBS for bulk downloads).
- vnstock is licensed for non-commercial/research use only.

### 2.2 Download historical OHLCV data

```python
from vnstock import Quote
import time

def download_ohlcv(ticker, start='2020-01-01', end='2025-12-31', source='VCI'):
    """Download daily OHLCV for one ticker. Returns DataFrame with columns:
    time, open, high, low, close, volume."""
    quote = Quote(symbol=ticker, source=source)
    df = quote.history(start=start, end=end, interval='1D')
    df = df.rename(columns={'time': 'date'})
    df['ticker'] = ticker
    return df

def download_all(tickers, start, end):
    """Download OHLCV for all tickers with rate limiting."""
    all_data = {}
    for i, ticker in enumerate(tickers):
        try:
            df = download_ohlcv(ticker, start, end)
            all_data[ticker] = df
            if i % 50 == 0:
                print(f"Downloaded {i}/{len(tickers)}")
            time.sleep(0.3)
        except Exception as e:
            print(f"Failed {ticker}: {e}")
    return all_data
```

**Data period recommendation:** Download **2020-01-01 to 2025-12-31** (6 years). Use 60/20/20 split for train/validation/test following the paper. This gives ~1,500 trading days — comparable to the paper's ~965 days for Taiwan.

### 2.3 Data quality filtering

Not all ~400+ HOSE stocks will be usable. Filter stocks that:
- Have been listed for the entire study period (no IPOs mid-period)
- Have no gaps longer than 5 consecutive trading days
- Have average daily trading volume above a minimum threshold (e.g., 10,000 shares) to avoid illiquid stocks
- Target: **150–300 stocks** after filtering (the paper used 100 for Taiwan, 424 for S&P 500)

```python
def filter_stocks(all_data, min_days=1200, min_avg_volume=10000):
    """Filter to stocks with sufficient history and liquidity."""
    valid = {}
    for ticker, df in all_data.items():
        if len(df) >= min_days and df['volume'].mean() >= min_avg_volume:
            valid[ticker] = df
    return valid
```

---

## 3. Sector/industry classification for HOSE

Vietnam's stock exchanges use the **ICB (Industry Classification Benchmark)** system with **11 Level-1 industries**: Technology, Telecommunications, Health Care, Financials, Real Estate, Consumer Discretionary, Consumer Staples, Industrials, Basic Materials, Energy, and Utilities. HOSE also uses GICS for its VNAllshare sector indices. **Use ICB for this project** since it aligns with the paper's use of industry sectors and is available through vnstock.

### 3.1 Build the sector mapping

```python
# data/sector_map.py
from vnstock import Vnstock
import time

def build_sector_mapping(tickers):
    """Get ICB industry classification for each ticker via TCBS API.
    Returns dict: {ticker: {'industry': str, 'industryID': int}}."""
    stock = Vnstock(show_log=False)
    mapping = {}
    for ticker in tickers:
        try:
            s = stock.stock(symbol=ticker, source='TCBS')
            overview = s.overview()
            mapping[ticker] = {
                'industry': overview.get('industry', 'Unknown'),
                'industryID': overview.get('industryID', -1),
            }
            time.sleep(0.3)
        except Exception as e:
            print(f"Sector lookup failed for {ticker}: {e}")
    return mapping
```

**Fallback approach:** The `listing_companies()` DataFrame includes boolean sector columns (VNIT, VNMAT, VNREAL, VNUTI, etc.) corresponding to VNAllshare sector indices. These can serve as a backup classification:

```python
def get_sector_from_listing(listing_df, ticker):
    sector_cols = [c for c in listing_df.columns if c.startswith('VN') and c not in ['VNINDEX', 'VN30']]
    row = listing_df[listing_df['ticker'] == ticker]
    for col in sector_cols:
        if row[col].values[0]:
            return col
    return 'OTHER'
```

### 3.2 Vietnam-specific sector considerations

- **Finance + Real Estate dominate** (~47% of VN-Index market cap). These two sectors will have far more stocks than others, creating imbalanced sector sizes.
- Consider **merging small sectors** with fewer than 5 stocks into an "Other" category, or using Level-2 ICB sub-industries to split large sectors.
- The paper's Taiwan dataset had **5 sectors with 100 stocks** (average 20 stocks/sector). Target a similar ratio: ~10 sectors with 15-30 stocks each.
- Map Vietnamese industry names (returned in Vietnamese, e.g., "Ngân hàng" for Banking) to English for readability.

---

## 4. Feature engineering — 15 features per stock per day

The paper defines exactly **15 features** per stock per trading day, organized in three groups. All features are implicitly normalized through ratio-based design (no explicit min-max or z-score normalization is described).

### 4.1 Feature definitions

```python
# data/features.py
import numpy as np
import pandas as pd

def compute_features(df):
    """Compute 15 FinGAT features from OHLCV DataFrame.
    Input columns: date, open, high, low, close, volume
    Output: DataFrame with 15 feature columns per row (day)."""

    # --- Group 1: Basic daily features (6) ---
    df['return_ratio'] = df['close'].pct_change()  # R = (p_j - p_{j-1}) / p_{j-1}
    # Features: open, close, high, low, close (as adj_close proxy), return_ratio

    # --- Group 2: Price-ratio features (3) ---
    # F_μ = μ / close - 1, where μ ∈ {open, high, low}
    df['pr_open'] = df['open'] / df['close'] - 1
    df['pr_high'] = df['high'] / df['close'] - 1
    df['pr_low'] = df['low'] / df['close'] - 1

    # --- Group 3: Moving-average features (6) ---
    # F_φ = (MA_φ / close) - 1, where φ ∈ {5, 10, 15, 20, 25, 30}
    for window in [5, 10, 15, 20, 25, 30]:
        ma = df['close'].rolling(window=window).mean()
        df[f'ma_{window}'] = ma / df['close'] - 1

    # Select final 15 feature columns
    feature_cols = [
        'open', 'close', 'high', 'low', 'close',  # adj_close ≈ close for HOSE
        'return_ratio',
        'pr_open', 'pr_high', 'pr_low',
        'ma_5', 'ma_10', 'ma_15', 'ma_20', 'ma_25', 'ma_30'
    ]
    return df, feature_cols
```

**Key adaptation for HOSE:** Vietnamese stocks do not have a separate "adjusted close" like US stocks (no dividends adjustment in vnstock OHLCV). Use `close` as `adjclose`. If dividend-adjusted data becomes available, substitute it.

### 4.2 Normalization decision

The paper's features are already **self-normalizing** — price-ratio and MA features divide by close price, and return ratio is a percentage change. However, the raw OHLC prices (features 1-5) are NOT normalized. Two options:

- **Option A (recommended):** Replace raw OHLC with **price-relative features** — divide all prices by the previous day's close price. This makes all 15 features unitless and comparable across stocks with different price levels.
- **Option B:** Apply per-stock z-score normalization over a rolling window (e.g., 60 days) to all features.

```python
def normalize_prices(df):
    """Convert absolute prices to price-relative features."""
    prev_close = df['close'].shift(1)
    for col in ['open', 'high', 'low', 'close']:
        df[f'{col}_norm'] = df[col] / prev_close - 1
    return df
```

---

## 5. Dataset construction — weekly sliding windows

The paper structures data as **weekly windows**: 3 consecutive weeks (15 trading days) of features predict the return on day 16. A daily sliding window generates overlapping instances.

### 5.1 Window construction logic

```python
# data/dataset.py
import numpy as np
import pickle

def create_weekly_windows(stock_features, num_weeks=3, days_per_week=5):
    """Convert daily feature matrices into weekly sliding windows.

    Args:
        stock_features: dict {ticker: np.array of shape [num_days, num_features]}
        num_weeks: number of lookback weeks (default 3)
        days_per_week: trading days per week (default 5)

    Returns:
        x_weeks: list of num_weeks arrays, each [num_samples, num_stocks, days_per_week, num_features]
        y_return: [num_samples, num_stocks] — return ratio on day after last window
        y_binary: [num_samples, num_stocks] — 1 if positive return, 0 otherwise
    """
    window_size = num_weeks * days_per_week  # 15 days
    tickers = sorted(stock_features.keys())
    num_stocks = len(tickers)
    num_days = min(len(v) for v in stock_features.values())
    num_features = list(stock_features.values())[0].shape[1]

    num_samples = num_days - window_size  # Each day after day 15 is a target

    x_weeks = [[] for _ in range(num_weeks)]  # x1, x2, x3
    y_return = []
    y_binary = []

    for t in range(num_samples):
        for w in range(num_weeks):
            week_start = t + w * days_per_week
            week_end = week_start + days_per_week
            week_data = np.stack([
                stock_features[ticker][week_start:week_end]
                for ticker in tickers
            ])  # [num_stocks, days_per_week, num_features]
            x_weeks[w].append(week_data)

        # Target: return ratio on day after the 3-week window
        target_day = t + window_size
        returns = np.array([
            stock_features[ticker][target_day, RETURN_RATIO_IDX]
            for ticker in tickers
        ])
        y_return.append(returns)
        y_binary.append((returns > 0).astype(float))

    # Stack into arrays
    for w in range(num_weeks):
        x_weeks[w] = np.stack(x_weeks[w])  # [num_samples, num_stocks, 5, 15]

    return x_weeks, np.stack(y_return), np.stack(y_binary), tickers
```

### 5.2 Train/validation/test split

```python
def split_data(x_weeks, y_return, y_binary, train_ratio=0.6, val_ratio=0.2):
    """Chronological split — no shuffling (time series data)."""
    n = x_weeks[0].shape[0]
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    splits = {}
    for name, start, end in [('train', 0, train_end),
                               ('val', train_end, val_end),
                               ('test', val_end, n)]:
        splits[name] = {
            **{f'x{w+1}': x_weeks[w][start:end] for w in range(len(x_weeks))},
            'y_return_ratio': y_return[start:end],
            'y_up_or_down': y_binary[start:end],
        }
    return splits
```

### 5.3 Pickle format (matching original repo)

Save the processed data as a pickle file matching the original format:

```python
def save_dataset(splits, path='data/cache/hose_fingat_data.pickle'):
    with open(path, 'wb') as f:
        pickle.dump(splits, f)
```

**Expected tensor shapes** for ~200 stocks, 3-week lookback, 5 days/week, 15 features:
- `x1, x2, x3`: `[num_samples, 200, 5, 15]`
- `y_return_ratio`: `[num_samples, 200]`
- `y_up_or_down`: `[num_samples, 200]`

---

## 6. Graph construction — intra-sector and inter-sector edges

FinGAT uses **two fully-connected graphs**: one connecting stocks within the same sector (intra-sector), and one connecting all sectors to each other (inter-sector). The GAT attention mechanism learns which edges are important — no pre-defined correlation or relationship data is needed.

### 6.1 Intra-sector edge index

```python
# data/graph_builder.py
import torch
import numpy as np

def build_intra_sector_edges(tickers, sector_mapping):
    """Build fully-connected graph within each sector.
    Returns edge_index tensor of shape [2, num_edges] in PyG format."""
    # Group tickers by sector
    sectors = {}
    for i, ticker in enumerate(tickers):
        sector = sector_mapping[ticker]['industry']
        sectors.setdefault(sector, []).append(i)

    src, dst = [], []
    for sector, stock_indices in sectors.items():
        # Fully-connected: every pair of stocks in this sector
        for i in stock_indices:
            for j in stock_indices:
                if i != j:
                    src.append(i)
                    dst.append(j)

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    return edge_index  # Shape: [2, num_intra_edges]
```

### 6.2 Inter-sector edge index

```python
def build_inter_sector_edges(sector_list):
    """Build fully-connected graph between all sectors.
    Returns edge_index tensor of shape [2, num_sector_pairs]."""
    n = len(sector_list)
    src, dst = [], []
    for i in range(n):
        for j in range(n):
            if i != j:
                src.append(i)
                dst.append(j)
    return torch.tensor([src, dst], dtype=torch.long)
```

### 6.3 Sector assignment tensor

```python
def build_sector_assignments(tickers, sector_mapping, sector_list):
    """Map each stock index to its sector index.
    Returns tensor of shape [num_stocks] with sector IDs."""
    sector_to_idx = {s: i for i, s in enumerate(sector_list)}
    assignments = torch.tensor([
        sector_to_idx[sector_mapping[t]['industry']]
        for t in tickers
    ], dtype=torch.long)
    return assignments
```

**Save all edge data as .npy files** for compatibility with the original approach:

```python
def save_edges(intra_edges, inter_edges, sector_assignments, path='data/cache/'):
    np.save(f'{path}/hose_inner_edge.npy', intra_edges.numpy().T)
    np.save(f'{path}/hose_outer_edge.npy', inter_edges.numpy().T)
    np.save(f'{path}/hose_sector_assignments.npy', sector_assignments.numpy())
```

---

## 7. Model architecture — FinGAT in PyTorch Geometric

The model has three components: stock-level modeling (Attentive GRU + Intra-sector GAT), sector-level modeling (Graph Pooling + Inter-sector GAT), and multi-task prediction heads. **All hidden dimensions are 16** throughout the model.

### 7.1 Attentive GRU module

Used at two levels: (1) short-term over daily features within a week, and (2) long-term over weekly embeddings across weeks. Uses feed-forward attention (Luong-style) to weight hidden states.

```python
# model/attentive_gru.py
import torch
import torch.nn as nn

class AttentiveGRU(nn.Module):
    """GRU with feed-forward attention aggregation over time steps.
    Input: [batch, seq_len, input_dim]
    Output: [batch, hidden_dim] — attention-weighted sum of hidden states."""

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.attention_W = nn.Linear(hidden_dim, 1, bias=False)  # W_0

    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        h, _ = self.gru(x)  # h: [batch, seq_len, hidden_dim]

        # Attention: α_j = softmax(tanh(W_0 · h_j))
        attn_scores = torch.tanh(self.attention_W(h))  # [batch, seq_len, 1]
        attn_weights = torch.softmax(attn_scores, dim=1)  # [batch, seq_len, 1]

        # Weighted sum: a = Σ α_j · h_j
        output = torch.sum(attn_weights * h, dim=1)  # [batch, hidden_dim]
        return output
```

### 7.2 Intra-sector GAT layer

Applies GAT on the fully-connected intra-sector stock graph. Uses PyTorch Geometric's `GATConv` (or `GATv2Conv` for improved dynamic attention).

```python
# model/intra_sector_gat.py
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv  # or GATv2Conv

class IntraSectorGAT(nn.Module):
    """GAT over fully-connected intra-sector stock graph.
    Input: stock embeddings [num_stocks, hidden_dim]
    Output: graph-enriched embeddings [num_stocks, hidden_dim]."""

    def __init__(self, hidden_dim, heads=1, dropout=0.0):
        super().__init__()
        self.gat = GATConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            heads=heads,
            concat=False,       # Average heads → output stays hidden_dim
            dropout=dropout,
            add_self_loops=True,
        )

    def forward(self, x, edge_index):
        # x: [num_stocks, hidden_dim]
        # edge_index: [2, num_intra_edges] — fully-connected within sectors
        return torch.relu(self.gat(x, edge_index))
```

**Design choice — GATConv vs GATv2Conv:** GATv2Conv (Brody et al., 2022) fixes GAT's "static attention" limitation, allowing dynamic attention that depends on both source and target nodes. Since FinGAT learns latent stock relationships (no pre-defined relations), **GATv2Conv is recommended** for better expressiveness. Simply replace `GATConv` with `GATv2Conv` — the API is identical.

### 7.3 Inter-sector GAT layer

Same architecture as intra-sector, but operates on sector-level embeddings.

```python
# model/inter_sector_gat.py
class InterSectorGAT(nn.Module):
    """GAT over fully-connected inter-sector graph.
    Input: sector embeddings [num_sectors, hidden_dim]
    Output: enriched sector embeddings [num_sectors, hidden_dim]."""

    def __init__(self, hidden_dim, heads=1, dropout=0.0):
        super().__init__()
        self.gat = GATConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            heads=heads,
            concat=False,
            dropout=dropout,
            add_self_loops=True,
        )

    def forward(self, x, edge_index):
        return torch.relu(self.gat(x, edge_index))
```

### 7.4 Full FinGAT model (CategoricalGraphAtt)

```python
# model/fingat.py
import torch
import torch.nn as nn
from model.attentive_gru import AttentiveGRU
from model.intra_sector_gat import IntraSectorGAT
from model.inter_sector_gat import InterSectorGAT

class FinGAT(nn.Module):
    def __init__(self, input_dim, time_step, hidden_dim, intra_edge_index,
                 inter_edge_index, sector_assignments, num_sectors,
                 agg_week_num=3, use_gru=True, device='cuda'):
        """
        Args:
            input_dim: number of features per stock per day (15)
            time_step: days per week (5)
            hidden_dim: hidden dimension for GRU and GAT (16)
            intra_edge_index: [2, E_intra] intra-sector edges
            inter_edge_index: [2, E_inter] inter-sector edges
            sector_assignments: [num_stocks] sector ID per stock
            num_sectors: number of unique sectors
            agg_week_num: number of past weeks to aggregate (3)
            use_gru: whether to use GRU (True) or simple attention
            device: torch device
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.agg_week_num = agg_week_num
        self.device = device

        # Register edge indices as buffers (non-trainable, move with model)
        self.register_buffer('intra_edge_index', intra_edge_index)
        self.register_buffer('inter_edge_index', inter_edge_index)
        self.register_buffer('sector_assignments', sector_assignments)
        self.num_sectors = num_sectors

        # Component 1a: Short-term Attentive GRU (daily → weekly)
        self.short_term_gru = AttentiveGRU(input_dim, hidden_dim)

        # Component 1b: Intra-sector GAT
        self.intra_gat = IntraSectorGAT(hidden_dim)

        # Component 1c: Long-term Attentive GRU (weekly → multi-week)
        # Two separate long-term GRUs for GAT-based and raw embeddings
        self.long_term_gru_G = AttentiveGRU(hidden_dim, hidden_dim)
        self.long_term_gru_A = AttentiveGRU(hidden_dim, hidden_dim)

        # Component 2: Inter-sector GAT
        self.inter_gat = InterSectorGAT(hidden_dim)

        # Component 3: Fusion + prediction heads
        # Fusion: concatenate τ_G, τ_A, τ_sector → linear
        self.fusion = nn.Linear(hidden_dim * 3, hidden_dim)

        # Regression head: predict return ratio
        self.reg_head = nn.Linear(hidden_dim, 1)

        # Classification head: predict up/down probability
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, weekly_inputs):
        """
        Args:
            weekly_inputs: list of num_weeks tensors, each [num_stocks, time_step, input_dim]

        Returns:
            reg_out: [num_stocks, 1] predicted return ratios
            cls_out: [num_stocks, 1] predicted up probability
        """
        weekly_A = []  # Short-term embeddings per week
        weekly_G = []  # GAT-enhanced embeddings per week

        for week_data in weekly_inputs:
            # week_data: [num_stocks, time_step, input_dim]

            # Short-term: daily features → weekly embedding
            a = self.short_term_gru(week_data)  # [num_stocks, hidden_dim]
            weekly_A.append(a)

            # Intra-sector GAT
            g = self.intra_gat(a, self.intra_edge_index)  # [num_stocks, hidden_dim]
            weekly_G.append(g)

        # Long-term: aggregate across weeks
        # Stack weekly embeddings: [num_stocks, num_weeks, hidden_dim]
        stacked_A = torch.stack(weekly_A, dim=1)
        stacked_G = torch.stack(weekly_G, dim=1)

        tau_A = self.long_term_gru_A(stacked_A)  # [num_stocks, hidden_dim]
        tau_G = self.long_term_gru_G(stacked_G)  # [num_stocks, hidden_dim]

        # Sector-level: graph pooling (element-wise max)
        sector_embeddings = torch.zeros(
            self.num_sectors, self.hidden_dim, device=self.device
        )
        for s in range(self.num_sectors):
            mask = self.sector_assignments == s
            if mask.any():
                sector_embeddings[s] = tau_G[mask].max(dim=0).values

        # Inter-sector GAT
        sector_out = self.inter_gat(
            sector_embeddings, self.inter_edge_index
        )  # [num_sectors, hidden_dim]

        # Map sector embeddings back to stocks
        tau_sector = sector_out[self.sector_assignments]  # [num_stocks, hidden_dim]

        # Fusion
        fused = torch.cat([tau_G, tau_A, tau_sector], dim=-1)  # [num_stocks, 3*hidden_dim]
        fused = torch.relu(self.fusion(fused))  # [num_stocks, hidden_dim]

        # Prediction heads
        reg_out = self.reg_head(fused)     # [num_stocks, 1]
        cls_out = self.cls_head(fused)     # [num_stocks, 1]

        return reg_out, cls_out

    def predict_toprank(self, test_weeks_list, device, top_k=5):
        """Run inference over all test samples. Returns predictions."""
        self.eval()
        all_reg, all_cls = [], []
        with torch.no_grad():
            num_samples = test_weeks_list[0].shape[0]
            for t in range(num_samples):
                weekly = [w[t].to(device) for w in test_weeks_list[-self.agg_week_num:]]
                reg, cls = self.forward(weekly)
                all_reg.append(reg.cpu().numpy())
                all_cls.append(cls.cpu().numpy())
        return np.concatenate(all_reg), np.concatenate(all_cls)
```

### 7.5 PyTorch Geometric GATConv — key parameters to tune

```python
GATConv(
    in_channels=16,       # Input feature size
    out_channels=16,      # Output size per head
    heads=1,              # Multi-head attention (paper uses 1; try 4-8)
    concat=False,         # False = average heads; True = concatenate (output *= heads)
    negative_slope=0.2,   # LeakyReLU slope for attention
    dropout=0.0,          # Dropout on attention weights
    add_self_loops=True,  # Automatically add self-loops
    residual=False,       # Skip connection (try True for stability)
)
```

---

## 8. Multi-task loss function

The paper combines three loss components. The code implementation differs slightly from the paper formulation — follow the code for faithful reproduction.

```python
# model/loss.py
import torch
import torch.nn as nn

class FinGATLoss(nn.Module):
    """Multi-task loss: ranking + classification + L2 regularization.

    Paper formula: L = (1-δ)·L_rank + δ·L_move + λ·||Θ||²
    Code formula: L = α·L_reg + β·L_cls + γ·L_rank
    (L2 handled by optimizer weight_decay)
    """

    def __init__(self, alpha=1.0, beta=0.01, gamma=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta  # Paper optimal δ = 0.01
        self.gamma = gamma
        self.reg_loss_fn = nn.L1Loss(reduction='mean')
        self.cls_loss_fn = nn.BCELoss(reduction='mean')

    def forward(self, reg_out, cls_out, y_return, y_binary):
        """
        reg_out: [N, 1] predicted returns
        cls_out: [N, 1] predicted up probability
        y_return: [N, 1] actual returns
        y_binary: [N, 1] actual binary labels
        """
        # MAE regression loss
        reg_loss = self.reg_loss_fn(reg_out, y_return)

        # Binary cross-entropy classification loss
        cls_loss = self.cls_loss_fn(cls_out, y_binary)

        # Pairwise ranking loss:
        # For all pairs (i,j): relu(-(pred_i * pred_j) * (y_i * y_j))
        # Penalizes when predicted sign of difference disagrees with actual
        pairwise_pred = reg_out.view(-1, 1) * reg_out.view(1, -1)
        pairwise_true = y_return.view(-1, 1) * y_return.view(1, -1)
        rank_loss = torch.relu(-pairwise_pred * pairwise_true).sum()

        total = self.alpha * reg_loss + self.beta * cls_loss + self.gamma * rank_loss
        return total, reg_loss.item(), cls_loss.item(), rank_loss.item()
```

**Important note on the ranking loss**: The pairwise ranking loss is O(N²) in the number of stocks. For 200+ stocks this creates a ~40,000-element matrix per sample. This is computationally expensive. Consider:
- Sampling a random subset of pairs (e.g., 1000 pairs per sample)
- Or computing ranking loss only within each sector (smaller subgroups)

---

## 9. Training pipeline

### 9.1 Training loop

```python
# train.py
import torch
from torch import optim
from model.fingat import FinGAT
from model.loss import FinGATLoss

def train(config):
    # Load data
    data = load_pickle(config.data_path)
    intra_edges = torch.tensor(np.load(config.intra_edge_path).T, dtype=torch.long)
    inter_edges = torch.tensor(np.load(config.inter_edge_path).T, dtype=torch.long)
    sector_assignments = torch.tensor(np.load(config.sector_path), dtype=torch.long)

    # Initialize model
    model = FinGAT(
        input_dim=15, time_step=5, hidden_dim=config.hidden_dim,
        intra_edge_index=intra_edges.to(config.device),
        inter_edge_index=inter_edges.to(config.device),
        sector_assignments=sector_assignments.to(config.device),
        num_sectors=config.num_sectors,
        agg_week_num=config.week_num,
        device=config.device,
    ).to(config.device)

    # Xavier initialization (matching original)
    for p in model.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_uniform_(p)

    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.l2)
    criterion = FinGATLoss(alpha=config.alpha, beta=config.beta, gamma=config.gamma)

    best_mrr = 0
    for epoch in range(config.epochs):
        model.train()
        num_samples = data['train']['x1'].shape[0]

        for t in range(num_samples):
            # Get weekly inputs for this time step
            weekly = [
                torch.tensor(data['train'][f'x{w+1}'][t], dtype=torch.float32).to(config.device)
                for w in range(config.week_num)
            ]
            y_ret = torch.tensor(data['train']['y_return_ratio'][t]).float().view(-1, 1).to(config.device)
            y_bin = torch.tensor(data['train']['y_up_or_down'][t]).float().view(-1, 1).to(config.device)

            reg_out, cls_out = model(weekly)
            loss, r, c, rk = criterion(reg_out, cls_out, y_ret, y_bin)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluate at end of epoch
        mrr = evaluate(model, data, config)
        if mrr > best_mrr:
            best_mrr = mrr
            torch.save(model.state_dict(), 'best_model.pt')
            print(f"Epoch {epoch+1}: New best MRR = {mrr:.4f}")
```

### 9.2 Hyperparameter configuration

```python
# config.py
from dataclasses import dataclass

@dataclass
class Config:
    # Data
    data_path: str = 'data/cache/hose_fingat_data.pickle'
    intra_edge_path: str = 'data/cache/hose_inner_edge.npy'
    inter_edge_path: str = 'data/cache/hose_outer_edge.npy'
    sector_path: str = 'data/cache/hose_sector_assignments.npy'

    # Model
    hidden_dim: int = 16         # Paper optimal
    week_num: int = 3            # Paper optimal (tested 1-4)
    num_sectors: int = 11        # ICB Level-1 count for HOSE

    # Training
    epochs: int = 30             # Paper uses 10; increase for HOSE
    lr: float = 0.001            # Paper searches {0.0005, 0.001, 0.005}
    l2: float = 1e-4             # Paper λ = 0.0001
    alpha: float = 1.0           # MAE loss weight
    beta: float = 0.01           # BCE loss weight (paper optimal δ = 0.01)
    gamma: float = 1.0           # Ranking loss weight
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Evaluation
    top_k_values: list = (5, 10, 20)
```

---

## 10. Evaluation metrics

The paper evaluates stock recommendation quality, not point prediction accuracy. Implement four metrics.

```python
# evaluate.py
import numpy as np
import pandas as pd

def mrr_at_k(y_true, y_pred, k=5):
    """Mean Reciprocal Rank: do predicted top-K match actual top-K?"""
    df = pd.DataFrame({'pred': y_pred, 'true': y_true})
    df = df.sort_values('pred', ascending=False).reset_index(drop=True)
    df['pred_rank'] = df.index + 1
    df = df.sort_values('true', ascending=False)
    return float(sum(1.0 / df['pred_rank'].values[:k]))

def precision_at_k(y_true, y_pred, k=5):
    """Overlap between predicted top-K and actual top-K."""
    pred_top = set(np.argsort(y_pred)[-k:])
    true_top = set(np.argsort(y_true)[-k:])
    return len(pred_top & true_top) / k

def irr_at_k(y_true, y_pred, k=5):
    """Investment Return Ratio: difference between optimal and predicted returns."""
    pred_top = np.argsort(y_pred)[-k:]
    true_top = np.argsort(y_true)[-k:]
    return float(y_true[true_top].sum() - y_true[pred_top].sum())

def accuracy(y_true_binary, y_pred_return):
    """Binary up/down accuracy."""
    pred_binary = (y_pred_return > 0).astype(int)
    return float(np.mean(y_true_binary == pred_binary))
```

**Evaluation procedure (matching the paper):** For each test time step, compute MRR@K, Precision@K, and IRR@K over all stocks simultaneously. Average across all test time steps. Report for K ∈ {5, 10, 20}.

---

## 11. Vietnam-specific adaptations and considerations

Several aspects of the Vietnamese market require adaptation beyond a direct port of the original code.

### Trading calendar differences
Vietnamese stocks trade **Monday–Friday** but have different holidays than Taiwan/US. Ensure the sliding window logic handles holiday gaps correctly. vnstock returns only trading days, so no explicit filtering is needed — just verify that consecutive rows are indeed consecutive trading days.

### Price limit rules
HOSE imposes **±7% daily price limits** (±10% for HNX, ±15% for UPCOM). This compresses the return distribution compared to US markets. The ranking loss and MRR metrics should still work, but expected return magnitudes will be smaller. Consider adjusting the regression loss scaling if MAE values are very small.

### Market microstructure
- **T+2 settlement** in Vietnam vs T+1 in Taiwan. The paper predicts next-day returns. For practical use, consider predicting **T+2 returns** instead.
- Vietnam is upgrading to **FTSE Secondary Emerging Market** status in September 2026, which may change market dynamics and data availability.
- **Foreign ownership limits** exist for many sectors. This does not affect the model but matters for practical trading.

### Sector imbalance
Finance and Real Estate account for ~47% of HOSE market cap. Options to handle this:
1. **Cap stocks per sector** at 20-30 to prevent Financials from dominating the graph
2. **Use ICB Level-2 sub-industries** to split Financials into Banks, Insurance, Securities, etc.
3. **Weighted sampling** during training to balance sector representation

### Recommended pilot configuration
Start with a smaller, well-curated dataset to validate the pipeline before scaling:
- **50-100 stocks** across 8-10 sectors (similar to the paper's Taiwan dataset)
- Focus on VN30 and VN100 index components (most liquid, best data quality)
- **2 years** of data for initial testing (2023-2024), then expand

---

## 12. Step-by-step implementation order for Claude Code

Execute these steps sequentially. Each step should be a working, testable unit before proceeding.

**Phase 1 — Data pipeline (Steps 1-4)**

1. **Set up project structure.** Create all directories and `requirements.txt`. Install dependencies.

2. **Implement `data/collector.py`.** Download HOSE stock list and OHLCV data. Start with VN30 stocks (30 tickers) for rapid iteration. Cache all raw data to `data/cache/raw/`. Verify data integrity — check for missing dates, zero volumes, duplicates.

3. **Implement `data/sector_map.py`.** Build ticker-to-sector mapping. Save as JSON. Verify all tickers have a valid sector. Handle edge cases (conglomerates, newly reclassified stocks).

4. **Implement `data/features.py` and `data/dataset.py`.** Compute 15 features. Build weekly sliding windows. Create train/val/test splits. Save as pickle. **Validation checkpoint:** print tensor shapes, verify they match expected dimensions. Spot-check a few data points manually against vnstock raw data.

**Phase 2 — Graph and model (Steps 5-8)**

5. **Implement `data/graph_builder.py`.** Build intra-sector and inter-sector edge indices. Save as .npy. **Validation:** verify edge count = Σ(n_i × (n_i − 1)) for intra-sector (where n_i is number of stocks in sector i), and n_sectors × (n_sectors − 1) for inter-sector.

6. **Implement `model/attentive_gru.py`.** Unit test: feed random tensor of shape [50, 5, 15], verify output shape is [50, 16].

7. **Implement `model/intra_sector_gat.py` and `model/inter_sector_gat.py`.** Unit test: feed random node features and edge_index, verify output shapes.

8. **Implement `model/fingat.py`.** Integrate all components. Unit test: full forward pass with random data — verify reg_out shape [num_stocks, 1] and cls_out shape [num_stocks, 1]. Verify gradients flow (loss.backward() doesn't error).

**Phase 3 — Training and evaluation (Steps 9-11)**

9. **Implement `model/loss.py`.** Unit test: verify loss computation with known inputs. Check that ranking loss gradient is correct with a simple 4-stock example.

10. **Implement `train.py`.** Run a full training loop on VN30 data. Monitor loss convergence. Expect training to take 10-30 minutes on GPU for the small dataset.

11. **Implement `evaluate.py`.** Compute MRR@5, MRR@10, MRR@20, Precision@K, Accuracy on test set. Compare against baselines: random ranking and simple return-momentum ranking.

**Phase 4 — Scale and refine (Steps 12-14)**

12. **Scale to full HOSE dataset.** Download all ~200+ filtered stocks. Retrain. Monitor memory usage — the pairwise ranking loss with 200 stocks needs ~200² = 40K element matrix.

13. **Hyperparameter tuning.** Grid search over: `lr ∈ {0.0005, 0.001, 0.005}`, `hidden_dim ∈ {8, 16, 32}`, `week_num ∈ {2, 3, 4}`, `beta ∈ {0.001, 0.01, 0.1}`, `heads ∈ {1, 4, 8}`.

14. **Add GATv2Conv variant.** Replace GATConv with GATv2Conv and compare performance. Add attention weight visualization using `return_attention_weights=True` to interpret learned stock/sector relationships.

---

## 13. Baseline models for comparison

Implement these simpler baselines to contextualize FinGAT's performance on HOSE:

- **Momentum baseline:** Rank stocks by past-week return. No learning required.
- **LSTM baseline:** Single LSTM per stock, no graph structure. Tests whether the graph adds value.
- **GRU + Attention (no graph):** The FinGAT-NT variant from the paper — removes all graph components.
- **Simple GAT (no hierarchy):** Single fully-connected graph over all stocks, one GAT layer, no sector structure. Tests whether the hierarchical design adds value.

---

## 14. Key reference repositories

- **Original FinGAT:** github.com/Roytsai27/Financial-GraphAttention — the authoritative reference for data format, training loop, and evaluation
- **HGAIT:** github.com/finxlab/hgait — most modern PyG-based stock GAT; best reference for GATConv patterns with financial data (PyTorch 2.1, PyG 2.5)
- **SP100 with GNNs:** github.com/timothewt/SP100AnalysisWithGNNs — end-to-end PyG pipeline with sector-based graphs; good reference for dataset construction
- **FinAdaptGAT:** github.com/blueBoy019/FinAdaptGAT — cites FinGAT, addresses oversmoothing
- **GNN4Fintech list:** github.com/jwwthu/GNN4Fintech — curated list of GNN papers for finance with code links
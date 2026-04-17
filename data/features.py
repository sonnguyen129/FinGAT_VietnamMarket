"""Feature engineering — exactly 15 features per stock per day (paper spec).

Features:
  1-5: Price-relative: open, close, high, low, adjclose (normalized by prev close)
  6:   Return ratio: (close_j - close_{j-1}) / close_{j-1}
  7-9: Price-ratio: open/close-1, high/close-1, low/close-1
  10-15: Moving-average ratios: MA(w)/adjclose_{j-1} - 1, w in {5,10,15,20,25,30}
"""

import numpy as np
import pandas as pd


FEATURE_COLS = [
    'open_norm', 'close_norm', 'high_norm', 'low_norm', 'adjclose_norm',
    'return_ratio',
    'pr_open', 'pr_high', 'pr_low',
    'ma_5', 'ma_10', 'ma_15', 'ma_20', 'ma_25', 'ma_30',
]

NUM_FEATURES = 15
RETURN_RATIO_IDX = 5  # Index of return_ratio in FEATURE_COLS


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 15 FinGAT features from OHLCV DataFrame.

    Expected input columns: Date, Open, High, Low, Close, Adj Close, Volume
    Returns DataFrame with 15 feature columns.
    """
    df = df.copy()
    df = df.sort_values('Date').reset_index(drop=True)

    close = df['Close']
    adjclose = df['Adj Close']
    prev_close = close.shift(1)
    prev_adjclose = adjclose.shift(1)

    # Group 1: Price-relative features (5) — divide by previous close
    df['open_norm'] = df['Open'] / prev_close - 1
    df['close_norm'] = close / prev_close - 1
    df['high_norm'] = df['High'] / prev_close - 1
    df['low_norm'] = df['Low'] / prev_close - 1
    df['adjclose_norm'] = adjclose / prev_adjclose - 1

    # Group 2: Return ratio (1)
    df['return_ratio'] = close.pct_change()

    # Group 3: Price-ratio features (3) — intra-day ratios
    df['pr_open'] = df['Open'] / close - 1
    df['pr_high'] = df['High'] / close - 1
    df['pr_low'] = df['Low'] / close - 1

    # Group 4: Moving-average features (6)
    for window in [5, 10, 15, 20, 25, 30]:
        ma = adjclose.rolling(window=window).mean()
        df[f'ma_{window}'] = ma / prev_adjclose - 1

    return df


def extract_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """Extract the 15-feature matrix from a DataFrame with computed features.

    Returns np.ndarray of shape [num_days, 15]. NaN rows at the beginning
    (due to rolling windows) are dropped.
    """
    features = df[FEATURE_COLS].values
    # Find first row without NaN (MA_30 needs 30 days warmup + 1 for prev_close)
    valid_mask = ~np.isnan(features).any(axis=1)
    first_valid = np.argmax(valid_mask)
    return features[first_valid:]


def compute_all_features(all_data: dict) -> dict:
    """Compute features for all stocks.

    Args:
        all_data: dict {ticker: DataFrame with OHLCV columns}

    Returns:
        dict {ticker: np.ndarray of shape [num_valid_days, 15]}
    """
    result = {}
    for ticker, df in all_data.items():
        df_feat = compute_features(df)
        matrix = extract_feature_matrix(df_feat)
        if len(matrix) > 0:
            result[ticker] = matrix
    return result


def align_features(stock_features: dict, min_days: int = 200) -> dict:
    """Align all stocks to the same number of days (trim to shortest).

    Drops stocks with fewer than min_days of valid data.
    Returns dict {ticker: np.ndarray [common_days, 15]}.
    """
    # Filter by minimum days
    filtered = {t: f for t, f in stock_features.items() if len(f) >= min_days}
    if not filtered:
        raise ValueError(f"No stocks have >= {min_days} valid days")

    # Trim all to the shortest common length
    min_len = min(len(f) for f in filtered.values())
    aligned = {t: f[-min_len:] for t, f in filtered.items()}

    print(f"Aligned {len(aligned)} stocks to {min_len} common days")
    return aligned

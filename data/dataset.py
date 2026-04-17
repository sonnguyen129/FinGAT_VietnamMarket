"""Dataset construction — weekly sliding windows and train/val/test split.

Converts daily feature matrices into the paper's format:
  x_weeks: list of num_weeks arrays, each [num_samples, num_stocks, days_per_week, num_features]
  y_return: [num_samples, num_stocks]
  y_binary: [num_samples, num_stocks]
"""

import os
import pickle

import numpy as np

from data.features import RETURN_RATIO_IDX


def create_weekly_windows(stock_features: dict, num_weeks: int = 3,
                          days_per_week: int = 5) -> tuple:
    """Convert daily feature matrices into weekly sliding windows.

    Args:
        stock_features: dict {ticker: np.array [num_days, num_features]}
        num_weeks: number of lookback weeks (default 3)
        days_per_week: trading days per week (default 5)

    Returns:
        x_weeks: list of num_weeks arrays, each [num_samples, num_stocks, days_per_week, num_features]
        y_return: [num_samples, num_stocks]
        y_binary: [num_samples, num_stocks]
        tickers: sorted list of ticker names
    """
    tickers = sorted(stock_features.keys())
    num_stocks = len(tickers)
    num_days = min(len(stock_features[t]) for t in tickers)
    num_features = stock_features[tickers[0]].shape[1]
    window_size = num_weeks * days_per_week
    num_samples = num_days - window_size

    if num_samples <= 0:
        raise ValueError(
            f"Not enough days ({num_days}) for window_size ({window_size})")

    x_weeks = [[] for _ in range(num_weeks)]
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

        # Target: return ratio on day after the window
        target_day = t + window_size
        returns = np.array([
            stock_features[ticker][target_day, RETURN_RATIO_IDX]
            for ticker in tickers
        ])
        y_return.append(returns)
        y_binary.append((returns > 0).astype(np.float32))

    # Stack into arrays
    for w in range(num_weeks):
        x_weeks[w] = np.stack(x_weeks[w]).astype(np.float32)

    y_return = np.stack(y_return).astype(np.float32)
    y_binary = np.stack(y_binary).astype(np.float32)

    print(f"Created {num_samples} samples: {num_stocks} stocks, "
          f"{num_weeks} weeks x {days_per_week} days, {num_features} features")

    return x_weeks, y_return, y_binary, tickers


def split_data(x_weeks: list, y_return: np.ndarray, y_binary: np.ndarray,
               train_ratio: float = 0.6, val_ratio: float = 0.2) -> dict:
    """Chronological split — no shuffling (time series data).

    Returns dict with keys 'train', 'val', 'test', each containing:
      {x1, x2, ..., x{num_weeks}, y_return_ratio, y_up_or_down}
    """
    n = x_weeks[0].shape[0]
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    num_weeks = len(x_weeks)

    splits = {}
    for name, start, end in [('train', 0, train_end),
                              ('val', train_end, val_end),
                              ('test', val_end, n)]:
        split = {}
        for w in range(num_weeks):
            split[f'x{w + 1}'] = x_weeks[w][start:end]
        split['y_return_ratio'] = y_return[start:end]
        split['y_up_or_down'] = y_binary[start:end]
        splits[name] = split

    for name in ['train', 'val', 'test']:
        print(f"  {name}: {splits[name]['x1'].shape[0]} samples")

    return splits


def save_dataset(splits: dict, tickers: list,
                 path: str = "data/cache/hose_fingat_data.pickle"):
    """Save processed data as pickle."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {'splits': splits, 'tickers': tickers}
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Dataset saved to {path}")


def load_dataset(path: str = "data/cache/hose_fingat_data.pickle") -> tuple:
    """Load processed data from pickle.

    Returns (splits_dict, tickers_list).
    """
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data['splits'], data['tickers']


def print_data_statistics(splits: dict, tickers: list, sector_mapping: dict = None):
    """Print dataset statistics table similar to Table 1 in the paper."""
    num_stocks = splits['train']['x1'].shape[1]
    num_features = splits['train']['x1'].shape[3]
    num_weeks = len([k for k in splits['train'] if k.startswith('x')])

    if sector_mapping:
        sectors = set()
        for t in tickers:
            if t in sector_mapping:
                sectors.add(sector_mapping[t]['industry'])
        num_sectors = len(sectors)
    else:
        num_sectors = "N/A"

    print(f"\n{'='*60}")
    print(f"Dataset Statistics (HOSE)")
    print(f"{'='*60}")
    print(f"  # Stocks:       {num_stocks}")
    print(f"  # Sectors:      {num_sectors}")
    print(f"  # Features:     {num_features}")
    print(f"  # Weeks:        {num_weeks}")
    print(f"  # Train days:   {splits['train']['x1'].shape[0]}")
    print(f"  # Val days:     {splits['val']['x1'].shape[0]}")
    print(f"  # Test days:    {splits['test']['x1'].shape[0]}")
    print(f"  Total days:     {sum(splits[s]['x1'].shape[0] for s in ['train','val','test'])}")
    print(f"{'='*60}\n")

"""Build 5 stock subsets for EQ2 (without sector info experiment).

Subsets based on market cap (or volume as proxy):
  1. Best 10: highest market cap
  2. Worst 10: lowest market cap
  3. Best 5 Worst 5: 5 highest + 5 lowest
  4. Random 10: random sample
  5. Uniform 10: 1 from each of 10 zones
"""

import numpy as np


def get_stock_ranking(all_data: dict) -> list:
    """Rank stocks by average volume (proxy for market cap).

    Returns list of (ticker, avg_volume) sorted descending.
    """
    rankings = []
    for ticker, df in all_data.items():
        avg_vol = df['Volume'].mean() if 'Volume' in df.columns else 0
        rankings.append((ticker, avg_vol))
    rankings.sort(key=lambda x: -x[1])
    return rankings


def build_subsets(tickers: list, all_data: dict, seed: int = 42) -> dict:
    """Build 5 subsets of 10 stocks each.

    Args:
        tickers: sorted list of all ticker names
        all_data: dict {ticker: DataFrame} for ranking
        seed: random seed for reproducible sampling

    Returns:
        dict {subset_name: list of 10 tickers}
    """
    rng = np.random.RandomState(seed)

    # Rank by volume
    rankings = get_stock_ranking(all_data)
    # Only keep tickers that are in our dataset
    valid_tickers = [t for t, _ in rankings if t in tickers]

    n = len(valid_tickers)
    subsets = {}

    # 1. Best 10: highest volume/market cap
    subsets['Best 10'] = valid_tickers[:10]

    # 2. Worst 10: lowest
    subsets['Worst 10'] = valid_tickers[-10:]

    # 3. Best 5 Worst 5
    subsets['Best5 Worst5'] = valid_tickers[:5] + valid_tickers[-5:]

    # 4. Random 10
    random_idx = rng.choice(n, size=10, replace=False)
    subsets['Random 10'] = [valid_tickers[i] for i in sorted(random_idx)]

    # 5. Uniform 10: divide into 10 zones, pick 1 from each
    zone_size = n // 10
    uniform = []
    for z in range(10):
        start = z * zone_size
        end = start + zone_size if z < 9 else n
        idx = rng.randint(start, end)
        uniform.append(valid_tickers[idx])
    subsets['Uniform 10'] = uniform

    for name, stocks in subsets.items():
        print(f"  {name}: {stocks}")

    return subsets


def extract_subset_data(data: dict, tickers: list,
                        subset_tickers: list) -> dict:
    """Extract data for a subset of stocks from the full dataset.

    Args:
        data: splits dict {train/val/test: {x1, x2, x3, y_return_ratio, y_up_or_down}}
        tickers: full list of tickers (matching data ordering)
        subset_tickers: list of tickers to extract

    Returns:
        new splits dict with only the subset stocks
    """
    # Find indices of subset tickers in the full list
    ticker_to_idx = {t: i for i, t in enumerate(tickers)}
    indices = [ticker_to_idx[t] for t in subset_tickers if t in ticker_to_idx]
    indices = np.array(indices)

    new_data = {}
    for split_name in ['train', 'val', 'test']:
        split = data[split_name]
        new_split = {}
        for key in split:
            if key.startswith('x'):
                # x: [num_samples, num_stocks, days, features]
                new_split[key] = split[key][:, indices, :, :]
            elif key in ('y_return_ratio', 'y_up_or_down'):
                # y: [num_samples, num_stocks]
                new_split[key] = split[key][:, indices]
        new_data[split_name] = new_split

    return new_data

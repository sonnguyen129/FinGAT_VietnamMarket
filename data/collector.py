"""Data collector — load HOSE stock data from local CSV files.

Reads pre-downloaded OHLCV data from datasets/VN_datasets/*.csv.
Each CSV has columns: Date, High, Low, Open, Close, Adj Close, Volume
"""

import os

import pandas as pd


def get_available_tickers(data_dir: str = "datasets/VN_datasets") -> list:
    """Return sorted list of tickers from CSV files in data_dir."""
    tickers = []
    for f in os.listdir(data_dir):
        if f.endswith('.csv'):
            tickers.append(f.replace('.csv', ''))
    return sorted(tickers)


def load_stock_csv(ticker: str, data_dir: str = "datasets/VN_datasets") -> pd.DataFrame:
    """Load a single stock's OHLCV data from CSV."""
    path = os.path.join(data_dir, f"{ticker}.csv")
    df = pd.read_csv(path, parse_dates=['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    df['ticker'] = ticker
    return df


def load_all_stocks(data_dir: str = "datasets/VN_datasets") -> dict:
    """Load all stock CSVs. Returns dict {ticker: DataFrame}."""
    tickers = get_available_tickers(data_dir)
    all_data = {}
    failed = []
    for ticker in tickers:
        try:
            df = load_stock_csv(ticker, data_dir)
            if len(df) > 0:
                all_data[ticker] = df
        except Exception as e:
            failed.append((ticker, str(e)))

    print(f"Loaded {len(all_data)}/{len(tickers)} stocks")
    if failed:
        print(f"Failed: {len(failed)} stocks")
    return all_data


def filter_stocks(all_data: dict, min_days: int = 800,
                  min_avg_volume: int = 10000,
                  start_date: str = None, end_date: str = None) -> dict:
    """Filter stocks by data quality criteria.

    Args:
        all_data: dict {ticker: DataFrame}
        min_days: minimum number of trading days required
        min_avg_volume: minimum average daily volume
        start_date: if set, only keep data from this date onward
        end_date: if set, only keep data up to this date

    Returns:
        dict {ticker: filtered DataFrame}
    """
    filtered = {}
    for ticker, df in all_data.items():
        d = df.copy()
        if start_date:
            d = d[d['Date'] >= pd.Timestamp(start_date)]
        if end_date:
            d = d[d['Date'] <= pd.Timestamp(end_date)]

        if len(d) < min_days:
            continue
        if d['Volume'].mean() < min_avg_volume:
            continue

        # Check for excessive gaps (>5 consecutive trading days missing)
        dates = d['Date'].sort_values()
        gaps = dates.diff().dt.days
        if gaps.max() > 10:  # Allow weekends + holidays, flag >10 day gaps
            continue

        filtered[ticker] = d.reset_index(drop=True)

    print(f"Filtered: {len(filtered)}/{len(all_data)} stocks "
          f"(min_days={min_days}, min_vol={min_avg_volume})")
    return filtered

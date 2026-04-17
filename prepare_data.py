"""Prepare HOSE data for FinGAT experiments.

This script runs the full data pipeline:
  1. Load stock CSVs from datasets/VN_datasets/
  2. Filter by quality criteria
  3. Compute 15 features per stock per day
  4. Build sector mapping from company_industries.csv
  5. Create weekly sliding windows
  6. Build intra/inter-sector graph edges
  7. Save everything to data/cache/
"""

import os
import sys

from config import Config
from data.collector import load_all_stocks, filter_stocks
from data.features import compute_all_features, align_features
from data.sector_map import build_and_save_sector_mapping, get_sector_list
from data.dataset import (create_weekly_windows, split_data,
                          save_dataset, print_data_statistics)
from data.graph_builder import (build_intra_sector_edges, build_inter_sector_edges,
                                build_sector_assignments, save_edges)


def prepare_data(config: Config = None):
    """Full data preparation pipeline."""
    if config is None:
        config = Config()

    os.makedirs(config.cache_dir, exist_ok=True)

    # Step 1: Load raw data
    print("Step 1: Loading stock data...")
    all_data = load_all_stocks(config.data_dir)

    # Step 2: Filter stocks
    print("\nStep 2: Filtering stocks...")
    filtered = filter_stocks(
        all_data,
        min_days=config.min_trading_days,
        min_avg_volume=config.min_avg_volume,
        start_date=config.start_date,
        end_date=config.end_date,
    )

    # Step 3: Build sector mapping (only for filtered tickers)
    print("\nStep 3: Building sector mapping...")
    tickers = sorted(filtered.keys())
    sector_mapping = build_and_save_sector_mapping(
        companies_file=config.companies_file,
        tickers=tickers,
        output_path=os.path.join(config.cache_dir, "sector_mapping.json"),
    )

    # Only keep tickers that have sector info
    tickers = [t for t in tickers if t in sector_mapping]
    filtered = {t: filtered[t] for t in tickers}
    print(f"\nStocks with sector info: {len(tickers)}")

    # Step 4: Compute features
    print("\nStep 4: Computing 15 features...")
    stock_features = compute_all_features(filtered)
    stock_features = align_features(stock_features, min_days=200)

    # Update ticker list
    tickers = sorted(stock_features.keys())
    print(f"Final stocks after feature alignment: {len(tickers)}")

    # Step 5: Create weekly sliding windows
    print("\nStep 5: Creating weekly sliding windows...")
    x_weeks, y_return, y_binary, tickers = create_weekly_windows(
        stock_features, num_weeks=config.week_num,
        days_per_week=config.days_per_week)

    # Step 6: Split data
    print("\nStep 6: Splitting data (60/20/20)...")
    splits = split_data(x_weeks, y_return, y_binary,
                        train_ratio=config.train_ratio,
                        val_ratio=config.val_ratio)

    # Step 7: Save dataset
    print("\nStep 7: Saving dataset...")
    save_dataset(splits, tickers,
                 path=os.path.join(config.cache_dir, "hose_fingat_data.pickle"))

    # Step 8: Build graph edges
    print("\nStep 8: Building graph edges...")
    # Re-filter sector mapping to final tickers
    sector_mapping_final = {t: sector_mapping[t] for t in tickers
                            if t in sector_mapping}

    intra_edges = build_intra_sector_edges(tickers, sector_mapping_final)
    sector_assignments, sector_list = build_sector_assignments(
        tickers, sector_mapping_final)
    num_sectors = len(sector_list)
    inter_edges = build_inter_sector_edges(num_sectors)

    save_edges(intra_edges, inter_edges, sector_assignments,
               path=config.cache_dir)

    # Print statistics
    print_data_statistics(splits, tickers, sector_mapping_final)

    print(f"\nSector list ({num_sectors} sectors):")
    for i, s in enumerate(sector_list):
        count = (sector_assignments == i).sum().item()
        print(f"  {i}: {s} ({count} stocks)")

    print(f"\nData preparation complete!")
    print(f"  Cache directory: {config.cache_dir}/")
    print(f"  Dataset: hose_fingat_data.pickle")
    print(f"  Intra edges: hose_inner_edge.npy ({intra_edges.shape[1]} edges)")
    print(f"  Inter edges: hose_outer_edge.npy ({inter_edges.shape[1]} edges)")
    print(f"  Sector assignments: hose_sector_assignments.npy")

    return splits, tickers, sector_mapping_final


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--min-days', type=int, default=800)
    parser.add_argument('--min-volume', type=int, default=10000)
    parser.add_argument('--start-date', type=str, default='2020-01-01')
    parser.add_argument('--end-date', type=str, default='2025-12-31')
    args = parser.parse_args()

    config = Config()
    config.min_trading_days = args.min_days
    config.min_avg_volume = args.min_volume
    config.start_date = args.start_date
    config.end_date = args.end_date
    prepare_data(config)

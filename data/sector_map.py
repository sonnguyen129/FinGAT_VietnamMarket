"""Sector mapping from company_industries.csv (ICB classification).

Reads datasets/company_industries.csv with columns:
  ticker, icb_code2, icb_name2, icb_name3, icb_name4
"""

import json
import os

import pandas as pd

# Vietnamese → English ICB Level-2 mapping
VN_TO_EN = {
    'Ngân hàng': 'Banks',
    'Bảo hiểm': 'Insurance',
    'Dịch vụ tài chính': 'Financial Services',
    'Bất động sản': 'Real Estate',
    'Xây dựng và Vật liệu': 'Construction & Materials',
    'Hàng & Dịch vụ Công nghiệp': 'Industrial Goods & Services',
    'Thực phẩm và đồ uống': 'Food & Beverage',
    'Hàng cá nhân & Gia dụng': 'Personal & Household Goods',
    'Y tế': 'Health Care',
    'Hóa chất': 'Chemicals',
    'Tài nguyên Cơ bản': 'Basic Resources',
    'Dầu khí': 'Oil & Gas',
    'Điện, nước & xăng dầu khí đốt': 'Utilities',
    'Công nghệ Thông tin': 'Technology',
    'Truyền thông': 'Media',
    'Du lịch và Giải trí': 'Travel & Leisure',
    'Bán lẻ': 'Retail',
    'Ô tô và phụ tùng': 'Automobiles & Parts',
}


def load_sector_mapping(companies_file: str = "datasets/company_industries.csv") -> dict:
    """Load raw sector mapping from CSV.

    Returns dict {ticker: {icb_code2, icb_name2, icb_name3, icb_name4, industry}}
    """
    df = pd.read_csv(companies_file, encoding='utf-8-sig')
    mapping = {}
    for _, row in df.iterrows():
        ticker = row['ticker']
        vn_name = row['icb_name2']
        en_name = VN_TO_EN.get(vn_name, vn_name)
        mapping[ticker] = {
            'icb_code2': int(row['icb_code2']),
            'icb_name2': vn_name,
            'icb_name3': row['icb_name3'],
            'icb_name4': row['icb_name4'],
            'industry': en_name,
        }
    return mapping


def clean_sectors(mapping: dict, min_stocks: int = 5) -> dict:
    """Merge small sectors (< min_stocks) into 'Other'.

    Returns updated mapping with 'sector_idx' added.
    """
    # Count stocks per sector
    sector_counts = {}
    for info in mapping.values():
        s = info['industry']
        sector_counts[s] = sector_counts.get(s, 0) + 1

    # Identify small sectors
    small_sectors = {s for s, c in sector_counts.items() if c < min_stocks}
    if small_sectors:
        print(f"Merging small sectors into 'Other': {small_sectors}")

    # Reassign small sectors
    for ticker, info in mapping.items():
        if info['industry'] in small_sectors:
            info['industry'] = 'Other'

    # Build sorted sector list and assign indices
    sectors = sorted(set(info['industry'] for info in mapping.values()))
    sector_to_idx = {s: i for i, s in enumerate(sectors)}
    for info in mapping.values():
        info['sector_idx'] = sector_to_idx[info['industry']]

    return mapping


def get_sector_list(mapping: dict) -> list:
    """Return sorted list of unique sector names."""
    return sorted(set(info['industry'] for info in mapping.values()))


def get_sector_stock_counts(mapping: dict) -> dict:
    """Return dict {sector: count}."""
    counts = {}
    for info in mapping.values():
        s = info['industry']
        counts[s] = counts.get(s, 0) + 1
    return dict(sorted(counts.items(), key=lambda x: -x[1]))


def build_and_save_sector_mapping(companies_file: str = "datasets/company_industries.csv",
                                   tickers: list = None,
                                   output_path: str = "data/cache/sector_mapping.json",
                                   min_stocks: int = 5) -> dict:
    """Build complete sector mapping, filter to available tickers, and save.

    Args:
        companies_file: path to company_industries.csv
        tickers: if provided, only keep these tickers
        output_path: where to save JSON
        min_stocks: merge sectors smaller than this

    Returns:
        cleaned mapping dict
    """
    mapping = load_sector_mapping(companies_file)

    # Filter to available tickers
    if tickers:
        mapping = {t: mapping[t] for t in tickers if t in mapping}

    mapping = clean_sectors(mapping, min_stocks)

    # Print summary
    counts = get_sector_stock_counts(mapping)
    print(f"\nSector mapping: {len(mapping)} stocks, {len(counts)} sectors")
    for sector, count in counts.items():
        print(f"  {sector}: {count} stocks")

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {output_path}")

    return mapping

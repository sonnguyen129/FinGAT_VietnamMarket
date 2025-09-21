import yaml
import numpy as np
from src.data.data_processor import VietnamDataProcessor
from src.data.data_loader import MultiAssetWindowDataset, build_dataloader

if __name__ == "__main__":
    # Load configs
    with open('configs/config.yaml', 'r', encoding='utf-8') as f:
        main_config = yaml.safe_load(f)
    with open('configs/data_config.yaml', 'r', encoding='utf-8') as f:
        data_config = yaml.safe_load(f)
    merged_config = main_config.copy()
    merged_config['data_sources'] = data_config.get('data_sources', {})
    merged_config['processing'] = data_config.get('processing', {})
    merged_config['feature_engineering'] = data_config.get('feature_engineering', {})
    merged_config['paths'] = main_config.get('paths', {})

    # Data processing
    processor = VietnamDataProcessor(merged_config)
    processor.load_sector_mapping()
    processor.load_stock_data()
    processor.process_all_stocks()
    # Generate sliding windows
    all_windows = processor.generate_sliding_windows(window_size=16, week_size=5, target_shift=1)
    # Flatten all windows to a list
    windowed_data = []
    for symbol, windows in all_windows.items():
        for w in windows:
            # Convert input_window DataFrame to numpy array (features only)
            # Exclude Date, Symbol, Sector, and target columns
            input_df = w['input_window']
            feature_cols = [col for col in input_df.columns if col not in ['Date','Symbol','Sector','y_return','y_move']]
            w['window'] = input_df[feature_cols].values
            w['ticker'] = symbol
            w['sector'] = processor.sector_mapping.get(symbol, 'Unknown')
            w['feature_cols'] = feature_cols
            # Ensure 'date' key is present for grouping (use target_date)
            w['date'] = w.get('target_date', None)
            windowed_data.append(w)
    # Get sector map
    sector_map = processor.sector_mapping
    # Use feature_cols from first window
    feature_cols = windowed_data[0]['feature_cols'] if windowed_data else []
    # Build dataset
    dataset = MultiAssetWindowDataset(windowed_data, sector_map, feature_cols)
    print(f"Total prediction dates: {len(dataset)}")
    # Build a DataLoader with padding collate
    loader = build_dataloader(dataset, batch_size=2, shuffle=False)
    # Fetch one batch from the loader
    for batch in loader:
        print(f"Batch keys: {list(batch.keys())}")
        print(f"x shape: {batch['x'].shape}")       # (B, N_max, W, F)
        print(f"adj shape: {batch['adj'].shape}")   # (B, 2, N_max, N_max)
        print(f"dates: {batch['date']}")
        print(f"asset_mask shape: {batch['asset_mask'].shape}")
        break

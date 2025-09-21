import yaml
import math
import numpy as np
import torch
from torch.utils.data import Subset
from datetime import datetime
from typing import List, Tuple

from src.utils.seed_utils import set_seed
from src.utils.logging_utils import get_logger
from src.data.data_processor import VietnamDataProcessor
from src.data.data_loader import MultiAssetWindowDataset, build_dataloader
from src.models.fingat_weekly import FinGATWeekly
from src.training.weekly_trainer import WeeklyTrainer, WeeklyLossConfig

logger = get_logger(__name__)


def build_windowed_dataset(config_path_main: str = 'configs/config.yaml',
                           config_path_data: str = 'configs/data_config.yaml') -> Tuple[MultiAssetWindowDataset, List[str]]:
    with open(config_path_main, 'r', encoding='utf-8') as f:
        main_config = yaml.safe_load(f)
    with open(config_path_data, 'r', encoding='utf-8') as f:
        data_config = yaml.safe_load(f)
    merged_config = main_config.copy()
    merged_config['data_sources'] = data_config.get('data_sources', {})
    merged_config['processing'] = data_config.get('processing', {})
    merged_config['feature_engineering'] = data_config.get('feature_engineering', {})
    merged_config['paths'] = main_config.get('paths', {})

    processor = VietnamDataProcessor(merged_config)
    processor.load_sector_mapping()
    processor.load_stock_data()
    processor.process_all_stocks()

    all_windows = processor.generate_sliding_windows(window_size=16, week_size=5, target_shift=1)

    windowed_data = []
    feature_cols_ref = None
    for symbol, windows in all_windows.items():
        for w in windows:
            input_df = w['input_window']
            feature_cols = [col for col in input_df.columns if col not in ['Date','Symbol','Sector','y_return','y_move']]
            window = input_df[feature_cols].values
            item = {
                'window': window,
                'ticker': symbol,
                'sector': processor.sector_mapping.get(symbol, 'Unknown'),
                'feature_cols': feature_cols,
                'y_return': w['y_return'],
                'y_move': w['y_move'],
                'date': w.get('target_date', None)
            }
            windowed_data.append(item)
            if feature_cols_ref is None:
                feature_cols_ref = feature_cols

    sector_map = processor.sector_mapping
    dataset = MultiAssetWindowDataset(windowed_data, sector_map, feature_cols_ref or [])
    return dataset, dataset.dates


def split_by_time(dates: List, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2,
                  start_year=2018, end_year=2023) -> Tuple[List[int], List[int], List[int]]:
    # dates list elements can be pandas.Timestamp or numpy.datetime64 or str
    # Convert to comparable timestamps and filter by year range
    parsed = []
    for i, d in enumerate(dates):
        if hasattr(d, 'to_pydatetime'):
            dt = d.to_pydatetime()
        elif isinstance(d, np.datetime64):
            dt = np.datetime_as_string(d, unit='D')
            dt = datetime.strptime(dt, '%Y-%m-%d')
        elif isinstance(d, str):
            try:
                dt = datetime.strptime(d[:10], '%Y-%m-%d')
            except Exception:
                # Try alternative formats if needed
                dt = datetime.fromisoformat(str(d))
        else:
            dt = datetime.fromisoformat(str(d))
        parsed.append((i, dt))
    # Filter by year
    parsed = [p for p in parsed if start_year <= p[1].year <= end_year]
    # Sort by date
    parsed.sort(key=lambda x: x[1])
    n = len(parsed)
    n_train = int(math.floor(train_ratio * n))
    n_val = int(math.floor(val_ratio * n))
    n_test = n - n_train - n_val
    train_idx = [parsed[i][0] for i in range(n_train)]
    val_idx = [parsed[n_train + i][0] for i in range(n_val)]
    test_idx = [parsed[n_train + n_val + i][0] for i in range(n_test)]
    return train_idx, val_idx, test_idx


def main():
    # Build dataset
    dataset, dates = build_windowed_dataset()
    # Time split 60/20/20 within 2018-2023
    train_idx, val_idx, test_idx = split_by_time(dates, 0.6, 0.2, 0.2, 2018, 2023)

    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    test_set = Subset(dataset, test_idx)

    # Build DataLoaders with custom collate
    from src.data.data_loader import pad_collate_fn
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_set, batch_size=2, shuffle=False, collate_fn=pad_collate_fn)
    val_loader = DataLoader(val_set, batch_size=2, shuffle=False, collate_fn=pad_collate_fn)
    test_loader = DataLoader(test_set, batch_size=2, shuffle=False, collate_fn=pad_collate_fn)

    # Model
    input_dim = len(dataset.feature_cols)
    model = FinGATWeekly(input_dim=input_dim)
    trainer = WeeklyTrainer(model, lr=1e-3, weight_decay=0.0, loss_cfg=WeeklyLossConfig(alpha=0.0, beta=0.01, gamma=0.99, lambda_l2=1e-4, ranking_margin=0.0))

    # Run 10 seeds
    seeds = [i for i in range(1, 11)]
    all_metrics = []
    num_epochs = 10

    for s in seeds:
        set_seed(s)
        # Re-init model per seed
        model = FinGATWeekly(input_dim=input_dim)
        trainer = WeeklyTrainer(model, lr=1e-3, weight_decay=0.0, loss_cfg=WeeklyLossConfig(alpha=0.0, beta=0.01, gamma=0.99, lambda_l2=1e-4, ranking_margin=0.0))
        best_val = float('inf')
        best_state = None
        for epoch in range(num_epochs):
            train_metrics = trainer.run_epoch(train_loader, train=True)
            val_metrics = trainer.run_epoch(val_loader, train=False)
            print(f"Seed {s} Epoch {epoch+1}: train {train_metrics} | val {val_metrics}")
            if val_metrics['loss'] < best_val:
                best_val = val_metrics['loss']
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        # Evaluate on test with best model
        if best_state is not None:
            model.load_state_dict(best_state)
        test_metrics = trainer.run_epoch(test_loader, train=False)
        print(f"Seed {s} Test: {test_metrics}")
        all_metrics.append(test_metrics)

    # Aggregate over seeds
    keys = ['mrr@5', 'mrr@10', 'p@5', 'p@10', 'acc']
    avg = {k: float(np.mean([m[k] for m in all_metrics])) for k in keys}
    std = {k: float(np.std([m[k] for m in all_metrics])) for k in keys}
    print("Averaged over 10 seeds:")
    for k in keys:
        print(f"{k}: {avg[k]:.4f} ± {std[k]:.4f}")


if __name__ == '__main__':
    main()

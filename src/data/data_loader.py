import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from typing import List, Dict, Any

class MultiAssetWindowDataset(Dataset):
    """
    Dataset for multi-asset, windowed time series with per-sample graph structure.
    Each sample is a batch of all stocks for a given prediction date.
    """
    def __init__(self, windowed_data, sector_map, feature_cols, device='cpu'):
        """
        windowed_data: list of dicts, each dict contains:
            - 'date': prediction date
            - 'ticker': stock ticker
            - 'window': np.ndarray (window_size, n_features)
            - 'y_return': float
            - 'y_move': int
            - 'sector': str
        sector_map: dict mapping ticker to sector
        feature_cols: list of feature column names
        """
        self.feature_cols = feature_cols
        self.device = device
        # Group by prediction date
        self.samples_by_date = defaultdict(list)
        for item in windowed_data:
            # Accept either 'date' or 'target_date' as prediction date key
            pred_date = item.get('date', item.get('target_date'))
            item['date'] = pred_date
            self.samples_by_date[pred_date].append(item)
        self.dates = sorted(self.samples_by_date.keys())
        self.sector_map = sector_map
        self.sector_list = sorted(set(sector_map.values()))

    def __len__(self):
        return len(self.dates)

    def __getitem__(self, idx):
        date = self.dates[idx]
        batch = self.samples_by_date[date]
        n_assets = len(batch)
        window_size = batch[0]['window'].shape[0]
        n_features = batch[0]['window'].shape[1]
        # Build tensors
        x = torch.stack([torch.tensor(item['window'], dtype=torch.float32) for item in batch])  # (n_assets, window, n_features)
        y_return = torch.tensor([item['y_return'] for item in batch], dtype=torch.float32)  # (n_assets,)
        y_move = torch.tensor([item['y_move'] for item in batch], dtype=torch.long)  # (n_assets,)
        tickers = [item['ticker'] for item in batch]
        sectors = [item['sector'] for item in batch]
        # Build graph adjacency (two channels: intra-sector, inter-sector)
        adj = self.build_graph(sectors)
        return {
            'x': x.to(self.device),  # (n_assets, window, n_features)
            'y_return': y_return.to(self.device),
            'y_move': y_move.to(self.device),
            'tickers': tickers,
            'sectors': sectors,
            'adj': adj.to(self.device),  # (2, n_assets, n_assets)
            'date': date
        }

    def build_graph(self, sectors):
        n = len(sectors)
        adj_intra = torch.zeros((n, n), dtype=torch.float32)
        adj_inter = torch.zeros((n, n), dtype=torch.float32)
        # Group indices by sector
        sector_indices = defaultdict(list)
        for i, s in enumerate(sectors):
            sector_indices[s].append(i)
        # Intra-sector fully-connected (no self-loops)
        for idxs in sector_indices.values():
            for i in idxs:
                for j in idxs:
                    if i != j:
                        adj_intra[i, j] = 1.0
        # Inter-sector fully-connected across different sectors (no self-loops)
        for i in range(n):
            for j in range(n):
                if i != j and sectors[i] != sectors[j]:
                    adj_inter[i, j] = 1.0
        # Stack as relation channels: (2, n, n)
        return torch.stack([adj_intra, adj_inter], dim=0)


def pad_collate_fn(batch: List[Dict[str, Any]]):
    """
    Collate function that pads variable number of assets per prediction date
    to build 4D tensors: (B, N_max, W, F) and adjacency (B, 2, N_max, N_max).
    Returns a dict with tensors and per-sample metadata lists.
    """
    # Batch is a list of samples (dicts) from __getitem__
    B = len(batch)
    n_assets_list = [sample['x'].shape[0] for sample in batch]
    W = batch[0]['x'].shape[1]
    F = batch[0]['x'].shape[2]
    N_max = max(n_assets_list)

    x_pad = torch.zeros((B, N_max, W, F), dtype=torch.float32)
    y_return_pad = torch.zeros((B, N_max), dtype=torch.float32)
    # Use -100 as ignore index by convention for classification loss, adjust as needed
    y_move_pad = torch.full((B, N_max), fill_value=-100, dtype=torch.long)
    adj_pad = torch.zeros((B, 2, N_max, N_max), dtype=torch.float32)
    asset_mask = torch.zeros((B, N_max), dtype=torch.bool)

    tickers_list: List[List[str]] = []
    sectors_list: List[List[str]] = []
    dates: List[Any] = []

    for b, sample in enumerate(batch):
        n = sample['x'].shape[0]
        x_pad[b, :n] = sample['x']
        y_return_pad[b, :n] = sample['y_return']
        y_move_pad[b, :n] = sample['y_move']
        adj_pad[b, :, :n, :n] = sample['adj']
        asset_mask[b, :n] = True
        tickers_list.append(sample['tickers'])
        sectors_list.append(sample['sectors'])
        dates.append(sample['date'])

    return {
        'x': x_pad,
        'y_return': y_return_pad,
        'y_move': y_move_pad,
        'adj': adj_pad,
        'asset_mask': asset_mask,  # True where an asset exists in the padded dim
        'tickers': tickers_list,   # list per sample (length varies)
        'sectors': sectors_list,   # list per sample (length varies)
        'date': dates
    }


def build_dataloader(dataset: MultiAssetWindowDataset, batch_size: int = 1, shuffle: bool = False, num_workers: int = 0) -> DataLoader:
    """
    Helper to create a DataLoader with the padding collate function.
    Note: variable number of assets per date requires this custom collate.
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=pad_collate_fn)

# Example usage:
# dataset = MultiAssetWindowDataset(windowed_data, sector_map, feature_cols)
# loader = build_dataloader(dataset, batch_size=2, shuffle=False)

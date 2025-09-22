"""
FinGAT DataModule
Complete data pipeline with proper graph construction for Vietnamese stocks
"""

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from .vn_data_loader import VNDataLoader
from .feature_engineering import FeatureEngineer
from .graph_constructor import GraphConstructor


class FinGATDataset(Dataset):
    """
    Custom dataset for FinGAT model
    Handles sliding windows, graph construction, and batching
    """
    
    def __init__(self,
                 features: np.ndarray,
                 returns: np.ndarray,
                 movements: np.ndarray,
                 stock_ids: np.ndarray,
                 dates: List,
                 graph_constructor: GraphConstructor):
        """
        Initialize dataset
        
        Args:
            features: Feature array (n_samples, n_weeks, n_days, n_features)
            returns: Target returns (n_samples,)
            movements: Target movements (n_samples,)
            stock_ids: Stock indices for each sample
            dates: Date for each sample
            graph_constructor: GraphConstructor instance
        """
        self.features = torch.FloatTensor(features)
        self.returns = torch.FloatTensor(returns)
        self.movements = torch.FloatTensor(movements)
        self.stock_ids = torch.LongTensor(stock_ids)
        self.dates = dates
        self.graph_constructor = graph_constructor
        
        # Get number of stocks for proper batching
        self.num_stocks = len(graph_constructor.stocks)
        self.num_sectors = len(graph_constructor.sectors)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        """
        Get a single sample
        Note: Graph edges will be created in collate_fn for batch efficiency
        """
        return {
            'features': self.features[idx],
            'returns': self.returns[idx],
            'movements': self.movements[idx],
            'stock_id': self.stock_ids[idx],
            'date': self.dates[idx]
        }


class FinGATDataModule:
    """
    Complete data module for FinGAT training and evaluation
    """
    
    def __init__(self,
                 data_path: str = 'datasets/VN_datasets/',
                 companies_file: str = 'datasets/VN_Companies.csv',
                 start_date: str = '2023-01-01',
                 end_date: str = '2024-11-30',
                 window_size: int = 15,
                 num_weeks: int = 3,
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.15,
                 batch_size: int = 32,
                 num_stocks_limit: Optional[int] = None,
                 device: str = 'cpu'):
        """
        Initialize data module
        
        Args:
            data_path: Path to stock data files
            companies_file: Path to companies info file
            start_date: Start date for data
            end_date: End date for data
            window_size: Number of days for input window (15 = 3 weeks)
            num_weeks: Number of weeks to organize data into
            train_ratio: Ratio for training data
            val_ratio: Ratio for validation data
            batch_size: Batch size for dataloaders
            num_stocks_limit: Limit number of stocks (for testing)
            device: Device to use
        """
        self.data_path = data_path
        self.companies_file = companies_file
        self.start_date = start_date
        self.end_date = end_date
        self.window_size = window_size
        self.num_weeks = num_weeks
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.batch_size = batch_size
        self.num_stocks_limit = num_stocks_limit
        self.device = device
        
        # Initialize components
        self.data_loader = VNDataLoader(data_path, companies_file, start_date, end_date)
        self.feature_engineer = FeatureEngineer()
        self.graph_constructor = None
        
        # Data storage
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
    def prepare_data(self):
        """
        Load and prepare all data
        """
        print("📊 Preparing FinGAT data...")
        
        # Load stock data
        self.data_loader.load_all_stocks(limit=self.num_stocks_limit)
        
        # Create graph constructor
        self.graph_constructor = GraphConstructor(
            stocks=self.data_loader.valid_stocks,
            sector_mapping=self.data_loader.sector_mapping,
            intra_sector_connectivity=1.0,  # Fully connected within sectors
            inter_sector_connectivity=0.3   # 30% connected between sectors
        )
        
        # Create features for all stocks
        print("🔧 Creating features...")
        all_features = {}
        for symbol in self.data_loader.valid_stocks:
            stock_df = self.data_loader.stock_data[symbol]
            features_df = self.feature_engineer.create_all_features(stock_df, symbol)
            all_features[symbol] = features_df
        
        # Create sliding windows
        print("📦 Creating sliding windows...")
        X, y_returns, y_movements, stock_ids, dates = self._create_sliding_windows(all_features)
        
        # Temporal split
        print("✂️ Splitting data temporally...")
        self._create_temporal_split(X, y_returns, y_movements, stock_ids, dates)
        
        # Print statistics
        self._print_data_statistics()
    
    def _create_sliding_windows(self, features_dict: Dict) -> Tuple:
        """
        Create sliding windows from feature data
        
        Returns:
            Tuple of (features, returns, movements, stock_ids, dates)
        """
        X_list = []
        y_returns = []
        y_movements = []
        stock_ids = []
        dates = []
        
        for symbol in features_dict:
            if symbol not in self.graph_constructor.stock_to_idx:
                continue
            
            stock_id = self.graph_constructor.stock_to_idx[symbol]
            features_df = features_dict[symbol]
            
            # Need at least window_size + 1 days
            if len(features_df) < self.window_size + 1:
                continue
            
            for i in range(self.window_size, len(features_df) - 1):
                # Get window of features
                window = features_df.iloc[i-self.window_size:i][self.feature_engineer.feature_columns].values
                
                # Reshape to (num_weeks, days_per_week, features)
                days_per_week = self.window_size // self.num_weeks
                window_reshaped = window.reshape(self.num_weeks, days_per_week, -1)
                
                # Get next day's return as target
                next_return = features_df.iloc[i]['returns']
                next_movement = 1 if next_return > 0 else 0
                
                X_list.append(window_reshaped)
                y_returns.append(next_return)
                y_movements.append(next_movement)
                stock_ids.append(stock_id)
                dates.append(features_df.index[i])
        
        return (
            np.array(X_list),
            np.array(y_returns),
            np.array(y_movements),
            np.array(stock_ids),
            dates
        )
    
    def _create_temporal_split(self, X, y_returns, y_movements, stock_ids, dates):
        """
        Create temporal train/val/test split
        CRITICAL: Use time-based split to avoid data leakage
        """
        dates_pd = pd.to_datetime(dates)
        
        # Calculate split points
        total_days = (dates_pd.max() - dates_pd.min()).days
        train_end_date = dates_pd.min() + pd.Timedelta(days=int(total_days * self.train_ratio))
        val_end_date = dates_pd.min() + pd.Timedelta(days=int(total_days * (self.train_ratio + self.val_ratio)))
        
        # Create masks
        train_mask = dates_pd <= train_end_date
        val_mask = (dates_pd > train_end_date) & (dates_pd <= val_end_date)
        test_mask = dates_pd > val_end_date
        
        # Create datasets
        self.train_dataset = FinGATDataset(
            X[train_mask], y_returns[train_mask], y_movements[train_mask],
            stock_ids[train_mask], [dates[i] for i, m in enumerate(train_mask) if m],
            self.graph_constructor
        )
        
        self.val_dataset = FinGATDataset(
            X[val_mask], y_returns[val_mask], y_movements[val_mask],
            stock_ids[val_mask], [dates[i] for i, m in enumerate(val_mask) if m],
            self.graph_constructor
        )
        
        self.test_dataset = FinGATDataset(
            X[test_mask], y_returns[test_mask], y_movements[test_mask],
            stock_ids[test_mask], [dates[i] for i, m in enumerate(test_mask) if m],
            self.graph_constructor
        )
    
    def _print_data_statistics(self):
        """Print data statistics"""
        print("\n" + "="*60)
        print("DATA STATISTICS")
        print("="*60)
        print(f"Total Stocks: {len(self.graph_constructor.stocks)}")
        print(f"Total Sectors: {len(self.graph_constructor.sectors)}")
        print(f"\nDataset Sizes:")
        print(f"  Train: {len(self.train_dataset):6d} samples")
        print(f"  Val:   {len(self.val_dataset):6d} samples")
        print(f"  Test:  {len(self.test_dataset):6d} samples")
        
        # Graph statistics
        stats = self.graph_constructor.get_graph_statistics()
        print(f"\nGraph Structure:")
        print(f"  Intra-sector edges: {stats['num_intra_edges']}")
        print(f"  Inter-sector edges: {stats['num_inter_edges']}")
        print("="*60)
    
    def custom_collate_fn(self, batch: List[Dict]) -> Dict:
        """
        Custom collate function that creates graph edges for the batch
        
        Args:
            batch: List of samples from dataset
            
        Returns:
            Batched data with graph edges
        """
        # Stack regular tensors
        features = torch.stack([sample['features'] for sample in batch])
        returns = torch.stack([sample['returns'] for sample in batch])
        movements = torch.stack([sample['movements'] for sample in batch])
        stock_ids = torch.stack([sample['stock_id'] for sample in batch])
        
        # Expand features for all stocks in batch
        batch_size = len(batch)
        num_stocks = self.graph_constructor.stock_to_idx.__len__()
        
        # Create expanded features tensor (batch_size, num_stocks, num_weeks, days_per_week, features)
        expanded_features = torch.zeros(
            batch_size, num_stocks, 
            features.shape[1], features.shape[2], features.shape[3]
        )
        
        # Fill in features for corresponding stocks
        for i, stock_id in enumerate(stock_ids):
            expanded_features[i, stock_id] = features[i]
        
        # Create graph edges for batch
        edge_index_intra, edge_index_inter, stock_to_sector = \
            self.graph_constructor.create_batch_edges(batch_size, self.device)
        
        return {
            'features': expanded_features.to(self.device),
            'returns': returns.to(self.device),
            'movements': movements.to(self.device),
            'stock_ids': stock_ids.to(self.device),
            'edge_index_intra': edge_index_intra,
            'edge_index_inter': edge_index_inter,
            'stock_to_sector': stock_to_sector
        }
    
    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Get train, validation, and test dataloaders
        
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        if self.train_dataset is None:
            self.prepare_data()
        
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.custom_collate_fn
        )
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.custom_collate_fn
        )
        
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.custom_collate_fn
        )
        
        return train_loader, val_loader, test_loader
    
    def get_model_config(self) -> Dict:
        """
        Get model configuration based on data
        
        Returns:
            Dictionary with model configuration
        """
        return {
            'input_dim': 19,  # Number of features
            'hidden_dim': 16,  # Hidden dimension
            'time_steps': self.window_size // self.num_weeks,  # Days per week
            'num_weeks': self.num_weeks,
            'num_stocks': len(self.graph_constructor.stocks),
            'num_sectors': len(self.graph_constructor.sectors),
            'dropout': 0.2,
            'device': self.device
        }


# Test the data module
if __name__ == "__main__":
    print("Testing FinGAT DataModule")
    print("-"*60)
    
    # Initialize data module
    datamodule = FinGATDataModule(
        data_path='../../datasets/VN_datasets/',
        companies_file='../../datasets/VN_Companies.csv',
        num_stocks_limit=10,  # Limit to 10 stocks for testing
        batch_size=4,
        device='cpu'
    )
    
    # Prepare data
    datamodule.prepare_data()
    
    # Get dataloaders
    train_loader, val_loader, test_loader = datamodule.get_dataloaders()
    
    print(f"\n✅ DataLoaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # Test one batch
    if len(train_loader) > 0:
        batch = next(iter(train_loader))
        print(f"\n📦 Sample batch:")
        print(f"  Features shape: {batch['features'].shape}")
        print(f"  Returns shape: {batch['returns'].shape}")
        print(f"  Movements shape: {batch['movements'].shape}")
        print(f"  Intra edges shape: {batch['edge_index_intra'].shape}")
        print(f"  Inter edges shape: {batch['edge_index_inter'].shape}")
        print(f"  Stock-to-sector shape: {batch['stock_to_sector'].shape}")
    
    # Get model config
    model_config = datamodule.get_model_config()
    print(f"\n🤖 Model configuration:")
    for key, value in model_config.items():
        print(f"  {key}: {value}")
    
    print("\n✅ DataModule working correctly!")
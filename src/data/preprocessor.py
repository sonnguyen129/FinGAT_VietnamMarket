"""
Data Preprocessor for Vietnamese Stock Market
Integrates data loading, feature engineering, and graph building
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split

from .vn_data_loader import VNDataLoader
from .feature_engineering import FeatureEngineer
from .graph_builder import GraphBuilder


class VNPreprocessor:
    """Complete preprocessing pipeline for Vietnamese stock market data"""
    
    def __init__(self,
                 data_path: str = 'datasets/VN_datasets/',
                 companies_file: str = 'datasets/VN_Companies.csv',
                 start_date: Optional[str] = None,
                 end_date: Optional[str] = None,
                 min_history_days: int = 365,
                 window_size: int = 15,
                 num_weeks: int = 3,
                 train_ratio: float = 0.6,
                 val_ratio: float = 0.2,
                 test_ratio: float = 0.2):
        """
        Initialize preprocessor
        
        Args:
            data_path: Path to stock data files
            companies_file: Path to companies info file
            start_date: Start date for data
            end_date: End date for data
            min_history_days: Minimum required history
            window_size: Input window size (days)
            num_weeks: Number of weeks for FinGAT
            train_ratio: Training data ratio
            val_ratio: Validation data ratio
            test_ratio: Test data ratio
        """
        self.data_path = data_path
        self.companies_file = companies_file
        self.start_date = start_date
        self.end_date = end_date
        self.min_history_days = min_history_days
        self.window_size = window_size
        self.num_weeks = num_weeks
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        
        # Initialize components
        self.data_loader = VNDataLoader(
            data_path=data_path,
            companies_file=companies_file,
            start_date=start_date,
            end_date=end_date,
            min_history_days=min_history_days
        )
        
        self.feature_engineer = FeatureEngineer()
        self.graph_builder = GraphBuilder()
        
        # Storage for processed data
        self.processed_data = {}
        self.train_data = {}
        self.val_data = {}
        self.test_data = {}
        
    def prepare_data(self, limit_stocks: Optional[int] = None) -> Dict:
        """
        Complete data preparation pipeline
        
        Args:
            limit_stocks: Limit number of stocks (for testing)
        
        Returns:
            Dictionary with all processed data
        """
        print("="*50)
        print("Starting data preparation pipeline...")
        print("="*50)
        
        # Step 1: Load stock data
        print("\n[Step 1/6] Loading stock data...")
        self.data_loader.load_all_stocks(limit=limit_stocks)
        
        # Step 2: Align dates
        print("\n[Step 2/6] Aligning trading dates...")
        self.data_loader.align_dates()
        
        # Step 3: Calculate returns
        print("\n[Step 3/6] Calculating returns...")
        self.data_loader.calculate_returns()
        
        # Step 4: Create features for all stocks
        print("\n[Step 4/6] Engineering features...")
        self._create_features()
        
        # Step 5: Build graphs
        print("\n[Step 5/6] Building graph structures...")
        self._build_graphs()
        
        # Step 6: Create datasets
        print("\n[Step 6/6] Creating train/val/test datasets...")
        self._create_datasets()
        
        print("\n" + "="*50)
        print("Data preparation completed!")
        print("="*50)
        
        return self.processed_data
    
    def _create_features(self) -> None:
        """Create features for all stocks"""
        for symbol in self.data_loader.valid_stocks:
            stock_data = self.data_loader.stock_data[symbol]['data']
            
            # Create features
            featured_data = self.feature_engineer.create_all_features(
                stock_data, symbol=symbol
            )
            
            # Store featured data
            self.data_loader.stock_data[symbol]['featured_data'] = featured_data
        
        print(f"  Created features for {len(self.data_loader.valid_stocks)} stocks")
    
    def _build_graphs(self) -> None:
        """Build graph structures"""
        # Build stock and sector mappings
        self.graph_builder.build_stock_mapping(self.data_loader.valid_stocks)
        sectors = list(self.data_loader.sector_mapping.keys())
        self.graph_builder.build_sector_mapping(sectors)
        
        # Build intra-sector edges
        intra_edges = self.graph_builder.build_intra_sector_edges(
            self.data_loader.sector_mapping,
            fully_connected=True
        )
        
        # Build inter-sector edges  
        inter_edges = self.graph_builder.build_inter_sector_edges(
            sectors,
            fully_connected=True
        )
        
        print(f"  Intra-sector edges: {intra_edges.shape[1]}")
        print(f"  Inter-sector edges: {inter_edges.shape[1]}")
        
        # Store in processed data
        self.processed_data['edges'] = {
            'intra_sector': intra_edges,
            'inter_sector': inter_edges
        }
        
        self.processed_data['mappings'] = {
            'stock_to_idx': self.graph_builder.stock_to_idx,
            'idx_to_stock': self.graph_builder.idx_to_stock,
            'sector_to_idx': self.graph_builder.sector_to_idx,
            'idx_to_sector': self.graph_builder.idx_to_sector
        }
    
    def _create_datasets(self) -> None:
        """Create train/val/test datasets"""
        all_features = []
        all_returns = []
        all_movements = []
        
        # Collect data from all stocks
        for symbol in self.data_loader.valid_stocks:
            featured_data = self.data_loader.stock_data[symbol]['featured_data']
            
            # Create sliding windows
            features, returns, movements = self.feature_engineer.create_sliding_windows(
                featured_data,
                window_size=self.window_size,
                target_size=1
            )
            
            if len(features) > 0:
                all_features.append(features)
                all_returns.append(returns)
                all_movements.append(movements)
        
        # Stack all stock data
        if all_features:
            # Shape: (num_stocks, num_samples, window_size, num_features)
            all_features = np.array(all_features)
            all_returns = np.array(all_returns)
            all_movements = np.array(all_movements)
            
            # Transpose to (num_samples, num_stocks, window_size, num_features)
            all_features = np.transpose(all_features, (1, 0, 2, 3))
            all_returns = np.transpose(all_returns, (1, 0, 2))
            all_movements = np.transpose(all_movements, (1, 0, 2))
            
            # Squeeze target dimension
            all_returns = all_returns.squeeze(-1)
            all_movements = all_movements.squeeze(-1)
            
            print(f"  Total samples: {len(all_features)}")
            print(f"  Feature shape: {all_features.shape}")
            
            # Split into train/val/test
            self._split_data(all_features, all_returns, all_movements)
    
    def _split_data(self, features, returns, movements) -> None:
        """Split data into train/val/test sets"""
        n_samples = len(features)
        
        # Calculate split indices
        train_end = int(n_samples * self.train_ratio)
        val_end = train_end + int(n_samples * self.val_ratio)
        
        # Split data (temporal split, not random)
        train_features = features[:train_end]
        train_returns = returns[:train_end]
        train_movements = movements[:train_end]
        
        val_features = features[train_end:val_end]
        val_returns = returns[train_end:val_end]
        val_movements = movements[train_end:val_end]
        
        test_features = features[val_end:]
        test_returns = returns[val_end:]
        test_movements = movements[val_end:]
        
        # Group into weeks for FinGAT
        train_weeks = self._group_into_weeks(train_features)
        val_weeks = self._group_into_weeks(val_features)
        test_weeks = self._group_into_weeks(test_features)
        
        # Store processed data
        self.train_data = {
            'features': train_features,
            'weekly_features': train_weeks,
            'returns': train_returns,
            'movements': train_movements
        }
        
        self.val_data = {
            'features': val_features,
            'weekly_features': val_weeks,
            'returns': val_returns,
            'movements': val_movements
        }
        
        self.test_data = {
            'features': test_features,
            'weekly_features': test_weeks,
            'returns': test_returns,
            'movements': test_movements
        }
        
        self.processed_data['train'] = self.train_data
        self.processed_data['val'] = self.val_data
        self.processed_data['test'] = self.test_data
        
        print(f"  Train samples: {len(train_features)}")
        print(f"  Val samples: {len(val_features)}")
        print(f"  Test samples: {len(test_features)}")
    
    def _group_into_weeks(self, features) -> Dict:
        """Group daily features into weekly format"""
        # features shape: (samples, stocks, days, features)
        samples, stocks, days, num_features = features.shape
        days_per_week = 5
        
        weekly_data = {}
        num_weeks = min(self.num_weeks, days // days_per_week)
        
        for week_idx in range(num_weeks):
            start_day = week_idx * days_per_week
            end_day = min(start_day + days_per_week, days)
            
            week_key = f'x{week_idx + 1}'
            weekly_data[week_key] = features[:, :, start_day:end_day, :]
        
        # Add return and movement targets
        weekly_data['y_return_ratio'] = None  # Will be set during training
        weekly_data['y_up_or_down'] = None
        
        return weekly_data
    
    def save_processed_data(self, save_path: str = 'datasets/processed/') -> None:
        """Save processed data to files"""
        os.makedirs(save_path, exist_ok=True)
        
        # Save main processed data
        with open(f'{save_path}/vn_processed_data.pkl', 'wb') as f:
            pickle.dump(self.processed_data, f)
        
        # Save edge indices separately
        if 'edges' in self.processed_data:
            np.save(f'{save_path}/intra_sector_edges.npy', 
                   self.processed_data['edges']['intra_sector'])
            np.save(f'{save_path}/inter_sector_edges.npy',
                   self.processed_data['edges']['inter_sector'])
        
        print(f"Processed data saved to {save_path}")
    
    def load_processed_data(self, load_path: str = 'datasets/processed/') -> Dict:
        """Load processed data from files"""
        # Load main data
        with open(f'{load_path}/vn_processed_data.pkl', 'rb') as f:
            self.processed_data = pickle.load(f)
        
        # Restore train/val/test data
        self.train_data = self.processed_data.get('train', {})
        self.val_data = self.processed_data.get('val', {})
        self.test_data = self.processed_data.get('test', {})
        
        print(f"Processed data loaded from {load_path}")
        
        return self.processed_data


# Example usage
if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = VNPreprocessor(
        data_path='datasets/VN_datasets/',
        companies_file='datasets/VN_Companies.csv',
        start_date='2019-01-01',
        end_date='2023-12-31',
        min_history_days=250,
        window_size=15,
        num_weeks=3
    )
    
    # Prepare data (limit to 30 stocks for testing)
    processed_data = preprocessor.prepare_data(limit_stocks=30)
    
    # Save processed data
    preprocessor.save_processed_data()
    
    # Print summary
    print("\n" + "="*50)
    print("Processing Summary:")
    print("="*50)
    
    if 'train' in processed_data:
        print(f"Train samples: {len(processed_data['train']['features'])}")
        print(f"Val samples: {len(processed_data['val']['features'])}")
        print(f"Test samples: {len(processed_data['test']['features'])}")
        
    if 'edges' in processed_data:
        print(f"\nGraph edges:")
        print(f"  Intra-sector: {processed_data['edges']['intra_sector'].shape}")
        print(f"  Inter-sector: {processed_data['edges']['inter_sector'].shape}")
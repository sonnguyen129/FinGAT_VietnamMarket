#!/usr/bin/env python
"""
Test script for Vietnamese stock market data pipeline
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from src.data.preprocessor import VNPreprocessor


def test_data_pipeline():
    """Test the complete data pipeline"""
    
    print("="*70)
    print("TESTING VIETNAMESE STOCK MARKET DATA PIPELINE")
    print("="*70)
    
    # Initialize preprocessor
    print("\n1. Initializing preprocessor...")
    preprocessor = VNPreprocessor(
        data_path='datasets/VN_datasets/',
        companies_file='datasets/VN_Companies.csv',
        start_date='2020-01-01',  # Use recent data for testing
        end_date='2023-12-31',
        min_history_days=250,
        window_size=15,  # 15 days input (3 weeks)
        num_weeks=3,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2
    )
    
    # Test with limited stocks first
    print("\n2. Testing with 20 stocks...")
    processed_data = preprocessor.prepare_data(limit_stocks=20)
    
    # Print detailed information
    print("\n" + "="*70)
    print("DATA PIPELINE RESULTS")
    print("="*70)
    
    # Data loader info
    print("\n📊 Data Loader Statistics:")
    print(f"  - Valid stocks loaded: {len(preprocessor.data_loader.valid_stocks)}")
    print(f"  - Sectors found: {len(preprocessor.data_loader.sector_mapping)}")
    
    # Print sector distribution
    print("\n📈 Sector Distribution:")
    for sector, stocks in preprocessor.data_loader.sector_mapping.items():
        print(f"  - {sector}: {len(stocks)} stocks")
    
    # Feature engineering info
    if preprocessor.data_loader.valid_stocks:
        sample_symbol = preprocessor.data_loader.valid_stocks[0]
        sample_data = preprocessor.data_loader.stock_data[sample_symbol]
        
        if 'featured_data' in sample_data:
            featured_df = sample_data['featured_data']
            print(f"\n🔧 Feature Engineering:")
            print(f"  - Original columns: {len(sample_data['data'].columns)}")
            print(f"  - Featured columns: {len(featured_df.columns)}")
            print(f"  - New features created: {len(featured_df.columns) - len(sample_data['data'].columns)}")
            
            # List feature names
            feature_cols = [col for col in featured_df.columns 
                          if col not in ['Date', 'Open', 'High', 'Low', 'Close', 
                                       'Adj Close', 'Volume']]
            print(f"\n  Feature names ({len(feature_cols)} features):")
            for i in range(0, len(feature_cols), 5):
                print(f"    {', '.join(feature_cols[i:i+5])}")
    
    # Graph statistics
    if 'edges' in processed_data:
        print(f"\n🕸️ Graph Construction:")
        print(f"  - Intra-sector edges: {processed_data['edges']['intra_sector'].shape}")
        print(f"  - Inter-sector edges: {processed_data['edges']['inter_sector'].shape}")
        
        stats = preprocessor.graph_builder.get_edge_statistics()
        for key, value in stats.items():
            print(f"  - {key}: {value:.2f}" if isinstance(value, float) else f"  - {key}: {value}")
    
    # Dataset statistics
    if 'train' in processed_data:
        print(f"\n📚 Dataset Split:")
        
        train_data = processed_data['train']
        val_data = processed_data['val']
        test_data = processed_data['test']
        
        print(f"  Training set:")
        print(f"    - Samples: {len(train_data['features'])}")
        print(f"    - Shape: {train_data['features'].shape}")
        print(f"    - Returns shape: {train_data['returns'].shape}")
        
        print(f"  Validation set:")
        print(f"    - Samples: {len(val_data['features'])}")
        print(f"    - Shape: {val_data['features'].shape}")
        
        print(f"  Test set:")
        print(f"    - Samples: {len(test_data['features'])}")
        print(f"    - Shape: {test_data['features'].shape}")
        
        # Check weekly grouping
        if 'weekly_features' in train_data:
            print(f"\n  Weekly grouping:")
            for key in train_data['weekly_features']:
                if key.startswith('x'):
                    print(f"    - {key} shape: {train_data['weekly_features'][key].shape}")
    
    # Save processed data
    print("\n3. Saving processed data...")
    preprocessor.save_processed_data('datasets/processed/')
    
    # Test loading
    print("\n4. Testing data loading...")
    preprocessor2 = VNPreprocessor()
    loaded_data = preprocessor2.load_processed_data('datasets/processed/')
    
    print("  ✅ Data loaded successfully!")
    
    # Memory usage
    print("\n💾 Memory Usage:")
    if 'train' in processed_data:
        train_size = processed_data['train']['features'].nbytes / (1024 * 1024)
        val_size = processed_data['val']['features'].nbytes / (1024 * 1024)
        test_size = processed_data['test']['features'].nbytes / (1024 * 1024)
        
        print(f"  - Training data: {train_size:.2f} MB")
        print(f"  - Validation data: {val_size:.2f} MB")
        print(f"  - Test data: {test_size:.2f} MB")
        print(f"  - Total: {train_size + val_size + test_size:.2f} MB")
    
    print("\n" + "="*70)
    print("✅ DATA PIPELINE TEST COMPLETED SUCCESSFULLY!")
    print("="*70)
    
    return processed_data


if __name__ == "__main__":
    try:
        processed_data = test_data_pipeline()
    except Exception as e:
        print(f"\n❌ Error in data pipeline: {e}")
        import traceback
        traceback.print_exc()
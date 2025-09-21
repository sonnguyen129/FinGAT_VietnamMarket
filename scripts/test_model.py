#!/usr/bin/env python
"""
Test script for FinGAT model implementation
Verifies model architecture and forward pass
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from src.models.fingat import FinGAT
from src.data.datamodule import VNStockDataModule


def test_model_components():
    """Test individual model components"""
    print("="*70)
    print("TESTING FINGAT MODEL COMPONENTS")
    print("="*70)
    
    # Test parameters
    batch_size = 2
    num_stocks = 19
    num_sectors = 5
    time_steps = 5
    input_dim = 19
    hidden_dim = 16
    num_weeks = 3
    
    print(f"\nTest Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Number of stocks: {num_stocks}")
    print(f"  Number of sectors: {num_sectors}")
    print(f"  Time steps per week: {time_steps}")
    print(f"  Input features: {input_dim}")
    print(f"  Hidden dimension: {hidden_dim}")
    print(f"  Number of weeks: {num_weeks}")
    
    # Initialize model
    print("\n1. Initializing FinGAT model...")
    model = FinGAT(
        input_dim=input_dim,
        time_steps=time_steps,
        hidden_dim=hidden_dim,
        num_weeks=num_weeks,
        num_stocks=num_stocks,
        num_sectors=num_sectors,
        dropout=0.2,
        device='cpu'
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Parameters:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Create dummy data
    print("\n2. Creating dummy input data...")
    weekly_data = []
    for week in range(num_weeks):
        week_tensor = torch.randn(batch_size, num_stocks, time_steps, input_dim)
        weekly_data.append(week_tensor)
    
    # Create dummy edges
    # Intra-sector edges (within sector connections)
    intra_edges = []
    stocks_per_sector = num_stocks // num_sectors
    for sector in range(num_sectors):
        start = sector * stocks_per_sector
        end = min(start + stocks_per_sector, num_stocks)
        for i in range(start, end):
            for j in range(start, end):
                if i != j:
                    intra_edges.append([i, j])
    
    intra_edge_index = torch.LongTensor(intra_edges).t()
    
    # Inter-sector edges (between sectors)
    inter_edges = []
    for i in range(num_sectors):
        for j in range(num_sectors):
            if i != j:
                inter_edges.append([i, j])
    
    inter_edge_index = torch.LongTensor(inter_edges).t()
    
    # Stock to sector mapping
    stock_to_sector = torch.LongTensor([i // stocks_per_sector for i in range(num_stocks)])
    
    print(f"\nGraph Structure:")
    print(f"  Intra-sector edges: {intra_edge_index.shape}")
    print(f"  Inter-sector edges: {inter_edge_index.shape}")
    print(f"  Stock-to-sector mapping: {stock_to_sector.shape}")
    
    # Test forward pass
    print("\n3. Testing forward pass...")
    model.eval()
    with torch.no_grad():
        try:
            returns, movements = model(
                weekly_data=weekly_data,
                intra_edge_index=intra_edge_index,
                inter_edge_index=inter_edge_index,
                stock_to_sector=stock_to_sector
            )
            
            print(f"\n✅ Forward pass successful!")
            print(f"  Returns shape: {returns.shape}")
            print(f"  Movements shape: {movements.shape}")
            print(f"  Returns range: [{returns.min():.4f}, {returns.max():.4f}]")
            print(f"  Movements range: [{movements.min():.4f}, {movements.max():.4f}]")
            
        except Exception as e:
            print(f"\n❌ Forward pass failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # Test prediction
    print("\n4. Testing prediction...")
    try:
        top_k_indices, top_k_returns, all_movements = model.predict(
            weekly_data=weekly_data,
            intra_edge_index=intra_edge_index,
            inter_edge_index=inter_edge_index,
            stock_to_sector=stock_to_sector,
            top_k=5
        )
        
        print(f"  Top-5 stock indices: {top_k_indices.tolist()}")
        print(f"  Top-5 returns: {top_k_returns.tolist()}")
        print(f"  Movement predictions: {all_movements[:10].tolist()[:5]}...")
        print(f"\n✅ Prediction successful!")
        
    except Exception as e:
        print(f"\n❌ Prediction failed: {e}")
        return False
    
    return True


def test_with_real_data():
    """Test model with real Vietnamese stock data"""
    print("\n" + "="*70)
    print("TESTING WITH REAL DATA")
    print("="*70)
    
    try:
        # Load data module
        print("\n1. Loading Vietnamese stock data...")
        data_module = VNStockDataModule(
            data_path='datasets/processed/',
            batch_size=4,
            num_workers=0
        )
        
        # Get data statistics
        stats = data_module.get_data_stats()
        print(f"\nData loaded successfully:")
        print(f"  Stocks: {stats['num_stocks']}")
        print(f"  Sectors: {stats['num_sectors']}")
        print(f"  Features: {stats['num_features']}")
        print(f"  Days: {stats['num_days']}")
        print(f"  Train samples: {stats['train_samples']}")
        
        # Get sample batch
        print("\n2. Getting sample batch...")
        sample_batch = data_module.get_sample_batch()
        
        if not sample_batch:
            print("❌ No sample batch available")
            return False
        
        # Initialize model with real dimensions
        print("\n3. Initializing model with real dimensions...")
        model = FinGAT(
            input_dim=stats['num_features'],
            time_steps=5,  # 5 days per week
            hidden_dim=16,
            num_weeks=3,
            num_stocks=stats['num_stocks'],
            num_sectors=stats['num_sectors'],
            dropout=0.2,
            device='cpu'
        )
        
        # Test forward pass with real data
        print("\n4. Testing forward pass with real data...")
        model.eval()
        
        weekly_data = sample_batch['weekly_data']
        
        # Need to adjust weekly data format
        # Currently: List of (1, stocks, days, features)
        # Need: List of (batch, stocks, days, features)
        
        # For testing, create a small batch
        batch_weekly_data = []
        for week_tensor in weekly_data[:3]:  # Use first 3 weeks
            # Repeat to create batch
            batch_week = week_tensor.repeat(4, 1, 1, 1)  # batch_size=4
            batch_weekly_data.append(batch_week)
        
        with torch.no_grad():
            try:
                returns, movements = model(
                    weekly_data=batch_weekly_data,
                    intra_edge_index=sample_batch['intra_edges'],
                    inter_edge_index=sample_batch['inter_edges'],
                    stock_to_sector=sample_batch['stock_to_sector']
                )
                
                print(f"\n✅ Model works with real data!")
                print(f"  Returns shape: {returns.shape}")
                print(f"  Movements shape: {movements.shape}")
                
                # Check outputs are reasonable
                if torch.isnan(returns).any() or torch.isnan(movements).any():
                    print("⚠️  Warning: NaN values in output")
                else:
                    print("  Output values are valid (no NaN)")
                
            except Exception as e:
                print(f"\n❌ Forward pass with real data failed: {e}")
                import traceback
                traceback.print_exc()
                return False
        
    except FileNotFoundError:
        print("\n⚠️  Processed data not found. Run data pipeline first.")
        return False
    except Exception as e:
        print(f"\n❌ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def main():
    """Main test function"""
    print("="*70)
    print("FINGAT MODEL TEST SUITE")
    print("="*70)
    
    # Test model components
    components_ok = test_model_components()
    
    # Test with real data if available
    real_data_ok = test_with_real_data()
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"  Model components: {'✅ PASSED' if components_ok else '❌ FAILED'}")
    print(f"  Real data test: {'✅ PASSED' if real_data_ok else '⚠️ SKIPPED/FAILED'}")
    
    if components_ok:
        print("\n✅ Model implementation is working correctly!")
    else:
        print("\n❌ Model implementation has issues that need to be fixed.")
    
    return components_ok


if __name__ == "__main__":
    success = main()
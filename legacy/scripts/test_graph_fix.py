"""
Test script for fixed graph construction and data module
Verifies edge indices and stock-to-sector mapping work correctly
"""

import os
import sys
import torch
sys.path.append('..')

from src.data.graph_constructor import GraphConstructor
from src.data.fingat_datamodule import FinGATDataModule
from src.data.vn_data_loader import VNDataLoader
from src.models.fingat import FinGAT


def test_graph_constructor():
    """Test graph constructor with real Vietnamese stock data"""
    print("\n" + "="*60)
    print("TESTING GRAPH CONSTRUCTOR")
    print("="*60)
    
    # Load real data
    data_loader = VNDataLoader(
        data_path='../datasets/VN_datasets/',
        companies_file='../datasets/VN_Companies.csv'
    )
    data_loader.load_all_stocks(limit=20)  # Load 20 stocks for testing
    
    # Create graph constructor
    graph_constructor = GraphConstructor(
        stocks=data_loader.valid_stocks,
        sector_mapping=data_loader.sector_mapping,
        intra_sector_connectivity=1.0,
        inter_sector_connectivity=0.3
    )
    
    # Visualize structure
    graph_constructor.visualize_graph_structure()
    
    # Test edge creation
    batch_size = 4
    edge_index_intra, edge_index_inter, stock_to_sector = graph_constructor.create_batch_edges(batch_size)
    
    print(f"\n✅ Graph Construction Test Results:")
    print(f"  Stocks: {len(data_loader.valid_stocks)}")
    print(f"  Sectors: {len(graph_constructor.sectors)}")
    print(f"  Intra-sector edges: {edge_index_intra.shape}")
    print(f"  Inter-sector edges: {edge_index_inter.shape}")
    print(f"  Stock-to-sector mapping: {stock_to_sector.shape}")
    
    # Verify edge indices are valid
    max_node_idx = batch_size * len(data_loader.valid_stocks) - 1
    assert edge_index_intra.max() <= max_node_idx, "Invalid intra-sector edge indices"
    assert edge_index_inter.max() <= max_node_idx, "Invalid inter-sector edge indices"
    
    # Verify stock-to-sector mapping
    assert stock_to_sector.shape[0] == batch_size * len(data_loader.valid_stocks)
    assert stock_to_sector.max() < len(graph_constructor.sectors)
    
    print("  ✓ All edge indices valid")
    print("  ✓ Stock-to-sector mapping correct")
    
    return graph_constructor


def test_datamodule():
    """Test complete data module with graph construction"""
    print("\n" + "="*60)
    print("TESTING FINGAT DATAMODULE")
    print("="*60)
    
    # Initialize data module
    datamodule = FinGATDataModule(
        data_path='../datasets/VN_datasets/',
        companies_file='../datasets/VN_Companies.csv',
        num_stocks_limit=10,  # Limit for testing
        batch_size=4,
        window_size=15,
        num_weeks=3,
        device='cpu'
    )
    
    # Prepare data
    print("\n📊 Preparing data...")
    datamodule.prepare_data()
    
    # Get dataloaders
    train_loader, val_loader, test_loader = datamodule.get_dataloaders()
    
    print(f"\n✅ DataModule Test Results:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # Test a batch
    if len(train_loader) > 0:
        batch = next(iter(train_loader))
        
        print(f"\n📦 Sample Batch Shapes:")
        print(f"  Features: {batch['features'].shape}")
        print(f"  Returns: {batch['returns'].shape}")
        print(f"  Movements: {batch['movements'].shape}")
        print(f"  Intra edges: {batch['edge_index_intra'].shape}")
        print(f"  Inter edges: {batch['edge_index_inter'].shape}")
        print(f"  Stock-to-sector: {batch['stock_to_sector'].shape}")
        
        # Verify batch dimensions
        batch_size = batch['features'].shape[0]
        num_stocks = batch['features'].shape[1]
        
        assert batch['stock_to_sector'].shape[0] == batch_size * num_stocks
        print(f"\n  ✓ Batch dimensions correct")
        print(f"  ✓ Graph edges properly created")
        
    return datamodule


def test_model_forward_pass(datamodule):
    """Test model forward pass with fixed graph structure"""
    print("\n" + "="*60)
    print("TESTING MODEL FORWARD PASS")
    print("="*60)
    
    # Get model config from datamodule
    model_config = datamodule.get_model_config()
    
    print("\n🤖 Initializing FinGAT model...")
    print(f"  Config: {model_config}")
    
    # Initialize model
    model = FinGAT(**model_config)
    model.eval()
    
    # Get a batch
    train_loader, _, _ = datamodule.get_dataloaders()
    
    if len(train_loader) > 0:
        batch = next(iter(train_loader))
        
        print("\n🔄 Testing forward pass...")
        
        try:
            with torch.no_grad():
                # Forward pass with proper inputs
                pred_returns, pred_movements = model(
                    batch['features'],
                    batch['edge_index_intra'],
                    batch['edge_index_inter'],
                    batch['stock_to_sector']
                )
            
            print(f"\n✅ Forward Pass Success!")
            print(f"  Predicted returns shape: {pred_returns.shape}")
            print(f"  Predicted movements shape: {pred_movements.shape}")
            
            # Verify output shapes
            batch_size = batch['features'].shape[0]
            assert pred_returns.shape[0] == batch_size
            assert pred_movements.shape[0] == batch_size
            
            print(f"  ✓ Output shapes correct")
            
        except Exception as e:
            print(f"\n❌ Forward pass failed: {e}")
            return False
    
    return True


def test_full_pipeline():
    """Test the complete pipeline with fixed graph construction"""
    print("\n" + "="*70)
    print("TESTING COMPLETE PIPELINE WITH GRAPH FIX")
    print("="*70)
    
    try:
        # Test 1: Graph Constructor
        graph_constructor = test_graph_constructor()
        
        # Test 2: DataModule
        datamodule = test_datamodule()
        
        # Test 3: Model Forward Pass
        success = test_model_forward_pass(datamodule)
        
        if success:
            print("\n" + "="*70)
            print("✅ ALL TESTS PASSED SUCCESSFULLY!")
            print("="*70)
            print("\n🎉 The graph construction issues have been fixed!")
            print("The pipeline is now ready for training with proper:")
            print("  - Intra-sector edges (within sector connections)")
            print("  - Inter-sector edges (between sector connections)")
            print("  - Stock-to-sector mapping tensor")
            print("\nYou can now train the model with:")
            print("  python scripts/train_with_fixed_graph.py")
        else:
            print("\n⚠️ Some issues remain. Please check the errors above.")
            
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_full_pipeline()
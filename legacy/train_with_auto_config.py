"""
Training script with auto-detected input_dim and configuration management
Demonstrates the new centralized config system
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data.fingat_datamodule import FinGATDataModule
from src.models.fingat import FinGAT
from src.utils.config_manager import load_config


def main():
    print("🚀 FinGAT Training with Auto-Config Detection")
    print("=" * 60)
    
    # Step 1: Load configuration
    print("\n📋 Step 1: Loading configuration...")
    config_manager = load_config('configs/default_config.yaml')
    config_manager.print_config()
    
    # Step 2: Initialize data module
    print("\n📊 Step 2: Initializing data module...")
    datamodule = FinGATDataModule(
        config_manager=config_manager,
        num_stocks_limit=20  # Small test
    )
    
    # Step 3: Prepare data (this will auto-detect input_dim)
    print("\n🔍 Step 3: Preparing data and auto-detecting features...")
    datamodule.prepare_data()
    
    # Get auto-detected input_dim
    input_dim = datamodule.get_input_dim()
    print(f"✅ Auto-detected input_dim: {input_dim} features")
    
    # Step 4: Get model configuration
    print("\n🤖 Step 4: Getting model configuration...")
    model_config = datamodule.get_model_config()
    print("Model config:")
    for key, value in model_config.items():
        print(f"  {key}: {value}")
    
    # Step 5: Initialize model with auto-detected parameters
    print(f"\n🏗️ Step 5: Building model...")
    model_config = datamodule.get_model_config()
    
    model = FinGAT(
        input_dim=model_config['input_dim'],
        num_stocks=model_config['num_stocks'],
        num_sectors=model_config['num_sectors'],
        config_manager=config_manager
    )
    
    # Print model configuration
    model.print_config()
    
    # Model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n📊 Model Parameters:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Step 6: Get dataloaders
    print("\n📦 Step 6: Creating dataloaders...")
    train_loader, val_loader, test_loader = datamodule.get_dataloaders()
    
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # Step 7: Test forward pass with real data
    print("\n🧪 Step 7: Testing forward pass with real data...")
    model.eval()
    
    # Get one batch from train_loader
    sample_batch = next(iter(train_loader))
    features = sample_batch['features']
    edge_index_intra = sample_batch['edge_index_intra']
    edge_index_inter = sample_batch['edge_index_inter']
    stock_to_sector = sample_batch['stock_to_sector']
    
    print(f"  Input shapes:")
    print(f"    Features: {features.shape}")
    print(f"    Edge intra: {edge_index_intra.shape}")
    print(f"    Edge inter: {edge_index_inter.shape}")
    print(f"    Stock-to-sector: {stock_to_sector.shape}")
    
    # Forward pass
    with torch.no_grad():
        ranking_pred, movement_pred = model(
            features, edge_index_intra, edge_index_inter, stock_to_sector
        )
    
    print(f"  Output shapes:")
    print(f"    Ranking predictions: {ranking_pred.shape}")
    print(f"    Movement predictions: {movement_pred.shape}")
    
    # Step 8: Setup training components
    print("\n⚙️ Step 8: Setting up training components...")
    training_config = config_manager.get_training_config()
    
    # Loss function components
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCELoss()
    
    # Optimizer
    optimizer = optim.Adam(
        model.parameters(), 
        lr=training_config['learning_rate'],
        weight_decay=training_config['weight_decay']
    )
    
    print(f"  Optimizer: {training_config['optimizer']}")
    print(f"  Learning rate: {training_config['learning_rate']}")
    print(f"  Weight decay: {training_config['weight_decay']}")
    print(f"  Ranking weight: {training_config['ranking_weight']}")
    print(f"  Movement weight: {training_config['movement_weight']}")
    
    # Step 9: Training loop demo (1 batch)
    print("\n🏃 Step 9: Demo training step...")
    model.train()
    
    # Get one batch
    batch = next(iter(train_loader))
    features = batch['features']
    returns = batch['returns']
    movements = batch['movements']
    edge_index_intra = batch['edge_index_intra']
    edge_index_inter = batch['edge_index_inter']
    stock_to_sector = batch['stock_to_sector']
    
    # Forward pass
    ranking_pred, movement_pred = model(
        features, edge_index_intra, edge_index_inter, stock_to_sector
    )
    
    # Calculate losses
    ranking_loss = mse_loss(ranking_pred, returns)
    movement_loss = bce_loss(movement_pred, movements.float())
    
    # Combined loss
    total_loss = (training_config['ranking_weight'] * ranking_loss + 
                  training_config['movement_weight'] * movement_loss)
    
    # Backward pass
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    print(f"  Ranking loss: {ranking_loss.item():.4f}")
    print(f"  Movement loss: {movement_loss.item():.4f}")
    print(f"  Total loss: {total_loss.item():.4f}")
    
    print("\n🎉 Auto-config training setup completed successfully!")
    print("\n📝 Summary:")
    print(f"  ✅ Auto-detected {input_dim} input features")
    print(f"  ✅ Configured {len(datamodule.graph_constructor.stocks)} stocks")
    print(f"  ✅ Configured {len(datamodule.graph_constructor.sectors)} sectors")
    print(f"  ✅ Model initialized with {total_params:,} parameters")
    print(f"  ✅ Training components ready")
    
    print("\n💡 Next steps:")
    print("  - Run full training loop")
    print("  - Add validation and early stopping")
    print("  - Implement evaluation metrics")
    print("  - Add checkpointing and logging")


if __name__ == "__main__":
    main()
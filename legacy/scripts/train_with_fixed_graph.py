"""
Training script for FinGAT with fixed graph construction
Uses proper edge indices and stock-to-sector mapping
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import argparse
import json
from tqdm import tqdm

sys.path.append('..')

from src.data.fingat_datamodule import FinGATDataModule
from src.models.fingat import FinGAT
from src.training.losses import FinGATLoss
from src.evaluation.metrics import calculate_mrr_at_k, calculate_precision_at_k


class FinGATTrainer:
    """
    Trainer for FinGAT model with fixed graph construction
    """
    
    def __init__(self,
                 model: nn.Module,
                 datamodule: FinGATDataModule,
                 experiment_name: str = None,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize trainer
        
        Args:
            model: FinGAT model
            datamodule: Data module with graph construction
            experiment_name: Name for experiment
            device: Device to use
        """
        self.model = model.to(device)
        self.datamodule = datamodule
        self.device = device
        
        # Set experiment name
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_name = f"fingat_fixed_{timestamp}"
        else:
            self.experiment_name = experiment_name
        
        # Create directories
        self.exp_dir = Path(f'../experiments/{self.experiment_name}')
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        (self.exp_dir / 'checkpoints').mkdir(exist_ok=True)
        (self.exp_dir / 'metrics').mkdir(exist_ok=True)
        
        # Initialize loss and optimizer
        self.criterion = FinGATLoss(
            ranking_weight=0.99,
            movement_weight=0.01,
            l2_lambda=1e-4
        )
        
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Metrics storage
        self.train_history = []
        self.val_history = []
        self.best_val_loss = float('inf')
        
        print(f"✅ Trainer initialized")
        print(f"  Experiment: {self.experiment_name}")
        print(f"  Device: {device}")
        print(f"  Output dir: {self.exp_dir}")
    
    def train_epoch(self, train_loader):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        ranking_losses = []
        movement_losses = []
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Get batch data
            features = batch['features']
            target_returns = batch['returns']
            target_movements = batch['movements']
            edge_index_intra = batch['edge_index_intra']
            edge_index_inter = batch['edge_index_inter']
            stock_to_sector = batch['stock_to_sector']
            
            # Forward pass
            pred_returns, pred_movements = self.model(
                features,
                edge_index_intra,
                edge_index_inter,
                stock_to_sector
            )
            
            # Calculate loss
            loss, ranking_loss, movement_loss = self.criterion(
                pred_returns.squeeze(),
                pred_movements.squeeze(),
                target_returns,
                target_movements,
                self.model.parameters()
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Track losses
            total_loss += loss.item()
            ranking_losses.append(ranking_loss.item())
            movement_losses.append(movement_loss.item())
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item(),
                'rank_loss': ranking_loss.item(),
                'move_loss': movement_loss.item()
            })
        
        return {
            'total_loss': total_loss / len(train_loader),
            'ranking_loss': np.mean(ranking_losses),
            'movement_loss': np.mean(movement_losses)
        }
    
    def validate(self, val_loader):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        all_pred_returns = []
        all_true_returns = []
        all_pred_movements = []
        all_true_movements = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Get batch data
                features = batch['features']
                target_returns = batch['returns']
                target_movements = batch['movements']
                edge_index_intra = batch['edge_index_intra']
                edge_index_inter = batch['edge_index_inter']
                stock_to_sector = batch['stock_to_sector']
                
                # Forward pass
                pred_returns, pred_movements = self.model(
                    features,
                    edge_index_intra,
                    edge_index_inter,
                    stock_to_sector
                )
                
                # Calculate loss
                loss, _, _ = self.criterion(
                    pred_returns.squeeze(),
                    pred_movements.squeeze(),
                    target_returns,
                    target_movements,
                    self.model.parameters()
                )
                
                total_loss += loss.item()
                
                # Store predictions
                all_pred_returns.extend(pred_returns.squeeze().cpu().numpy())
                all_true_returns.extend(target_returns.cpu().numpy())
                all_pred_movements.extend(pred_movements.squeeze().cpu().numpy())
                all_true_movements.extend(target_movements.cpu().numpy())
        
        # Calculate metrics
        val_loss = total_loss / len(val_loader)
        
        # Calculate ranking metrics (sample for efficiency)
        sample_size = min(1000, len(all_pred_returns))
        sample_indices = np.random.choice(len(all_pred_returns), sample_size, replace=False)
        
        mrr_5 = calculate_mrr_at_k(
            np.array(all_pred_returns)[sample_indices],
            np.array(all_true_returns)[sample_indices],
            k=5
        )
        
        precision_5 = calculate_precision_at_k(
            np.array(all_pred_returns)[sample_indices],
            np.array(all_true_returns)[sample_indices],
            k=5
        )
        
        # Movement accuracy
        movement_acc = np.mean(
            (np.array(all_pred_movements) >= 0.5) == np.array(all_true_movements)
        )
        
        return {
            'val_loss': val_loss,
            'mrr@5': mrr_5,
            'precision@5': precision_5,
            'movement_accuracy': movement_acc
        }
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_config': self.datamodule.get_model_config(),
            'val_loss': self.val_history[-1]['val_loss'] if self.val_history else None,
            'train_history': self.train_history,
            'val_history': self.val_history
        }
        
        # Save epoch checkpoint
        checkpoint_path = self.exp_dir / 'checkpoints' / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.exp_dir / 'checkpoints' / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"  💾 Saved best model (val_loss: {self.val_history[-1]['val_loss']:.4f})")
    
    def save_metrics(self):
        """Save training metrics to CSV"""
        # Combine train and val metrics
        metrics_df = pd.DataFrame()
        
        for i, (train_metrics, val_metrics) in enumerate(zip(self.train_history, self.val_history)):
            row = {
                'epoch': i + 1,
                'train_loss': train_metrics['total_loss'],
                'train_ranking_loss': train_metrics['ranking_loss'],
                'train_movement_loss': train_metrics['movement_loss'],
                'val_loss': val_metrics['val_loss'],
                'val_mrr@5': val_metrics['mrr@5'],
                'val_precision@5': val_metrics['precision@5'],
                'val_movement_accuracy': val_metrics['movement_accuracy'],
                'learning_rate': self.optimizer.param_groups[0]['lr']
            }
            metrics_df = pd.concat([metrics_df, pd.DataFrame([row])], ignore_index=True)
        
        # Save to CSV
        metrics_path = self.exp_dir / 'metrics' / 'training_metrics.csv'
        metrics_df.to_csv(metrics_path, index=False)
    
    def train(self, epochs: int = 100):
        """
        Train the model
        
        Args:
            epochs: Number of epochs to train
        """
        print(f"\n🚀 Starting training for {epochs} epochs...")
        
        # Get dataloaders
        train_loader, val_loader, test_loader = self.datamodule.get_dataloaders()
        
        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")
            print("-" * 60)
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            self.train_history.append(train_metrics)
            
            print(f"  Train Loss: {train_metrics['total_loss']:.4f}")
            print(f"    - Ranking: {train_metrics['ranking_loss']:.4f}")
            print(f"    - Movement: {train_metrics['movement_loss']:.4f}")
            
            # Validate
            val_metrics = self.validate(val_loader)
            self.val_history.append(val_metrics)
            
            print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
            print(f"    - MRR@5: {val_metrics['mrr@5']:.4f}")
            print(f"    - Precision@5: {val_metrics['precision@5']:.4f}")
            print(f"    - Movement Acc: {val_metrics['movement_accuracy']:.4f}")
            
            # Learning rate scheduling
            self.scheduler.step(val_metrics['val_loss'])
            
            # Save checkpoint
            is_best = val_metrics['val_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['val_loss']
            
            self.save_checkpoint(epoch, is_best)
            
            # Save metrics every epoch
            self.save_metrics()
        
        print("\n✅ Training completed!")
        print(f"  Best val loss: {self.best_val_loss:.4f}")
        print(f"  Checkpoints saved to: {self.exp_dir / 'checkpoints'}")
        print(f"  Metrics saved to: {self.exp_dir / 'metrics'}")


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train FinGAT with fixed graph')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--num-stocks', type=int, default=None, help='Number of stocks (None for all)')
    parser.add_argument('--experiment-name', type=str, default=None, help='Experiment name')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    print("="*70)
    print("FINGAT TRAINING WITH FIXED GRAPH CONSTRUCTION")
    print("="*70)
    
    # Initialize data module
    print("\n📊 Initializing data module...")
    datamodule = FinGATDataModule(
        data_path='./datasets/VN_datasets/',
        companies_file='./datasets/VN_Companies.csv',
        batch_size=args.batch_size,
        num_stocks_limit=args.num_stocks,
        device=args.device
    )
    
    # Prepare data
    datamodule.prepare_data()
    
    # Initialize model
    print("\n🤖 Initializing model...")
    model_config = datamodule.get_model_config()
    model = FinGAT(**model_config)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total_params:,}")
    
    # Initialize trainer
    trainer = FinGATTrainer(
        model=model,
        datamodule=datamodule,
        experiment_name=args.experiment_name,
        device=args.device
    )
    
    # Train model
    trainer.train(epochs=args.epochs)
    
    print("\n🎉 Training pipeline completed successfully!")


if __name__ == "__main__":
    main()
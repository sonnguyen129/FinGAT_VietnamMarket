"""
Trainer module for FinGAT model
Handles training loop, validation, checkpointing, and metric tracking
"""

import os
import csv
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from datetime import datetime
from pathlib import Path
import yaml
from tqdm import tqdm

from ..models.fingat import FinGAT
from .losses import FinGATLoss
from ..evaluation.metrics import calculate_metrics


class FinGATTrainer:
    """
    Trainer class for FinGAT model with comprehensive logging and checkpointing
    """
    
    def __init__(self,
                 model: FinGAT,
                 config: Dict[str, Any],
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 test_loader: Optional[DataLoader] = None,
                 experiment_name: Optional[str] = None):
        """
        Initialize trainer
        
        Args:
            model: FinGAT model instance
            config: Configuration dictionary
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Optional test data loader
            experiment_name: Name for this experiment
        """
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        # Setup experiment
        self.experiment_name = experiment_name or config['experiment']['name']
        self.setup_experiment_dir()
        
        # Device
        self.device = torch.device(config['system']['device'])
        self.model = self.model.to(self.device)
        
        # Loss function
        self.criterion = FinGATLoss(
            ranking_weight=config['training']['ranking_weight'],
            movement_weight=config['training']['movement_weight'],
            l2_lambda=config['training']['l2_lambda']
        )
        
        # Optimizer
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_metric = -float('inf')
        self.patience_counter = 0
        
        # Metrics tracking
        self.train_metrics_history = []
        self.val_metrics_history = []
        self.metrics_file = None
        self.setup_metrics_logging()
        
    def setup_experiment_dir(self):
        """Setup directories for experiment"""
        # Create timestamp for unique experiment
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.experiment_dir = Path(f"experiments/{self.experiment_name}_{timestamp}")
        
        # Create subdirectories
        self.checkpoint_dir = self.experiment_dir / "checkpoints"
        self.metrics_dir = self.experiment_dir / "metrics"
        self.logs_dir = self.experiment_dir / "logs"
        
        for dir_path in [self.checkpoint_dir, self.metrics_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Save config
        config_path = self.experiment_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        print(f"Experiment directory created: {self.experiment_dir}")
    
    def setup_metrics_logging(self):
        """Setup CSV file for metrics logging"""
        metrics_file_path = self.metrics_dir / "training_metrics.csv"
        self.metrics_file = open(metrics_file_path, 'w', newline='')
        
        # Define fields
        self.metric_fields = [
            'epoch', 'step', 'phase',  # Basic info
            'loss_total', 'loss_ranking', 'loss_movement', 'loss_l2',  # Losses
            'MRR@5', 'MRR@10', 'MRR@20',  # MRR metrics
            'Precision@5', 'Precision@10', 'Precision@20',  # Precision
            'movement_accuracy',  # Movement prediction accuracy
            'learning_rate', 'time_elapsed'  # Training info
        ]
        
        self.metrics_writer = csv.DictWriter(
            self.metrics_file, 
            fieldnames=self.metric_fields
        )
        self.metrics_writer.writeheader()
        self.metrics_file.flush()
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on config"""
        opt_config = self.config['training']
        
        if opt_config['optimizer'].lower() == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=opt_config['learning_rate'],
                weight_decay=opt_config['weight_decay']
            )
        elif opt_config['optimizer'].lower() == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=opt_config['learning_rate'],
                momentum=0.9,
                weight_decay=opt_config['weight_decay']
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_config['optimizer']}")
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler"""
        if not self.config['training'].get('scheduler'):
            return None
        
        scheduler_type = self.config['training']['scheduler']
        
        if scheduler_type == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config['training']['scheduler_step_size'],
                gamma=self.config['training']['scheduler_gamma']
            )
        elif scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['epochs']
            )
        else:
            return None
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch
        
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        epoch_metrics = {
            'loss_total': 0,
            'loss_ranking': 0,
            'loss_movement': 0,
            'loss_l2': 0
        }
        
        num_batches = len(self.train_loader)
        
        with tqdm(total=num_batches, desc=f"Epoch {self.current_epoch+1} Training") as pbar:
            for batch_idx, batch in enumerate(self.train_loader):
                # Move batch to device
                features = batch['features'].to(self.device)
                target_returns = batch['returns'].to(self.device)
                target_movements = batch['movements'].to(self.device)
                intra_edges = batch['intra_edges'].to(self.device)
                inter_edges = batch['inter_edges'].to(self.device)
                stock_to_sector = batch['stock_to_sector'].to(self.device)
                
                # Convert features to weekly format
                weekly_data = self._prepare_weekly_data(features)
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                pred_returns, pred_movements = self.model(
                    weekly_data, intra_edges, inter_edges, stock_to_sector
                )
                
                # Reshape targets to match predictions
                target_returns = target_returns.view(-1)
                target_movements = target_movements.view(-1)
                
                # Calculate loss
                loss, loss_dict = self.criterion(
                    pred_returns, pred_movements,
                    target_returns, target_movements,
                    self.model.parameters()
                )
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.config['training'].get('gradient_clip'):
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['gradient_clip']
                    )
                
                # Optimizer step
                self.optimizer.step()
                
                # Update metrics
                for key in ['loss_total', 'loss_ranking', 'loss_movement', 'loss_l2']:
                    epoch_metrics[key] += loss_dict[key]
                
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
                })
                
                self.global_step += 1
                
                # Log metrics every n steps
                if self.global_step % self.config['training']['log_every_n_steps'] == 0:
                    self._log_step_metrics(loss_dict, 'train')
        
        # Average epoch metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        return epoch_metrics
    
    def validate(self) -> Dict[str, float]:
        """
        Validate model on validation set
        
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        
        all_pred_returns = []
        all_pred_movements = []
        all_target_returns = []
        all_target_movements = []
        
        val_loss = 0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            with tqdm(total=num_batches, desc="Validation") as pbar:
                for batch in self.val_loader:
                    # Move batch to device
                    features = batch['features'].to(self.device)
                    target_returns = batch['returns'].to(self.device)
                    target_movements = batch['movements'].to(self.device)
                    intra_edges = batch['intra_edges'].to(self.device)
                    inter_edges = batch['inter_edges'].to(self.device)
                    stock_to_sector = batch['stock_to_sector'].to(self.device)
                    
                    # Convert to weekly format
                    weekly_data = self._prepare_weekly_data(features)
                    
                    # Forward pass
                    pred_returns, pred_movements = self.model(
                        weekly_data, intra_edges, inter_edges, stock_to_sector
                    )
                    
                    # Reshape targets
                    target_returns = target_returns.view(-1)
                    target_movements = target_movements.view(-1)
                    
                    # Calculate loss
                    loss, _ = self.criterion(
                        pred_returns, pred_movements,
                        target_returns, target_movements
                    )
                    
                    val_loss += loss.item()
                    
                    # Collect predictions
                    all_pred_returns.append(pred_returns.cpu())
                    all_pred_movements.append(pred_movements.cpu())
                    all_target_returns.append(target_returns.cpu())
                    all_target_movements.append(target_movements.cpu())
                    
                    pbar.update(1)
        
        # Concatenate all predictions
        all_pred_returns = torch.cat(all_pred_returns)
        all_pred_movements = torch.cat(all_pred_movements)
        all_target_returns = torch.cat(all_target_returns)
        all_target_movements = torch.cat(all_target_movements)
        
        # Calculate metrics
        val_metrics = calculate_metrics(
            all_pred_returns.numpy(),
            all_pred_movements.numpy(),
            all_target_returns.numpy(),
            all_target_movements.numpy(),
            k_values=self.config['evaluation']['top_k']
        )
        
        val_metrics['loss_total'] = val_loss / num_batches
        
        return val_metrics
    
    def _prepare_weekly_data(self, features: torch.Tensor) -> List[torch.Tensor]:
        """
        Convert features to weekly format for model
        
        Args:
            features: Features tensor (batch, stocks, days, features)
            
        Returns:
            List of weekly tensors
        """
        batch_size, num_stocks, num_days, num_features = features.shape
        days_per_week = 5
        num_weeks = num_days // days_per_week
        
        weekly_data = []
        for week_idx in range(num_weeks):
            start_day = week_idx * days_per_week
            end_day = min(start_day + days_per_week, num_days)
            week_features = features[:, :, start_day:end_day, :]
            weekly_data.append(week_features)
        
        return weekly_data
    
    def save_checkpoint(self, is_best: bool = False):
        """
        Save model checkpoint
        
        Args:
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_metric': self.best_val_metric,
            'config': self.config
        }
        
        # Save epoch checkpoint
        epoch_path = self.checkpoint_dir / f"checkpoint_epoch_{self.current_epoch}.pth"
        torch.save(checkpoint, epoch_path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"  💾 Saved best model with val_MRR@10: {self.best_val_metric:.4f}")
        
        # Save latest checkpoint
        latest_path = self.checkpoint_dir / "latest_checkpoint.pth"
        torch.save(checkpoint, latest_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_metric = checkpoint['best_val_metric']
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def _log_step_metrics(self, metrics: Dict, phase: str):
        """Log metrics for a training step"""
        log_entry = {
            'epoch': self.current_epoch,
            'step': self.global_step,
            'phase': phase,
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'time_elapsed': time.time() - self.training_start_time
        }
        log_entry.update(metrics)
        
        self.metrics_writer.writerow(log_entry)
        self.metrics_file.flush()
    
    def _log_epoch_metrics(self, train_metrics: Dict, val_metrics: Dict):
        """Log metrics for an epoch"""
        log_entry = {
            'epoch': self.current_epoch,
            'step': self.global_step,
            'phase': 'epoch_end',
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'time_elapsed': time.time() - self.training_start_time
        }
        
        # Add train metrics with prefix
        for key, value in train_metrics.items():
            log_entry[f'train_{key}'] = value
        
        # Add val metrics with prefix
        for key, value in val_metrics.items():
            log_entry[f'val_{key}'] = value
        
        self.metrics_writer.writerow(log_entry)
        self.metrics_file.flush()
        
        # Store in history
        self.train_metrics_history.append(train_metrics)
        self.val_metrics_history.append(val_metrics)
    
    def train(self):
        """Main training loop"""
        self.training_start_time = time.time()
        
        print("="*60)
        print(f"Starting training: {self.experiment_name}")
        print(f"Device: {self.device}")
        print(f"Total epochs: {self.config['training']['epochs']}")
        print(f"Batch size: {self.config['training']['batch_size']}")
        print("="*60)
        
        for epoch in range(self.config['training']['epochs']):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Log epoch metrics
            self._log_epoch_metrics(train_metrics, val_metrics)
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{self.config['training']['epochs']} Summary:")
            print(f"  Train Loss: {train_metrics['loss_total']:.4f}")
            print(f"  Val Loss: {val_metrics['loss_total']:.4f}")
            print(f"  Val MRR@10: {val_metrics.get('MRR@10', 0):.4f}")
            print(f"  Val Movement Acc: {val_metrics.get('movement_accuracy', 0):.4f}")
            
            # Check for best model
            val_metric = val_metrics.get('MRR@10', 0)
            is_best = val_metric > self.best_val_metric
            if is_best:
                self.best_val_metric = val_metric
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Save checkpoint
            if self.config['training']['save_every_epoch']:
                self.save_checkpoint(is_best=is_best)
            
            # Learning rate scheduling
            if self.scheduler:
                self.scheduler.step()
            
            # Early stopping
            if (self.config['training']['early_stopping'] and 
                self.patience_counter >= self.config['training']['patience']):
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
        
        # Training complete
        training_time = time.time() - self.training_start_time
        print("\n" + "="*60)
        print("Training Complete!")
        print(f"Total time: {training_time/3600:.2f} hours")
        print(f"Best val MRR@10: {self.best_val_metric:.4f}")
        print(f"Experiment saved to: {self.experiment_dir}")
        print("="*60)
        
        # Close metrics file
        if self.metrics_file:
            self.metrics_file.close()
    
    def test(self):
        """Test model on test set"""
        if not self.test_loader:
            print("No test loader provided")
            return
        
        print("\nTesting best model...")
        
        # Load best model
        best_model_path = self.checkpoint_dir / "best_model.pth"
        if best_model_path.exists():
            self.load_checkpoint(str(best_model_path))
        
        # Run validation on test set
        self.model.eval()
        test_metrics = self.validate()  # Use same validation function
        
        print("\nTest Results:")
        for key, value in test_metrics.items():
            if key != 'loss_total':
                print(f"  {key}: {value:.4f}")
        
        # Save test results
        test_results_path = self.metrics_dir / "test_results.yaml"
        with open(test_results_path, 'w') as f:
            yaml.dump(test_metrics, f)
        
        return test_metrics
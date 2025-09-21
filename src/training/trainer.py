"""
Training Module for FinGAT Vietnam Project

This module handles the training process, including loss functions,
optimization, and training loop implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch_geometric.data import DataLoader as GraphDataLoader
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
from pathlib import Path
import time
import json
from tqdm import tqdm

from src.utils.logging_utils import get_logger
from src.utils.seed_utils import set_seed

logger = get_logger(__name__)


class PairwiseRankingLoss(nn.Module):
    """
    Pairwise ranking loss for stock ranking task.

    This loss encourages the model to rank stocks with higher returns
    above stocks with lower returns.
    """

    def __init__(self, margin: float = 1.0):
        """
        Initialize pairwise ranking loss.

        Args:
            margin: Margin for ranking loss
        """
        super(PairwiseRankingLoss, self).__init__()
        self.margin = margin

    def forward(self, scores: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise ranking loss.

        Args:
            scores: Predicted ranking scores (batch_size,)
            returns: True returns for ranking (batch_size,)

        Returns:
            Ranking loss value
        """
        batch_size = scores.size(0)

        # Create all pairwise comparisons
        scores_i = scores.unsqueeze(1)  # (batch_size, 1)
        scores_j = scores.unsqueeze(0)  # (1, batch_size)

        returns_i = returns.unsqueeze(1)  # (batch_size, 1)
        returns_j = returns.unsqueeze(0)  # (1, batch_size)

        # Compute score differences and return differences
        score_diff = scores_i - scores_j  # (batch_size, batch_size)
        return_diff = returns_i - returns_j  # (batch_size, batch_size)

        # Only consider pairs where returns are significantly different
        mask = (return_diff > 0).float()

        # Ranking loss: max(0, margin - score_diff) when return_i > return_j
        loss = F.relu(self.margin - score_diff) * mask

        # Average over valid pairs
        num_pairs = mask.sum()
        if num_pairs > 0:
            loss = loss.sum() / num_pairs
        else:
            loss = torch.tensor(0.0, device=scores.device, requires_grad=True)

        return loss


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss combining movement classification and ranking.
    """

    def __init__(self, movement_weight: float = 0.5, ranking_weight: float = 0.5, 
                 ranking_margin: float = 1.0):
        """
        Initialize multi-task loss.

        Args:
            movement_weight: Weight for movement classification loss
            ranking_weight: Weight for ranking loss
            ranking_margin: Margin for ranking loss
        """
        super(MultiTaskLoss, self).__init__()

        self.movement_weight = movement_weight
        self.ranking_weight = ranking_weight

        self.movement_loss = nn.CrossEntropyLoss()
        self.ranking_loss = PairwiseRankingLoss(margin=ranking_margin)

    def forward(self, outputs: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss.

        Args:
            outputs: Model outputs containing movement_logits and ranking_scores
            targets: Targets containing movement_labels and returns

        Returns:
            Dictionary containing individual and total losses
        """
        # Movement classification loss
        movement_loss = self.movement_loss(
            outputs['movement_logits'], 
            targets['movement_labels']
        )

        # Ranking loss
        ranking_loss = self.ranking_loss(
            outputs['ranking_scores'], 
            targets['returns']
        )

        # Total loss
        total_loss = (self.movement_weight * movement_loss + 
                     self.ranking_weight * ranking_loss)

        return {
            'total_loss': total_loss,
            'movement_loss': movement_loss,
            'ranking_loss': ranking_loss
        }


class FinGATTrainer:
    """
    Trainer class for FinGAT model.

    Handles the complete training process including optimization,
    validation, checkpointing, and logging.
    """

    def __init__(self, model: nn.Module, config: Dict, 
                 experiment_logger: Optional[ExperimentLogger] = None):
        """
        Initialize trainer.

        Args:
            model: FinGAT model to train
            config: Configuration dictionary
            experiment_logger: Logger for experiment tracking
        """
        self.model = model
        self.config = config
        self.training_config = config.get('training', {})
        self.device_config = config.get('device', {})

        # Setup device
        self.device = self._setup_device()
        self.model.to(self.device)

        # Setup loss function
        loss_config = self.training_config.get('loss', {})
        self.criterion = MultiTaskLoss(
            movement_weight=loss_config.get('movement_weight', 0.5),
            ranking_weight=loss_config.get('ranking_weight', 0.5),
            ranking_margin=loss_config.get('margin', 1.0)
        )

        # Setup optimizer
        self.optimizer = self._setup_optimizer()

        # Setup scheduler
        self.scheduler = self._setup_scheduler()

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []

        # Experiment tracking
        self.exp_logger = experiment_logger or ExperimentLogger("FinGAT_Trainer")

        # Paths
        self.model_save_path = Path(config.get('paths', {}).get('models', 'outputs/models'))
        self.model_save_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"FinGATTrainer initialized on device: {self.device}")

    def _setup_device(self) -> torch.device:
        """Setup training device (CPU/GPU)."""
        if self.device_config.get('use_cuda', True) and torch.cuda.is_available():
            device = torch.device(f"cuda:{self.device_config.get('gpu_id', 0)}")
        else:
            device = torch.device('cpu')

        logger.info(f"Using device: {device}")
        return device

    def _setup_optimizer(self):
        """Setup optimizer."""
        optimizer_name = self.training_config.get('optimizer', 'Adam')
        lr = self.training_config.get('learning_rate', 0.001)
        weight_decay = self.training_config.get('weight_decay', 0.0001)

        if optimizer_name.lower() == 'adam':
            optimizer = Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name.lower() == 'adamw':
            optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        logger.info(f"Optimizer: {optimizer_name}, LR: {lr}, Weight Decay: {weight_decay}")
        return optimizer

    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        scheduler_config = self.training_config.get('scheduler', {})

        if not scheduler_config:
            return None

        scheduler_name = scheduler_config.get('name', 'ReduceLROnPlateau')

        if scheduler_name == 'ReduceLROnPlateau':
            return ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=scheduler_config.get('factor', 0.5),
                patience=scheduler_config.get('patience', 10),
                verbose=True
            )
        elif scheduler_name == 'CosineAnnealingLR':
            return CosineAnnealingLR(
                self.optimizer,
                T_max=scheduler_config.get('T_max', 50),
                eta_min=scheduler_config.get('eta_min', 1e-6)
            )

        return None

    def train_epoch(self, train_loader) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader

        Returns:
            Dictionary containing training metrics
        """
        self.model.train()

        total_loss = 0.0
        total_movement_loss = 0.0
        total_ranking_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")

        for batch_idx, batch in enumerate(progress_bar):
            try:
                # Move batch to device
                temporal_data = batch['temporal_data'].to(self.device)
                intra_graph = batch['intra_graph'].to(self.device)
                inter_graph = batch['inter_graph'].to(self.device)

                targets = {
                    'movement_labels': batch['movement_labels'].to(self.device),
                    'returns': batch['returns'].to(self.device)
                }

                # Forward pass
                outputs = self.model(temporal_data, intra_graph, inter_graph)

                # Compute loss
                losses = self.criterion(outputs, targets)

                # Backward pass
                self.optimizer.zero_grad()
                losses['total_loss'].backward()

                # Gradient clipping
                if self.training_config.get('grad_clip', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.training_config['grad_clip']
                    )

                self.optimizer.step()

                # Update metrics
                total_loss += losses['total_loss'].item()
                total_movement_loss += losses['movement_loss'].item()
                total_ranking_loss += losses['ranking_loss'].item()
                num_batches += 1

                # Update progress bar
                progress_bar.set_postfix({
                    'Loss': f"{losses['total_loss'].item():.4f}",
                    'MoveLoss': f"{losses['movement_loss'].item():.4f}",
                    'RankLoss': f"{losses['ranking_loss'].item():.4f}"
                })

            except Exception as e:
                logger.error(f"Error in training batch {batch_idx}: {e}")
                continue

        # Calculate average metrics
        metrics = {
            'train_loss': total_loss / max(num_batches, 1),
            'train_movement_loss': total_movement_loss / max(num_batches, 1),
            'train_ranking_loss': total_ranking_loss / max(num_batches, 1)
        }

        return metrics

    def validate_epoch(self, val_loader) -> Dict[str, float]:
        """
        Validate for one epoch.

        Args:
            val_loader: Validation data loader

        Returns:
            Dictionary containing validation metrics
        """
        self.model.eval()

        total_loss = 0.0
        total_movement_loss = 0.0
        total_ranking_loss = 0.0
        num_batches = 0

        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                try:
                    # Move batch to device
                    temporal_data = batch['temporal_data'].to(self.device)
                    intra_graph = batch['intra_graph'].to(self.device)
                    inter_graph = batch['inter_graph'].to(self.device)

                    targets = {
                        'movement_labels': batch['movement_labels'].to(self.device),
                        'returns': batch['returns'].to(self.device)
                    }

                    # Forward pass
                    outputs = self.model(temporal_data, intra_graph, inter_graph)

                    # Compute loss
                    losses = self.criterion(outputs, targets)

                    # Update metrics
                    total_loss += losses['total_loss'].item()
                    total_movement_loss += losses['movement_loss'].item()
                    total_ranking_loss += losses['ranking_loss'].item()
                    num_batches += 1

                    # Collect predictions for accuracy calculation
                    predictions = torch.argmax(outputs['movement_probs'], dim=1)
                    all_predictions.extend(predictions.cpu().numpy())
                    all_labels.extend(targets['movement_labels'].cpu().numpy())

                except Exception as e:
                    logger.error(f"Error in validation batch: {e}")
                    continue

        # Calculate metrics
        accuracy = np.mean(np.array(all_predictions) == np.array(all_labels)) if all_labels else 0.0

        metrics = {
            'val_loss': total_loss / max(num_batches, 1),
            'val_movement_loss': total_movement_loss / max(num_batches, 1),
            'val_ranking_loss': total_ranking_loss / max(num_batches, 1),
            'val_accuracy': accuracy
        }

        return metrics

    def train(self, train_loader, val_loader, num_epochs: Optional[int] = None) -> Dict:
        """
        Full training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs (uses config if not provided)

        Returns:
            Training history dictionary
        """
        if num_epochs is None:
            num_epochs = self.training_config.get('num_epochs', 100)

        early_stopping_config = self.training_config.get('early_stopping', {})
        patience = early_stopping_config.get('patience', 15)
        min_delta = early_stopping_config.get('min_delta', 0.001)

        best_val_loss = float('inf')
        patience_counter = 0

        training_history = {
            'train_losses': [],
            'val_losses': [], 
            'val_accuracies': [],
            'learning_rates': []
        }

        logger.info(f"Starting training for {num_epochs} epochs")

        start_time = time.time()

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Training
            train_metrics = self.train_epoch(train_loader)

            # Validation
            val_metrics = self.validate_epoch(val_loader)

            # Update learning rate scheduler
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['val_loss'])
                else:
                    self.scheduler.step()

            # Log metrics
            current_lr = self.optimizer.param_groups[0]['lr']

            epoch_metrics = {**train_metrics, **val_metrics, 'learning_rate': current_lr}
            self.exp_logger.log_metrics(epoch_metrics, step=epoch)

            # Update history
            training_history['train_losses'].append(train_metrics['train_loss'])
            training_history['val_losses'].append(val_metrics['val_loss'])
            training_history['val_accuracies'].append(val_metrics['val_accuracy'])
            training_history['learning_rates'].append(current_lr)

            # Early stopping check
            if val_metrics['val_loss'] < best_val_loss - min_delta:
                best_val_loss = val_metrics['val_loss']
                patience_counter = 0

                # Save best model
                self.save_checkpoint(epoch, val_metrics['val_loss'], is_best=True)

            else:
                patience_counter += 1

            # Regular checkpoint saving
            if (epoch + 1) % self.training_config.get('save_interval', 25) == 0:
                self.save_checkpoint(epoch, val_metrics['val_loss'])

            # Log progress
            logger.info(
                f"Epoch {epoch}: "
                f"Train Loss: {train_metrics['train_loss']:.4f}, "
                f"Val Loss: {val_metrics['val_loss']:.4f}, "
                f"Val Acc: {val_metrics['val_accuracy']:.4f}, "
                f"LR: {current_lr:.6f}"
            )

            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {patience} epochs without improvement")
                break

        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")

        return training_history

    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """
        Save model checkpoint.

        Args:
            epoch: Current epoch
            val_loss: Validation loss
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'config': self.config
        }

        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        # Save regular checkpoint
        checkpoint_path = self.model_save_path / f"checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)

        # Save best model
        if is_best:
            best_path = self.model_save_path / "best_model.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"Best model saved at epoch {epoch} with val_loss: {val_loss:.4f}")

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'scheduler_state_dict' in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['val_loss']

        logger.info(f"Checkpoint loaded from epoch {self.current_epoch}")


# Example usage
if __name__ == "__main__":
    # This would be used in a training script
    print("FinGAT Trainer module loaded successfully")
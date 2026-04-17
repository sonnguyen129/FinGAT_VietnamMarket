"""
Loss functions for FinGAT model
Implements multi-task loss as described in the paper:
L = (1-δ)L_rank + δL_move + λ||Θ||²
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class PairwiseRankingLoss(nn.Module):
    """
    Pairwise ranking loss for return prediction
    From paper: L_rank = max(0, -Δ̂ · Δ) where:
    - Δ = y_return(s) - y_return(k) (ground truth difference)
    - Δ̂ = ŷ_return(s) - ŷ_return(k) (predicted difference)
    """
    
    def __init__(self, margin: float = 0.0, reduction: str = 'mean'):
        """
        Args:
            margin: Margin for ranking loss (default 0 as per paper)
            reduction: How to reduce the loss ('none', 'mean', 'sum')
        """
        super().__init__()
        self.margin = margin
        self.reduction = reduction
    
    def forward(self, 
                predictions: torch.Tensor,
                targets: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute pairwise ranking loss
        
        Args:
            predictions: Predicted returns (batch_size,) or (batch_size, num_stocks)
            targets: Ground truth returns (batch_size,) or (batch_size, num_stocks)
            mask: Optional mask for valid pairs
            
        Returns:
            Ranking loss value
        """
        # Flatten if needed
        if predictions.dim() > 1:
            batch_size = predictions.shape[0]
            predictions = predictions.view(-1)
            targets = targets.view(-1)
        
        n = len(predictions)
        
        # Create all pairs
        # pred_diff[i,j] = pred[i] - pred[j]
        pred_diff = predictions.unsqueeze(0) - predictions.unsqueeze(1)
        target_diff = targets.unsqueeze(0) - targets.unsqueeze(1)
        
        # Compute loss: max(0, -pred_diff * target_diff + margin)
        # When target[i] > target[j], we want pred[i] > pred[j]
        # So target_diff > 0 should have pred_diff > 0
        losses = torch.relu(-pred_diff * target_diff + self.margin)
        
        # Apply mask if provided
        if mask is not None:
            losses = losses * mask
        
        # Remove diagonal (comparing with itself)
        mask_diagonal = torch.eye(n, device=predictions.device)
        losses = losses * (1 - mask_diagonal)
        
        # Reduce
        if self.reduction == 'none':
            return losses
        elif self.reduction == 'sum':
            return losses.sum()
        elif self.reduction == 'mean':
            # Mean over valid pairs (excluding diagonal)
            num_pairs = n * (n - 1)
            return losses.sum() / max(num_pairs, 1)
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")


class BatchPairwiseRankingLoss(nn.Module):
    """
    Efficient batch-wise pairwise ranking loss
    Computes ranking loss within each batch to reduce memory usage
    """
    
    def __init__(self, 
                 sample_ratio: float = 1.0,
                 margin: float = 0.0,
                 reduction: str = 'mean'):
        """
        Args:
            sample_ratio: Ratio of pairs to sample (for efficiency)
            margin: Margin for ranking loss
            reduction: How to reduce the loss
        """
        super().__init__()
        self.sample_ratio = sample_ratio
        self.margin = margin
        self.reduction = reduction
        
    def forward(self,
                predictions: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        Compute batch-wise ranking loss
        
        Args:
            predictions: Predicted returns (batch_size, num_stocks)
            targets: Ground truth returns (batch_size, num_stocks)
            
        Returns:
            Ranking loss value
        """
        batch_size, num_stocks = predictions.shape
        total_loss = 0
        
        # Process each sample in batch
        for b in range(batch_size):
            pred_b = predictions[b]
            target_b = targets[b]
            
            # Sample pairs if needed
            if self.sample_ratio < 1.0:
                num_pairs = int(num_stocks * (num_stocks - 1) * self.sample_ratio)
                # Random sampling of pairs
                idx1 = torch.randint(0, num_stocks, (num_pairs,))
                idx2 = torch.randint(0, num_stocks, (num_pairs,))
                # Ensure different indices
                mask = idx1 != idx2
                idx1 = idx1[mask]
                idx2 = idx2[mask]
                
                pred_diff = pred_b[idx1] - pred_b[idx2]
                target_diff = target_b[idx1] - target_b[idx2]
            else:
                # All pairs
                pred_diff = pred_b.unsqueeze(0) - pred_b.unsqueeze(1)
                target_diff = target_b.unsqueeze(0) - target_b.unsqueeze(1)
                # Flatten and remove diagonal
                n = num_stocks
                mask = ~torch.eye(n, dtype=torch.bool, device=pred_b.device)
                pred_diff = pred_diff[mask]
                target_diff = target_diff[mask]
            
            # Compute ranking loss
            loss = torch.relu(-pred_diff * target_diff + self.margin)
            
            if self.reduction == 'mean':
                total_loss += loss.mean()
            else:
                total_loss += loss.sum()
        
        # Average over batch
        return total_loss / batch_size


class MovementLoss(nn.Module):
    """
    Binary cross-entropy loss for movement prediction
    From paper: L_move = -[y_move log ŷ_move + (1-y_move) log(1-ŷ_move)]
    """
    
    def __init__(self, reduction: str = 'mean'):
        """
        Args:
            reduction: How to reduce the loss
        """
        super().__init__()
        self.bce_loss = nn.BCELoss(reduction=reduction)
    
    def forward(self,
                predictions: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        Compute movement prediction loss
        
        Args:
            predictions: Predicted movements (batch_size,) with sigmoid applied
            targets: Ground truth movements (batch_size,) binary values
            
        Returns:
            BCE loss value
        """
        # Ensure targets are float for BCE
        targets = targets.float()
        
        # Clip predictions to avoid log(0)
        predictions = torch.clamp(predictions, min=1e-7, max=1-1e-7)
        
        return self.bce_loss(predictions, targets)


class FinGATLoss(nn.Module):
    """
    Combined multi-task loss for FinGAT
    L = (1-δ)L_rank + δL_move + λ||Θ||²
    
    Paper recommendations:
    - δ = 0.01 (mostly ranking loss with small movement loss)
    - λ = 1e-4 (L2 regularization)
    """
    
    def __init__(self,
                 ranking_weight: float = 0.99,  # (1-δ) in paper
                 movement_weight: float = 0.01,  # δ in paper
                 l2_lambda: float = 1e-4,       # λ in paper
                 use_batch_ranking: bool = True,
                 sample_ratio: float = 1.0):
        """
        Args:
            ranking_weight: Weight for ranking loss (1-δ)
            movement_weight: Weight for movement loss (δ)
            l2_lambda: L2 regularization weight (λ)
            use_batch_ranking: Use batch-wise ranking for efficiency
            sample_ratio: Ratio of pairs to sample in ranking loss
        """
        super().__init__()
        
        # Ensure weights sum to 1
        total = ranking_weight + movement_weight
        self.ranking_weight = ranking_weight / total
        self.movement_weight = movement_weight / total
        self.l2_lambda = l2_lambda
        
        # Initialize loss functions
        if use_batch_ranking:
            self.ranking_loss = BatchPairwiseRankingLoss(
                sample_ratio=sample_ratio,
                reduction='mean'
            )
        else:
            self.ranking_loss = PairwiseRankingLoss(reduction='mean')
        
        self.movement_loss = MovementLoss(reduction='mean')
        
        print(f"FinGATLoss initialized:")
        print(f"  Ranking weight: {self.ranking_weight:.4f}")
        print(f"  Movement weight: {self.movement_weight:.4f}")
        print(f"  L2 lambda: {self.l2_lambda}")
        
    def forward(self,
                pred_returns: torch.Tensor,
                pred_movements: torch.Tensor,
                target_returns: torch.Tensor,
                target_movements: torch.Tensor,
                model_params: Optional[list] = None) -> Tuple[torch.Tensor, dict]:
        """
        Compute combined loss
        
        Args:
            pred_returns: Predicted returns
            pred_movements: Predicted movements (after sigmoid)
            target_returns: Ground truth returns
            target_movements: Ground truth movements
            model_params: Model parameters for L2 regularization
            
        Returns:
            total_loss: Combined loss value
            loss_dict: Dictionary with individual loss components
        """
        # Compute individual losses
        loss_rank = self.ranking_loss(pred_returns, target_returns)
        loss_move = self.movement_loss(pred_movements, target_movements)
        
        # Combine losses
        total_loss = (self.ranking_weight * loss_rank + 
                     self.movement_weight * loss_move)
        
        # Add L2 regularization if model parameters provided
        loss_l2 = 0
        if model_params is not None and self.l2_lambda > 0:
            for param in model_params:
                if param.requires_grad:
                    loss_l2 += param.norm(2).pow(2)
            loss_l2 = self.l2_lambda * loss_l2
            total_loss = total_loss + loss_l2
        
        # Create loss dictionary for logging
        loss_dict = {
            'loss_total': total_loss.item(),
            'loss_ranking': loss_rank.item(),
            'loss_movement': loss_move.item(),
            'loss_l2': loss_l2.item() if isinstance(loss_l2, torch.Tensor) else loss_l2,
            'loss_weighted_ranking': (self.ranking_weight * loss_rank).item(),
            'loss_weighted_movement': (self.movement_weight * loss_move).item()
        }
        
        return total_loss, loss_dict


# Test functions
if __name__ == "__main__":
    print("Testing FinGAT Loss Functions")
    print("="*50)
    
    # Test data
    batch_size = 4
    num_stocks = 10
    
    pred_returns = torch.randn(batch_size, num_stocks)
    target_returns = torch.randn(batch_size, num_stocks)
    
    pred_movements = torch.sigmoid(torch.randn(batch_size, num_stocks))
    target_movements = (torch.rand(batch_size, num_stocks) > 0.5).float()
    
    # Test individual losses
    print("\n1. Testing PairwiseRankingLoss...")
    ranking_loss = PairwiseRankingLoss()
    loss_rank = ranking_loss(pred_returns[0], target_returns[0])
    print(f"   Ranking loss: {loss_rank:.4f}")
    
    print("\n2. Testing BatchPairwiseRankingLoss...")
    batch_ranking_loss = BatchPairwiseRankingLoss()
    loss_batch_rank = batch_ranking_loss(pred_returns, target_returns)
    print(f"   Batch ranking loss: {loss_batch_rank:.4f}")
    
    print("\n3. Testing MovementLoss...")
    movement_loss = MovementLoss()
    loss_move = movement_loss(pred_movements.flatten(), target_movements.flatten())
    print(f"   Movement loss: {loss_move:.4f}")
    
    print("\n4. Testing FinGATLoss...")
    fingat_loss = FinGATLoss()
    total_loss, loss_dict = fingat_loss(
        pred_returns.flatten(),
        pred_movements.flatten(),
        target_returns.flatten(),
        target_movements.flatten()
    )
    
    print(f"   Total loss: {total_loss:.4f}")
    print("   Loss components:")
    for key, value in loss_dict.items():
        print(f"     {key}: {value:.4f}")
    
    print("\n✅ All loss functions working correctly!")
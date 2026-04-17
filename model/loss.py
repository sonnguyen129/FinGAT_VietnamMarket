"""Multi-task loss function for FinGAT.

NOTE: The implementation follows the ORIGINAL REPO
(https://github.com/Roytsai27/Financial-GraphAttention), not the paper's
Eq. 15 literally. The paper writes:

    L = (1-delta)*L_rank + delta*L_move + lambda*||Theta||^2

But the authors' actual code adds an auxiliary MAE regression term and uses
three independent weights (alpha, beta, gamma) instead of the (1-delta, delta)
trade-off:

    total = alpha * mean(L1)  +  beta * mean(BCE)  +  gamma * mean(rank)

Differences from the original repo:
  - Rank loss uses DIFFERENCE-based formula (paper Eq. 16):
        max(0, -(pred_i - pred_j) * (y_i - y_j))
    Original repo uses PRODUCT-based:
        relu(-(pred_i * pred_j) * (y_i * y_j))   # non-standard
  - Rank loss uses MEAN over sampled pairs instead of SUM over all pairs
    (see _ranking_loss for sampling rationale).

L2 regularization (lambda * ||Theta||^2) is handled via optimizer weight_decay.
"""

import torch
import torch.nn as nn


class FinGATLoss(nn.Module):
    """Combined multi-task loss for FinGAT.

    Args:
        alpha: MAE regression loss weight (default 1.0)
        beta: classification loss weight (default 0.01, paper's delta)
        gamma: pairwise ranking loss weight (default 1.0)
        use_mse: if True, use MSE instead of BCE for classification
        max_pairs: max number of random pairs for ranking loss (efficiency)
    """

    def __init__(self, alpha: float = 1.0, beta: float = 0.01,
                 gamma: float = 1.0, use_mse: bool = False,
                 max_pairs: int = 1000):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.use_mse = use_mse
        self.max_pairs = max_pairs
        self.reg_loss_fn = nn.L1Loss(reduction='mean')
        if use_mse:
            self.cls_loss_fn = nn.MSELoss(reduction='mean')
        else:
            self.cls_loss_fn = nn.BCELoss(reduction='mean')

    def forward(self, reg_out: torch.Tensor, cls_out: torch.Tensor,
                y_return: torch.Tensor, y_binary: torch.Tensor) -> tuple:
        """
        Args:
            reg_out: [N, 1] predicted returns
            cls_out: [N, 1] predicted up probability (or None)
            y_return: [N] or [N, 1] actual returns
            y_binary: [N] or [N, 1] actual binary labels

        Returns:
            (total_loss, reg_loss_val, cls_loss_val, rank_loss_val)
        """
        y_ret = y_return.view(-1, 1)
        y_bin = y_binary.view(-1, 1)
        reg_out_flat = reg_out.view(-1, 1)

        # MAE regression loss
        reg_loss = self.reg_loss_fn(reg_out_flat, y_ret)

        # Classification loss
        cls_loss_val = 0.0
        if cls_out is not None and self.beta > 0:
            cls_out_flat = cls_out.view(-1, 1)
            cls_loss = self.cls_loss_fn(cls_out_flat, y_bin)
            cls_loss_val = cls_loss.item()
        else:
            cls_loss = torch.tensor(0.0, device=reg_out.device)

        # Pairwise ranking loss with sampling
        rank_loss = self._ranking_loss(reg_out_flat, y_ret)

        total = (self.alpha * reg_loss +
                 self.beta * cls_loss +
                 self.gamma * rank_loss)

        return total, reg_loss.item(), cls_loss_val, rank_loss.item()

    def _ranking_loss(self, pred: torch.Tensor,
                      target: torch.Tensor) -> torch.Tensor:
        """Pairwise ranking loss with random pair sampling.

        Eq. 16: sum max(0, -(pred_i - pred_j) * (y_i - y_j))

        Why sample pairs? Full pairwise loss is O(N^2). For N=200 stocks
        that is ~40k pairs PER training step. Monte-Carlo sampling
        `max_pairs` random (i, j) indices each step gives an unbiased
        estimate of the mean loss with lower compute.

        TRADEOFF NOTE: RankLSTM (Feng et al. 2019) and the original FinGAT
        repo both compute the full N^2 pairs via vectorized outer-products
        (single GPU op, very fast). For N <= ~1000 the vectorized approach
        is usually preferable to sampling. Sampling is kept here for
        flexibility with very large N and as a baseline; consider switching
        to outer-product if you observe high variance in the rank loss.
        """
        n = pred.shape[0]
        if n <= 1:
            return torch.tensor(0.0, device=pred.device)

        # Sample random pairs for efficiency
        num_pairs = min(self.max_pairs, n * (n - 1))
        idx_i = torch.randint(0, n, (num_pairs,), device=pred.device)
        idx_j = torch.randint(0, n, (num_pairs,), device=pred.device)
        # Ensure i != j
        mask = idx_i != idx_j
        idx_i = idx_i[mask]
        idx_j = idx_j[mask]

        if len(idx_i) == 0:
            return torch.tensor(0.0, device=pred.device)

        pred_diff = pred[idx_i] - pred[idx_j]
        true_diff = target[idx_i] - target[idx_j]

        loss = torch.relu(-pred_diff * true_diff).mean()
        return loss


class RankingOnlyLoss(nn.Module):
    """Ranking loss only (for w/o MTL ablation)."""

    def __init__(self, gamma: float = 1.0, max_pairs: int = 1000):
        super().__init__()
        self.gamma = gamma
        self.max_pairs = max_pairs
        self.reg_loss_fn = nn.L1Loss(reduction='mean')

    def forward(self, reg_out, cls_out, y_return, y_binary):
        y_ret = y_return.view(-1, 1)
        reg_out_flat = reg_out.view(-1, 1)

        reg_loss = self.reg_loss_fn(reg_out_flat, y_ret)

        n = reg_out_flat.shape[0]
        num_pairs = min(self.max_pairs, n * (n - 1))
        idx_i = torch.randint(0, n, (num_pairs,), device=reg_out.device)
        idx_j = torch.randint(0, n, (num_pairs,), device=reg_out.device)
        mask = idx_i != idx_j
        idx_i, idx_j = idx_i[mask], idx_j[mask]

        if len(idx_i) > 0:
            pred_diff = reg_out_flat[idx_i] - reg_out_flat[idx_j]
            true_diff = y_ret[idx_i] - y_ret[idx_j]
            rank_loss = torch.relu(-pred_diff * true_diff).mean()
        else:
            rank_loss = torch.tensor(0.0, device=reg_out.device)

        total = reg_loss + self.gamma * rank_loss
        return total, reg_loss.item(), 0.0, rank_loss.item()

import math
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class WeeklyLossConfig:
    # Weights per Deepwiki: alpha (MAE), beta (CE), gamma (Ranking), plus L2
    alpha: float = 0.0
    beta: float = 0.01
    gamma: float = 0.99
    lambda_l2: float = 1e-4
    ranking_margin: float = 0.0  # hinge margin


class MultiTaskWeeklyLoss(nn.Module):
    """
    (1-δ) L_rank + δ L_move + λ ||Θ||² with δ in [0,1].
    - L_move: CrossEntropy with ignore_index=-100 and masking by asset_mask
    - L_rank: pairwise hinge over assets per date (only valid assets per sample)
    """
    def __init__(self, cfg: WeeklyLossConfig):
        super().__init__()
        self.cfg = cfg
        self.ce = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
        self.mae = nn.L1Loss(reduction='none')

    def ranking_loss_per_sample(self, scores: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
        # scores, returns: (N_valid,)
        if scores.numel() <= 1:
            return scores.new_zeros(())
        s_i = scores.unsqueeze(1)
        s_j = scores.unsqueeze(0)
        r_i = returns.unsqueeze(1)
        r_j = returns.unsqueeze(0)
        mask = (r_i > r_j)  # only pairs where return_i > return_j
        if mask.sum() == 0:
            return scores.new_zeros(())
        # hinge loss: max(0, margin - (s_i - s_j)) for valid pairs
        diff = s_i - s_j
        loss = F.relu(self.cfg.ranking_margin - diff)[mask]
        return loss.mean()

    def forward(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], model: Optional[nn.Module] = None) -> Dict[str, torch.Tensor]:
        logits = outputs['movement_logits']  # (B,N,C)
        scores = outputs['ranking_scores']   # (B,N)
        ret_pred = outputs.get('ret_pred', None)
        labels = batch['y_move']             # (B,N)
        rets = batch['y_return']            # (B,N)
        mask = batch['asset_mask']          # (B,N) bool

        # Movement CE with mask
        B, N, C = logits.shape
        ce_loss = self.ce(logits.view(B*N, C), labels.view(B*N))  # (B*N,)
        ce_loss = ce_loss.view(B, N)
        if mask is not None:
            ce_loss = ce_loss * mask.float()
            denom = mask.float().sum().clamp_min(1.0)
            L_move = ce_loss.sum() / denom
        else:
            L_move = ce_loss.mean()

    # Ranking pairwise per sample using valid assets
        L_rank_list = []
        for b in range(B):
            if mask is not None:
                idx = mask[b].nonzero(as_tuple=False).squeeze(-1)
            else:
                idx = torch.arange(N, device=scores.device)
            if idx.numel() <= 1:
                L_rank_list.append(scores.new_zeros(()))
                continue
            s_b = scores[b, idx]
            r_b = rets[b, idx]
            L_rank_list.append(self.ranking_loss_per_sample(s_b, r_b))
        L_rank = torch.stack(L_rank_list).mean() if len(L_rank_list) > 0 else scores.new_zeros(())

        # MAE regression on returns with mask (if ret_pred available)
        if ret_pred is not None:
            mae = self.mae(ret_pred, rets)
            if mask is not None:
                mae = mae * mask.float()
                denom = mask.float().sum().clamp_min(1.0)
                L_mae = mae.sum() / denom
            else:
                L_mae = mae.mean()
        else:
            L_mae = scores.new_zeros(())

        # L2 regularization
        L2 = torch.tensor(0.0, device=scores.device)
        if model is not None and self.cfg.lambda_l2 > 0:
            for p in model.parameters():
                if p.requires_grad:
                    L2 = L2 + p.pow(2).sum()
            L2 = self.cfg.lambda_l2 * L2

        total = self.cfg.alpha * L_mae + self.cfg.beta * L_move + self.cfg.gamma * L_rank + L2

        return {
            'total_loss': total,
            'movement_loss': L_move.detach(),
            'ranking_loss': L_rank.detach(),
            'mae_loss': L_mae.detach(),
            'l2_reg': L2.detach()
        }


def mrr_at_k(y_true: torch.Tensor, y_scores: torch.Tensor, k: int) -> float:
    # y_true, y_scores: (N,)
    N = y_scores.numel()
    if N == 0:
        return 0.0
    k = min(k, N)
    _, idx = torch.topk(y_scores, k)
    rel = y_true[idx]
    pos = (rel == 1).nonzero(as_tuple=False)
    if pos.numel() == 0:
        return 0.0
    first_rank = pos[0].item() + 1
    return 1.0 / first_rank


def precision_at_k(y_true: torch.Tensor, y_scores: torch.Tensor, k: int) -> float:
    N = y_scores.numel()
    if N == 0:
        return 0.0
    k = min(k, N)
    _, idx = torch.topk(y_scores, k)
    rel = y_true[idx]
    return rel.float().mean().item()


class WeeklyTrainer:
    def __init__(self, model: nn.Module, device: Optional[torch.device] = None,
                 lr: float = 1e-3, weight_decay: float = 0.0,
                 loss_cfg: WeeklyLossConfig = WeeklyLossConfig()):
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.loss_fn = MultiTaskWeeklyLoss(loss_cfg)

    def step(self, batch: Dict[str, Any], train: bool = True) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        # Move tensors to device
        for k in ['x', 'y_return', 'y_move', 'adj', 'asset_mask']:
            if k in batch and isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(self.device)
        self.model.train(mode=train)
        outputs = self.model(batch)
        losses = self.loss_fn(outputs, batch, model=self.model)
        if train:
            self.opt.zero_grad()
            losses['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.opt.step()
        # Prepare metrics
        with torch.no_grad():
            mask = batch['asset_mask']
            logits = outputs['movement_logits']  # (B,N,C)
            scores = outputs['ranking_scores']   # (B,N)
            labels = batch['y_move']             # (B,N)
            returns = batch['y_return']          # (B,N)
            preds = logits.argmax(dim=-1)
            # Masked accuracy
            correct = ((preds == labels) & mask).sum().item()
            total = mask.sum().item()
            acc = correct / total if total > 0 else 0.0
            # Ranking metrics per sample then average
            mrr5 = 0.0
            mrr10 = 0.0
            p5 = 0.0
            p10 = 0.0
            B = scores.size(0)
            for b in range(B):
                idx = mask[b].nonzero(as_tuple=False).squeeze(-1)
                if idx.numel() == 0:
                    continue
                s = scores[b, idx]
                r = returns[b, idx]
                y_bin = (r > 0).long()
                mrr5 += mrr_at_k(y_bin, s, 5)
                mrr10 += mrr_at_k(y_bin, s, 10)
                p5 += precision_at_k(y_bin, s, 5)
                p10 += precision_at_k(y_bin, s, 10)
            denom = max(B, 1)
        metrics = {
            'loss': losses['total_loss'].item(),
            'loss_move': losses['movement_loss'].item(),
            'loss_rank': losses['ranking_loss'].item(),
            'acc': acc,
            'mrr@5': mrr5 / denom,
            'mrr@10': mrr10 / denom,
            'p@5': p5 / denom,
            'p@10': p10 / denom,
        }
        return metrics, outputs

    def run_epoch(self, loader, train: bool = True) -> Dict[str, float]:
        agg = {'loss': 0.0, 'loss_move': 0.0, 'loss_rank': 0.0, 'acc': 0.0, 'mrr@5': 0.0, 'mrr@10': 0.0, 'p@5': 0.0, 'p@10': 0.0}
        n = 0
        for batch in tqdm(loader, desc='Train' if train else 'Eval'):
            metrics, _ = self.step(batch, train=train)
            for k in agg:
                agg[k] += metrics[k]
            n += 1
        for k in agg:
            agg[k] = agg[k] / max(n, 1)
        return agg

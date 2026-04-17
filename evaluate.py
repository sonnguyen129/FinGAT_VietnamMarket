"""Evaluation metrics — MRR@K, Precision@K, Accuracy, IRR (Paper Eq. 17-18)."""

import numpy as np


def mrr_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int = 5) -> float:
    """Mean Reciprocal Rank @ K.

    Eq. 17: MRR@K = (1/K) * sum(1/rank_true(sq)) for sq in predicted top-K.
    """
    n = len(y_true)
    if n == 0 or k <= 0:
        return 0.0

    # Predicted top-K indices (highest predicted return)
    pred_top_k = np.argsort(y_pred.flatten())[-k:]

    # Ground-truth ranking (1-based, higher return = better rank)
    true_ranks = np.zeros(n)
    true_order = np.argsort(y_true.flatten())[::-1]  # descending
    for rank, idx in enumerate(true_order):
        true_ranks[idx] = rank + 1  # 1-based

    # MRR: average reciprocal of true ranks for predicted top-K
    reciprocal_sum = sum(1.0 / true_ranks[i] for i in pred_top_k)
    return reciprocal_sum / k


def precision_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int = 5) -> float:
    """Precision @ K — overlap between predicted and actual top-K.

    Eq. 18: P@K = |predicted_top_K ∩ actual_top_K| / K
    """
    pred_top = set(np.argsort(y_pred.flatten())[-k:])
    true_top = set(np.argsort(y_true.flatten())[-k:])
    return len(pred_top & true_top) / k


def accuracy(y_true_binary: np.ndarray, y_pred_return: np.ndarray) -> float:
    """Binary up/down movement accuracy."""
    pred_binary = (y_pred_return.flatten() > 0).astype(int)
    true_binary = y_true_binary.flatten().astype(int)
    return float(np.mean(pred_binary == true_binary))


def irr_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int = 5) -> float:
    """Investment Return Ratio @ K.

    Sum of actual returns for predicted top-K stocks.
    """
    pred_top = np.argsort(y_pred.flatten())[-k:]
    return float(y_true.flatten()[pred_top].sum())


def evaluate_model(model, data: dict, config, split: str = 'test') -> dict:
    """Evaluate model on a data split.

    Args:
        model: trained model with forward(weekly_inputs) -> (reg_out, cls_out)
        data: splits dict with keys like 'test'
        config: Config object with top_k_values, week_num, device
        split: which split to evaluate on

    Returns:
        dict of metric_name -> value
    """
    import torch

    model.eval()
    split_data = data[split]
    num_samples = split_data['x1'].shape[0]
    week_num = config.week_num

    all_mrr = {k: [] for k in config.top_k_values}
    all_prec = {k: [] for k in config.top_k_values}
    all_acc = []
    all_irr = {k: [] for k in config.top_k_values}

    with torch.no_grad():
        for t in range(num_samples):
            weekly = [
                torch.tensor(split_data[f'x{w + 1}'][t],
                             dtype=torch.float32).to(config.device)
                for w in range(week_num)
            ]
            reg_out, cls_out = model(weekly)
            y_true = split_data['y_return_ratio'][t]
            y_binary = split_data['y_up_or_down'][t]
            preds = reg_out.cpu().numpy().flatten()

            for k in config.top_k_values:
                all_mrr[k].append(mrr_at_k(y_true, preds, k))
                all_prec[k].append(precision_at_k(y_true, preds, k))
                all_irr[k].append(irr_at_k(y_true, preds, k))
            all_acc.append(accuracy(y_binary, preds))

    results = {}
    for k in config.top_k_values:
        results[f'MRR@{k}'] = float(np.mean(all_mrr[k]))
        results[f'Precision@{k}'] = float(np.mean(all_prec[k]))
        results[f'IRR@{k}'] = float(np.mean(all_irr[k]))
    results['ACC'] = float(np.mean(all_acc))

    return results


def evaluate_baseline(model, data: dict, config, split: str = 'test',
                      is_graph_model: bool = False) -> dict:
    """Evaluate a baseline model that takes flattened input.

    Baselines expect input shape [num_stocks, seq_len, features] or flattened.
    """
    import torch

    model.eval()
    split_data = data[split]
    num_samples = split_data['x1'].shape[0]
    week_num = config.week_num

    all_mrr = {k: [] for k in config.top_k_values}
    all_prec = {k: [] for k in config.top_k_values}
    all_acc = []

    with torch.no_grad():
        for t in range(num_samples):
            weekly = [
                split_data[f'x{w + 1}'][t] for w in range(week_num)
            ]
            # Concatenate weeks: [num_stocks, total_days, features]
            x_concat = np.concatenate(weekly, axis=1)
            x_tensor = torch.tensor(x_concat, dtype=torch.float32).to(config.device)

            if is_graph_model:
                reg_out, cls_out = model(x_tensor)
            else:
                reg_out, cls_out = model(x_tensor)

            y_true = split_data['y_return_ratio'][t]
            y_binary = split_data['y_up_or_down'][t]
            preds = reg_out.cpu().numpy().flatten()

            for k in config.top_k_values:
                all_mrr[k].append(mrr_at_k(y_true, preds, k))
                all_prec[k].append(precision_at_k(y_true, preds, k))
            all_acc.append(accuracy(y_binary, preds))

    results = {}
    for k in config.top_k_values:
        results[f'MRR@{k}'] = float(np.mean(all_mrr[k]))
        results[f'Precision@{k}'] = float(np.mean(all_prec[k]))
    results['ACC'] = float(np.mean(all_acc))
    return results

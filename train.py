"""Training loop for FinGAT and baseline models."""

import copy

import numpy as np
import torch
from torch import optim
from tqdm import tqdm

from config import Config
from evaluate import evaluate_model, mrr_at_k
from utils import set_seed, init_weights


def train_model(model, data: dict, config: Config,
                criterion, name: str = "model",
                verbose: bool = True) -> dict:
    """Train a model and return best test results.

    Args:
        model: model with forward(weekly_inputs) -> (reg_out, cls_out)
        data: splits dict {train/val/test: {x1, x2, x3, y_return_ratio, y_up_or_down}}
        config: Config object
        criterion: loss function with forward(reg_out, cls_out, y_return, y_binary)
        name: model name for logging
        verbose: print progress

    Returns:
        dict with best_val_mrr, test_results, training_history
    """
    model = model.to(config.device)
    init_weights(model)

    optimizer = optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    best_val_mrr = -1
    best_model_state = None
    history = []
    patience = 10
    patience_counter = 0

    num_train = data['train']['x1'].shape[0]
    week_num = config.week_num

    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_reg = 0.0
        epoch_cls = 0.0
        epoch_rank = 0.0

        indices = list(range(num_train))
        for t in indices:
            weekly = [
                torch.tensor(data['train'][f'x{w + 1}'][t],
                             dtype=torch.float32).to(config.device)
                for w in range(week_num)
            ]
            y_ret = torch.tensor(
                data['train']['y_return_ratio'][t],
                dtype=torch.float32).to(config.device)
            y_bin = torch.tensor(
                data['train']['y_up_or_down'][t],
                dtype=torch.float32).to(config.device)

            reg_out, cls_out = model(weekly)
            loss, reg_l, cls_l, rank_l = criterion(reg_out, cls_out, y_ret, y_bin)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_reg += reg_l
            epoch_cls += cls_l
            epoch_rank += rank_l

        avg_loss = epoch_loss / num_train

        # Validate
        val_results = evaluate_model(model, data, config, split='val')
        val_mrr = val_results['MRR@5']

        history.append({
            'epoch': epoch + 1,
            'train_loss': avg_loss,
            'val_mrr5': val_mrr,
        })

        if val_mrr > best_val_mrr:
            best_val_mrr = val_mrr
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose and (epoch + 1) % 5 == 0:
            print(f"  [{name}] Epoch {epoch+1}/{config.epochs} "
                  f"loss={avg_loss:.4f} val_MRR@5={val_mrr:.4f}")

        if patience_counter >= patience:
            if verbose:
                print(f"  [{name}] Early stopping at epoch {epoch+1}")
            break

    # Load best model and evaluate on test set
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    test_results = evaluate_model(model, data, config, split='test')

    if verbose:
        print(f"  [{name}] Best val MRR@5={best_val_mrr:.4f} | "
              f"Test: MRR@5={test_results['MRR@5']:.4f} "
              f"ACC={test_results['ACC']:.4f}")

    return {
        'best_val_mrr': best_val_mrr,
        'test_results': test_results,
        'history': history,
        'model_state': best_model_state,
    }


def train_baseline(model, data: dict, config: Config,
                   criterion, name: str = "baseline",
                   verbose: bool = True) -> dict:
    """Train a baseline model that takes concatenated weekly input.

    Baselines receive [num_stocks, total_days, features] instead of weekly list.
    """
    model = model.to(config.device)
    init_weights(model)

    optimizer = optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    best_val_mrr = -1
    best_model_state = None
    patience = 10
    patience_counter = 0
    history = []

    num_train = data['train']['x1'].shape[0]
    week_num = config.week_num

    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0

        for t in range(num_train):
            # Concatenate weeks for baselines
            weekly_arrays = [
                data['train'][f'x{w + 1}'][t] for w in range(week_num)
            ]
            x_concat = np.concatenate(weekly_arrays, axis=1)
            x_tensor = torch.tensor(
                x_concat, dtype=torch.float32).to(config.device)

            y_ret = torch.tensor(
                data['train']['y_return_ratio'][t],
                dtype=torch.float32).to(config.device)
            y_bin = torch.tensor(
                data['train']['y_up_or_down'][t],
                dtype=torch.float32).to(config.device)

            reg_out, cls_out = model(x_tensor)
            loss, _, _, _ = criterion(reg_out, cls_out, y_ret, y_bin)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / num_train

        # Validate
        val_results = _evaluate_baseline_split(
            model, data, config, split='val')
        val_mrr = val_results['MRR@5']

        history.append({'epoch': epoch + 1, 'train_loss': avg_loss,
                        'val_mrr5': val_mrr})

        if val_mrr > best_val_mrr:
            best_val_mrr = val_mrr
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose and (epoch + 1) % 5 == 0:
            print(f"  [{name}] Epoch {epoch+1}/{config.epochs} "
                  f"loss={avg_loss:.4f} val_MRR@5={val_mrr:.4f}")

        if patience_counter >= patience:
            if verbose:
                print(f"  [{name}] Early stopping at epoch {epoch+1}")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    test_results = _evaluate_baseline_split(model, data, config, split='test')

    if verbose:
        print(f"  [{name}] Best val MRR@5={best_val_mrr:.4f} | "
              f"Test: MRR@5={test_results['MRR@5']:.4f} "
              f"ACC={test_results['ACC']:.4f}")

    return {
        'best_val_mrr': best_val_mrr,
        'test_results': test_results,
        'history': history,
        'model_state': best_model_state,
    }


def _evaluate_baseline_split(model, data, config, split='val'):
    """Evaluate baseline on a split."""
    from evaluate import mrr_at_k, precision_at_k, accuracy

    model.eval()
    split_data = data[split]
    num_samples = split_data['x1'].shape[0]
    week_num = config.week_num

    all_mrr = {k: [] for k in config.top_k_values}
    all_prec = {k: [] for k in config.top_k_values}
    all_acc = []

    with torch.no_grad():
        for t in range(num_samples):
            weekly_arrays = [
                split_data[f'x{w + 1}'][t] for w in range(week_num)
            ]
            x_concat = np.concatenate(weekly_arrays, axis=1)
            x_tensor = torch.tensor(
                x_concat, dtype=torch.float32).to(config.device)

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


def train_multi_run(model_factory, data: dict, config: Config,
                    criterion_factory, name: str = "model",
                    num_runs: int = None, is_baseline: bool = False,
                    verbose: bool = True) -> dict:
    """Train multiple runs with different seeds and aggregate results.

    Args:
        model_factory: callable(config) -> model
        data: splits dict
        config: Config object
        criterion_factory: callable(config) -> loss function
        name: model name
        num_runs: override config.num_runs
        is_baseline: if True, use train_baseline instead of train_model
        verbose: print per-run results

    Returns:
        dict with 'mean', 'std', 'all_runs' for each metric
    """
    if num_runs is None:
        num_runs = config.num_runs

    all_results = []
    for run in range(num_runs):
        seed = config.seed_base + run
        set_seed(seed)
        if verbose:
            print(f"\n--- Run {run+1}/{num_runs} (seed={seed}) ---")

        model = model_factory(config)
        criterion = criterion_factory(config)

        train_fn = train_baseline if is_baseline else train_model
        result = train_fn(
            model, data, config, criterion, name=name, verbose=verbose)
        all_results.append(result['test_results'])

    # Aggregate
    all_metrics = {}
    for key in all_results[0]:
        values = [r[key] for r in all_results]
        all_metrics[key] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'values': values,
        }

    if verbose:
        print(f"\n=== {name} ({num_runs} runs) ===")
        for key, stats in all_metrics.items():
            print(f"  {key}: {stats['mean']:.4f} ± {stats['std']:.4f}")

    return all_metrics

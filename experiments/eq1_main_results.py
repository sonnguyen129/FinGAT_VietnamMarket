"""EQ1: Main Results — Table 2.

Compare FinGAT vs 5 baselines on HOSE data.
Models: MLP, GRU, GRU+Att, FineNet, RankLSTM, FinGAT + Momentum, Random
Metrics: MRR@K, Precision@K, ACC for K in {5, 10, 20}
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from config import Config
from utils import set_seed, save_results, Timer
from train import train_model, train_baseline, train_multi_run
from evaluate import mrr_at_k, precision_at_k, accuracy
from model.fingat import FinGAT
from model.loss import FinGATLoss
from model.baselines import get_baseline_factory
from data.dataset import load_dataset
from data.graph_builder import load_edges
from data.sector_map import get_sector_list


def make_fingat(config, intra_edges, inter_edges, sector_assignments, num_sectors):
    """Factory for FinGAT model."""
    def factory(cfg):
        return FinGAT(
            input_dim=cfg.num_features,
            hidden_dim=cfg.hidden_dim,
            intra_edge_index=intra_edges.to(cfg.device),
            inter_edge_index=inter_edges.to(cfg.device),
            sector_assignments=sector_assignments.to(cfg.device),
            num_sectors=num_sectors,
            week_num=cfg.week_num,
            gat_heads=cfg.gat_heads,
            gat_dropout=cfg.gat_dropout,
            device=cfg.device,
        )
    return factory


def make_criterion(config):
    """Factory for loss function."""
    def factory(cfg):
        return FinGATLoss(
            alpha=cfg.alpha_reg, beta=cfg.beta_cls, gamma=cfg.gamma_rank,
            max_pairs=cfg.max_pairs)
    return factory


def momentum_baseline(data, config):
    """Momentum baseline: rank by past-week return (no learning)."""
    results = {}
    for k in config.top_k_values:
        results[f'MRR@{k}'] = {'mean': 0, 'std': 0}
        results[f'Precision@{k}'] = {'mean': 0, 'std': 0}
    results['ACC'] = {'mean': 0, 'std': 0}

    test_data = data['test']
    num_samples = test_data['x1'].shape[0]
    week_num = config.week_num

    all_mrr = {k: [] for k in config.top_k_values}
    all_prec = {k: [] for k in config.top_k_values}
    all_acc = []

    for t in range(num_samples):
        # Use last week's return as prediction
        last_week = test_data[f'x{week_num}'][t]  # [num_stocks, 5, 15]
        # Return ratio is feature index 5 (RETURN_RATIO_IDX)
        last_returns = last_week[:, -1, 5]  # last day of last week
        y_true = test_data['y_return_ratio'][t]
        y_binary = test_data['y_up_or_down'][t]

        for k in config.top_k_values:
            all_mrr[k].append(mrr_at_k(y_true, last_returns, k))
            all_prec[k].append(precision_at_k(y_true, last_returns, k))
        all_acc.append(accuracy(y_binary, last_returns))

    for k in config.top_k_values:
        results[f'MRR@{k}'] = {'mean': float(np.mean(all_mrr[k])), 'std': 0.0,
                                'values': all_mrr[k]}
        results[f'Precision@{k}'] = {'mean': float(np.mean(all_prec[k])), 'std': 0.0,
                                      'values': all_prec[k]}
    results['ACC'] = {'mean': float(np.mean(all_acc)), 'std': 0.0,
                      'values': all_acc}
    return results


def random_baseline(data, config):
    """Random baseline: random ranking."""
    rng = np.random.RandomState(config.seed_base)
    test_data = data['test']
    num_samples = test_data['x1'].shape[0]
    num_stocks = test_data['x1'].shape[1]

    all_mrr = {k: [] for k in config.top_k_values}
    all_prec = {k: [] for k in config.top_k_values}
    all_acc = []

    for t in range(num_samples):
        random_pred = rng.randn(num_stocks)
        y_true = test_data['y_return_ratio'][t]
        y_binary = test_data['y_up_or_down'][t]

        for k in config.top_k_values:
            all_mrr[k].append(mrr_at_k(y_true, random_pred, k))
            all_prec[k].append(precision_at_k(y_true, random_pred, k))
        all_acc.append(accuracy(y_binary, random_pred))

    results = {}
    for k in config.top_k_values:
        results[f'MRR@{k}'] = {'mean': float(np.mean(all_mrr[k])), 'std': 0.0}
        results[f'Precision@{k}'] = {'mean': float(np.mean(all_prec[k])), 'std': 0.0}
    results['ACC'] = {'mean': float(np.mean(all_acc)), 'std': 0.0}
    return results


def run_eq1(config: Config = None, num_runs: int = None):
    """Run EQ1 experiment — main results comparison."""
    if config is None:
        config = Config()
    if num_runs is not None:
        config.num_runs = num_runs

    print("=" * 60)
    print("EQ1: Main Results (Table 2)")
    print("=" * 60)

    # Load data
    data, tickers = load_dataset()
    intra_edges, inter_edges, sector_assignments = load_edges()
    num_sectors = int(sector_assignments.max().item()) + 1

    all_results = {}

    # Non-learning baselines
    print("\n--- Momentum Baseline ---")
    all_results['Momentum'] = momentum_baseline(data, config)
    print(f"  MRR@5={all_results['Momentum']['MRR@5']['mean']:.4f}")

    print("\n--- Random Baseline ---")
    all_results['Random'] = random_baseline(data, config)
    print(f"  MRR@5={all_results['Random']['MRR@5']['mean']:.4f}")

    # Learning baselines
    criterion_factory = make_criterion(config)
    for bname in ['MLP', 'GRU', 'GRU+Att', 'FineNet', 'RankLSTM']:
        print(f"\n--- {bname} ---")
        model_factory = get_baseline_factory(bname)
        results = train_multi_run(
            model_factory, data, config, criterion_factory,
            name=bname, num_runs=config.num_runs, is_baseline=True)
        all_results[bname] = results

    # FinGAT
    print("\n--- FinGAT ---")
    fingat_factory = make_fingat(
        config, intra_edges, inter_edges, sector_assignments, num_sectors)
    results = train_multi_run(
        fingat_factory, data, config, criterion_factory,
        name='FinGAT', num_runs=config.num_runs, is_baseline=False)
    all_results['FinGAT'] = results

    # Print Table 2
    print_table2(all_results, config)

    # Save
    save_results(all_results, "results/eq1_main_results.json")
    return all_results


def print_table2(results: dict, config: Config):
    """Print formatted Table 2."""
    print("\n" + "=" * 100)
    print("TABLE 2: Main Results on HOSE")
    print("=" * 100)

    header = f"{'Model':<12}"
    for k in config.top_k_values:
        header += f" | MRR@{k:<3} Prec@{k:<3}"
    header += " | ACC"
    print(header)
    print("-" * 100)

    models = ['Random', 'Momentum', 'MLP', 'GRU', 'GRU+Att',
              'FineNet', 'RankLSTM', 'FinGAT']
    for model in models:
        if model not in results:
            continue
        r = results[model]
        row = f"{model:<12}"
        for k in config.top_k_values:
            mrr = r[f'MRR@{k}']['mean']
            prec = r[f'Precision@{k}']['mean']
            row += f" | {mrr:.4f}  {prec:.4f}"
        acc = r['ACC']['mean']
        row += f" | {acc:.4f}"
        print(row)

    print("=" * 100)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=30)
    args = parser.parse_args()

    config = Config()
    config.num_runs = args.runs
    config.epochs = args.epochs
    run_eq1(config)

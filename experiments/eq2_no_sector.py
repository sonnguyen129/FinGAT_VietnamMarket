"""EQ2: Without Sector Info — Figure 3.

FinGAT-NT vs baselines on 5 stock subsets (10 stocks each).
Metric: MRR@3
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from config import Config
from utils import set_seed, save_results
from train import train_model, train_baseline, train_multi_run
from model.fingat_nt import FinGAT_NT
from model.loss import FinGATLoss
from model.baselines import get_baseline_factory
from data.dataset import load_dataset
from data.subset_builder import build_subsets, extract_subset_data
from data.collector import load_all_stocks


def make_fingat_nt(num_stocks):
    """Factory for FinGAT-NT model."""
    def factory(cfg):
        return FinGAT_NT(
            input_dim=cfg.num_features,
            hidden_dim=cfg.hidden_dim,
            num_stocks=num_stocks,
            week_num=cfg.week_num,
            gat_heads=cfg.gat_heads,
            device=cfg.device,
        )
    return factory


def run_eq2(config: Config = None, num_runs: int = None):
    """Run EQ2 experiment."""
    if config is None:
        config = Config()
    if num_runs is not None:
        config.num_runs = num_runs

    # Override top_k for EQ2: paper uses MRR@3
    eq2_config = Config()
    eq2_config.__dict__.update(config.__dict__)
    eq2_config.top_k_values = (3,)

    print("=" * 60)
    print("EQ2: Without Sector Info (Figure 3)")
    print("=" * 60)

    # Load full data
    data, tickers = load_dataset()
    all_raw_data = load_all_stocks(config.data_dir)

    # Build 5 subsets
    print("\nBuilding 5 stock subsets...")
    subsets = build_subsets(tickers, all_raw_data, seed=config.seed_base)

    all_results = {}
    criterion_factory = lambda cfg: FinGATLoss(
        alpha=cfg.alpha_reg, beta=cfg.beta_cls, gamma=cfg.gamma_rank,
        max_pairs=cfg.max_pairs)

    for subset_name, subset_tickers in subsets.items():
        print(f"\n{'='*40}")
        print(f"Subset: {subset_name}")
        print(f"{'='*40}")

        subset_data = extract_subset_data(data, tickers, subset_tickers)
        num_stocks = len(subset_tickers)
        subset_results = {}

        # Baselines
        for bname in ['MLP', 'GRU', 'GRU+Att', 'FineNet', 'RankLSTM']:
            print(f"\n  --- {bname} ---")
            model_factory = get_baseline_factory(bname)
            results = train_multi_run(
                model_factory, subset_data, eq2_config, criterion_factory,
                name=bname, num_runs=config.num_runs, is_baseline=True,
                verbose=False)
            subset_results[bname] = results
            print(f"  MRR@3={results['MRR@3']['mean']:.4f} ± {results['MRR@3']['std']:.4f}")

        # FinGAT-NT
        print(f"\n  --- FinGAT-NT ---")
        fingat_nt_factory = make_fingat_nt(num_stocks)
        results = train_multi_run(
            fingat_nt_factory, subset_data, eq2_config, criterion_factory,
            name='FinGAT-NT', num_runs=config.num_runs, is_baseline=False,
            verbose=False)
        subset_results['FinGAT-NT'] = results
        print(f"  MRR@3={results['MRR@3']['mean']:.4f} ± {results['MRR@3']['std']:.4f}")

        all_results[subset_name] = subset_results

    save_results(all_results, "results/eq2_no_sector.json")
    return all_results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=30)
    args = parser.parse_args()

    config = Config()
    config.num_runs = args.runs
    config.epochs = args.epochs
    run_eq2(config)

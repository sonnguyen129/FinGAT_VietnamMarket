"""EQ4: Hyperparameter Analysis — Figure 4(a-f).

Sweep 3 hyperparameters:
  - week_num in {1, 2, 3, 4}
  - hidden_dim in {8, 16, 32, 64}
  - delta in {0, 1e-4, 1e-3, 1e-2, 1e-1, 1}
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import copy

from config import Config
from utils import save_results
from train import train_multi_run
from model.fingat import FinGAT
from model.loss import FinGATLoss, RankingOnlyLoss
from data.dataset import load_dataset
from data.graph_builder import load_edges


def make_fingat_factory(intra_edges, inter_edges, sector_assignments, num_sectors):
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
            device=cfg.device,
        )
    return factory


def run_eq4(config: Config = None, num_runs: int = None):
    """Run EQ4 hyperparameter sweep."""
    if config is None:
        config = Config()
    if num_runs is not None:
        config.num_runs = num_runs

    print("=" * 60)
    print("EQ4: Hyperparameter Analysis (Figure 4)")
    print("=" * 60)

    data, tickers = load_dataset()
    intra_edges, inter_edges, sector_assignments = load_edges()
    num_sectors = int(sector_assignments.max().item()) + 1

    all_results = {}

    # Experiment 4a+4b: Number of weeks
    print("\n--- Sweep: week_num ---")
    week_results = {}
    for wn in [1, 2, 3, 4]:
        print(f"\n  week_num={wn}")
        cfg = copy.deepcopy(config)
        cfg.week_num = wn

        # Need to rebuild data for different week_num
        # For simplicity, we reuse the 3-week data but only use first wn weeks
        # This works because we stored x1, x2, x3 separately
        if wn <= 3:  # Can reuse existing data
            factory = make_fingat_factory(
                intra_edges, inter_edges, sector_assignments, num_sectors)
            criterion_factory = lambda c: FinGATLoss(
                alpha=c.alpha_reg, beta=c.beta_cls, gamma=c.gamma_rank,
                max_pairs=c.max_pairs)
            results = train_multi_run(
                factory, data, cfg, criterion_factory,
                name=f'week_{wn}', num_runs=cfg.num_runs, verbose=False)
            week_results[wn] = results
        else:
            print(f"  Skipping week_num={wn} (need 4-week data)")

    all_results['weeks'] = week_results

    # Experiment 4c+4d: Hidden dimension
    print("\n--- Sweep: hidden_dim ---")
    dim_results = {}
    for hd in [8, 16, 32, 64]:
        print(f"\n  hidden_dim={hd}")
        cfg = copy.deepcopy(config)
        cfg.hidden_dim = hd

        factory = make_fingat_factory(
            intra_edges, inter_edges, sector_assignments, num_sectors)
        criterion_factory = lambda c: FinGATLoss(
            alpha=c.alpha_reg, beta=c.beta_cls, gamma=c.gamma_rank,
            max_pairs=c.max_pairs)
        results = train_multi_run(
            factory, data, cfg, criterion_factory,
            name=f'dim_{hd}', num_runs=cfg.num_runs, verbose=False)
        dim_results[hd] = results

    all_results['hidden_dim'] = dim_results

    # Experiment 4e+4f: Delta (loss balance)
    print("\n--- Sweep: delta ---")
    delta_results = {}
    for delta in [0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]:
        print(f"\n  delta={delta}")
        cfg = copy.deepcopy(config)
        cfg.beta_cls = delta

        factory = make_fingat_factory(
            intra_edges, inter_edges, sector_assignments, num_sectors)
        if delta == 0:
            criterion_factory = lambda c: RankingOnlyLoss(
                gamma=c.gamma_rank, max_pairs=c.max_pairs)
            # Need model without MTL
            def factory_no_mtl(c):
                return FinGAT(
                    input_dim=c.num_features, hidden_dim=c.hidden_dim,
                    intra_edge_index=intra_edges.to(c.device),
                    inter_edge_index=inter_edges.to(c.device),
                    sector_assignments=sector_assignments.to(c.device),
                    num_sectors=num_sectors, week_num=c.week_num,
                    use_mtl=False, device=c.device)
            factory = factory_no_mtl
        else:
            criterion_factory = lambda c: FinGATLoss(
                alpha=c.alpha_reg, beta=c.beta_cls, gamma=c.gamma_rank,
                max_pairs=c.max_pairs)

        results = train_multi_run(
            factory, data, cfg, criterion_factory,
            name=f'delta_{delta}', num_runs=cfg.num_runs, verbose=False)
        delta_results[str(delta)] = results

    all_results['delta'] = delta_results

    save_results(all_results, "results/eq4_hyperparam.json")
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
    run_eq4(config)

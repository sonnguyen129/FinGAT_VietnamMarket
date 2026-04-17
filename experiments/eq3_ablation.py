"""EQ3: Ablation Study — Table 3.

5 model variants:
  1. Full model (all components)
  2. w/o intra-sector GAT
  3. w/o inter-sector GAT
  4. w/o MTL (ranking loss only)
  5. w/ MSE (replace BCE with MSE)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from utils import save_results
from train import train_multi_run
from model.fingat import FinGAT
from model.loss import FinGATLoss, RankingOnlyLoss
from data.dataset import load_dataset
from data.graph_builder import load_edges


def make_fingat_variant(intra_edges, inter_edges, sector_assignments,
                        num_sectors, use_intra=True, use_inter=True,
                        use_mtl=True, use_mse=False):
    """Factory for FinGAT variant."""
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
            use_intra=use_intra,
            use_inter=use_inter,
            use_mtl=use_mtl,
            use_mse=use_mse,
            device=cfg.device,
        )
    return factory


def run_eq3(config: Config = None, num_runs: int = None):
    """Run EQ3 ablation experiment."""
    if config is None:
        config = Config()
    if num_runs is not None:
        config.num_runs = num_runs

    print("=" * 60)
    print("EQ3: Ablation Study (Table 3)")
    print("=" * 60)

    data, tickers = load_dataset()
    intra_edges, inter_edges, sector_assignments = load_edges()
    num_sectors = int(sector_assignments.max().item()) + 1

    variants = [
        ("Full model", dict(use_intra=True, use_inter=True, use_mtl=True, use_mse=False)),
        ("w/o intra", dict(use_intra=False, use_inter=True, use_mtl=True, use_mse=False)),
        ("w/o inter", dict(use_intra=True, use_inter=False, use_mtl=True, use_mse=False)),
        ("w/o MTL", dict(use_intra=True, use_inter=True, use_mtl=False, use_mse=False)),
        ("w/ MSE", dict(use_intra=True, use_inter=True, use_mtl=True, use_mse=True)),
    ]

    all_results = {}
    for vname, vkwargs in variants:
        print(f"\n--- {vname} ---")

        model_factory = make_fingat_variant(
            intra_edges, inter_edges, sector_assignments, num_sectors,
            **vkwargs)

        # Use appropriate loss
        if not vkwargs['use_mtl']:
            criterion_factory = lambda cfg: RankingOnlyLoss(
                gamma=cfg.gamma_rank, max_pairs=cfg.max_pairs)
        else:
            criterion_factory = lambda cfg: FinGATLoss(
                alpha=cfg.alpha_reg, beta=cfg.beta_cls, gamma=cfg.gamma_rank,
                use_mse=vkwargs['use_mse'], max_pairs=cfg.max_pairs)

        results = train_multi_run(
            model_factory, data, config, criterion_factory,
            name=vname, num_runs=config.num_runs)
        all_results[vname] = results

    # Print Table 3
    print("\n" + "=" * 100)
    print("TABLE 3: Ablation Study on HOSE")
    print("=" * 100)
    header = f"{'Variant':<14}"
    for k in config.top_k_values:
        header += f" | MRR@{k:<3} Prec@{k:<3}"
    header += " | ACC"
    print(header)
    print("-" * 100)
    for vname, _ in variants:
        r = all_results[vname]
        row = f"{vname:<14}"
        for k in config.top_k_values:
            row += f" | {r[f'MRR@{k}']['mean']:.4f}  {r[f'Precision@{k}']['mean']:.4f}"
        row += f" | {r['ACC']['mean']:.4f}"
        print(row)
    print("=" * 100)

    save_results(all_results, "results/eq3_ablation.json")
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
    run_eq3(config)

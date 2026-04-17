"""EQ5: Attention Visualization — Figures 5, 6, 7.

Extract and visualize attention weights from trained FinGAT model:
  - Figure 5: Intra-sector attention heatmaps (2 sectors)
  - Figure 6: Inter-sector attention heatmap
  - Figure 7: Attention weight distributions and variances
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from config import Config
from utils import set_seed, save_results
from train import train_model
from model.fingat import FinGAT
from model.loss import FinGATLoss
from data.dataset import load_dataset
from data.graph_builder import load_edges, build_sector_assignments
from data.sector_map import load_sector_mapping, clean_sectors, get_sector_list


def run_eq5(config: Config = None):
    """Run EQ5 attention visualization experiment."""
    if config is None:
        config = Config()

    print("=" * 60)
    print("EQ5: Attention Visualization (Figures 5-7)")
    print("=" * 60)

    # Load data and model
    data, tickers = load_dataset()
    intra_edges, inter_edges, sector_assignments = load_edges()
    num_sectors = int(sector_assignments.max().item()) + 1

    # Load sector mapping for labels
    mapping = load_sector_mapping(config.companies_file)
    mapping = {t: mapping[t] for t in tickers if t in mapping}
    mapping = clean_sectors(mapping)
    sector_list = get_sector_list(mapping)

    # Train (or load) best model
    set_seed(config.seed_base)
    model = FinGAT(
        input_dim=config.num_features,
        hidden_dim=config.hidden_dim,
        intra_edge_index=intra_edges.to(config.device),
        inter_edge_index=inter_edges.to(config.device),
        sector_assignments=sector_assignments.to(config.device),
        num_sectors=num_sectors,
        week_num=config.week_num,
        device=config.device,
    )

    criterion = FinGATLoss(
        alpha=config.alpha_reg, beta=config.beta_cls, gamma=config.gamma_rank,
        max_pairs=config.max_pairs)

    print("\nTraining model for attention extraction...")
    result = train_model(model, data, config, criterion, name='FinGAT-viz')
    model.load_state_dict(result['model_state'])
    model = model.to(config.device)

    # Extract attention weights from test samples
    print("\nExtracting attention weights from test set...")
    test_data = data['test']
    num_test = test_data['x1'].shape[0]
    week_num = config.week_num

    all_intra_attns = []
    all_inter_attns = []

    for t in range(min(num_test, 50)):  # Use first 50 test samples
        weekly = [
            torch.tensor(test_data[f'x{w+1}'][t],
                         dtype=torch.float32).to(config.device)
            for w in range(week_num)
        ]
        attn_dict = model.extract_attention(weekly)
        if attn_dict['intra_attention']:
            all_intra_attns.append(
                [a.numpy() for a in attn_dict['intra_attention']])
        if attn_dict['inter_attention'] is not None:
            all_inter_attns.append(attn_dict['inter_attention'].numpy())

    # Build sector-stock mapping for heatmap labels
    sector_stocks = {}
    for i, t in enumerate(tickers):
        if t in mapping:
            s = mapping[t]['industry']
            sector_stocks.setdefault(s, []).append((i, t))

    results = {
        'intra_attns': all_intra_attns,
        'inter_attns': all_inter_attns,
        'tickers': tickers,
        'sector_list': sector_list,
        'sector_stocks': {s: [(i, t) for i, t in v]
                          for s, v in sector_stocks.items()},
        'intra_edge_index': intra_edges.numpy().tolist(),
        'inter_edge_index': inter_edges.numpy().tolist(),
    }

    # Generate plots
    try:
        from visualize import (plot_intra_sector_heatmap,
                                plot_inter_sector_heatmap,
                                plot_attention_distribution,
                                plot_attention_variance)

        # Figure 5: Intra-sector heatmaps
        sorted_sectors = sorted(sector_stocks.items(),
                                key=lambda x: -len(x[1]))
        for fig_idx, (sector, stocks) in enumerate(sorted_sectors[:2]):
            stock_indices = [i for i, _ in stocks]
            stock_names = [t for _, t in stocks]
            plot_intra_sector_heatmap(
                all_intra_attns, intra_edges.numpy(),
                stock_indices, stock_names, sector,
                f"results/eq5_fig5{'ab'[fig_idx]}_{sector.replace(' ', '_')}.png")

        # Figure 6: Inter-sector heatmap
        if all_inter_attns:
            plot_inter_sector_heatmap(
                all_inter_attns, inter_edges.numpy(),
                sector_list, num_sectors,
                "results/eq5_fig6_inter_sector.png")

        # Figure 7: Distributions
        if all_intra_attns and all_inter_attns:
            plot_attention_distribution(
                all_intra_attns, all_inter_attns,
                "results/eq5_fig7a_attn_dist.png")
            plot_attention_variance(
                all_intra_attns, all_inter_attns,
                "results/eq5_fig7b_attn_var.png")

        print("Figures saved to results/eq5_*.png")
    except ImportError:
        print("Warning: visualize.py not found, skipping plots")

    save_results(
        {'sector_list': sector_list,
         'num_test_samples': len(all_intra_attns)},
        "results/eq5_attention_viz.json")
    return results


if __name__ == '__main__':
    config = Config()
    config.epochs = 30
    run_eq5(config)

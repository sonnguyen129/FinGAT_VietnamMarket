"""Visualization functions for FinGAT experiments.

7 plotting functions:
  1. plot_intra_sector_heatmap — Figure 5
  2. plot_inter_sector_heatmap — Figure 6
  3. plot_attention_distribution — Figure 7a
  4. plot_attention_variance — Figure 7b
  5. generate_table — LaTeX/markdown table (Table 2, 3)
  6. plot_eq2_barchart — grouped bar chart (Figure 3)
  7. plot_eq4_line_charts — line charts (Figure 4)
"""

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Style settings
plt.rcParams.update({
    'font.size': 12,
    'figure.dpi': 300,
    'figure.figsize': (10, 6),
    'axes.grid': True,
    'grid.alpha': 0.3,
})

COLORS = {
    'MLP': '#808080',
    'GRU': '#1f77b4',
    'GRU+Att': '#2ca02c',
    'FineNet': '#ff7f0e',
    'RankLSTM': '#d62728',
    'FinGAT': '#9467bd',
    'FinGAT-NT': '#9467bd',
    'Momentum': '#8c564b',
    'Random': '#e377c2',
}


def _save_fig(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches='tight', dpi=300)
    # Also save PDF
    fig.savefig(path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_intra_sector_heatmap(all_intra_attns, intra_edge_index,
                                stock_indices, stock_names, sector_name,
                                save_path):
    """Figure 5: Intra-sector attention heatmap for one sector."""
    try:
        import seaborn as sns
    except ImportError:
        print("Warning: seaborn not installed, skipping heatmap")
        return

    n = len(stock_indices)
    idx_set = set(stock_indices)
    idx_map = {idx: i for i, idx in enumerate(stock_indices)}

    # Average attention weights across test samples and weeks
    attn_matrix = np.zeros((n, n))
    count_matrix = np.zeros((n, n))

    edge_src = intra_edge_index[0]
    edge_dst = intra_edge_index[1]

    for sample_attns in all_intra_attns:
        for week_attn in sample_attns:
            for edge_idx in range(len(edge_src)):
                s, d = int(edge_src[edge_idx]), int(edge_dst[edge_idx])
                if s in idx_set and d in idx_set:
                    i, j = idx_map[s], idx_map[d]
                    if edge_idx < len(week_attn):
                        attn_matrix[i, j] += float(week_attn[edge_idx])
                        count_matrix[i, j] += 1

    # Average
    mask = count_matrix > 0
    attn_matrix[mask] /= count_matrix[mask]

    fig, ax = plt.subplots(figsize=(max(8, n * 0.5), max(6, n * 0.4)))
    sns.heatmap(attn_matrix, xticklabels=stock_names, yticklabels=stock_names,
                cmap='Blues', ax=ax, square=True)
    ax.set_title(f'Intra-sector Attention: {sector_name}')
    ax.set_xlabel('Target Stock')
    ax.set_ylabel('Source Stock')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    _save_fig(fig, save_path)


def plot_inter_sector_heatmap(all_inter_attns, inter_edge_index,
                                sector_names, num_sectors, save_path):
    """Figure 6: Inter-sector attention heatmap."""
    try:
        import seaborn as sns
    except ImportError:
        print("Warning: seaborn not installed")
        return

    attn_matrix = np.zeros((num_sectors, num_sectors))
    count_matrix = np.zeros((num_sectors, num_sectors))

    edge_src = inter_edge_index[0]
    edge_dst = inter_edge_index[1]

    for attn in all_inter_attns:
        for edge_idx in range(len(edge_src)):
            s, d = int(edge_src[edge_idx]), int(edge_dst[edge_idx])
            if edge_idx < len(attn):
                attn_matrix[s, d] += float(attn[edge_idx])
                count_matrix[s, d] += 1

    mask = count_matrix > 0
    attn_matrix[mask] /= count_matrix[mask]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(attn_matrix, xticklabels=sector_names,
                yticklabels=sector_names, cmap='Blues', ax=ax,
                annot=True, fmt='.3f', square=True)
    ax.set_title('Inter-sector Attention Weights')
    ax.set_xlabel('Target Sector')
    ax.set_ylabel('Source Sector')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    _save_fig(fig, save_path)


def plot_attention_distribution(all_intra_attns, all_inter_attns, save_path):
    """Figure 7a: Distribution of attention weights."""
    intra_weights = []
    for sample in all_intra_attns:
        for week in sample:
            intra_weights.extend(week.flatten().tolist())

    inter_weights = []
    for attn in all_inter_attns:
        inter_weights.extend(attn.flatten().tolist())

    fig, ax = plt.subplots()
    ax.hist(intra_weights, bins=50, alpha=0.6, color='orange',
            label='Intra-sector', density=True)
    ax.hist(inter_weights, bins=50, alpha=0.6, color='blue',
            label='Inter-sector', density=True)
    ax.set_xlabel('Attention Score')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of Attention Weights')
    ax.legend()
    _save_fig(fig, save_path)


def plot_attention_variance(all_intra_attns, all_inter_attns, save_path):
    """Figure 7b: Variance of attention weights across test instances."""
    # Stack attention weights per edge across samples
    if not all_intra_attns or not all_inter_attns:
        return

    # Intra: compute variance per edge across samples
    n_edges_intra = len(all_intra_attns[0][0]) if all_intra_attns[0] else 0
    if n_edges_intra > 0:
        intra_per_edge = np.zeros((len(all_intra_attns), n_edges_intra))
        for i, sample in enumerate(all_intra_attns):
            # Use first week's attention
            intra_per_edge[i] = sample[0].flatten()[:n_edges_intra]
        intra_vars = np.var(intra_per_edge, axis=0)
    else:
        intra_vars = np.array([])

    n_edges_inter = len(all_inter_attns[0]) if all_inter_attns else 0
    if n_edges_inter > 0:
        inter_per_edge = np.zeros((len(all_inter_attns), n_edges_inter))
        for i, attn in enumerate(all_inter_attns):
            inter_per_edge[i] = attn.flatten()[:n_edges_inter]
        inter_vars = np.var(inter_per_edge, axis=0)
    else:
        inter_vars = np.array([])

    fig, ax = plt.subplots()
    if len(intra_vars) > 0:
        ax.hist(intra_vars, bins=50, alpha=0.6, color='orange',
                label='Intra-sector', density=True)
    if len(inter_vars) > 0:
        ax.hist(inter_vars, bins=50, alpha=0.6, color='blue',
                label='Inter-sector', density=True)
    ax.set_xlabel('Attention Weight Variance')
    ax.set_ylabel('Density')
    ax.set_title('Variance of Attention Weights Across Test Instances')
    ax.legend()
    _save_fig(fig, save_path)


def generate_table(results: dict, title: str, save_path: str,
                   top_k_values=(5, 10, 20)):
    """Generate formatted markdown/CSV table."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    lines = [f"# {title}\n"]
    header = "| Model |"
    for k in top_k_values:
        header += f" MRR@{k} | Prec@{k} |"
    header += " ACC |"
    lines.append(header)
    lines.append("|" + "---|" * (1 + 2 * len(top_k_values) + 1))

    for model, r in results.items():
        row = f"| {model} |"
        for k in top_k_values:
            mrr = r.get(f'MRR@{k}', {}).get('mean', 0)
            prec = r.get(f'Precision@{k}', {}).get('mean', 0)
            row += f" {mrr:.4f} | {prec:.4f} |"
        acc = r.get('ACC', {}).get('mean', 0)
        row += f" {acc:.4f} |"
        lines.append(row)

    with open(save_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  Table saved: {save_path}")


def plot_eq2_barchart(results: dict, save_path: str):
    """Figure 3: Grouped bar chart for EQ2 (5 subsets, 6 models, MRR@3)."""
    subsets = list(results.keys())
    models = ['MLP', 'GRU', 'GRU+Att', 'FineNet', 'RankLSTM', 'FinGAT-NT']

    x = np.arange(len(subsets))
    width = 0.12
    n_models = len(models)

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, model in enumerate(models):
        values = []
        errors = []
        for subset in subsets:
            if model in results[subset]:
                m = results[subset][model].get('MRR@3', {})
                values.append(m.get('mean', 0))
                errors.append(m.get('std', 0))
            else:
                values.append(0)
                errors.append(0)

        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, yerr=errors,
                       label=model, color=COLORS.get(model, '#333'),
                       capsize=2)

    ax.set_xlabel('Stock Subset')
    ax.set_ylabel('MRR@3')
    ax.set_title('EQ2: FinGAT-NT vs Baselines on Stock Subsets')
    ax.set_xticks(x)
    ax.set_xticklabels(subsets, rotation=15)
    ax.legend(loc='upper right')
    ax.set_ylim(bottom=0)
    _save_fig(fig, save_path)


def plot_eq4_line_charts(results: dict, param_name: str,
                          param_values: list, metric_name: str,
                          top_k_values=(5, 10, 20), save_path: str = None):
    """Figure 4: Line chart for hyperparameter sweep."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for k in top_k_values:
        metric_key = f'{metric_name}@{k}'
        values = []
        for pv in param_values:
            pv_key = str(pv) if isinstance(pv, float) else pv
            if pv_key in results and metric_key in results[pv_key]:
                values.append(results[pv_key][metric_key]['mean'])
            elif pv in results and metric_key in results[pv]:
                values.append(results[pv][metric_key]['mean'])
            else:
                values.append(0)
        ax.plot(range(len(param_values)), values, 'o-', label=f'K={k}')

    ax.set_xlabel(param_name)
    ax.set_ylabel(metric_name)
    ax.set_title(f'{metric_name} vs {param_name}')
    ax.set_xticks(range(len(param_values)))
    ax.set_xticklabels([str(v) for v in param_values])
    ax.legend()

    if save_path:
        _save_fig(fig, save_path)
    else:
        plt.show()

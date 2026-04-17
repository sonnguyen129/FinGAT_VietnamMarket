"""Unified experiment runner — runs all EQ1-EQ5 sequentially."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from datetime import datetime

from config import Config
from utils import Timer


def run_all(config: Config):
    """Run all 5 experiment groups."""
    os.makedirs("results", exist_ok=True)

    from experiments.eq1_main_results import run_eq1
    from experiments.eq2_no_sector import run_eq2
    from experiments.eq3_ablation import run_eq3
    from experiments.eq4_hyperparam import run_eq4
    from experiments.eq5_attention_viz import run_eq5

    results = {}

    with Timer("EQ1: Main Results"):
        results['eq1'] = run_eq1(config)

    with Timer("EQ2: No Sector Info"):
        results['eq2'] = run_eq2(config)

    with Timer("EQ3: Ablation Study"):
        results['eq3'] = run_eq3(config)

    with Timer("EQ4: Hyperparameter Analysis"):
        results['eq4'] = run_eq4(config)

    with Timer("EQ5: Attention Visualization"):
        results['eq5'] = run_eq5(config)

    # Generate summary report
    generate_report(results, config)
    print("\n" + "=" * 60)
    print("All experiments complete! Results in results/")
    print("=" * 60)


def generate_report(results: dict, config: Config):
    """Generate a markdown summary report."""
    report = f"""# FinGAT Experiment Results on HOSE

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Configuration: {config.num_runs} runs, {config.epochs} epochs

## EQ1: Main Results (Table 2)

"""
    # Add EQ1 table
    if 'eq1' in results:
        eq1 = results['eq1']
        report += "| Model | MRR@5 | MRR@10 | MRR@20 | Prec@5 | Prec@10 | Prec@20 | ACC |\n"
        report += "|-------|-------|--------|--------|--------|---------|---------|-----|\n"
        for model in ['Random', 'Momentum', 'MLP', 'GRU', 'GRU+Att',
                       'FineNet', 'RankLSTM', 'FinGAT']:
            if model in eq1:
                r = eq1[model]
                row = f"| {model} "
                for k in [5, 10, 20]:
                    row += f"| {r.get(f'MRR@{k}', {}).get('mean', 0):.4f} "
                for k in [5, 10, 20]:
                    row += f"| {r.get(f'Precision@{k}', {}).get('mean', 0):.4f} "
                row += f"| {r.get('ACC', {}).get('mean', 0):.4f} |"
                report += row + "\n"

    report += "\n## EQ3: Ablation Study (Table 3)\n\n"
    if 'eq3' in results:
        eq3 = results['eq3']
        report += "| Variant | MRR@5 | Prec@5 | ACC |\n"
        report += "|---------|-------|--------|-----|\n"
        for vname in ['Full model', 'w/o intra', 'w/o inter', 'w/o MTL', 'w/ MSE']:
            if vname in eq3:
                r = eq3[vname]
                report += (f"| {vname} "
                           f"| {r.get('MRR@5', {}).get('mean', 0):.4f} "
                           f"| {r.get('Precision@5', {}).get('mean', 0):.4f} "
                           f"| {r.get('ACC', {}).get('mean', 0):.4f} |\n")

    report += "\n## Key Findings\n\n"
    report += "- TODO: Fill in after experiments complete\n"

    with open("results/full_report.md", "w") as f:
        f.write(report)
    print(f"\nReport saved to results/full_report.md")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run all FinGAT experiments")
    parser.add_argument('--runs', type=int, default=3,
                        help='Number of runs per experiment (default: 3, paper: 10)')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Training epochs per run')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu)')
    parser.add_argument('--eq', type=str, default='all',
                        help='Which experiment to run (1-5 or all)')
    args = parser.parse_args()

    config = Config()
    config.num_runs = args.runs
    config.epochs = args.epochs
    if args.device:
        config.device = args.device

    if args.eq == 'all':
        run_all(config)
    else:
        from experiments.eq1_main_results import run_eq1
        from experiments.eq2_no_sector import run_eq2
        from experiments.eq3_ablation import run_eq3
        from experiments.eq4_hyperparam import run_eq4
        from experiments.eq5_attention_viz import run_eq5

        eq_map = {'1': run_eq1, '2': run_eq2, '3': run_eq3,
                  '4': run_eq4, '5': run_eq5}
        if args.eq in eq_map:
            eq_map[args.eq](config)
        else:
            print(f"Unknown experiment: {args.eq}")

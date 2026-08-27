#!/usr/bin/env python3
"""
Standalone plotting script for SciWorld evaluation results.

Can be used to regenerate plots from existing eval_results_*.jsonl files
without re-running the evaluation.

Usage:
    # Plot from a single results file
    python plot_eval_results.py --results-file path/to/eval_results_20260127.jsonl

    # Plot from the most recent results in a directory
    python plot_eval_results.py --results-dir path/to/eval_results/

    # Compare multiple configurations
    python plot_eval_results.py --compare \
        --results-files config1/eval_results.jsonl config2/eval_results.jsonl \
        --config-names "Baseline" "Fine-grained"
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
from datetime import datetime
import re

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


def load_results_from_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """Load evaluation results from a JSONL file."""
    results = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def find_latest_results(results_dir: str) -> Optional[str]:
    """Find the most recent eval_results_*.jsonl file in a directory."""
    results_path = Path(results_dir)
    if not results_path.exists():
        return None

    jsonl_files = list(results_path.glob("eval_results_*.jsonl"))
    if not jsonl_files:
        return None

    # Sort by modification time, newest first
    jsonl_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return str(jsonl_files[0])


def setup_academic_style():
    """Configure matplotlib for academic publication-quality plots."""
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    plt.style.use('seaborn-v0_8-whitegrid')

    ACADEMIC_COLORS = {
        'grounding': '#0072B2',
        'action': '#D55E00',
        'temporal': '#009E73',
        'aggregate': '#CC79A7',
    }

    mpl.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
        'font.size': 11,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 13,
        'axes.linewidth': 1.0,
        'grid.linewidth': 0.5,
        'lines.linewidth': 1.5,
        'lines.markersize': 6,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
    })

    return ACADEMIC_COLORS


def extract_trajectory_id(sample_id: str) -> str:
    """Extract trajectory ID from sample ID."""
    parts = sample_id.rsplit('_step', 1)
    return parts[0] if len(parts) > 1 else sample_id


def extract_step_number(sample_id: str) -> int:
    """Extract step number from sample ID."""
    match = re.search(r'_step(\d+)$', sample_id)
    return int(match.group(1)) if match else 0


def group_results_by_trajectory(results: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group evaluation results by trajectory ID."""
    grouped = defaultdict(list)
    for r in results:
        traj_id = extract_trajectory_id(r.get('id', ''))
        grouped[traj_id].append(r)

    for traj_id in grouped:
        grouped[traj_id].sort(key=lambda x: extract_step_number(x.get('id', '')))

    return dict(grouped)


def compute_per_turn_statistics(
    grouped_results: Dict[str, List[Dict[str, Any]]],
    dimensions: List[str] = ['grounding', 'action', 'temporal'],
) -> Tuple[Dict[str, List[float]], Dict[str, List[float]], List[int], int]:
    """Compute per-turn mean and std error across trajectories."""
    import numpy as np

    max_turns = max(len(results) for results in grouped_results.values())
    scores_per_turn = {dim: [[] for _ in range(max_turns)] for dim in dimensions}

    for traj_id, results in grouped_results.items():
        for turn_idx, result in enumerate(results):
            extra = result.get('extra') or {}
            scalar_scores = extra.get('scalar_scores') or {}

            for dim in dimensions:
                if dim in scalar_scores and scalar_scores[dim] is not None:
                    scores_per_turn[dim][turn_idx].append(scalar_scores[dim])

    means = {dim: [] for dim in dimensions}
    stderrs = {dim: [] for dim in dimensions}
    turn_counts = []

    for turn_idx in range(max_turns):
        count = 0
        for dim in dimensions:
            scores = scores_per_turn[dim][turn_idx]
            if scores:
                means[dim].append(np.mean(scores))
                stderrs[dim].append(np.std(scores, ddof=1) / np.sqrt(len(scores)) if len(scores) > 1 else 0.0)
                count = max(count, len(scores))
            else:
                means[dim].append(np.nan)
                stderrs[dim].append(np.nan)
        turn_counts.append(count)

    return means, stderrs, turn_counts, max_turns


def plot_scores_over_turns(
    results: List[Dict[str, Any]],
    output_dir: str,
    config_name: Optional[str] = None,
    timestamp: Optional[str] = None,
    show_individual: bool = False,
) -> List[str]:
    """Create academic-quality plots of per-dimension scores over turns."""
    import matplotlib.pyplot as plt
    import numpy as np

    colors = setup_academic_style()
    grouped = group_results_by_trajectory(results)
    n_trajectories = len(grouped)

    plots_dir = Path(output_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    config_str = f"_{config_name}" if config_name else ""

    dimensions = ['grounding', 'action', 'temporal']
    dim_labels = {
        'grounding': 'Observation Grounding',
        'action': 'Action Coherence',
        'temporal': 'Temporal Consistency',
    }

    plot_paths = []

    if n_trajectories == 1:
        # Single trajectory plot
        traj_id = list(grouped.keys())[0]
        traj_results = grouped[traj_id]

        fig, ax = plt.subplots(figsize=(8, 5))
        turns = list(range(1, len(traj_results) + 1))

        for dim in dimensions:
            scores = []
            for r in traj_results:
                extra = r.get('extra') or {}
                scalar_scores = extra.get('scalar_scores') or {}
                scores.append(scalar_scores.get(dim, np.nan))

            ax.plot(turns, scores, 'o-', color=colors[dim], label=dim_labels[dim],
                   markersize=5, linewidth=1.5)

        ax.set_xlabel('Turn', fontweight='medium')
        ax.set_ylabel('Score', fontweight='medium')
        ax.set_title(f'Per-Dimension Scores Over Turns\n(Single Trajectory: {traj_id})',
                    fontweight='bold')
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(0.5, len(turns) + 0.5)
        ax.legend(loc='lower left', framealpha=0.9)
        ax.grid(True, alpha=0.3)

        plot_path = plots_dir / f"scores_over_turns{config_str}_{timestamp}.pdf"
        plt.savefig(plot_path, format='pdf', bbox_inches='tight')
        plt.savefig(plot_path.with_suffix('.png'), format='png', bbox_inches='tight')
        plt.close()
        plot_paths.append(str(plot_path))

    else:
        # Multiple trajectories: mean ± stderr
        means, stderrs, turn_counts, max_turns = compute_per_turn_statistics(grouped, dimensions)

        fig, ax = plt.subplots(figsize=(10, 6))
        turns = np.arange(1, max_turns + 1)

        for dim in dimensions:
            mean_vals = np.array(means[dim])
            stderr_vals = np.array(stderrs[dim])

            line, = ax.plot(turns, mean_vals, 'o-', color=colors[dim],
                           label=dim_labels[dim], markersize=5, linewidth=1.5)

            valid_mask = ~np.isnan(mean_vals)
            ax.fill_between(
                turns[valid_mask],
                (mean_vals - stderr_vals)[valid_mask],
                (mean_vals + stderr_vals)[valid_mask],
                color=colors[dim], alpha=0.2
            )

        ax.set_xlabel('Turn', fontweight='medium')
        ax.set_ylabel('Score (Mean ± SE)', fontweight='medium')
        ax.set_title(f'Per-Dimension Scores Over Turns\n(Averaged over {n_trajectories} trajectories)',
                    fontweight='bold')
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(0.5, max_turns + 0.5)
        ax.legend(loc='lower left', framealpha=0.9)
        ax.grid(True, alpha=0.3)

        # Secondary x-axis for trajectory counts
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        tick_interval = max(1, max_turns // 10)
        tick_positions = turns[::tick_interval]
        ax2.set_xticks(tick_positions)
        ax2.set_xticklabels([f'n={turn_counts[i-1]}' for i in tick_positions],
                           fontsize=8, color='gray')
        ax2.set_xlabel('Trajectories at turn', fontsize=9, color='gray')

        plot_path = plots_dir / f"scores_over_turns_avg{config_str}_{timestamp}.pdf"
        plt.savefig(plot_path, format='pdf', bbox_inches='tight')
        plt.savefig(plot_path.with_suffix('.png'), format='png', bbox_inches='tight')
        plt.close()
        plot_paths.append(str(plot_path))

        # Individual trajectory plots
        if show_individual and n_trajectories <= 6:
            n_rows = (n_trajectories + 2) // 3
            n_cols = min(3, n_trajectories)
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5*n_cols, 4*n_rows), squeeze=False)
            axes = axes.flatten()

            for idx, (traj_id, traj_results) in enumerate(grouped.items()):
                ax = axes[idx]
                turns = list(range(1, len(traj_results) + 1))

                for dim in dimensions:
                    scores = []
                    for r in traj_results:
                        extra = r.get('extra') or {}
                        scalar_scores = extra.get('scalar_scores') or {}
                        scores.append(scalar_scores.get(dim, np.nan))

                    ax.plot(turns, scores, 'o-', color=colors[dim], label=dim_labels[dim],
                           markersize=3, linewidth=1)

                ax.set_xlabel('Turn', fontsize=9)
                ax.set_ylabel('Score', fontsize=9)
                ax.set_title(f'Trajectory: {traj_id}', fontsize=10)
                ax.set_ylim(-0.05, 1.05)
                ax.grid(True, alpha=0.3)
                if idx == 0:
                    ax.legend(loc='lower left', fontsize=8, framealpha=0.9)

            for idx in range(n_trajectories, len(axes)):
                axes[idx].set_visible(False)

            plt.suptitle('Individual Trajectory Scores', fontweight='bold', y=1.02)
            plt.tight_layout()

            indiv_path = plots_dir / f"scores_individual{config_str}_{timestamp}.pdf"
            plt.savefig(indiv_path, format='pdf', bbox_inches='tight')
            plt.savefig(indiv_path.with_suffix('.png'), format='png', bbox_inches='tight')
            plt.close()
            plot_paths.append(str(indiv_path))

    return plot_paths


def plot_dimension_comparison(
    results: List[Dict[str, Any]],
    output_dir: str,
    config_name: Optional[str] = None,
    timestamp: Optional[str] = None,
) -> str:
    """Create bar chart comparing average scores across dimensions."""
    import matplotlib.pyplot as plt
    import numpy as np

    colors = setup_academic_style()

    dimensions = ['grounding', 'action', 'temporal']
    dim_labels = ['Observation\nGrounding', 'Action\nCoherence', 'Temporal\nConsistency']

    scores_by_dim = {dim: [] for dim in dimensions}

    for r in results:
        extra = r.get('extra') or {}
        scalar_scores = extra.get('scalar_scores') or {}
        for dim in dimensions:
            if dim in scalar_scores and scalar_scores[dim] is not None:
                scores_by_dim[dim].append(scalar_scores[dim])

    means = [np.mean(scores_by_dim[dim]) if scores_by_dim[dim] else 0 for dim in dimensions]
    stderrs = [np.std(scores_by_dim[dim], ddof=1) / np.sqrt(len(scores_by_dim[dim]))
               if len(scores_by_dim[dim]) > 1 else 0 for dim in dimensions]

    fig, ax = plt.subplots(figsize=(6, 4))

    x = np.arange(len(dimensions))
    bar_colors = [colors[dim] for dim in dimensions]

    bars = ax.bar(x, means, yerr=stderrs, capsize=5, color=bar_colors,
                  edgecolor='black', linewidth=0.5, alpha=0.85)

    ax.set_ylabel('Score (Mean ± SE)', fontweight='medium')
    title = 'Average Scores by Evaluation Dimension'
    if config_name:
        title += f'\n({config_name})'
    ax.set_title(title, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(dim_labels)
    ax.set_ylim(0, 1.1)
    ax.grid(True, axis='y', alpha=0.3)

    for bar, mean, stderr in zip(bars, means, stderrs):
        height = bar.get_height()
        ax.annotate(f'{mean:.2f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=10, fontweight='medium')

    plots_dir = Path(output_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    config_str = f"_{config_name}" if config_name else ""

    plot_path = plots_dir / f"dimension_comparison{config_str}_{timestamp}.pdf"
    plt.savefig(plot_path, format='pdf', bbox_inches='tight')
    plt.savefig(plot_path.with_suffix('.png'), format='png', bbox_inches='tight')
    plt.close()

    return str(plot_path)


def plot_multi_config_comparison(
    results_list: List[List[Dict[str, Any]]],
    config_names: List[str],
    output_dir: str,
    timestamp: Optional[str] = None,
) -> List[str]:
    """Compare multiple configurations side by side."""
    import matplotlib.pyplot as plt
    import numpy as np

    colors = setup_academic_style()

    # Use a different color palette for configurations
    config_colors = plt.cm.Set2(np.linspace(0, 1, len(config_names)))

    dimensions = ['grounding', 'action', 'temporal']
    dim_labels = ['Grounding', 'Action', 'Temporal']

    # Compute means and stderrs for each config
    all_means = []
    all_stderrs = []

    for results in results_list:
        scores_by_dim = {dim: [] for dim in dimensions}
        for r in results:
            extra = r.get('extra') or {}
            scalar_scores = extra.get('scalar_scores') or {}
            for dim in dimensions:
                if dim in scalar_scores and scalar_scores[dim] is not None:
                    scores_by_dim[dim].append(scalar_scores[dim])

        means = [np.mean(scores_by_dim[dim]) if scores_by_dim[dim] else 0 for dim in dimensions]
        stderrs = [np.std(scores_by_dim[dim], ddof=1) / np.sqrt(len(scores_by_dim[dim]))
                   if len(scores_by_dim[dim]) > 1 else 0 for dim in dimensions]
        all_means.append(means)
        all_stderrs.append(stderrs)

    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(dimensions))
    width = 0.8 / len(config_names)

    for i, (means, stderrs, name) in enumerate(zip(all_means, all_stderrs, config_names)):
        offset = (i - len(config_names)/2 + 0.5) * width
        bars = ax.bar(x + offset, means, width, yerr=stderrs, capsize=3,
                     label=name, color=config_colors[i], edgecolor='black', linewidth=0.5)

    ax.set_ylabel('Score (Mean ± SE)', fontweight='medium')
    ax.set_title('Configuration Comparison by Dimension', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(dim_labels)
    ax.set_ylim(0, 1.15)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, axis='y', alpha=0.3)

    plots_dir = Path(output_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")

    plot_path = plots_dir / f"config_comparison_{timestamp}.pdf"
    plt.savefig(plot_path, format='pdf', bbox_inches='tight')
    plt.savefig(plot_path.with_suffix('.png'), format='png', bbox_inches='tight')
    plt.close()

    return [str(plot_path)]


def main():
    parser = argparse.ArgumentParser(
        description="Generate plots from SciWorld evaluation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot from a single results file
  python plot_eval_results.py --results-file eval_results.jsonl --output-dir ./plots

  # Plot from the most recent results in a directory
  python plot_eval_results.py --results-dir ./eval_results/

  # Compare multiple configurations
  python plot_eval_results.py --compare \\
      --results-files config1/results.jsonl config2/results.jsonl \\
      --config-names "Baseline" "Fine-grained"
        """
    )

    parser.add_argument(
        "--results-file",
        type=str,
        help="Path to eval_results_*.jsonl file",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        help="Directory containing eval_results_*.jsonl files (uses most recent)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for plots (default: same as results)",
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default=None,
        help="Configuration name for plot titles/filenames",
    )
    parser.add_argument(
        "--show-individual",
        action="store_true",
        help="Also plot individual trajectories (if <= 6)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare multiple configurations",
    )
    parser.add_argument(
        "--results-files",
        type=str,
        nargs="+",
        help="Multiple results files for comparison (use with --compare)",
    )
    parser.add_argument(
        "--config-names",
        type=str,
        nargs="+",
        help="Names for each configuration (use with --compare)",
    )

    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.compare:
        # Multi-configuration comparison
        if not args.results_files:
            print("Error: --results-files required for --compare mode")
            return

        results_list = [load_results_from_jsonl(f) for f in args.results_files]
        config_names = args.config_names or [Path(f).stem for f in args.results_files]

        output_dir = args.output_dir or str(Path(args.results_files[0]).parent)

        print(f"Comparing {len(results_list)} configurations...")
        paths = plot_multi_config_comparison(results_list, config_names, output_dir, timestamp)

        for path in paths:
            print(f"Saved: {path}")

    else:
        # Single configuration
        if args.results_file:
            results_path = args.results_file
        elif args.results_dir:
            results_path = find_latest_results(args.results_dir)
            if not results_path:
                print(f"Error: No eval_results_*.jsonl files found in {args.results_dir}")
                return
        else:
            print("Error: Either --results-file or --results-dir required")
            return

        print(f"Loading results from: {results_path}")
        results = load_results_from_jsonl(results_path)
        print(f"Loaded {len(results)} evaluation results")

        output_dir = args.output_dir or str(Path(results_path).parent)

        # Generate plots
        print("Generating plots...")

        paths = plot_scores_over_turns(
            results, output_dir, args.config_name, timestamp, args.show_individual
        )
        for path in paths:
            print(f"Saved: {path}")

        path = plot_dimension_comparison(results, output_dir, args.config_name, timestamp)
        print(f"Saved: {path}")

    print("\nDone!")


if __name__ == "__main__":
    main()

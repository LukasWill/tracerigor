"""
Summarize the multi-step ALFWorld judge eval across training-step snapshots.

For each step_X subdirectory under a run root, this script:
  1. Loads the LATEST `eval_metrics_*.json` from `step_X/eval_results/`
     (this is the metrics file produced by `eval_alfworld_samples.py`)
  2. Loads the env-side trajectory metrics from `step_X/samples.jsonl`
     (success rate, avg trajectory length, termination reasons)
  3. Prints a single per-step row combining judge + env metrics
  4. Saves a consolidated CSV + JSON summary

Usage:
    python -m tracerigor.verifier.scripts.summarize_alfworld_multistep_eval \\
        --run-root /path/to/alfworld/run \\
        --steps 0 10 20 30 40 50 60 70 100 120 140 150 \\
        --step-70-source run1_v2
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def find_latest_metrics(eval_dir: Path) -> Optional[Path]:
    """Return the most recent `eval_metrics_*.json` in `eval_dir`."""
    if not eval_dir.exists():
        return None
    candidates = sorted(eval_dir.glob("eval_metrics_*.json"))
    return candidates[-1] if candidates else None


def load_env_metrics(samples_path: Path) -> Dict[str, Any]:
    """Aggregate env-side metrics from samples.jsonl."""
    if not samples_path.exists():
        return {}
    n = 0
    success = 0
    total_steps = 0
    reasons: Dict[str, int] = {}
    avg_action_eff = 0.0
    avg_action_valid = 0.0
    with open(samples_path, "r") as f:
        for line in f:
            o = json.loads(line)
            m = o.get("metrics", {}) or {}
            n += 1
            if m.get("success"):
                success += 1
            total_steps += int(m.get("step", 0))
            r = m.get("termination_reason", "none") or "none"
            reasons[r] = reasons.get(r, 0) + 1
            avg_action_eff += float(m.get("action_is_effective", 0.0))
            avg_action_valid += float(m.get("action_is_valid", 0.0))
    if not n:
        return {}
    return {
        "env_n_trajectories": n,
        "env_success_rate": success / n,
        "env_avg_traj_len": total_steps / n,
        "env_termination_reasons": reasons,
        "env_avg_action_effective": avg_action_eff / n,
        "env_avg_action_valid": avg_action_valid / n,
    }


def main():
    parser = argparse.ArgumentParser(description="Multi-step ALFWorld eval summary")
    parser.add_argument(
        "--run-root",
        required=True,
        help="Root directory containing step_N run directories",
    )
    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=[0, 10, 20, 30, 40, 50, 60, 70, 100, 120, 140, 150],
        help="Training-step snapshots to summarise",
    )
    parser.add_argument(
        "--step-70-source",
        type=str,
        default=None,
        help=(
            "If set, for step_70 load metrics from `eval_results/<this>/eval_metrics_*.json` "
            "instead of the default `eval_results/eval_metrics_*.json`. "
            "Use this when step_70 was evaluated under a per-run subdir (e.g. run1_v2)."
        ),
    )
    parser.add_argument(
        "--source-overrides",
        type=str,
        nargs="*",
        default=[],
        help=(
            "Per-step subdir overrides for the eval_metrics source, format "
            "'<step>=<subdir>' (e.g. --source-overrides 200=postfix 210=postfix). "
            "Use when later steps were re-evaluated under a per-step subdir."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Where to save the CSV + JSON summary (default: <run-root>/eval_summary)",
    )
    args = parser.parse_args()

    overrides: Dict[int, str] = {}
    for tok in args.source_overrides:
        if "=" in tok:
            k, v = tok.split("=", 1)
            try:
                overrides[int(k)] = v
            except ValueError:
                pass

    run_root = Path(args.run_root)
    out_dir = Path(args.output_dir) if args.output_dir else run_root / "eval_summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []

    for step in args.steps:
        step_dir = run_root / f"step_{step}"
        samples_path = step_dir / "samples.jsonl"
        # Locate the right eval_metrics file
        if step in overrides:
            metrics_path = find_latest_metrics(step_dir / "eval_results" / overrides[step])
        elif step == 70 and args.step_70_source:
            metrics_path = find_latest_metrics(step_dir / "eval_results" / args.step_70_source)
        else:
            metrics_path = find_latest_metrics(step_dir / "eval_results")

        env = load_env_metrics(samples_path)

        row: Dict[str, Any] = {"step": step}
        row.update(env)

        if metrics_path and metrics_path.exists():
            with open(metrics_path, "r") as f:
                jm = json.load(f)
            row["judge_metrics_file"] = str(metrics_path)
            for k in (
                "num_samples", "query_success_rate", "parse_success_rate", "avg_score",
                "avg_observation_grounding", "avg_action_coherence", "avg_temporal_consistency",
                "mechanical_empty_reflections", "mechanical_inadmissible_actions",
                "mechanical_repeating_actions", "num_trajectories",
            ):
                if k in jm:
                    row[k] = jm[k]
        else:
            print(f"  [warn] no eval_metrics found for step_{step}")

        rows.append(row)

    # Console
    print("\n" + "=" * 100)
    print(f"{'step':>5} {'env_succ':>8} {'avg_len':>8} {'judge_g':>8} {'judge_a':>8} {'judge_t':>8} {'judge_avg':>9} {'n_judge':>8}")
    print("-" * 100)
    for r in rows:
        print(
            f"{r['step']:>5} "
            f"{r.get('env_success_rate', float('nan')):>8.3f} "
            f"{r.get('env_avg_traj_len', float('nan')):>8.1f} "
            f"{r.get('avg_observation_grounding', float('nan')):>8.3f} "
            f"{r.get('avg_action_coherence', float('nan')):>8.3f} "
            f"{r.get('avg_temporal_consistency', float('nan')):>8.3f} "
            f"{r.get('avg_score', float('nan')):>9.3f} "
            f"{r.get('num_samples', 0):>8d}"
        )
    print("=" * 100)

    # Save consolidated JSON
    out_json = out_dir / "multistep_eval_summary.json"
    with open(out_json, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved JSON: {out_json}")

    # Save CSV (flat columns)
    out_csv = out_dir / "multistep_eval_summary.csv"
    flat_cols = [
        "step",
        "env_n_trajectories", "env_success_rate", "env_avg_traj_len",
        "env_avg_action_effective", "env_avg_action_valid",
        "num_samples", "num_trajectories",
        "avg_score", "avg_observation_grounding", "avg_action_coherence", "avg_temporal_consistency",
        "query_success_rate", "parse_success_rate",
        "mechanical_empty_reflections", "mechanical_inadmissible_actions",
        "mechanical_repeating_actions",
    ]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=flat_cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in flat_cols})
    print(f"Saved CSV:  {out_csv}")


if __name__ == "__main__":
    main()

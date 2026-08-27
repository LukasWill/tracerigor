"""
Analyze inter-judge consistency across multiple repeated runs of the ALFWorld
LLM judge on the SAME 64 trajectories at the SAME training-step snapshot.

Inputs: 3+ run directories each containing one `eval_results_*.jsonl` file
produced by `eval_alfworld_samples.py`.

Outputs:
  - inter_judge_agreement.json   per-rubric agreement statistics
  - inter_judge_disagreements.jsonl   one entry per disagreed sample,
                                      with the per-run verdicts/evidence
                                      and a compact diff trace

Usage:
    python -m tracerigor.verifier.scripts.analyze_alfworld_judge_consistency \\
        --runs-dir /path/to/alfworld/step_N/eval_results \\
        --run-subdirs run1 run2 run3
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


RUBRIC_KEYS = ("grounding", "action", "temporal")
RUBRIC_LABEL = {
    "grounding": "observation_grounding",
    "action": "action_coherence",
    "temporal": "temporal_consistency",
}


def find_latest_results_file(run_dir: Path) -> Optional[Path]:
    """Return the most recent eval_results_<ts>.jsonl in a run dir."""
    candidates = sorted(run_dir.glob("eval_results_*.jsonl"))
    return candidates[-1] if candidates else None


def load_results(jsonl_path: Path) -> Dict[str, Dict[str, Any]]:
    """Load a result JSONL file into a dict keyed by sample_id."""
    out: Dict[str, Dict[str, Any]] = {}
    with open(jsonl_path, "r") as f:
        for line in f:
            r = json.loads(line)
            sid = r.get("id")
            if sid:
                out[sid] = r
    return out


def get_per_rubric_verdicts(result: Dict[str, Any]) -> Dict[str, Optional[str]]:
    """Extract YES/NO verdicts per rubric from a result dict.

    Returns dict {grounding/action/temporal -> 'YES'|'NO'|None}.
    None means the rubric was not scored (parse failure or missing field).
    """
    out: Dict[str, Optional[str]] = {k: None for k in RUBRIC_KEYS}
    extra = result.get("extra") or {}
    for short, full in RUBRIC_LABEL.items():
        block = extra.get(full) or {}
        v = block.get("yes_no") if isinstance(block, dict) else None
        if v in ("YES", "NO"):
            out[short] = v
    return out


def is_synthetic(result: Dict[str, Any]) -> bool:
    extra = result.get("extra") or {}
    return bool(extra.get("_synthetic"))


def compute_agreement(
    run_results: List[Dict[str, Dict[str, Any]]],
    drop_synthetic: bool = True,
) -> Dict[str, Any]:
    """Compute inter-judge agreement statistics across runs.

    Args:
        run_results: list of {sample_id -> result-dict}, one entry per run.
        drop_synthetic: if True, exclude synthetic empty-reflection samples
                        (they always agree trivially across runs).

    Returns:
        dict with per-rubric and overall agreement metrics.
    """
    if len(run_results) < 2:
        raise ValueError("Need at least 2 runs to compute agreement.")

    # Intersect sample IDs across all runs
    shared_ids = set(run_results[0].keys())
    for run in run_results[1:]:
        shared_ids &= set(run.keys())
    shared_ids = sorted(shared_ids)

    # Optionally drop synthetic samples (all-NO without LLM)
    if drop_synthetic:
        shared_ids = [
            sid for sid in shared_ids
            if not any(is_synthetic(run.get(sid, {})) for run in run_results)
        ]

    n = len(shared_ids)
    n_runs = len(run_results)

    metrics: Dict[str, Any] = {
        "n_samples_shared": n,
        "n_runs": n_runs,
    }

    # Per-rubric stats
    for rubric in RUBRIC_KEYS:
        all_agree_yes = 0
        all_agree_no = 0
        any_disagree = 0
        parse_failures = 0
        # Distribution of (n_YES, n_NO) tuples
        verdict_distribution: Counter = Counter()

        # Pairwise agreement rate (across all run pairs)
        pairwise_agree = 0
        pairwise_total = 0

        for sid in shared_ids:
            verdicts = []
            for run in run_results:
                v = get_per_rubric_verdicts(run.get(sid, {})).get(rubric)
                verdicts.append(v)

            if any(v is None for v in verdicts):
                parse_failures += 1
                continue

            n_yes = sum(1 for v in verdicts if v == "YES")
            n_no = sum(1 for v in verdicts if v == "NO")
            verdict_distribution[(n_yes, n_no)] += 1

            if n_yes == n_runs:
                all_agree_yes += 1
            elif n_no == n_runs:
                all_agree_no += 1
            else:
                any_disagree += 1

            # Pairwise agreement
            for i in range(n_runs):
                for j in range(i + 1, n_runs):
                    pairwise_total += 1
                    if verdicts[i] == verdicts[j]:
                        pairwise_agree += 1

        n_scored = n - parse_failures
        rubric_metrics = {
            "n_scored": n_scored,
            "parse_failures": parse_failures,
            "all_agree_yes": all_agree_yes,
            "all_agree_no": all_agree_no,
            "any_disagree": any_disagree,
            "unanimous_agreement_rate": (all_agree_yes + all_agree_no) / n_scored if n_scored else 0,
            "disagreement_rate": any_disagree / n_scored if n_scored else 0,
            "pairwise_agreement_rate": pairwise_agree / pairwise_total if pairwise_total else 0,
            "verdict_distribution": {
                f"{ny}YES_{nn}NO": cnt
                for (ny, nn), cnt in sorted(verdict_distribution.items(), reverse=True)
            },
        }
        metrics[rubric] = rubric_metrics

    # All-rubrics-agreement = all 3 rubrics unanimously agreed
    all_three_rubrics_agree = 0
    any_rubric_disagrees = 0
    for sid in shared_ids:
        per_rubric_ok = True
        for rubric in RUBRIC_KEYS:
            verdicts = [
                get_per_rubric_verdicts(run.get(sid, {})).get(rubric)
                for run in run_results
            ]
            if any(v is None for v in verdicts):
                per_rubric_ok = False
                break
            if len(set(verdicts)) > 1:
                per_rubric_ok = False
                break
        if per_rubric_ok:
            all_three_rubrics_agree += 1
        else:
            any_rubric_disagrees += 1

    metrics["all_three_rubrics_unanimously_agree"] = all_three_rubrics_agree
    metrics["any_rubric_disagrees"] = any_rubric_disagrees
    metrics["full_unanimity_rate"] = (
        all_three_rubrics_agree / n if n else 0
    )

    return metrics, shared_ids


def collect_disagreements(
    run_results: List[Dict[str, Dict[str, Any]]],
    shared_ids: List[str],
) -> List[Dict[str, Any]]:
    """Collect per-sample disagreement records for case-study analysis."""
    out: List[Dict[str, Any]] = []
    n_runs = len(run_results)

    for sid in shared_ids:
        per_rubric_verdicts: Dict[str, List[Optional[str]]] = {}
        per_rubric_evidence: Dict[str, List[Any]] = {}
        any_disagree = False
        for rubric in RUBRIC_KEYS:
            verdicts = []
            evidences = []
            for run in run_results:
                r = run.get(sid, {})
                pv = get_per_rubric_verdicts(r).get(rubric)
                extra = r.get("extra") or {}
                ev = (extra.get(RUBRIC_LABEL[rubric]) or {}).get("evidence", "")
                verdicts.append(pv)
                evidences.append(ev)
            per_rubric_verdicts[rubric] = verdicts
            per_rubric_evidence[rubric] = evidences
            if any(v is None for v in verdicts):
                continue
            if len(set(verdicts)) > 1:
                any_disagree = True

        if not any_disagree:
            continue

        # Pull the sample inputs from any one run (they're identical inputs)
        anchor = run_results[0].get(sid, {})
        record = {
            "id": sid,
            "task_description": anchor.get("task_description"),
            "current_step": anchor.get("current_step"),
            "current_observation_text": anchor.get("current_observation_text"),
            "reflection_tokens": anchor.get("reflection_tokens"),
            "action_tokens": anchor.get("action_tokens"),
            "last_action_inadmissible": anchor.get("last_action_inadmissible"),
            "admissible_commands_count": len(anchor.get("admissible_commands") or []),
            "n_history_steps": len(anchor.get("history") or []),
            "mechanical": (anchor.get("extra") or {}).get("mechanical"),
            "per_rubric_verdicts": per_rubric_verdicts,
            "per_rubric_evidence": per_rubric_evidence,
            # Per-run response strings for deeper analysis if needed
            "per_run_response_snippets": [
                (r.get(sid, {}).get("response") or "")[:300]
                for r in run_results
            ],
        }
        out.append(record)

    return out


def categorize_disagreement(record: Dict[str, Any]) -> Dict[str, Any]:
    """Auto-classify a disagreement record into common patterns.

    Returns dict of boolean flags per pattern.
    """
    refl = (record.get("reflection_tokens") or "").lower()
    obs = (record.get("current_observation_text") or "").lower()
    act = (record.get("action_tokens") or "").lower()
    last_inadm = record.get("last_action_inadmissible")
    mech = record.get("mechanical") or {}

    # Patterns to detect
    flags: Dict[str, bool] = {
        "after_inadmissible": bool(last_inadm),
        "claims_completion": any(
            k in refl
            for k in (
                "task is complete", "completed the task", "no further",
                "have successfully", "have now completed",
            )
        ),
        "claims_change_with_now": bool(re.search(r"\b(now|having|have moved|have placed|is now)\b", refl)),
        "action_is_help_inventory_look": act in {"help", "inventory", "look"},
        "obs_is_help_dump": "available commands:" in obs,
        "obs_is_examine_reply": (
            "there's nothing special about" in obs
            or re.search(r"this is a (hot|cool|clean|sliced) ", obs) is not None
        ),
        "obs_is_arrive": obs.startswith("you arrive at"),
        "obs_is_pickup_stale": obs.startswith("you pick up"),
        "is_inadmissible_now": bool(mech.get("action_inadmissible")),
        "is_repeating_now": bool(mech.get("action_repetition")),
    }

    # Which rubric(s) disagree
    disagreed = []
    for rubric in RUBRIC_KEYS:
        v = record["per_rubric_verdicts"].get(rubric) or []
        if v and len(set(v)) > 1:
            disagreed.append(rubric)
    flags["disagreed_rubrics"] = disagreed
    flags["disagreed_rubrics_count"] = len(disagreed)

    return flags


def summarize_disagreement_patterns(
    disagreements: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Produce a summary of the most common patterns in disagreements."""
    rubric_disagreement_counts: Counter = Counter()
    pattern_counts: Counter = Counter()

    for rec in disagreements:
        flags = categorize_disagreement(rec)
        for rubric in flags["disagreed_rubrics"]:
            rubric_disagreement_counts[rubric] += 1
        for k, v in flags.items():
            if isinstance(v, bool) and v:
                pattern_counts[k] += 1

    return {
        "disagreed_rubrics_counts": dict(rubric_disagreement_counts),
        "pattern_co_occurrence": dict(pattern_counts.most_common()),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Analyze inter-judge consistency across repeated runs of ALFWorld judge."
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        required=True,
        help="Parent dir containing per-run subdirs (e.g. step_70/eval_results)",
    )
    parser.add_argument(
        "--run-subdirs",
        type=str,
        nargs="+",
        default=["run1", "run2", "run3"],
        help="Names of run subdirs to compare",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Where to save the analysis report (default: runs-dir/inter_judge_analysis)",
    )
    parser.add_argument(
        "--keep-synthetic",
        action="store_true",
        help="Include synthetic empty-reflection samples in agreement computation",
    )
    parser.add_argument(
        "--max-disagreement-cases",
        type=int,
        default=None,
        help="If set, cap the number of disagreement records written to disk",
    )

    args = parser.parse_args()

    runs_root = Path(args.runs_dir)
    out_dir = Path(args.output_dir) if args.output_dir else runs_root / "inter_judge_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from {len(args.run_subdirs)} runs under {runs_root}...")
    run_results: List[Dict[str, Dict[str, Any]]] = []
    run_files: List[Path] = []
    for sub in args.run_subdirs:
        run_dir = runs_root / sub
        f = find_latest_results_file(run_dir)
        if f is None:
            raise FileNotFoundError(f"No eval_results_*.jsonl in {run_dir}")
        print(f"  {sub}: {f.name}")
        results = load_results(f)
        run_results.append(results)
        run_files.append(f)

    metrics, shared_ids = compute_agreement(
        run_results, drop_synthetic=not args.keep_synthetic
    )

    disagreements = collect_disagreements(run_results, shared_ids)
    pattern_summary = summarize_disagreement_patterns(disagreements)

    metrics["disagreement_pattern_summary"] = pattern_summary
    metrics["run_files"] = [str(f) for f in run_files]
    metrics["timestamp"] = datetime.now().isoformat(timespec="seconds")

    # Optional cap
    if args.max_disagreement_cases and len(disagreements) > args.max_disagreement_cases:
        disagreements = disagreements[: args.max_disagreement_cases]

    # Save
    metrics_path = out_dir / "inter_judge_agreement.json"
    disagree_path = out_dir / "inter_judge_disagreements.jsonl"

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved agreement metrics: {metrics_path}")

    with open(disagree_path, "w") as f:
        for rec in disagreements:
            f.write(json.dumps(rec) + "\n")
    print(f"Saved {len(disagreements)} disagreement records: {disagree_path}")

    # Console summary
    print("\n" + "=" * 60)
    print(f"Inter-judge consistency summary  ({len(args.run_subdirs)} runs, "
          f"{metrics['n_samples_shared']} shared samples)")
    print("=" * 60)
    print(f"  Full unanimity (all 3 rubrics agree across runs): "
          f"{metrics['full_unanimity_rate'] * 100:.1f}%  "
          f"({metrics['all_three_rubrics_unanimously_agree']}/{metrics['n_samples_shared']})")
    print(f"  Any-rubric disagreement: {metrics['any_rubric_disagrees']} samples")
    print()
    for rubric in RUBRIC_KEYS:
        m = metrics[rubric]
        print(f"  [{RUBRIC_LABEL[rubric]:<22}]")
        print(f"    unanimous: {m['unanimous_agreement_rate'] * 100:.1f}%  "
              f"(all-YES={m['all_agree_yes']}, all-NO={m['all_agree_no']})")
        print(f"    pairwise:  {m['pairwise_agreement_rate'] * 100:.1f}%")
        print(f"    disagree:  {m['any_disagree']} samples")
    print()
    print("Disagreement patterns (co-occurrence counts):")
    for k, v in pattern_summary["pattern_co_occurrence"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

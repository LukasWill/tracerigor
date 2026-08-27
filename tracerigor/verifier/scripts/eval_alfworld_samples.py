"""
Evaluate ALFWorld LLM-agent validation samples with the LLM judge.

Adapted from eval_sciworld_samples.py for the ALFWorld domain.

Key differences from SciWorld:
- Text-only env (no images), but unlike SciWorld the env exposes an explicit
  `Admissible commands: [...]` list every turn. Admissibility is therefore an
  EXACT mechanical check (no "uncertain" cases as in SciWorld).
- Inadmissible actions leave the world frozen and re-emit the prior obs; the
  per-turn prompt template appends:
    "(Note: your previous action was not in admissible_commands;
     the environment state did not advance.)"
  We extract this Note flag for the current step + each history step so the
  judge prompt can use the inadmissible-action signal.
- NO ground-truth replay (per project design). Mechanical pre-filter is
  simplified (closer to Sokoban than SciWorld): only empty_reflection
  triggers a synthetic-NO LLM skip; everything else (including inadmissible
  actions) flows through the LLM, with admissibility recorded as metadata.
- Agent format: <reflection>...</reflection><action>...</action>, exactly ONE
  command per turn (max_actions_per_step=1 in the reflact config).

Usage:
    python -m tracerigor.verifier.scripts.eval_alfworld_samples \\
        --samples-dir /path/to/alfworld/step_N \\
        --model gpt-5-nano-2025-08-07 --mode direct --plot --max-trajectories 4
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root on path for direct script invocation.
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class EvalSample:
    """A single evaluation sample extracted from a multi-turn ALFWorld trajectory."""

    sample_id: str
    env_id: str  # e.g. "val1"
    step_index: int  # 1-based turn index within trajectory
    task_description: str
    current_observation_text: str
    reflection_tokens: str
    action_tokens: str
    admissible_commands: List[str] = field(default_factory=list)
    # Whether the action that PRODUCED current_observation_text was inadmissible
    # (i.e., the prior turn's action was rejected → current obs is stale +
    # prompt has the "(Note: ... did not advance.)" line). For step 1 this is
    # always False.
    last_action_inadmissible: bool = False
    history: List[Dict[str, Any]] = field(default_factory=list)

    # Bookkeeping
    original_score: Optional[float] = None
    original_success: Optional[bool] = None
    original_termination_reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


_REFLACT_REFL_RE = re.compile(r"<reflection>(.*?)</reflection>", re.DOTALL)
_REFLACT_ACT_RE = re.compile(r"<action>(.*?)</action>", re.DOTALL)
_TASK_RE = re.compile(r"Your task is:\s*(.+?)(?:\n|$)")
_NOTE_RE = re.compile(
    r"\(Note:\s*your previous action was not in admissible_commands;\s*"
    r"the environment state did not advance\.\)",
    re.DOTALL,
)


def parse_reflact_response(response_text: str) -> Tuple[str, str]:
    """Extract reflection and action from a ReflAct assistant turn."""
    refl = ""
    act = ""
    m_refl = _REFLACT_REFL_RE.search(response_text)
    if m_refl:
        refl = m_refl.group(1).strip()
    m_act = _REFLACT_ACT_RE.search(response_text)
    if m_act:
        act = m_act.group(1).strip()
    return refl, act


def extract_task_from_user_turn(user_text: str) -> str:
    """Extract the task instruction from a user-turn body.

    Both the initial and subsequent user templates start with
    `Your task is: <task>`. We use the first match (most reliable).
    """
    m = _TASK_RE.search(user_text)
    return m.group(1).strip() if m else "N/A"


def extract_observation_and_note(user_text: str) -> Tuple[str, bool]:
    """Extract current observation text + inadmissible-note flag.

    The user-turn template has two shapes:
      - Initial:  "Your task is: ...\n\nInitial observation: <obs>\n\n
                   Admissible commands: ..."
      - Step N:   "Your task is: ...\n\nEpisode history so far:\n<hist>\n\n
                   Step N. Current observation: <obs>[<note>]\n\n
                   Admissible commands: ..."

    Returns (observation_text, has_inadmissible_note).
    The observation text excludes the Note line (which is metadata, not obs).
    """
    has_note = bool(_NOTE_RE.search(user_text))

    # Try "Step N. Current observation: ..." first (subsequent turns)
    m = re.search(
        r"Step\s+\d+\.\s+Current observation:\s+(.+?)(?:\n\nAdmissible commands:|$)",
        user_text,
        re.DOTALL,
    )
    if m:
        obs = m.group(1).strip()
    else:
        # Initial observation
        m = re.search(
            r"Initial observation:\s+(.+?)(?:\n\nAdmissible commands:|$)",
            user_text,
            re.DOTALL,
        )
        obs = m.group(1).strip() if m else ""

    # Strip the Note line from the obs text if present
    obs = _NOTE_RE.sub("", obs).strip()
    return obs, has_note


def extract_admissible_commands(user_text: str) -> List[str]:
    """Extract the comma-separated admissible commands list from a user turn."""
    m = re.search(
        r"Admissible commands:\s*\[(.+?)\]",
        user_text,
        re.DOTALL,
    )
    if not m:
        return []
    raw = m.group(1)
    # Commands are comma-separated; commas appear inside command strings only as
    # the action_sep, but the template separates them with ", " for display.
    parts = [c.strip() for c in raw.split(",")]
    return [p for p in parts if p]


def extract_samples_from_trajectory(raw_sample: Dict[str, Any]) -> List[EvalSample]:
    """Extract per-turn EvalSamples from a multi-turn ALFWorld trajectory.

    Each assistant turn yields one EvalSample. The current observation for
    turn i is taken from the i-th user turn; the inadmissible-note flag for
    turn i+1 (about the action of turn i) is read from the (i+1)-th user turn
    and attached to BOTH the next sample (as `last_action_inadmissible`) and
    to the corresponding history entry of subsequent samples.
    """
    samples: List[EvalSample] = []
    output_str = raw_sample.get("output_str", "")
    sample_idx = raw_sample.get("sample_idx", 0)
    env_id = raw_sample.get("env_id", "unknown")
    metrics = raw_sample.get("metrics", {}) or {}

    # Split into user/assistant turns
    user_pattern = r"<\|im_start\|>user\n(.*?)<\|im_end\|>"
    assistant_pattern = r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>"
    user_turns = re.findall(user_pattern, output_str, re.DOTALL)
    assistant_turns = re.findall(assistant_pattern, output_str, re.DOTALL)

    if not user_turns or not assistant_turns:
        return samples

    # Task description (consistent across turns; use the first user turn)
    task_desc = extract_task_from_user_turn(user_turns[0])

    HISTORY_WINDOW = 5
    history: List[Dict[str, Any]] = []

    n_turns = min(len(user_turns), len(assistant_turns))
    for i in range(n_turns):
        u = user_turns[i]
        a = assistant_turns[i]

        refl, act = parse_reflact_response(a)
        obs_text, has_note = extract_observation_and_note(u)
        admissibles = extract_admissible_commands(u)

        sample = EvalSample(
            sample_id=f"{sample_idx}_{env_id}_step{i + 1}",
            env_id=env_id,
            step_index=i + 1,
            task_description=task_desc,
            current_observation_text=obs_text,
            reflection_tokens=refl,
            action_tokens=act,
            admissible_commands=admissibles,
            last_action_inadmissible=has_note,  # i.e., step i-1's action was rejected
            history=list(history),  # snapshot
            original_score=metrics.get("score"),
            original_success=metrics.get("success"),
            original_termination_reason=metrics.get("termination_reason"),
        )
        samples.append(sample)

        # Append THIS step into history for subsequent samples. The inadmissible
        # flag for THIS step's action is read from the NEXT user turn (the one
        # that reports the result). For the last turn no such report exists.
        next_inadm = False
        if i + 1 < len(user_turns):
            _, next_inadm = extract_observation_and_note(user_turns[i + 1])

        history.append(
            {
                "step": i + 1,
                "observation_text": obs_text,
                "reflection": refl,
                "action": act,
                "last_action_inadmissible": next_inadm,
            }
        )
        if len(history) > HISTORY_WINDOW:
            history = history[-HISTORY_WINDOW:]

    return samples


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_samples(
    samples_dir: str,
    max_samples: Optional[int] = None,
    max_trajectories: Optional[int] = None,
    trajectory_seed: int = 42,
) -> List[EvalSample]:
    """Load samples from samples.jsonl file."""
    samples_path = Path(samples_dir) / "samples.jsonl"
    if not samples_path.exists():
        raise FileNotFoundError(f"Samples file not found: {samples_path}")

    raw_lines: List[Tuple[int, str]] = []
    with open(samples_path, "r") as f:
        for line_num, line in enumerate(f):
            raw_lines.append((line_num, line.strip()))

    total_trajectories = len(raw_lines)

    if max_trajectories and max_trajectories < total_trajectories:
        import random

        rng = random.Random(trajectory_seed)
        selected = sorted(rng.sample(range(total_trajectories), max_trajectories))
        raw_lines = [raw_lines[i] for i in selected]
        print(
            f"Randomly subsampled {max_trajectories} of {total_trajectories} "
            f"trajectories (seed={trajectory_seed})"
        )

    all_eval_samples: List[EvalSample] = []
    trajectories_loaded = 0

    for line_num, line in raw_lines:
        if max_samples and len(all_eval_samples) >= max_samples:
            break
        try:
            raw_sample = json.loads(line)
            eval_samples = extract_samples_from_trajectory(raw_sample)
            all_eval_samples.extend(eval_samples)
            trajectories_loaded += 1
            if max_samples and len(all_eval_samples) >= max_samples:
                all_eval_samples = all_eval_samples[:max_samples]
                break
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse line {line_num}: {e}")

    print(
        f"Loaded {len(all_eval_samples)} evaluation samples from "
        f"{trajectories_loaded} trajectories"
    )
    return all_eval_samples


# ---------------------------------------------------------------------------
# Batch items (for verifier API)
# ---------------------------------------------------------------------------


def samples_to_batch_items(samples: List[EvalSample]) -> List[Dict[str, Any]]:
    """Convert EvalSample objects to batch items for the verifier API.

    The ALFWorld universal template expects:
      - reflection_tokens, action_tokens (raw text, not XML-wrapped)
      - task_description, current_step
      - current_observation_text
      - admissible_commands (list of strings)
      - last_action_inadmissible (bool) — surfaces the env's "(Note: ...)" line
      - history (list of dicts with step / observation_text / reflection /
                 action / last_action_inadmissible)

    Also surfaces `reasoning_tokens` (XML-wrapped reflection) as an alias used
    by the shared wandb logging helper. Mirrors the SciWorld pattern.
    """
    items: List[Dict[str, Any]] = []
    for sample in samples:
        item: Dict[str, Any] = {
            "id": sample.sample_id,
            "reflection_tokens": sample.reflection_tokens,
            "action_tokens": sample.action_tokens,
            # Alias for shared wandb logger (used as r['reasoning_tokens']).
            "reasoning_tokens": (
                f"<reflection>{sample.reflection_tokens}</reflection>"
                if sample.reflection_tokens
                else ""
            ),
            "task_description": sample.task_description,
            "current_step": sample.step_index,
            "current_observation_text": sample.current_observation_text,
            "admissible_commands": list(sample.admissible_commands),
            "last_action_inadmissible": bool(sample.last_action_inadmissible),
            "history": list(sample.history),
        }
        items.append(item)
    return items


# ---------------------------------------------------------------------------
# Mechanical Pre-filter (simplified, ALFWorld-specific)
# ---------------------------------------------------------------------------


def run_mechanical_prefilter(
    samples: List[EvalSample],
) -> Dict[str, Dict[str, Any]]:
    """Run lightweight mechanical checks on ALFWorld samples.

    Checks (all are metadata-only EXCEPT empty_reflection, which triggers a
    synthetic-NO LLM skip downstream):

      - empty_reflection: no reflection text → synthetic NO
      - action_inadmissible: action ∉ admissible_commands at this step
                              (exact, since the env surfaces the list)
      - empty_action: action token empty / could not be parsed
      - action_repetition: action equals each of the last K actions
                          (≥2 consecutive identical actions in history)
    """
    results: Dict[str, Dict[str, Any]] = {}

    for sample in samples:
        is_empty_refl = not sample.reflection_tokens or not sample.reflection_tokens.strip()
        is_empty_action = not sample.action_tokens or not sample.action_tokens.strip()

        # Admissibility: action ∈ admissible_commands (exact match, case-insensitive
        # trim). ALFWorld surfaces the list, so this check is authoritative.
        adm_set = {c.strip().lower() for c in sample.admissible_commands}
        action_norm = (sample.action_tokens or "").strip().lower()
        is_inadmissible = bool(adm_set) and (action_norm not in adm_set) and not is_empty_action

        # Repetition: same action as the last >=2 entries in history.
        consecutive_same = 0
        if not is_empty_action:
            for h in reversed(sample.history):
                past = (h.get("action") or "").strip().lower()
                if past == action_norm and past:
                    consecutive_same += 1
                else:
                    break
        is_repeating = consecutive_same >= 2

        results[sample.sample_id] = {
            "empty_reflection": is_empty_refl,
            "empty_action": is_empty_action,
            "action_inadmissible": is_inadmissible,
            "action_repetition": is_repeating,
            "consecutive_same_action_count": consecutive_same + (0 if is_empty_action else 1),
            # Convenience flags
            "any_failure": is_empty_refl,
            "has_format_or_validity_issue": is_empty_refl or is_empty_action or is_inadmissible,
        }

    # Summary
    n = len(samples)
    if n:
        n_empty_refl = sum(1 for r in results.values() if r["empty_reflection"])
        n_empty_act = sum(1 for r in results.values() if r["empty_action"])
        n_inadm = sum(1 for r in results.values() if r["action_inadmissible"])
        n_rep = sum(1 for r in results.values() if r["action_repetition"])
        print("\n[Mechanical Pre-filter] ALFWorld Results:")
        print(f"  Total samples: {n}")
        print(f"  Empty reflections: {n_empty_refl} ({100 * n_empty_refl / n:.1f}%)")
        print(f"  Empty actions: {n_empty_act} ({100 * n_empty_act / n:.1f}%)")
        print(f"  Inadmissible actions: {n_inadm} ({100 * n_inadm / n:.1f}%)")
        print(f"  Repetition (>=2 consecutive same): {n_rep} ({100 * n_rep / n:.1f}%)")

    return results


def create_synthetic_result_all_no(sample: EvalSample, reason: str) -> Dict[str, Any]:
    """Create a synthetic all-NO evaluation result without an LLM call."""
    return {
        "id": sample.sample_id,
        "response": f"[Synthetic: {reason}]",
        "query_success": True,
        "parse_success": True,
        "error": None,
        "score": 0.0,
        "verdict": "NO",
        "model": "mechanical",
        "retries": 0,
        "extra": {
            "observation_grounding": {"yes_no": "NO", "evidence": f"Synthetic NO: {reason}"},
            "action_coherence": {"yes_no": "NO", "evidence": f"Synthetic NO: {reason}"},
            "temporal_consistency": {"yes_no": "NO", "evidence": f"Synthetic NO: {reason}"},
            "scalar_scores": {
                "grounding": 0.0,
                "action": 0.0,
                "temporal": 0.0,
                "aggregate": 0.0,
            },
            "_synthetic": True,
            "_synthetic_reason": reason,
        },
    }


def split_samples_for_eval(
    samples: List[EvalSample],
    mechanical_results: Dict[str, Any],
) -> Tuple[List[EvalSample], List[EvalSample], List[Dict[str, Any]]]:
    """Split samples into LLM-eval vs. synthetic-NO (empty reflections).

    Following the sciworld design lesson: only empty reflections are
    synthetic-skipped to preserve trajectory continuity in per-step metrics.
    Inadmissible / repeating actions remain in the LLM stream (the judge
    prompt accepts them as signals).
    """
    samples_for_llm: List[EvalSample] = []
    empty_samples: List[EvalSample] = []
    synthetic_results: List[Dict[str, Any]] = []

    for sample in samples:
        mech = mechanical_results.get(sample.sample_id, {})
        is_empty = mech.get("empty_reflection", False)
        if not mech:
            is_empty = not sample.reflection_tokens or not sample.reflection_tokens.strip()

        if is_empty:
            empty_samples.append(sample)
            synthetic_results.append(
                create_synthetic_result_all_no(sample, "empty_reflection")
            )
        else:
            samples_for_llm.append(sample)

    if empty_samples:
        print(
            f"  [Post-hoc] {len(empty_samples)} empty reflections scored all-NO "
            f"(no LLM call)"
        )

    return samples_for_llm, empty_samples, synthetic_results


def attach_mechanical_metadata(
    results: List[Dict[str, Any]],
    mechanical_results: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Attach mechanical check metadata to evaluation results."""
    for result in results:
        sample_id = result.get("id", "")
        if sample_id in mechanical_results:
            mech = mechanical_results[sample_id]
            extra = result.get("extra") or {}
            extra["mechanical"] = {
                "empty_reflection": mech["empty_reflection"],
                "empty_action": mech["empty_action"],
                "action_inadmissible": mech["action_inadmissible"],
                "action_repetition": mech["action_repetition"],
                "consecutive_same_action_count": mech["consecutive_same_action_count"],
            }
            result["extra"] = extra
    return results


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


async def evaluate_via_api(
    samples: List[EvalSample],
    rubric: str = "universal",
    model: str = "gpt-5-nano-2025-08-07",
    api_url: str = "http://localhost:8000/batch_verify",
) -> List[Dict[str, Any]]:
    """Evaluate samples using the FastAPI endpoint."""
    import httpx

    items = samples_to_batch_items(samples)
    payload = {
        "items": items,
        "rubric": rubric,
        "models": [model],
        "model_params": {"temperature": 0.0},
    }
    async with httpx.AsyncClient(timeout=300.0) as client:
        response = await client.post(api_url, json=payload)
        response.raise_for_status()
        return response.json().get("results", [])


async def evaluate_direct(
    samples: List[EvalSample],
    rubric: str = "universal",
    model: str = "gpt-5-nano-2025-08-07",
    debug_prompts: int = 0,
) -> List[Dict[str, Any]]:
    """Evaluate samples directly using the ALFWorld verifier (no server needed)."""
    from tracerigor.verifier.verifier.openai_verifier import (
        _get_hydra_config,
        build_verifier,
        run_openai_verifier,
    )

    items = samples_to_batch_items(samples)

    # Debug prompts: generate and attach full prompts to the first N items
    if debug_prompts > 0:
        verifier_cls = build_verifier("alfworld", rubric)
        verifier = verifier_cls(
            _get_hydra_config(os.getpid()),
            model,
            {"temperature": 0.0},
        )
        for i, item in enumerate(items[:debug_prompts]):
            try:
                messages = verifier.assemble_messages(dict(item))
                item["_debug_system_prompt"] = (
                    messages[0]["content"] if messages else "N/A"
                )
                if len(messages) > 1:
                    user_content = messages[1]["content"]
                    if isinstance(user_content, list):
                        text_parts = [
                            p.get("text", "")
                            for p in user_content
                            if p.get("type") == "text"
                        ]
                        item["_debug_user_prompt"] = "\n".join(p for p in text_parts if p) or "N/A"
                    else:
                        item["_debug_user_prompt"] = user_content
                else:
                    item["_debug_user_prompt"] = "N/A"
            except Exception as e:
                item["_debug_system_prompt"] = f"Error: {e}"
                item["_debug_user_prompt"] = f"Error: {e}"

    results = await run_openai_verifier(
        input_data=items,
        rubric=rubric,
        model_name=model,
        model_params={"temperature": 0.0},
        env="alfworld",
    )

    # Copy debug prompts into results
    for i, result in enumerate(results):
        if i < len(items) and "_debug_system_prompt" in items[i]:
            result["debug_system_prompt"] = items[i]["_debug_system_prompt"]
            result["debug_user_prompt"] = items[i]["_debug_user_prompt"]

    return results


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_aggregate_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute aggregate metrics from evaluation results."""
    if not results:
        return {}

    n = len(results)
    metrics: Dict[str, Any] = {
        "num_samples": n,
        "query_success_rate": sum(1 for r in results if r.get("query_success", False)) / n,
        "parse_success_rate": sum(1 for r in results if r.get("parse_success", False)) / n,
        "avg_score": sum(r.get("score", 0) for r in results) / n,
    }

    grounding_scores: List[float] = []
    action_scores: List[float] = []
    temporal_scores: List[float] = []

    for r in results:
        extra = r.get("extra") or {}
        ss = extra.get("scalar_scores") or {}
        if "grounding" in ss:
            grounding_scores.append(ss["grounding"])
        if "action" in ss:
            action_scores.append(ss["action"])
        if "temporal" in ss:
            temporal_scores.append(ss["temporal"])

    if grounding_scores:
        metrics["avg_observation_grounding"] = sum(grounding_scores) / len(grounding_scores)
    if action_scores:
        metrics["avg_action_coherence"] = sum(action_scores) / len(action_scores)
    if temporal_scores:
        metrics["avg_temporal_consistency"] = sum(temporal_scores) / len(temporal_scores)

    return metrics


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------


def save_results(
    results: List[Dict[str, Any]],
    samples: List[EvalSample],
    output_dir: str,
    metrics: Dict[str, Any],
):
    """Save evaluation results, metrics, and a human-readable report."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Raw results
    results_file = output_path / f"eval_results_{timestamp}.jsonl"
    with open(results_file, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    print(f"Saved raw results to {results_file}")

    # Metrics summary
    metrics_file = output_path / f"eval_metrics_{timestamp}.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_file}")

    # Human-readable report
    report_file = output_path / f"eval_report_{timestamp}.txt"
    result_by_id = {r.get("id", ""): r for r in results}
    with open(report_file, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("ALFWorld LLM Judge Evaluation Report\n")
        f.write("=" * 60 + "\n\n")

        f.write("Aggregate Metrics:\n")
        f.write("-" * 40 + "\n")
        for k, v in metrics.items():
            if isinstance(v, float):
                f.write(f"  {k}: {v:.4f}\n")
            else:
                f.write(f"  {k}: {v}\n")

        f.write("\n\nSample Results (first 10):\n")
        f.write("-" * 40 + "\n")
        for i, sample in enumerate(samples[:10]):
            result = result_by_id.get(sample.sample_id, {})
            f.write(f"\n[{i + 1}] {sample.sample_id}\n")
            f.write(f"  Step: {sample.step_index}\n")
            f.write(f"  Task: {sample.task_description}\n")
            f.write(f"  Last-action inadmissible: {sample.last_action_inadmissible}\n")
            f.write(f"  Observation: {sample.current_observation_text[:280]}\n")
            f.write(f"  Reflection: {sample.reflection_tokens[:280]}\n")
            f.write(f"  Action: {sample.action_tokens}\n")
            f.write(f"  Score: {result.get('score', 'N/A')}\n")

            extra = result.get("extra") or {}
            ss = extra.get("scalar_scores") or {}
            if ss:
                f.write(
                    f"  Per-dim: grounding={ss.get('grounding', 'N/A')}, "
                    f"action={ss.get('action', 'N/A')}, "
                    f"temporal={ss.get('temporal', 'N/A')}\n"
                )
            mech = extra.get("mechanical") or {}
            if mech:
                f.write(
                    f"  Mechanical: empty_refl={mech.get('empty_reflection')}, "
                    f"inadm={mech.get('action_inadmissible')}, "
                    f"rep={mech.get('action_repetition')}\n"
                )
            f.write(f"  Parse OK: {result.get('parse_success', 'N/A')}\n")
            if result.get("response"):
                f.write(f"  LLM Response: {result['response'][:500]}\n")

            if result.get("debug_system_prompt"):
                f.write(f"\n  [DEBUG] System Prompt:\n  {'-' * 30}\n")
                for line in result["debug_system_prompt"].split("\n"):
                    f.write(f"    {line}\n")
            if result.get("debug_user_prompt"):
                f.write(f"\n  [DEBUG] User Prompt:\n  {'-' * 30}\n")
                for line in result["debug_user_prompt"].split("\n"):
                    f.write(f"    {line}\n")

    print(f"Saved report to {report_file}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def setup_academic_style():
    """Configure matplotlib for academic publication-quality plots."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    ACADEMIC_COLORS = {
        "grounding": "#0072B2",
        "action": "#D55E00",
        "temporal": "#009E73",
        "aggregate": "#CC79A7",
    }
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 13,
            "axes.linewidth": 1.0,
            "grid.linewidth": 0.5,
            "lines.linewidth": 1.5,
            "lines.markersize": 6,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.1,
        }
    )
    return ACADEMIC_COLORS


def extract_trajectory_id(sample_id: str) -> str:
    parts = sample_id.rsplit("_step", 1)
    return parts[0] if len(parts) > 1 else sample_id


def extract_step_number(sample_id: str) -> int:
    match = re.search(r"_step(\d+)$", sample_id)
    return int(match.group(1)) if match else 0


def group_results_by_trajectory(
    results: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for r in results:
        tid = extract_trajectory_id(r.get("id", ""))
        grouped.setdefault(tid, []).append(r)
    for tid in grouped:
        grouped[tid].sort(key=lambda r: extract_step_number(r.get("id", "")))
    return grouped


def compute_per_turn_statistics(
    results: List[Dict[str, Any]],
) -> Dict[int, Dict[str, Any]]:
    turn_data: Dict[int, Dict[str, List[float]]] = {}
    for r in results:
        step = extract_step_number(r.get("id", ""))
        if step < 1:
            continue
        extra = r.get("extra") or {}
        ss = extra.get("scalar_scores") or {}
        if step not in turn_data:
            turn_data[step] = {"grounding": [], "action": [], "temporal": []}
        for dim in ("grounding", "action", "temporal"):
            if dim in ss:
                turn_data[step][dim].append(ss[dim])

    stats: Dict[int, Dict[str, Any]] = {}
    import numpy as np
    for step in sorted(turn_data):
        stats[step] = {"step": step, "count": 0}
        for dim in ("grounding", "action", "temporal"):
            vals = turn_data[step][dim]
            if vals:
                stats[step][f"{dim}_mean"] = float(np.mean(vals))
                stats[step][f"{dim}_std"] = float(np.std(vals))
                stats[step]["count"] = max(stats[step]["count"], len(vals))
    return stats


def plot_scores_over_turns(
    results: List[Dict[str, Any]],
    output_dir: str,
    config_name: Optional[str] = None,
    timestamp: Optional[str] = None,
    show_individual: bool = False,
) -> str:
    import matplotlib.pyplot as plt
    import numpy as np

    colors = setup_academic_style()
    stats = compute_per_turn_statistics(results)
    if not stats:
        print("No per-turn statistics to plot.")
        return ""

    steps = sorted(stats.keys())
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for dim, label in [
        ("grounding", "Observation Grounding"),
        ("action", "Action Coherence"),
        ("temporal", "Temporal Consistency"),
    ]:
        means = [stats[s].get(f"{dim}_mean", float("nan")) for s in steps]
        stds = [stats[s].get(f"{dim}_std", 0) for s in steps]
        means_arr = np.array(means)
        stds_arr = np.array(stds)
        ax.plot(steps, means_arr, marker="o", label=label, color=colors[dim])
        ax.fill_between(
            steps,
            means_arr - stds_arr,
            np.minimum(means_arr + stds_arr, 1.0),
            alpha=0.15,
            color=colors[dim],
        )

    ax.set_xlabel("Turn Index")
    ax.set_ylabel("Score (YES=1, NO=0)")
    title = "ALFWorld: LLM Judge Scores by Turn"
    if config_name:
        title += f" ({config_name})"
    ax.set_title(title)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower left")

    ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"alfworld_scores_over_turns_{ts}.png"
    path = str(Path(output_dir) / fname)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved per-turn scores plot: {path}")

    if show_individual:
        grouped = group_results_by_trajectory(results)
        traj_ids = list(grouped.keys())[:6]
        if traj_ids:
            n_cols = min(3, len(traj_ids))
            n_rows = (len(traj_ids) + n_cols - 1) // n_cols
            fig2, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
            axes_flat = [axes] if len(traj_ids) == 1 else axes.flatten()
            for idx, tid in enumerate(traj_ids):
                ax2 = axes_flat[idx]
                traj_results = grouped[tid]
                t_steps = [extract_step_number(r.get("id", "")) for r in traj_results]
                for dim in ("grounding", "action", "temporal"):
                    vals = []
                    for r in traj_results:
                        extra = r.get("extra") or {}
                        ss = extra.get("scalar_scores") or {}
                        vals.append(ss.get(dim, float("nan")))
                    ax2.plot(t_steps, vals, marker="o", label=dim, color=colors[dim])
                ax2.set_title(f"Traj: {tid}", fontsize=9)
                ax2.set_ylim(-0.05, 1.05)
                if idx == 0:
                    ax2.legend(fontsize=8)
            for idx in range(len(traj_ids), len(axes_flat)):
                axes_flat[idx].set_visible(False)
            fig2.suptitle("Individual Trajectory Scores")
            fig2.tight_layout()
            fname2 = f"alfworld_individual_trajectories_{ts}.png"
            path2 = str(Path(output_dir) / fname2)
            fig2.savefig(path2)
            plt.close(fig2)
            print(f"Saved individual trajectory plot: {path2}")

    return path


def plot_dimension_comparison_bar(
    results: List[Dict[str, Any]],
    output_dir: str,
    config_name: Optional[str] = None,
    timestamp: Optional[str] = None,
) -> str:
    import matplotlib.pyplot as plt
    import numpy as np

    colors = setup_academic_style()
    dims = ["grounding", "action", "temporal"]
    labels = ["Observation\nGrounding", "Action\nCoherence", "Temporal\nConsistency"]
    means = []
    stds = []

    for dim in dims:
        vals = []
        for r in results:
            extra = r.get("extra") or {}
            ss = extra.get("scalar_scores") or {}
            if dim in ss:
                vals.append(ss[dim])
        if vals:
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals)))
        else:
            means.append(0.0)
            stds.append(0.0)

    fig, ax = plt.subplots(figsize=(5, 4))
    x = np.arange(len(dims))
    bar_colors = [colors[d] for d in dims]
    bars = ax.bar(x, means, yerr=stds, color=bar_colors, capsize=4, width=0.5)

    for bar, m in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{m:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean YES Rate")
    ax.set_ylim(0, 1.15)
    title = "ALFWorld: Dimension Comparison"
    if config_name:
        title += f" ({config_name})"
    ax.set_title(title)

    ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"alfworld_dimension_comparison_{ts}.png"
    path = str(Path(output_dir) / fname)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved dimension comparison plot: {path}")
    return path


def generate_all_plots(
    results: List[Dict[str, Any]],
    output_dir: str,
    config_name: Optional[str] = None,
    timestamp: Optional[str] = None,
    show_individual: bool = True,
) -> List[str]:
    plot_paths: List[str] = []
    try:
        path = plot_scores_over_turns(results, output_dir, config_name, timestamp, show_individual)
        if path:
            plot_paths.append(path)
        path = plot_dimension_comparison_bar(results, output_dir, config_name, timestamp)
        if path:
            plot_paths.append(path)
    except ImportError as e:
        print(f"Warning: Could not generate plots. Missing dependency: {e}")
    except Exception as e:
        print(f"Warning: Error generating plots: {e}")
        import traceback

        traceback.print_exc()
    return plot_paths


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ALFWorld LLM-agent validation samples with the LLM judge"
    )
    parser.add_argument(
        "--samples-dir",
        type=str,
        required=True,
        help="Directory containing samples.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results (default: samples-dir/eval_results)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of steps/turns to evaluate (for testing)",
    )
    parser.add_argument(
        "--max-trajectories",
        type=int,
        default=None,
        help="Maximum number of trajectories (samples.jsonl entries) to load",
    )
    parser.add_argument(
        "--trajectory-seed",
        type=int,
        default=42,
        help="Random seed for deterministic trajectory subsampling",
    )
    parser.add_argument(
        "--debug-prompts",
        type=int,
        default=0,
        help="Number of samples for which to save full prompts for debugging",
    )
    parser.add_argument(
        "--rubric",
        type=str,
        choices=["universal", "grounding", "action_coherence", "temporal_consistency"],
        default="universal",
        help="Evaluation rubric to use",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5-nano-2025-08-07",
        help="Model to use for evaluation",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["api", "direct"],
        default="direct",
        help="'api' uses FastAPI endpoint, 'direct' calls verifier directly",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:8000/batch_verify",
        help="FastAPI endpoint URL (for api mode)",
    )
    parser.add_argument("--plot", action="store_true", help="Generate evaluation plots")
    parser.add_argument(
        "--plot-individual",
        action="store_true",
        help="Also plot individual trajectories (up to 6)",
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default=None,
        help="Configuration name for plot titles/filenames",
    )
    parser.add_argument(
        "--posthoc-empty-no",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Post-hoc mark empty reflections as all-NO without LLM call (default: True).",
    )

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = str(Path(args.samples_dir) / "eval_results")

    # ---- Load samples ----
    print(f"Loading samples from: {args.samples_dir}")
    samples = load_samples(
        args.samples_dir,
        max_samples=args.max_samples,
        max_trajectories=args.max_trajectories,
        trajectory_seed=args.trajectory_seed,
    )
    if not samples:
        print("No samples found!")
        return

    # ---- Mechanical checks ----
    print("\n[Mechanical Checks] Running checks...")
    mechanical_results = run_mechanical_prefilter(samples)

    # ---- Split: posthoc empty -> synthetic all-NO ----
    synthetic_results: List[Dict[str, Any]] = []
    if args.posthoc_empty_no:
        samples_for_llm, empty_samples, synthetic_results = split_samples_for_eval(
            samples, mechanical_results
        )
    else:
        samples_for_llm = samples

    # ---- Evaluate via LLM ----
    print(
        f"\nEvaluating {len(samples_for_llm)} samples via LLM "
        f"using rubric '{args.rubric}'..."
    )
    if synthetic_results:
        print(f"  ({len(synthetic_results)} empty reflections scored all-NO without LLM)")
    print(f"Model: {args.model}")
    print(f"Mode: {args.mode}")
    if args.debug_prompts > 0:
        print(f"Debug prompts: saving full prompts for first {args.debug_prompts} samples")

    if args.mode == "api":
        llm_results = await evaluate_via_api(
            samples=samples_for_llm,
            rubric=args.rubric,
            model=args.model,
            api_url=args.api_url,
        )
    else:
        llm_results = await evaluate_direct(
            samples=samples_for_llm,
            rubric=args.rubric,
            model=args.model,
            debug_prompts=args.debug_prompts,
        )

    # Merge LLM + synthetic results
    results = llm_results + synthetic_results

    # Attach mechanical metadata
    results = attach_mechanical_metadata(results, mechanical_results)

    # ---- Compute metrics ----
    metrics = compute_aggregate_metrics(results)
    metrics["rubric"] = args.rubric
    metrics["model"] = args.model
    metrics["samples_dir"] = args.samples_dir
    metrics["llm_evaluated_samples"] = len(samples_for_llm)
    metrics["posthoc_empty_no_samples"] = len(synthetic_results)
    metrics["total_results"] = len(results)

    # Mechanical stats
    metrics["mechanical_empty_reflections"] = sum(
        1 for r in mechanical_results.values() if r["empty_reflection"]
    )
    metrics["mechanical_empty_actions"] = sum(
        1 for r in mechanical_results.values() if r["empty_action"]
    )
    metrics["mechanical_inadmissible_actions"] = sum(
        1 for r in mechanical_results.values() if r["action_inadmissible"]
    )
    metrics["mechanical_repeating_actions"] = sum(
        1 for r in mechanical_results.values() if r["action_repetition"]
    )

    grouped = group_results_by_trajectory(results)
    metrics["num_trajectories"] = len(grouped)
    metrics["trajectory_seed"] = args.trajectory_seed

    print("\n" + "=" * 40)
    print("Evaluation Results:")
    print("=" * 40)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # ---- Save ----
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_results(results, samples, args.output_dir, metrics)

    # ---- Plots ----
    if args.plot:
        print("\nGenerating plots...")
        plot_paths = generate_all_plots(
            results=results,
            output_dir=args.output_dir,
            config_name=args.config_name,
            timestamp=timestamp,
            show_individual=args.plot_individual,
        )
        if plot_paths:
            print(f"Generated {len(plot_paths)} plot(s)")

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())

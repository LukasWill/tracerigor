"""
Evaluate Navigation (AI2-THOR home-robot) VLM validation samples with LLM judge.

Adapted from eval_sokoban_samples.py. Shared traits with Sokoban:
  - VLM agent; current observation is an <image> each turn.
  - Short episodes (1–5 turns).
  - ReflAct format: <reflection>...</reflection><action>...</action>.
  - Comma-separated multi-action turns (1–5 sub-actions).

Navigation-specific (vs. Sokoban):
  - 8 admissible actions: moveahead, moveback, moveleft, moveright,
    rotateleft, rotateright, lookup, lookdown.
  - First-person egocentric scene; agent-relative spatial words.
  - Human Instruction (per-step) names the target object (directly or
    indirectly). The eval pipeline carries it through to the judge prompt.
  - **No ground-truth replay**: there is no lightweight way to recover the
    AI2-THOR scene state offline, and the image itself is the authoritative
    observation. We deliberately omit a `--replay-ground-truth` path.
  - **Mechanical pre-filter**: only empty reflections produce a synthetic
    all-NO. Malformed / oversize / oscillating actions are recorded as
    metadata but still sent to the LLM (the judge handles those signals
    inside Action Coherence and Temporal Consistency).

Usage:
    python -m tracerigor.verifier.scripts.eval_navigation_samples \\
        --samples-dir /path/to/navigation/step_N \\
        --model gpt-5-nano-2025-08-07 --mode direct --plot
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_ACTIONS = {
    "moveahead", "moveback", "moveleft", "moveright",
    "rotateleft", "rotateright", "lookup", "lookdown",
}

OPPOSITES = {
    "moveahead": "moveback", "moveback": "moveahead",
    "moveleft": "moveright", "moveright": "moveleft",
    "rotateleft": "rotateright", "rotateright": "rotateleft",
    "lookup": "lookdown", "lookdown": "lookup",
}

MAX_ACTIONS_PER_STEP = 5  # The agent's training prompt advertises this cap.


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class EvalSample:
    """A single evaluation sample extracted from a multi-turn Navigation trajectory."""

    sample_id: str
    env_id: str
    step_index: int
    instruction: str
    reflection_tokens: str
    action_tokens: str  # Comma-separated, e.g. "moveahead, rotateright"
    image_path: str
    history: List[Dict[str, Any]] = field(default_factory=list)

    # Metrics carried through from the original sample (for downstream analysis).
    original_score: Optional[float] = None
    original_success: Optional[bool] = None
    turn_reward: Optional[float] = None


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def parse_reflact_response(response_text: str) -> Tuple[str, str]:
    """Extract reflection / action from a ReflAct-style assistant turn."""
    reflection = ""
    action = ""
    m = re.search(r"<reflection>(.*?)</reflection>", response_text, re.DOTALL)
    if m:
        reflection = m.group(1).strip()
    m = re.search(r"<action>(.*?)</action>", response_text, re.DOTALL)
    if m:
        action = m.group(1).strip()
    return reflection, action


def extract_executed_actions_from_user_msg(user_text: str) -> Optional[List[str]]:
    """Extract the extracted-valid-action list emitted by the env after a turn.

    Navigation env writes: "After your answer, the extracted valid action is ['moveahead', 'rotateright']."
    Returns the list of action strings (possibly empty), or None if the marker
    is absent (e.g., the initial user message).
    """
    m = re.search(r"the extracted valid action is \[(.*?)\]", user_text, re.DOTALL)
    if not m:
        return None
    raw = m.group(1).strip()
    if not raw:
        return []
    return [a.strip().strip("'\"") for a in raw.split(",") if a.strip()]


def extract_env_feedback_from_user_msg(user_text: str) -> Optional[str]:
    """Extract the env_feedback line from the navigation env's next user message.

    Format: "The environment feedback is: Last action is executed successfully."
    """
    m = re.search(r"The environment feedback is:\s*([^\n]+)", user_text)
    return m.group(1).strip() if m else None


def extract_instruction_from_user_msg(user_text: str) -> str:
    """Extract the Human Instruction line."""
    m = re.search(r"Human Instruction:\s*([^\n]+)", user_text)
    return m.group(1).strip() if m else "N/A"


def extract_samples_from_trajectory(
    raw_sample: Dict[str, Any],
    step_dir: str,
) -> List[EvalSample]:
    """Extract one EvalSample per assistant turn from a navigation trajectory."""
    samples: List[EvalSample] = []
    output_str = raw_sample.get("output_str", "")
    sample_idx = raw_sample.get("sample_idx", 0)
    env_id = raw_sample.get("env_id", "unknown")
    image_paths = raw_sample.get("image_paths", [])
    turn_rewards = raw_sample.get("turn_rewards", [])
    metrics = raw_sample.get("metrics", {})

    assistant_pattern = r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>"
    user_pattern = r"<\|im_start\|>user\n(.*?)<\|im_end\|>"

    assistant_turns = re.findall(assistant_pattern, output_str, re.DOTALL)
    user_turns = re.findall(user_pattern, output_str, re.DOTALL)

    # Instruction is in the initial user message (and repeated in every later one).
    instruction = (
        extract_instruction_from_user_msg(user_turns[0]) if user_turns else "N/A"
    )

    history: List[Dict[str, Any]] = []
    HISTORY_WINDOW = 5  # Matches NavigationUniversalTemplateV2.DEFAULT_HISTORY_WINDOW.

    for i, assistant_content in enumerate(assistant_turns):
        reflection, action = parse_reflact_response(assistant_content)

        if i < len(image_paths):
            img_rel = image_paths[i]
        else:
            img_rel = f"images/sample_{sample_idx}_img_{i}.png"
        img_abs = str(Path(step_dir) / img_rel) if step_dir else img_rel

        reward = turn_rewards[i] if i < len(turn_rewards) else None

        sample = EvalSample(
            sample_id=f"{sample_idx}_{env_id}_step{i + 1}",
            env_id=env_id,
            step_index=i + 1,
            instruction=instruction,
            reflection_tokens=reflection,
            action_tokens=action,
            image_path=img_abs,
            history=list(history),
            original_score=metrics.get("score"),
            original_success=metrics.get("success"),
            turn_reward=reward,
        )
        samples.append(sample)

        # The NEXT user message records (a) which sub-actions the env extracted
        # and (b) the env_feedback for the last sub-action. Both are useful
        # signals for temporal-consistency judgment.
        executed = None
        env_feedback = None
        if i + 1 < len(user_turns):
            executed = extract_executed_actions_from_user_msg(user_turns[i + 1])
            env_feedback = extract_env_feedback_from_user_msg(user_turns[i + 1])

        history.append(
            {
                "step": i + 1,
                "reflection": reflection,
                "action": action,
                "executed_actions": executed,
                "env_feedback": env_feedback,
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
    """Load samples from samples.jsonl."""
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
            eval_samples = extract_samples_from_trajectory(
                raw_sample, step_dir=samples_dir
            )
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


def samples_to_batch_items(
    samples: List[EvalSample],
    rubric: str = "universal",
) -> List[Dict[str, Any]]:
    """Convert EvalSample objects to batch items consumed by the navigation verifier.

    Fields produced (consumed by NavigationUniversalTemplateV2 + binary variants):
      - reasoning_tokens, action_tokens     : raw text (no XML wrapping).
      - current_step / episode_step         : 1-based turn index.
      - history                             : list of {step, reflection, action,
                                              executed_actions_text, env_feedback}.
      - instruction                         : Human Instruction (per-step).
      - current_observation_image           : list with the per-turn image path.
      - admissible_actions                  : fixed 8-action set, for binary
                                              action_coherence rubric.
    """
    items: List[Dict[str, Any]] = []
    admissible = sorted(VALID_ACTIONS)

    for sample in samples:
        enriched_history = []
        for h in sample.history:
            entry = {
                "step": h.get("step"),
                "reflection": h.get("reflection", ""),
                "action": h.get("action", ""),
            }
            executed_actions = h.get("executed_actions")
            if executed_actions is not None:
                entry["executed_actions_text"] = (
                    ", ".join(executed_actions) if executed_actions else "(none)"
                )
                proposed = [a.strip() for a in entry["action"].split(",") if a.strip()]
                if proposed:
                    if executed_actions == proposed:
                        entry["action_outcome_note"] = "All proposed actions extracted."
                    elif not executed_actions:
                        entry["action_outcome_note"] = "No proposed actions extracted."
                    else:
                        entry["action_outcome_note"] = (
                            "Only these sub-actions extracted: "
                            f"{entry['executed_actions_text']}."
                        )
            env_feedback = h.get("env_feedback")
            if env_feedback:
                entry["env_feedback"] = env_feedback
            enriched_history.append(entry)

        item: Dict[str, Any] = {
            "id": sample.sample_id,
            "reasoning_tokens": sample.reflection_tokens,
            "action_tokens": sample.action_tokens,
            "current_step": sample.step_index,
            "episode_step": sample.step_index,
            "instruction": sample.instruction,
            "history": enriched_history,
            "current_observation_text": "",
            "admissible_actions": admissible,
        }

        if sample.image_path and os.path.isfile(sample.image_path):
            item["current_observation_image"] = [sample.image_path]

        items.append(item)

    return items


# ---------------------------------------------------------------------------
# Mechanical Pre-filter (navigation)
# ---------------------------------------------------------------------------
#
# Navigation pre-filter scope:
#  - Hard skip (synthetic all-NO, no LLM call) ONLY for empty reflections.
#  - Soft flags (metadata only, LLM still called):
#       * malformed_action       — action sequence has non-admissible tokens
#                                  OR is empty (often: late-step truncation).
#       * oversize               — proposed sequence has >5 sub-actions
#                                  (exceeds the agent's stated cap).
#       * is_oscillating         — last sub-action and prior turn's last
#                                  sub-action are opposites (e.g.,
#                                  moveleft → moveright).
#
# Rationale: Action Coherence and Temporal Consistency rubrics already
# encode these patterns; bypassing the LLM would break per-turn analysis.
# Empty reflections are the only deterministic all-NO case.


def run_mechanical_prefilter(
    samples: List[EvalSample],
) -> Dict[str, Dict[str, Any]]:
    """Run navigation-specific mechanical checks."""
    results: Dict[str, Dict[str, Any]] = {}

    for sample in samples:
        is_empty = (
            not sample.reflection_tokens or not sample.reflection_tokens.strip()
        )

        raw_actions = (
            [a.strip() for a in sample.action_tokens.split(",") if a.strip()]
            if sample.action_tokens
            else []
        )
        invalid_actions = [a for a in raw_actions if a not in VALID_ACTIONS]
        is_malformed = bool(invalid_actions) or not raw_actions
        is_oversize = len(raw_actions) > MAX_ACTIONS_PER_STEP

        is_oscillating = False
        if sample.history:
            prev = sample.history[-1]
            prev_acts = [
                a.strip() for a in prev.get("action", "").split(",") if a.strip()
            ]
            if prev_acts and raw_actions:
                p_last, cur_first = prev_acts[-1], raw_actions[0]
                if OPPOSITES.get(p_last) == cur_first:
                    is_oscillating = True

        results[sample.sample_id] = {
            "empty_reflection": is_empty,
            "malformed_action": is_malformed,
            "invalid_action_tokens": invalid_actions,
            "oversize_action": is_oversize,
            "is_oscillating": is_oscillating,
            # Only the empty-reflection signal is a hard pre-filter (synthetic
            # all-NO). Other flags are descriptive only.
            "any_failure": is_empty,
            "has_format_or_validity_issue": is_empty or is_malformed,
        }

    n = len(samples)
    if n:
        n_empty = sum(1 for r in results.values() if r["empty_reflection"])
        n_mal = sum(1 for r in results.values() if r["malformed_action"])
        n_inv = sum(1 for r in results.values() if r["invalid_action_tokens"])
        n_over = sum(1 for r in results.values() if r["oversize_action"])
        n_osc = sum(1 for r in results.values() if r["is_oscillating"])
        print("\n[Mechanical Pre-filter] Navigation Results:")
        print(f"  Total samples: {n}")
        print(f"  Empty reflections: {n_empty} ({100 * n_empty / n:.1f}%)")
        print(f"  Malformed (any invalid or empty action): {n_mal} ({100 * n_mal / n:.1f}%)")
        print(f"    of which contain non-admissible tokens: {n_inv}")
        print(f"  Oversize (>{MAX_ACTIONS_PER_STEP} sub-actions): {n_over} ({100 * n_over / n:.1f}%)")
        print(f"  Immediate oscillations: {n_osc} ({100 * n_osc / n:.1f}%)")

    return results


def create_synthetic_result_all_no(sample: EvalSample, reason: str) -> Dict[str, Any]:
    """Synthetic all-NO result for a sample that does not warrant an LLM call."""
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
            "observation_grounding": {
                "yes_no": "NO",
                "evidence": f"Synthetic NO: {reason}",
            },
            "action_coherence": {
                "yes_no": "NO",
                "evidence": f"Synthetic NO: {reason}",
            },
            "temporal_consistency": {
                "yes_no": "NO",
                "evidence": f"Synthetic NO: {reason}",
            },
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
    """Split into LLM-bound samples vs. those scored mechanically."""
    samples_for_llm: List[EvalSample] = []
    empty_samples: List[EvalSample] = []
    synthetic_results: List[Dict[str, Any]] = []

    for sample in samples:
        mech = mechanical_results.get(sample.sample_id, {})
        is_empty = mech.get(
            "empty_reflection",
            not sample.reflection_tokens or not sample.reflection_tokens.strip(),
        )
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
    """Attach mechanical-check metadata to LLM results (descriptive only)."""
    for result in results:
        sample_id = result.get("id", "")
        if sample_id in mechanical_results:
            mech = mechanical_results[sample_id]
            extra = result.get("extra") or {}
            extra["mechanical"] = {
                "empty_reflection": mech["empty_reflection"],
                "malformed_action": mech["malformed_action"],
                "invalid_action_tokens": mech["invalid_action_tokens"],
                "oversize_action": mech["oversize_action"],
                "is_oscillating": mech["is_oscillating"],
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
    import httpx

    items = samples_to_batch_items(samples, rubric)
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
    """Evaluate via the navigation verifier directly (no server)."""
    from tracerigor.verifier.verifier.openai_verifier import (
        _get_hydra_config,
        build_verifier,
        run_openai_verifier,
    )

    items = samples_to_batch_items(samples, rubric)

    if debug_prompts > 0:
        verifier_cls = build_verifier("navigation", rubric)
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
                            part.get("text", "")
                            for part in user_content
                            if part.get("type") == "text"
                        ]
                        n_images = sum(
                            1
                            for part in user_content
                            if part.get("type") == "image_url"
                        )
                        debug_user = "\n".join(p for p in text_parts if p)
                        if n_images:
                            debug_user += f"\n\n[Attached image parts: {n_images}]"
                        item["_debug_user_prompt"] = debug_user or "N/A"
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
        env="navigation",
    )

    for i, result in enumerate(results):
        if i < len(items) and "_debug_system_prompt" in items[i]:
            result["debug_system_prompt"] = items[i]["_debug_system_prompt"]
            result["debug_user_prompt"] = items[i]["_debug_user_prompt"]

    return results


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_aggregate_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {}

    n = len(results)
    metrics: Dict[str, Any] = {
        "num_samples": n,
        "query_success_rate": sum(
            1 for r in results if r.get("query_success", False)
        )
        / n,
        "parse_success_rate": sum(
            1 for r in results if r.get("parse_success", False)
        )
        / n,
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
        metrics["avg_observation_grounding"] = sum(grounding_scores) / len(
            grounding_scores
        )
    if action_scores:
        metrics["avg_action_coherence"] = sum(action_scores) / len(action_scores)
    if temporal_scores:
        metrics["avg_temporal_consistency"] = sum(temporal_scores) / len(
            temporal_scores
        )

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
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results_file = output_path / f"eval_results_{timestamp}.jsonl"
    with open(results_file, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    print(f"Saved raw results to {results_file}")

    metrics_file = output_path / f"eval_metrics_{timestamp}.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_file}")

    report_file = output_path / f"eval_report_{timestamp}.txt"
    result_by_id = {r.get("id", ""): r for r in results}
    with open(report_file, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("Navigation LLM Judge Evaluation Report\n")
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
            f.write(f"  Instruction: {sample.instruction}\n")
            f.write(f"  Image: {sample.image_path}\n")
            f.write(f"  Reflection: {sample.reflection_tokens[:200]}...\n")
            f.write(f"  Action: {sample.action_tokens}\n")
            f.write(f"  Turn Reward: {sample.turn_reward}\n")
            f.write(f"  Score: {result.get('score', 'N/A')}\n")

            extra = result.get("extra") or {}
            ss = extra.get("scalar_scores") or {}
            if ss:
                f.write(
                    f"  Per-dim: grounding={ss.get('grounding', 'N/A')}, "
                    f"action={ss.get('action', 'N/A')}, "
                    f"temporal={ss.get('temporal', 'N/A')}\n"
                )
            f.write(f"  Parse OK: {result.get('parse_success', 'N/A')}\n")
            if result.get("response"):
                f.write(f"  LLM Response: {result['response'][:300]}...\n")

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
    for step in sorted(turn_data):
        stats[step] = {"step": step, "count": 0}
        for dim in ("grounding", "action", "temporal"):
            vals = turn_data[step][dim]
            if vals:
                import numpy as np

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
    title = "Navigation: LLM Judge Scores by Turn"
    if config_name:
        title += f" ({config_name})"
    ax.set_title(title)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower left")

    ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"navigation_scores_over_turns_{ts}.png"
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
            fig2, axes = plt.subplots(
                n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows)
            )
            axes_flat = [axes] if len(traj_ids) == 1 else axes.flatten()
            for idx, tid in enumerate(traj_ids):
                ax2 = axes_flat[idx]
                traj_results = grouped[tid]
                t_steps = [
                    extract_step_number(r.get("id", "")) for r in traj_results
                ]
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
            fname2 = f"navigation_individual_trajectories_{ts}.png"
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
    means: List[float] = []
    stds: List[float] = []

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
    title = "Navigation: Dimension Comparison"
    if config_name:
        title += f" ({config_name})"
    ax.set_title(title)

    ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"navigation_dimension_comparison_{ts}.png"
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
        path = plot_scores_over_turns(
            results, output_dir, config_name, timestamp, show_individual
        )
        if path:
            plot_paths.append(path)

        path = plot_dimension_comparison_bar(
            results, output_dir, config_name, timestamp
        )
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
        description="Evaluate Navigation VLM validation samples with LLM judge"
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
        help="Maximum number of steps/turns to evaluate",
    )
    parser.add_argument(
        "--max-trajectories",
        type=int,
        default=None,
        help="Maximum number of trajectories to load",
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
        choices=[
            "universal",
            "universal_v2",
            "grounding",
            "action_coherence",
            "temporal_consistency",
            "self_consistency",
            "history_consistency",
        ],
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
    parser.add_argument(
        "--plot", action="store_true", help="Generate evaluation plots"
    )
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
        help="Post-hoc mark empty reflections as all-NO without LLM call.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load, run mechanical pre-filter, render prompts; SKIP LLM calls.",
    )

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = str(Path(args.samples_dir) / "eval_results")

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

    print("\n[Mechanical Checks] Running checks...")
    mechanical_results = run_mechanical_prefilter(samples)

    synthetic_results: List[Dict[str, Any]] = []
    if args.posthoc_empty_no:
        samples_for_llm, empty_samples, synthetic_results = split_samples_for_eval(
            samples, mechanical_results
        )
    else:
        samples_for_llm = samples

    print(
        f"\nEvaluating {len(samples_for_llm)} samples via LLM "
        f"using rubric '{args.rubric}'..."
    )
    if synthetic_results:
        print(
            f"  ({len(synthetic_results)} empty reflections scored all-NO without LLM)"
        )
    print(f"Model: {args.model}")
    print(f"Mode: {args.mode}")
    if args.debug_prompts > 0:
        print(
            f"Debug prompts: saving full prompts for first {args.debug_prompts} "
            "samples"
        )

    if args.dry_run:
        print("\n[Dry-run] Skipping LLM evaluation. Rendering first prompt only.")
        from tracerigor.verifier.verifier.openai_verifier import (
            _get_hydra_config,
            build_verifier,
        )

        verifier_cls = build_verifier("navigation", args.rubric)
        verifier = verifier_cls(
            _get_hydra_config(os.getpid()), args.model, {"temperature": 0.0}
        )
        items = samples_to_batch_items(samples_for_llm[:1], args.rubric)
        if items:
            messages = verifier.assemble_messages(dict(items[0]))
            print(f"  System prompt: {len(messages[0]['content'])} chars")
            user_content = messages[1]["content"]
            if isinstance(user_content, list):
                text_chars = sum(
                    len(p.get("text", "")) for p in user_content
                    if p.get("type") == "text"
                )
                n_images = sum(
                    1 for p in user_content if p.get("type") == "image_url"
                )
                print(
                    f"  User prompt: {text_chars} chars text + "
                    f"{n_images} image part(s)"
                )
            else:
                print(f"  User prompt: {len(user_content)} chars (text only)")
        print("[Dry-run] Done.")
        return

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

    results = llm_results + synthetic_results
    results = attach_mechanical_metadata(results, mechanical_results)

    metrics = compute_aggregate_metrics(results)
    metrics["rubric"] = args.rubric
    metrics["model"] = args.model
    metrics["samples_dir"] = args.samples_dir
    metrics["llm_evaluated_samples"] = len(samples_for_llm)
    metrics["posthoc_empty_no_samples"] = len(synthetic_results)
    metrics["total_results"] = len(results)

    n_total = len(samples)
    metrics["mechanical_empty_reflections"] = sum(
        1 for r in mechanical_results.values() if r["empty_reflection"]
    )
    metrics["mechanical_malformed_actions"] = sum(
        1 for r in mechanical_results.values() if r["malformed_action"]
    )
    metrics["mechanical_oversize"] = sum(
        1 for r in mechanical_results.values() if r["oversize_action"]
    )
    metrics["mechanical_oscillating"] = sum(
        1 for r in mechanical_results.values() if r["is_oscillating"]
    )
    metrics["mechanical_total_samples"] = n_total

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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_results(results, samples, args.output_dir, metrics)

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

"""
Evaluate Sokoban VLM validation samples with LLM judge.

Adapted from eval_sciworld_samples.py for the Sokoban domain.

Key differences from SciWorld:
- Observations are IMAGES, not text grids. The agent sees <image> each turn.
- Ground truth is extracted via SokobanEnv replay (sokoban_state_to_sentences).
- Agent format: <reflection>...</reflection><action>Up,Down,Left</action>
  with 1–3 comma-separated actions per turn.
- Short episodes (1–5 turns), 6×6 grid, 1 box.
- No valid_actions or task_description fields (fixed action set, single task type).
- Mechanical pre-filter is simplified (empty reflections, malformed actions).

Usage:
    python -m tracerigor.verifier.scripts.eval_sokoban_samples \
        --samples-dir /path/to/sokoban/step_N \
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
# Data structures
# ---------------------------------------------------------------------------

VALID_ACTIONS = {"Up", "Down", "Left", "Right"}


@dataclass
class EvalSample:
    """A single evaluation sample extracted from a multi-turn Sokoban trajectory."""

    sample_id: str
    env_id: str  # Validation env label (e.g. "val1"), not a stable seed
    step_index: int  # 1-based turn index within trajectory
    reflection_tokens: str
    action_tokens: str  # Comma-separated, e.g. "Left,Left"
    image_path: str  # Relative path to observation image for this turn
    history: List[Dict[str, Any]] = field(default_factory=list)

    # Metrics from original sample (for comparison)
    original_score: Optional[float] = None
    original_success: Optional[bool] = None
    turn_reward: Optional[float] = None


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def parse_reflact_response(response_text: str) -> Tuple[str, str]:
    """Extract reflection and action from ReflAct-style response.

    Format: <reflection>...</reflection><action>...</action>
    """
    reflection = ""
    action = ""

    reflection_match = re.search(
        r"<reflection>(.*?)</reflection>", response_text, re.DOTALL
    )
    if reflection_match:
        reflection = reflection_match.group(1).strip()

    action_match = re.search(r"<action>(.*?)</action>", response_text, re.DOTALL)
    if action_match:
        action = action_match.group(1).strip()

    return reflection, action


def extract_executed_actions_from_user_msg(user_text: str) -> Optional[List[str]]:
    """Extract the actions that the environment actually executed from the user message.

    Format: "After your answer, the extracted valid action is ['Left']."
    Returns None if not found (e.g. the initial user message).
    """
    match = re.search(
        r"the extracted valid action is \[(.*?)\]", user_text, re.DOTALL
    )
    if not match:
        return None
    raw = match.group(1).strip()
    if not raw:
        return []
    # Parse individual quoted action strings
    return [a.strip().strip("'\"") for a in raw.split(",") if a.strip()]


def extract_samples_from_trajectory(
    raw_sample: Dict[str, Any],
    step_dir: str,
) -> List[EvalSample]:
    """Extract individual turn evaluations from a multi-turn Sokoban trajectory.

    Each assistant turn produces one EvalSample.
    """
    samples: List[EvalSample] = []
    output_str = raw_sample.get("output_str", "")
    sample_idx = raw_sample.get("sample_idx", 0)
    env_id = raw_sample.get("env_id", "unknown")
    image_paths = raw_sample.get("image_paths", [])
    turn_rewards = raw_sample.get("turn_rewards", [])
    metrics = raw_sample.get("metrics", {})

    # Split by assistant turns
    assistant_pattern = r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>"
    user_pattern = r"<\|im_start\|>user\n(.*?)<\|im_end\|>"

    assistant_turns = re.findall(assistant_pattern, output_str, re.DOTALL)
    user_turns = re.findall(user_pattern, output_str, re.DOTALL)

    # Build history and samples
    history: List[Dict[str, Any]] = []
    HISTORY_WINDOW = 5

    for i, assistant_content in enumerate(assistant_turns):
        reflection, action = parse_reflact_response(assistant_content)

        # Image for this turn: images/sample_{idx}_img_{i}.png
        if i < len(image_paths):
            img_rel = image_paths[i]
        else:
            img_rel = f"images/sample_{sample_idx}_img_{i}.png"
        img_abs = str(Path(step_dir) / img_rel) if step_dir else img_rel

        # Turn reward
        reward = turn_rewards[i] if i < len(turn_rewards) else None

        sample = EvalSample(
            sample_id=f"{sample_idx}_{env_id}_step{i + 1}",
            env_id=env_id,
            step_index=i + 1,
            reflection_tokens=reflection,
            action_tokens=action,
            image_path=img_abs,
            history=list(history),  # copy
            original_score=metrics.get("score"),
            original_success=metrics.get("success"),
            turn_reward=reward,
        )
        samples.append(sample)

        # Accumulate history
        # Extract actions the env actually executed (from the NEXT user message)
        executed = None
        if i + 1 < len(user_turns):
            executed = extract_executed_actions_from_user_msg(user_turns[i + 1])

        history.append(
            {
                "step": i + 1,
                "reflection": reflection,
                "action": action,
                "executed_actions": executed,
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

    # First pass: load all raw trajectory lines
    raw_lines: List[Tuple[int, str]] = []
    with open(samples_path, "r") as f:
        for line_num, line in enumerate(f):
            raw_lines.append((line_num, line.strip()))

    total_trajectories = len(raw_lines)

    # Randomly subsample trajectories if requested
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
    ground_truth_states: Optional[Dict[str, Dict]] = None,
) -> List[Dict[str, Any]]:
    """Convert EvalSample objects to batch items for the verifier API.

        The Sokoban V2 template expects:
        - reasoning_tokens, action_tokens (raw text, NOT xml-wrapped)
        - history (list of dicts)
        - episode_step / current_step
        - ground_truth_state_text (optional, from replay)
        - current_observation_image (list of image paths for multimodal)

        Design note:
        - The current observation should remain the image whenever available.
        - Replay-derived ground truth is auxiliary and should stay in a separate
            prompt section, not replace the image observation.
    """
    items: List[Dict[str, Any]] = []

    for sample in samples:
        # Build history with optional ground truth text per turn
        enriched_history = []
        for h in sample.history:
            entry = dict(h)
            executed_actions = entry.get("executed_actions")
            if executed_actions is not None:
                entry["executed_actions_text"] = (
                    ", ".join(executed_actions) if executed_actions else "(none)"
                )
                proposed_actions = [
                    a.strip() for a in entry.get("action", "").split(",") if a.strip()
                ]
                if proposed_actions:
                    if executed_actions == proposed_actions:
                        entry["action_outcome_note"] = "All proposed actions executed."
                    elif not executed_actions:
                        entry["action_outcome_note"] = "No proposed actions executed."
                    else:
                        entry["action_outcome_note"] = (
                            "Only these actions executed: "
                            f"{entry['executed_actions_text']}."
                        )
            # If replay data is available, attach state text for history turns
            hist_id = f"{sample.sample_id.rsplit('_step', 1)[0]}_step{h['step']}"
            if ground_truth_states and hist_id in ground_truth_states:
                entry["observation_state_text"] = ground_truth_states[hist_id].get(
                    "state_text", ""
                )
            enriched_history.append(entry)

        item: Dict[str, Any] = {
            "id": sample.sample_id,
            "reasoning_tokens": sample.reflection_tokens,
            "action_tokens": sample.action_tokens,
            "current_step": sample.step_index,
            "episode_step": sample.step_index,
            "history": enriched_history,
            "current_observation_text": "",
            "ground_truth_state_text": "",
        }

        # Image observation for multimodal judge
        has_current_image = bool(sample.image_path and os.path.isfile(sample.image_path))
        if has_current_image:
            item["current_observation_image"] = [sample.image_path]

        # Ground truth from replay stays separate from the current observation.
        if ground_truth_states and sample.sample_id in ground_truth_states:
            gt = ground_truth_states[sample.sample_id]
            item["ground_truth_state_text"] = gt.get("state_text", "")
            if not has_current_image:
                item["current_observation_text"] = item["ground_truth_state_text"]

        items.append(item)

    return items


# ---------------------------------------------------------------------------
# Trajectory Replay for Ground Truth
# ---------------------------------------------------------------------------


def recover_test_seeds(
    env_config_kwargs: Optional[Dict[str, Any]] = None,
    test_size: int = 128,
    global_seed: int = 42,
    verbose: bool = True,
) -> List[int]:
    """Deterministically recover the test seeds used during dataset creation.

    The dataset is created via create_dataset.py which calls
    ``env_config.generate_seeds(test_size, seed=global_seed)``.
    By replaying this with the same parameters we recover the exact seed list.

    Returns:
        Ordered list of integer seeds (index *i* → seed for the *i*-th test
        instance, which was assigned ``env_id = "val{i+1}"`` during the first
        validation step, assuming sequential counter assignment).
    """
    try:
        from tracerigor.env.sokoban.env_config import SokobanEnvConfig
    except ImportError as e:
        if verbose:
            print(f"Warning: Cannot recover seeds — import failed: {e}")
        return []

    config_kwargs = env_config_kwargs or {}
    config = SokobanEnvConfig(**config_kwargs)
    if verbose:
        print(f"  Recovering {test_size} test seeds (global_seed={global_seed})...")
    seeds = config.generate_seeds(test_size, seed=global_seed)
    if verbose:
        print(f"  Recovered {len(seeds)} seeds (first 5: {seeds[:5]})")
    return seeds


def build_env_id_to_seed_map(
    seeds: List[int],
    env_ids: List[str],
) -> Dict[str, int]:
    """Build a mapping from env_id → original seed.

    IMPORTANT: This assumes env_ids were assigned in sequential counter order
    during the FIRST validation step (val1, val2, ..., valN).  This holds when
    all environments are freshly created (no reuse from a previous batch).
    For later validation steps where environments may be recycled/reused,
    the mapping may be INCORRECT.

    Args:
        seeds: Ordered list of seeds from ``recover_test_seeds``.
        env_ids: Unique env_ids observed in the samples (e.g. ["val1", "val10", ...]).

    Returns:
        Dict mapping env_id → seed.  Entries that cannot be mapped are omitted.
    """
    # Extract integer part from env_ids and sort by it
    id_to_int = {}
    for eid in env_ids:
        m = re.match(r"val(\d+)$", eid)
        if m:
            id_to_int[eid] = int(m.group(1))

    # Counter-based assignment: val1 = seeds[0], val2 = seeds[1], etc.
    mapping: Dict[str, int] = {}
    for eid, idx in id_to_int.items():
        # Counter starts at 1, seed list is 0-indexed
        seed_idx = idx - 1
        if 0 <= seed_idx < len(seeds):
            mapping[eid] = seeds[seed_idx]

    return mapping


def extract_explicit_env_id_to_seed_map(
    raw_samples: List[Dict[str, Any]],
    verbose: bool = True,
) -> Dict[str, int]:
    """Extract a trustworthy env_id -> seed map if samples already store seeds."""
    mapping: Dict[str, int] = {}
    conflicts: List[Tuple[str, int, int]] = []

    for raw_sample in raw_samples:
        env_id = raw_sample.get("env_id")
        if not env_id:
            continue

        seed = raw_sample.get("seed")
        if seed is None:
            extra_info = raw_sample.get("extra_info")
            if isinstance(extra_info, dict):
                seed = extra_info.get("seed")

        if seed is None:
            continue

        try:
            seed_value = int(seed)
        except (TypeError, ValueError):
            continue

        prior = mapping.get(env_id)
        if prior is not None and prior != seed_value:
            conflicts.append((env_id, prior, seed_value))
            continue
        mapping[env_id] = seed_value

    if conflicts:
        if verbose:
            first_env_id, first_a, first_b = conflicts[0]
            print(
                "  WARNING: Conflicting explicit seeds found for "
                f"{first_env_id} ({first_a} vs {first_b}). "
                "Discarding explicit replay seed metadata."
            )
        return {}

    if mapping and verbose:
        print(
            f"  Found explicit seed metadata for {len(mapping)} env_ids in samples.jsonl"
        )
    return mapping


def resolve_replay_seed_map(
    raw_samples: List[Dict[str, Any]],
    samples_dir: str,
    replay_global_seed: int,
    verbose: bool = True,
) -> Tuple[Optional[Dict[str, int]], str]:
    """Resolve a trustworthy env_id -> seed map for replay.

    Safe cases:
    - samples.jsonl already stores explicit seeds
    - the directory is the initial validation snapshot (step_0), where env_id
      counter assignment still matches the original seed order

    Unsafe case:
    - later checkpoints, where rollout code may recycle env_ids across seeds
    """
    explicit_map = extract_explicit_env_id_to_seed_map(raw_samples, verbose=verbose)
    if explicit_map:
        return explicit_map, "explicit_samples_metadata"

    step_name = Path(samples_dir).name
    if step_name == "step_0":
        env_ids = list({sample.get("env_id", "unknown") for sample in raw_samples})
        seeds = recover_test_seeds(
            env_config_kwargs={
                "dim_room": (6, 6),
                "max_steps": 100,
                "num_boxes": 1,
                "render_mode": "text",
            },
            test_size=len(env_ids),
            global_seed=replay_global_seed,
            verbose=verbose,
        )
        mapping = build_env_id_to_seed_map(seeds, env_ids) if seeds else {}
        if mapping and verbose:
            print(
                "  Using step_0 counter-based env_id -> seed mapping. "
                "This is only trusted for the initial validation snapshot."
            )
        return (mapping or None), "step0_counter_assignment"

    if verbose:
        print("  No trustworthy seed metadata found in samples.jsonl.")
        print(
            f"  {samples_dir} is not step_0, and later checkpoints may recycle "
            "env_ids across different seeds."
        )
        print(
            "  Replay-derived ground truth has been disabled to avoid "
            "incorrect observation labels."
        )
    return None, "disabled_untrusted_env_id_mapping"


def extract_ground_truth_via_replay(
    raw_samples: List[Dict[str, Any]],
    env_id_to_seed: Optional[Dict[str, int]] = None,
    verbose: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """Replay Sokoban trajectories to extract ground truth spatial state at each turn.

    Args:
        raw_samples: Raw trajectory dicts from samples.jsonl.
        env_id_to_seed: Trustworthy mapping from env_id (e.g. "val1") to the
            original integer seed used during training. Trajectories without a
            seed map are skipped.
        verbose: Print progress.

    Returns:
        Dict mapping sample_id -> {"state_text": str, "step_index": int, ...}
    """
    try:
        from tracerigor.env.sokoban.env import SokobanEnv, SokobanEnvConfig
        from tracerigor.env.sokoban.utils import sokoban_state_to_sentences
    except ImportError as e:
        print(f"Warning: Could not import Sokoban env: {e}")
        print("  Ground truth extraction requires 'gym' and the Sokoban environment.")
        return {}

    ground_truth: Dict[str, Dict[str, Any]] = {}
    config = SokobanEnvConfig(
        dim_room=(6, 6),
        max_steps=100,
        num_boxes=1,
        render_mode="text",  # Replay only needs env state, not image rendering
    )
    env = SokobanEnv(config)
    n_success = 0
    n_fail = 0
    n_no_seed = 0

    for raw_sample in raw_samples:
        sample_idx = raw_sample.get("sample_idx", 0)
        env_id = raw_sample.get("env_id", "unknown")
        output_str = raw_sample.get("output_str", "")

        if not env_id_to_seed or env_id not in env_id_to_seed:
            if verbose and n_no_seed < 5:
                print(
                    f"  Skipping {sample_idx}/{env_id}: no trustworthy seed mapping available"
                )
            n_no_seed += 1
            continue

        seed = env_id_to_seed[env_id]

        # Parse all actions from trajectory
        assistant_pattern = r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>"
        assistant_turns = re.findall(assistant_pattern, output_str, re.DOTALL)

        turn_actions: List[List[str]] = []
        for turn_text in assistant_turns:
            _, action_str = parse_reflact_response(turn_text)
            actions = [a.strip() for a in action_str.split(",") if a.strip()] if action_str else []
            turn_actions.append(actions)

        if not turn_actions:
            continue

        # Replay trajectory
        try:
            env.reset(seed=seed)

            # Ground truth at initial state (before any action)
            state = env.get_env_state(to_relative=False)
            state_text = "; ".join(sokoban_state_to_sentences(state))
            sample_id_step0 = f"{sample_idx}_{env_id}_step1"
            ground_truth[sample_id_step0] = {
                "state_text": state_text,
                "step_index": 1,
                "raw_state": state,
            }

            # Execute each turn's actions
            for turn_idx, actions in enumerate(turn_actions):
                for action_name in actions:
                    if action_name in SokobanEnv.ACTION_LOOKUP:
                        action_id = SokobanEnv.ACTION_LOOKUP[action_name]
                        env.env.step(action_id)

                # Record state after this turn's actions (= observation for next turn)
                if turn_idx + 1 < len(turn_actions):
                    state = env.get_env_state(to_relative=False)
                    state_text = "; ".join(sokoban_state_to_sentences(state))
                    next_step = turn_idx + 2
                    sample_id = f"{sample_idx}_{env_id}_step{next_step}"
                    ground_truth[sample_id] = {
                        "state_text": state_text,
                        "step_index": next_step,
                        "raw_state": state,
                    }

            n_success += 1
        except Exception as e:
            if verbose:
                print(f"  Warning: Replay failed for {sample_idx}/{env_id}: {e}")
            n_fail += 1

    if verbose:
        print(
            f"  Ground truth extraction: {n_success} trajectories replayed "
            f"({n_fail} failures, {n_no_seed} without seed map), "
            f"{len(ground_truth)} step states extracted."
        )
    return ground_truth


# ---------------------------------------------------------------------------
# Mechanical Pre-filter (simplified for Sokoban)
# ---------------------------------------------------------------------------


def run_mechanical_prefilter(
    samples: List[EvalSample],
) -> Dict[str, Dict[str, Any]]:
    """Run simplified mechanical checks on Sokoban samples.

    Checks:
    - empty_reflection: no reflection text
    - malformed_action: action tokens not in {Up, Down, Left, Right} (comma-sep)
    - action_is_oscillating: last 2+ turns alternate opposite directions
    """
    results: Dict[str, Dict[str, Any]] = {}

    for sample in samples:
        is_empty = not sample.reflection_tokens or not sample.reflection_tokens.strip()

        # Parse and validate actions
        raw_actions = [
            a.strip() for a in sample.action_tokens.split(",") if a.strip()
        ] if sample.action_tokens else []
        invalid_actions = [a for a in raw_actions if a not in VALID_ACTIONS]
        is_malformed = bool(invalid_actions) or not raw_actions

        # Check for oscillation in history
        is_oscillating = False
        opposites = {
            ("Up", "Down"),
            ("Down", "Up"),
            ("Left", "Right"),
            ("Right", "Left"),
        }
        if len(sample.history) >= 2:
            prev_actions = [h.get("action", "") for h in sample.history[-2:]]
            # Check if current action reverses the previous
            if len(prev_actions) == 2:
                a1_set = {a.strip() for a in prev_actions[0].split(",") if a.strip()}
                a2_set = {a.strip() for a in prev_actions[1].split(",") if a.strip()}
                cur_set = set(raw_actions)
                # Simple check: if prev two + current form A, B, A pattern
                if a1_set == cur_set and a2_set != cur_set and len(a1_set) > 0:
                    is_oscillating = True

        results[sample.sample_id] = {
            "empty_reflection": is_empty,
            "malformed_action": is_malformed,
            "invalid_action_tokens": invalid_actions,
            "is_oscillating": is_oscillating,
            "any_failure": is_empty,  # Only count empty reflection as hard failure
            "has_format_or_validity_issue": is_empty or is_malformed,
        }

    # Summary
    n = len(samples)
    if n:
        n_empty = sum(1 for r in results.values() if r["empty_reflection"])
        n_malformed = sum(1 for r in results.values() if r["malformed_action"])
        n_osc = sum(1 for r in results.values() if r["is_oscillating"])
        print(f"\n[Mechanical Pre-filter] Sokoban Results:")
        print(f"  Total samples: {n}")
        print(f"  Empty reflections: {n_empty} ({100 * n_empty / n:.1f}%)")
        print(f"  Malformed actions: {n_malformed} ({100 * n_malformed / n:.1f}%)")
        print(f"  Oscillating patterns: {n_osc} ({100 * n_osc / n:.1f}%)")

    return results


def create_synthetic_result_all_no(sample: EvalSample, reason: str) -> Dict[str, Any]:
    """Create a synthetic all-NO result for a sample without calling the LLM."""
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
    """Split samples into those needing LLM eval and those scored mechanically.

    Empty reflections → synthetic all-NO. All other samples go to LLM.
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
                "malformed_action": mech["malformed_action"],
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
    ground_truth_states: Optional[Dict[str, Dict]] = None,
) -> List[Dict[str, Any]]:
    """Evaluate samples using the FastAPI endpoint."""
    import httpx

    items = samples_to_batch_items(samples, rubric, ground_truth_states)
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
    ground_truth_states: Optional[Dict[str, Dict]] = None,
) -> List[Dict[str, Any]]:
    """Evaluate samples directly using the Sokoban verifier (no server needed)."""
    from tracerigor.verifier.verifier.openai_verifier import (
        _get_hydra_config,
        build_verifier,
        run_openai_verifier,
    )
    import os

    items = samples_to_batch_items(samples, rubric, ground_truth_states)

    # Debug prompts: generate and attach full prompts
    if debug_prompts > 0:
        verifier_cls = build_verifier("sokoban", rubric)
        verifier = verifier_cls(
            _get_hydra_config(os.getpid()),
            model,
            {"temperature": 0.0},
        )
        for i, item in enumerate(items[:debug_prompts]):
            try:
                messages = verifier.assemble_messages(dict(item))
                item["_debug_system_prompt"] = messages[0]["content"] if messages else "N/A"
                if len(messages) > 1:
                    user_content = messages[1]["content"]
                    if isinstance(user_content, list):
                        text_parts = [
                            part.get("text", "")
                            for part in user_content
                            if part.get("type") == "text"
                        ]
                        n_images = sum(
                            1 for part in user_content if part.get("type") == "image_url"
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
        env="sokoban",
    )

    # Copy debug prompts to results
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
        f.write("Sokoban LLM Judge Evaluation Report\n")
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
    """e.g. '0_val1_step1' -> '0_val1'"""
    parts = sample_id.rsplit("_step", 1)
    return parts[0] if len(parts) > 1 else sample_id


def extract_step_number(sample_id: str) -> int:
    """e.g. '0_val1_step1' -> 1"""
    match = re.search(r"_step(\d+)$", sample_id)
    return int(match.group(1)) if match else 0


def group_results_by_trajectory(
    results: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Group results by trajectory id."""
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for r in results:
        tid = extract_trajectory_id(r.get("id", ""))
        grouped.setdefault(tid, []).append(r)
    # Sort each group by step
    for tid in grouped:
        grouped[tid].sort(key=lambda r: extract_step_number(r.get("id", "")))
    return grouped


def compute_per_turn_statistics(
    results: List[Dict[str, Any]],
) -> Dict[int, Dict[str, Any]]:
    """Compute per-turn-index statistics across all trajectories."""
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
    """Plot per-dimension scores across turns (aggregate + optional individual)."""
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
        ax.plot(steps, means_arr, marker="o", label=label, color=colors[dim.split("_")[0] if "_" not in dim else dim])
        ax.fill_between(
            steps,
            means_arr - stds_arr,
            np.minimum(means_arr + stds_arr, 1.0),
            alpha=0.15,
            color=colors[dim.split("_")[0] if "_" not in dim else dim],
        )

    ax.set_xlabel("Turn Index")
    ax.set_ylabel("Score (YES=1, NO=0)")
    title = "Sokoban: LLM Judge Scores by Turn"
    if config_name:
        title += f" ({config_name})"
    ax.set_title(title)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower left")

    ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"sokoban_scores_over_turns_{ts}.png"
    path = str(Path(output_dir) / fname)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved per-turn scores plot: {path}")

    # Optional: individual trajectories
    if show_individual:
        grouped = group_results_by_trajectory(results)
        traj_ids = list(grouped.keys())[:6]  # Limit to 6
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
            fname2 = f"sokoban_individual_trajectories_{ts}.png"
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
    """Bar chart comparing mean YES rate across 3 dimensions."""
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

    # Annotate bars
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
    title = "Sokoban: Dimension Comparison"
    if config_name:
        title += f" ({config_name})"
    ax.set_title(title)

    ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"sokoban_dimension_comparison_{ts}.png"
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
    """Generate all evaluation plots."""
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
        description="Evaluate Sokoban VLM validation samples with LLM judge"
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
        choices=["universal", "universal_v2", "universal_v2_temporalfix", "grounding", "self_consistency"],
        default="universal",
        help="Evaluation rubric to use",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5.4-nano-2026-03-17",
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
        "--replay-ground-truth",
        action="store_true",
        help="Replay trajectories in SokobanEnv to get ground truth spatial state. "
        "Requires 'gym' and the Sokoban env.",
    )
    parser.add_argument(
        "--replay-global-seed",
        type=int,
        default=42,
        help="Global seed used during dataset creation (default: 42). "
        "Used to recover original seeds for replay.",
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

    # ---- Ground truth via replay (optional) ----
    replay_requested = args.replay_ground_truth
    replay_mode = "disabled"
    ground_truth_states: Optional[Dict[str, Dict]] = None
    if replay_requested:
        print("\n[Ground Truth] Recovering seeds and replaying trajectories...")
        # Load raw samples for replay
        raw_samples = []
        samples_path = Path(args.samples_dir) / "samples.jsonl"
        with open(samples_path, "r") as f:
            for line in f:
                raw_samples.append(json.loads(line.strip()))

        env_ids = list({sample.get("env_id", "unknown") for sample in raw_samples})
        env_id_to_seed, replay_mode = resolve_replay_seed_map(
            raw_samples,
            args.samples_dir,
            args.replay_global_seed,
        )
        if env_id_to_seed:
            print(
                f"  Trusting {len(env_id_to_seed)}/{len(env_ids)} env_ids for replay "
                f"({replay_mode})"
            )
            ground_truth_states = extract_ground_truth_via_replay(
                raw_samples, env_id_to_seed=env_id_to_seed
            )
        else:
            print(
                "  Continuing without replay-derived ground truth; "
                "observation grounding will use the current image only."
            )
    else:
        print(
            "\n[Ground Truth] Skipped (no --replay-ground-truth). "
            "Action coherence and temporal consistency are fully evaluable. "
            "Observation grounding requires a multimodal judge or replay data."
        )

    # ---- Split: posthoc empty → all-NO ----
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
        print(
            f"  ({len(synthetic_results)} empty reflections scored all-NO without LLM)"
        )
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
            ground_truth_states=ground_truth_states,
        )
    else:
        llm_results = await evaluate_direct(
            samples=samples_for_llm,
            rubric=args.rubric,
            model=args.model,
            debug_prompts=args.debug_prompts,
            ground_truth_states=ground_truth_states,
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
    metrics["replay_ground_truth"] = ground_truth_states is not None
    metrics["replay_ground_truth_requested"] = replay_requested
    metrics["replay_ground_truth_mode"] = replay_mode

    # Mechanical stats
    n_total = len(samples)
    metrics["mechanical_empty_reflections"] = sum(
        1 for r in mechanical_results.values() if r["empty_reflection"]
    )
    metrics["mechanical_malformed_actions"] = sum(
        1 for r in mechanical_results.values() if r["malformed_action"]
    )
    metrics["mechanical_oscillating"] = sum(
        1 for r in mechanical_results.values() if r["is_oscillating"]
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

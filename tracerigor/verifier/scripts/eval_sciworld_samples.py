#!/usr/bin/env python3
"""
SciWorld Validation Sample Evaluator

Loads validation samples from samples.jsonl and evaluates them using the LLM judge pipeline.
Supports both the FastAPI endpoint (online) and direct async evaluation (offline).

Usage:
    # Using FastAPI endpoint (requires server running)
    python eval_sciworld_samples.py --samples-dir /path/to/sciworld/step_N --mode api

    # Direct evaluation (no server needed)
    python eval_sciworld_samples.py --samples-dir /path/to/sciworld/step_N --mode direct

    # Test with small batch
    python eval_sciworld_samples.py --samples-dir /path/to/sciworld/step_N --max-samples 5 --mode direct
"""

import argparse
import asyncio
import json
import re
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import defaultdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


@dataclass
class EvalSample:
    """A single evaluation sample extracted from a multi-turn trajectory."""
    sample_id: str
    env_id: str
    step_index: int
    task_description: str
    current_observation_text: str
    reflection_tokens: str
    action_tokens: str
    valid_actions: str
    history: List[Dict[str, Any]] = field(default_factory=list)

    # Observation AFTER the action was taken (used for mechanical validity check)
    next_observation_text: str = ""

    # Metrics from original sample (for comparison)
    original_format_reward: Optional[float] = None
    original_task_reward: Optional[float] = None


def parse_reflact_response(response_text: str) -> tuple[str, str]:
    """Extract reflection and action from ReflAct-style response.

    Format: <reflection>...</reflection><action>...</action>
    """
    reflection = ""
    action = ""

    # Extract reflection
    reflection_match = re.search(r'<reflection>(.*?)</reflection>', response_text, re.DOTALL)
    if reflection_match:
        reflection = reflection_match.group(1).strip()

    # Extract action
    action_match = re.search(r'<action>(.*?)</action>', response_text, re.DOTALL)
    if action_match:
        action = action_match.group(1).strip()

    return reflection, action


def extract_task_description(output_str: str) -> str:
    """Extract task description from the output string."""
    # Look for "Your current task is:" pattern
    match = re.search(r'Your current task is:\s*(.+?)(?:\n|Your current observation)', output_str, re.DOTALL)
    if match:
        return match.group(1).strip()
    return "N/A"


def extract_current_observation(turn_text: str) -> str:
    """Extract current observation from a turn."""
    # Look for observation after "Your current observation is:" (case-insensitive)
    # Also handle variations like "your current observation is:" (lowercase)
    match = re.search(
        r'[Yy]our current observation is:\s*(.+?)(?:Current available actions:|Valid_actions:|$)',
        turn_text,
        re.DOTALL | re.IGNORECASE
    )
    if match:
        return match.group(1).strip()
    return "N/A"


def extract_valid_actions(turn_text: str) -> str:
    """Extract valid actions from a turn.

    Format: Valid_actions: ['action1 OBJ', ...], OBJ needs to be replaced with one of the following objects: ['obj1', ...]
    Returns the full string including both the action templates and the object substitution list.
    """
    # Match the full valid_actions string including the OBJ substitution list
    # Pattern captures: "Valid_actions: [...], OBJ needs to be replaced with one of the following objects: [...]"
    full_pattern = r"Valid_actions:\s*(\[.*?\]),\s*OBJ needs to be replaced with one of the following objects:\s*(\[.*?\])"
    match = re.search(full_pattern, turn_text, re.DOTALL)
    if match:
        action_templates = match.group(1)
        object_list = match.group(2)
        return f"Action templates: {action_templates}, Valid objects for OBJ substitution: {object_list}"

    # Fallback: try to match just the action list (for backward compatibility)
    simple_match = re.search(r"Valid_actions:\s*(\[.*?\])", turn_text, re.DOTALL)
    if simple_match:
        return f"Action templates: {simple_match.group(1)}"
    return "N/A"


def extract_samples_from_trajectory(raw_sample: Dict[str, Any]) -> List[EvalSample]:
    """Extract individual step evaluations from a multi-turn trajectory.

    The output_str contains the full conversation. We extract each assistant turn
    for evaluation.
    """
    samples = []
    output_str = raw_sample.get("output_str", "")
    sample_idx = raw_sample.get("sample_idx", 0)
    env_id = raw_sample.get("env_id", "unknown")

    # Extract task description (appears at the beginning)
    task_desc = extract_task_description(output_str)

    # Split by assistant turns
    # Pattern: <|im_start|>assistant\n...content...<|im_end|>
    assistant_pattern = r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>'
    user_pattern = r'<\|im_start\|>user\n(.*?)<\|im_end\|>'

    assistant_turns = re.findall(assistant_pattern, output_str, re.DOTALL)
    user_turns = re.findall(user_pattern, output_str, re.DOTALL)

    # Build history and samples
    history = []

    for i, (assistant_content, user_content) in enumerate(zip(assistant_turns, user_turns[1:] if len(user_turns) > 1 else [])):
        reflection, action = parse_reflact_response(assistant_content)

        if not reflection and not action:
            continue

        # Get observation from the current user turn (before this action)
        current_obs = extract_current_observation(user_turns[i] if i < len(user_turns) else "")
        valid_actions = extract_valid_actions(user_turns[i] if i < len(user_turns) else "")

        # Get observation AFTER the action was taken (from the NEXT user turn)
        # This is critical for mechanical action validity check
        next_obs = extract_current_observation(user_turns[i + 1] if i + 1 < len(user_turns) else "")

        sample = EvalSample(
            sample_id=f"{sample_idx}_{env_id}_step{i+1}",
            env_id=env_id,
            step_index=i + 1,
            task_description=task_desc,
            current_observation_text=current_obs,
            reflection_tokens=reflection,
            action_tokens=action,
            valid_actions=valid_actions,
            history=history.copy(),
            next_observation_text=next_obs,
            original_format_reward=raw_sample.get("metrics", {}).get("format_reward"),
            original_task_reward=raw_sample.get("metrics", {}).get("task_reward"),
        )
        samples.append(sample)

        # Add to history for next iteration
        # Keep history window aligned with rollout's window_size (default 5)
        # NOTE: Do NOT truncate observation/reflection text here.
        # The LLM agent saw full text during generation (rollout manager does not truncate).
        # The judge must see the same context to correctly assess temporal consistency.
        HISTORY_WINDOW = 5  # Match rollout config's window_size
        history.append({
            "step": i + 1,
            "observation_text": current_obs,
            "action": action,
            "reflection": reflection,
        })
        if len(history) > HISTORY_WINDOW:
            history = history[-HISTORY_WINDOW:]

    return samples


def load_samples(
    samples_dir: str,
    max_samples: Optional[int] = None,
    max_trajectories: Optional[int] = None,
    trajectory_seed: Optional[int] = 42,
) -> List[EvalSample]:
    """Load samples from samples.jsonl file.

    Args:
        samples_dir: Directory containing samples.jsonl
        max_samples: Maximum number of steps/turns to load (across all trajectories)
        max_trajectories: Maximum number of trajectories (lines in samples.jsonl) to load.
                         When set, randomly samples trajectories (seeded for reproducibility)
                         rather than taking the first N, to avoid ordering bias.
        trajectory_seed: Random seed for trajectory subsampling. Default 42.
    """
    samples_path = Path(samples_dir) / "samples.jsonl"

    if not samples_path.exists():
        raise FileNotFoundError(f"Samples file not found: {samples_path}")

    # First pass: load all raw trajectory lines
    raw_lines = []
    with open(samples_path, 'r') as f:
        for line_num, line in enumerate(f):
            raw_lines.append((line_num, line.strip()))

    total_trajectories = len(raw_lines)

    # Randomly subsample trajectories if max_trajectories is set
    if max_trajectories and max_trajectories < total_trajectories:
        import random
        rng = random.Random(trajectory_seed)
        selected_indices = sorted(rng.sample(range(total_trajectories), max_trajectories))
        raw_lines = [raw_lines[i] for i in selected_indices]
        print(f"Randomly subsampled {max_trajectories} of {total_trajectories} trajectories (seed={trajectory_seed})")

    # Second pass: extract eval samples from selected trajectories
    all_eval_samples = []
    trajectories_loaded = 0

    for line_num, line in raw_lines:
        # Check sample (step) limit
        if max_samples and len(all_eval_samples) >= max_samples:
            break

        try:
            raw_sample = json.loads(line)
            eval_samples = extract_samples_from_trajectory(raw_sample)
            all_eval_samples.extend(eval_samples)
            trajectories_loaded += 1

            # Trim to max_samples if exceeded
            if max_samples and len(all_eval_samples) >= max_samples:
                all_eval_samples = all_eval_samples[:max_samples]
                break

        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse line {line_num}: {e}")
            continue

    print(f"Loaded {len(all_eval_samples)} evaluation samples from {trajectories_loaded} trajectories")
    return all_eval_samples


def samples_to_batch_items(
    samples: List[EvalSample],
    rubric: str = "universal",
    ground_truth_states: Optional[Dict[str, Dict]] = None,
) -> List[Dict[str, Any]]:
    """Convert EvalSample objects to batch items for the verifier API.

    Args:
        samples: List of EvalSample objects
        rubric: Evaluation rubric name
        ground_truth_states: Optional dict mapping sample_id -> ground truth info
                            (from trajectory replay). If provided, ground truth
                            location will be included in the prompt.
    """
    items = []

    for sample in samples:
        item = {
            "id": sample.sample_id,
            "reasoning_tokens": f"<reflection>{sample.reflection_tokens}</reflection>",
            "current_observation_text": sample.current_observation_text,
            "current_step": sample.step_index,
            "history": sample.history,
            # SciWorld-specific fields: raw text without XML tags.
            # The user prompt template inserts these directly, so tags would
            # cause double-wrapping and inconsistency with how reflection is shown.
            "reflection_tokens": sample.reflection_tokens,
            "action_tokens": sample.action_tokens,
            "task_description": sample.task_description,
            "valid_actions": sample.valid_actions,
        }

        # Add ground truth if available (from trajectory replay)
        if ground_truth_states and sample.sample_id in ground_truth_states:
            gt = ground_truth_states[sample.sample_id]
            item["ground_truth_location"] = gt.get("location")
            item["ground_truth_inventory"] = gt.get("inventory", [])

        items.append(item)

    return items


# =============================================================================
# Trajectory Replay for Ground Truth Extraction
# =============================================================================

def extract_ground_truth_via_replay(
    raw_samples: List[Dict[str, Any]],
    verbose: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Replay trajectories to extract ground truth state at each step.

    This function launches a SciWorld environment, replays the recorded actions,
    and extracts ground truth location/inventory from the object tree.

    Args:
        raw_samples: List of raw sample dictionaries from samples.jsonl
                    Each should have 'output_str' with the full trajectory
        verbose: Whether to print progress

    Returns:
        Dictionary mapping sample_id -> ground truth state dict

    Note: This requires:
    1. scienceworld package installed
    2. Task info extractable from trajectory (we attempt heuristic extraction)
    """
    try:
        from tracerigor.verifier.verifier.sciworld_trajectory_replay import (
            TrajectoryReplayer,
            extract_task_info_from_description,
            GroundTruthState,
        )
    except ImportError as e:
        print(f"Warning: Could not import trajectory replay module: {e}")
        return {}

    # Check if scienceworld is available
    try:
        from scienceworld import ScienceWorldEnv
    except ImportError:
        print("Warning: scienceworld package not installed. Cannot replay trajectories.")
        print("  Install with: pip install scienceworld")
        return {}

    ground_truth_results = {}

    for raw_sample in raw_samples:
        sample_idx = raw_sample.get("sample_idx", 0)
        env_id = raw_sample.get("env_id", "unknown")
        output_str = raw_sample.get("output_str", "")

        # Extract task description
        task_desc_match = re.search(
            r'Your current task is:\s*(.+?)(?:\n|Your current observation)',
            output_str, re.DOTALL
        )
        if not task_desc_match:
            if verbose:
                print(f"  Warning: Could not extract task description for {sample_idx}")
            continue

        task_description = task_desc_match.group(1).strip()

        # Try to infer task name from description
        task_name = extract_task_info_from_description(task_description)
        if not task_name:
            # Default to a common task for testing
            if verbose:
                print(f"  Warning: Could not infer task name for '{task_description[:50]}...'")
            continue

        # Extract actions from trajectory
        assistant_pattern = r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>'
        assistant_turns = re.findall(assistant_pattern, output_str, re.DOTALL)

        actions = []
        for turn in assistant_turns:
            action_match = re.search(r'<action>(.*?)</action>', turn, re.DOTALL)
            if action_match:
                actions.append(action_match.group(1).strip())

        if not actions:
            continue

        # Replay the trajectory
        if verbose:
            print(f"  Replaying trajectory {sample_idx} ({len(actions)} actions, task={task_name})")

        try:
            replayer = TrajectoryReplayer(
                task_name=task_name,
                task_variation=0,  # Default variation
                simplification_str="easy",
            )

            states = replayer.replay_actions(actions)

            # Store ground truth for each step
            for i, state in enumerate(states[1:], 1):  # Skip initial state
                sample_id = f"{sample_idx}_{env_id}_step{i}"
                ground_truth_results[sample_id] = {
                    "location": state.location,
                    "inventory": state.inventory,
                    "action_accepted": state.action_accepted,
                    "step_index": i,
                }

            replayer.close()

        except Exception as e:
            if verbose:
                print(f"    Error replaying trajectory: {e}")
            continue

    if verbose:
        print(f"  Extracted ground truth for {len(ground_truth_results)} samples")

    return ground_truth_results


# =============================================================================
# Mechanical Pre-filter Integration (Revised Role)
# =============================================================================

def run_mechanical_prefilter_on_samples(
    samples: List[EvalSample],
    skip_llm_on_failure: bool = False,
    skip_format_and_validity: bool = False,  # NEW: Skip empty reflections + invalid actions (for v2.2)
    ground_truth_states: Optional[Dict[str, Dict]] = None,
) -> Tuple[List[EvalSample], Dict[str, Any]]:
    """
    Run mechanical checks on all samples before LLM evaluation.

    REVISED PURPOSE
    The primary role is to provide ground truth to ASSIST LLM judges, not to skip
    LLM calls. Skipping causes trajectory discontinuity in per-step metrics.

    Args:
        samples: List of EvalSample objects
        skip_llm_on_failure: If True, samples with mechanical failures will be marked
                            and can be excluded from LLM evaluation.
                            WARNING: This breaks trajectory continuity for per-step analysis.
        skip_format_and_validity: If True, skip samples with empty reflections OR invalid actions.
                                  This is for v2.2 prompt where LLM doesn't handle these cases.
        ground_truth_states: Optional dict mapping sample_id -> ground truth from replay.
                            If provided, enables ground truth location verification.

    Returns:
        Tuple of (filtered_samples, mechanical_results_dict)
        - filtered_samples: Samples to send to LLM (may exclude mechanical failures if skip_llm_on_failure=True)
        - mechanical_results_dict: Dict mapping sample_id -> mechanical check results
    """
    try:
        from tracerigor.verifier.verifier.sciworld_mechanical_checks import (
            run_mechanical_prefilter,
            MechanicalPrefilterResult,
            check_location_grounding,
            extract_location_claim_from_reflection,
        )
    except ImportError as e:
        print(f"Warning: Could not import mechanical checks module: {e}")
        return samples, {}

    mechanical_results = {}
    samples_with_failures = []
    samples_without_failures = []

    for sample in samples:
        # Build history in the format expected by mechanical checks
        history_for_check = [
            {"action": h.get("action", ""), "observation_text": h.get("observation_text", "")}
            for h in sample.history
        ]

        # Get ground truth location if available from replay
        gt_location = None
        if ground_truth_states and sample.sample_id in ground_truth_states:
            gt_location = ground_truth_states[sample.sample_id].get("location")

        # Run mechanical pre-filter
        result = run_mechanical_prefilter(
            reflection=sample.reflection_tokens,
            action=sample.action_tokens,
            current_observation=sample.current_observation_text,
            valid_actions_str=sample.valid_actions,
            history=history_for_check,
            object_tree=None,  # Object tree not available; use ground_truth_states instead
            next_observation=sample.next_observation_text,
        )

        # Extract location claim and check against ground truth if available
        location_claim = extract_location_claim_from_reflection(sample.reflection_tokens)
        location_gt_match = None
        if gt_location and location_claim:
            # Normalize and compare
            claim_norm = location_claim.lower().strip()
            gt_norm = gt_location.lower().strip()
            location_gt_match = (claim_norm == gt_norm or
                                 claim_norm in gt_norm or
                                 gt_norm in claim_norm)

        mechanical_results[sample.sample_id] = {
            "any_failure": result.any_failure,
            "empty_reflection": result.empty_reflection,
            "has_format_or_validity_issue": result.has_format_or_validity_issue,
            "failure_summary": result.failure_summary,
            "location_check": {
                "passed": result.location_check.passed,
                "evidence": result.location_check.evidence,
            },
            "action_validity_check": {
                "passed": result.action_validity_check.passed,
                "evidence": result.action_validity_check.evidence,
            },
            "action_repetition_check": {
                "passed": result.action_repetition_check.passed,
                "evidence": result.action_repetition_check.evidence,
            },
            # Ground truth comparison (from trajectory replay)
            "ground_truth": {
                "location": gt_location,
                "location_claim": location_claim,
                "location_match": location_gt_match,  # True/False/None
            }
        }

        if result.any_failure:
            samples_with_failures.append(sample)
        else:
            samples_without_failures.append(sample)

    # Summary statistics
    n_total = len(samples)
    n_failures = len(samples_with_failures)
    n_empty_reflections = sum(1 for r in mechanical_results.values() if r["empty_reflection"])
    n_validity_failures = sum(
        1 for r in mechanical_results.values()
        if not r["action_validity_check"]["passed"]
    )
    n_repetition_failures = sum(
        1 for r in mechanical_results.values()
        if not r["action_repetition_check"]["passed"]
    )
    n_format_or_validity = sum(
        1 for r in mechanical_results.values()
        if r["has_format_or_validity_issue"]
    )

    # Ground truth location statistics (if replay was used)
    gt_checks = [r["ground_truth"]["location_match"] for r in mechanical_results.values()
                 if r["ground_truth"]["location_match"] is not None]

    print(f"\n[Mechanical Pre-filter] Results:")
    print(f"  Total samples: {n_total}")
    print(f"  Empty reflections: {n_empty_reflections} ({100*n_empty_reflections/n_total:.1f}%)")
    print(f"  Action validity failures: {n_validity_failures} ({100*n_validity_failures/n_total:.1f}%)")
    print(f"  Action repetition failures: {n_repetition_failures} ({100*n_repetition_failures/n_total:.1f}%)")
    print(f"  Format OR validity issues: {n_format_or_validity} ({100*n_format_or_validity/n_total:.1f}%)")

    if gt_checks:
        gt_correct = sum(1 for m in gt_checks if m)
        print(f"  Ground truth location checks: {len(gt_checks)}")
        print(f"    - Correct location claims: {gt_correct} ({100*gt_correct/len(gt_checks):.1f}%)")

    # Filter based on mode
    if skip_format_and_validity:
        # v2.2 mode: skip empty reflections and invalid actions
        filtered_samples = [
            s for s in samples
            if not mechanical_results[s.sample_id]["has_format_or_validity_issue"]
        ]
        n_skipped = n_total - len(filtered_samples)
        print(f"  [v2.2 mode] Skipping {n_skipped} samples with empty reflection OR invalid action")
        return filtered_samples, mechanical_results
    elif skip_llm_on_failure:
        print(f"  WARNING: Skipping LLM for {n_failures} samples (breaks trajectory continuity)")
        return samples_without_failures, mechanical_results
    else:
        return samples, mechanical_results


def create_synthetic_result_all_no(sample: EvalSample, reason: str) -> Dict[str, Any]:
    """Create a synthetic all-NO evaluation result for a sample without calling the LLM.

    Used for samples where LLM evaluation is unnecessary and the outcome is
    deterministically known (e.g., empty reflections).

    Args:
        sample: The EvalSample to create a result for
        reason: Why this sample was scored all-NO (e.g., "empty_reflection")
    """
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
    """Split samples into those needing LLM eval and those scored mechanically.

    Empty reflections (as identified by mechanical checks) are scored all-NO
    without LLM calls. All other samples (including those with invalid actions)
    go to LLM.

    Uses mechanical_results as the single source of truth for empty reflection
    detection, rather than re-checking independently.

    Args:
        samples: List of EvalSample objects
        mechanical_results: Dict mapping sample_id -> mechanical check results
                           (from run_mechanical_prefilter_on_samples)

    Returns:
        (samples_for_llm, empty_samples, synthetic_results)
    """
    samples_for_llm = []
    empty_samples = []
    synthetic_results = []

    for sample in samples:
        mech = mechanical_results.get(sample.sample_id, {})
        is_empty = mech.get("empty_reflection", False)

        # Fallback: if mechanical_results somehow missing for this sample,
        # check directly (should not happen in normal flow)
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
        print(f"  [Post-hoc] {len(empty_samples)} empty reflections scored all-NO (no LLM call)")

    return samples_for_llm, empty_samples, synthetic_results


def attach_mechanical_metadata(
    results: List[Dict[str, Any]],
    mechanical_results: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Attach mechanical check metadata to evaluation results.

    Adds action_validity from mechanical checks to each result's extra dict.
    This allows downstream analysis to use action_validity as a separate dimension
    without overriding LLM scores.

    Args:
        results: List of evaluation result dicts
        mechanical_results: Dict mapping sample_id -> mechanical check results
    """
    for result in results:
        sample_id = result.get("id", "")
        if sample_id in mechanical_results:
            mech = mechanical_results[sample_id]
            extra = result.get("extra") or {}
            extra["mechanical"] = {
                "action_validity": mech["action_validity_check"]["passed"],
                "action_validity_evidence": mech["action_validity_check"]["evidence"],
                "empty_reflection": mech["empty_reflection"],
                "action_repetition": mech["action_repetition_check"]["passed"],
            }
            result["extra"] = extra
    return results


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
    """Evaluate samples directly using the verifier (no server needed).

    Args:
        samples: List of EvalSample objects to evaluate
        rubric: Evaluation rubric (universal, grounding, etc.)
        model: Model name for evaluation
        debug_prompts: Number of samples for which to include full prompts in results
        ground_truth_states: Optional ground truth from trajectory replay
    """
    from tracerigor.verifier.verifier.openai_verifier import run_openai_verifier, build_verifier

    items = samples_to_batch_items(samples, rubric, ground_truth_states)

    # If debug_prompts > 0, generate and attach prompts to the first N items
    if debug_prompts > 0:
        # Import template classes to generate prompts for debugging
        from tracerigor.verifier.prompt.sciworld import get_sciworld_template
        template = get_sciworld_template(rubric)

        for i, item in enumerate(items[:debug_prompts]):
            try:
                messages = template.build_messages(dict(item))
                item["_debug_system_prompt"] = messages[0]["content"] if messages else "N/A"
                item["_debug_user_prompt"] = messages[1]["content"] if len(messages) > 1 else "N/A"
            except Exception as e:
                item["_debug_system_prompt"] = f"Error generating prompt: {e}"
                item["_debug_user_prompt"] = f"Error generating prompt: {e}"

    # Use the SciWorld verifier for SciWorld samples
    results = await run_openai_verifier(
        input_data=items,
        rubric=rubric,
        model_name=model,
        model_params={"temperature": 0.0},
        env="sciworld",  # Specify SciWorld environment
    )

    # Copy debug prompts to results
    for i, result in enumerate(results):
        if i < len(items) and "_debug_system_prompt" in items[i]:
            result["debug_system_prompt"] = items[i]["_debug_system_prompt"]
            result["debug_user_prompt"] = items[i]["_debug_user_prompt"]

    return results


def compute_aggregate_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute aggregate metrics from evaluation results."""
    if not results:
        return {}

    n = len(results)

    metrics = {
        "num_samples": n,
        "query_success_rate": sum(1 for r in results if r.get("query_success", False)) / n,
        "parse_success_rate": sum(1 for r in results if r.get("parse_success", False)) / n,
        "avg_score": sum(r.get("score", 0) for r in results) / n,
    }

    # Extract per-dimension scores from extra.scalar_scores
    # Keys: grounding, action, temporal, aggregate (from SciWorld universal parser)
    grounding_scores = []
    action_scores = []
    temporal_scores = []

    for r in results:
        extra = r.get("extra") or {}
        scalar_scores = extra.get("scalar_scores") or {}

        if "grounding" in scalar_scores:
            grounding_scores.append(scalar_scores["grounding"])
        if "action" in scalar_scores:
            action_scores.append(scalar_scores["action"])
        if "temporal" in scalar_scores:
            temporal_scores.append(scalar_scores["temporal"])

    if grounding_scores:
        metrics["avg_observation_grounding"] = sum(grounding_scores) / len(grounding_scores)
    if action_scores:
        metrics["avg_action_coherence"] = sum(action_scores) / len(action_scores)
    if temporal_scores:
        metrics["avg_temporal_consistency"] = sum(temporal_scores) / len(temporal_scores)

    return metrics


def save_results(
    results: List[Dict[str, Any]],
    samples: List[EvalSample],
    output_dir: str,
    metrics: Dict[str, Any],
):
    """Save evaluation results to files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save raw results
    results_file = output_path / f"eval_results_{timestamp}.jsonl"
    with open(results_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    print(f"Saved raw results to {results_file}")

    # Save metrics summary
    metrics_file = output_path / f"eval_metrics_{timestamp}.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_file}")

    # Save human-readable report
    report_file = output_path / f"eval_report_{timestamp}.txt"
    with open(report_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("SciWorld LLM Judge Evaluation Report\n")
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
        # Build result lookup by id for correct matching (results may be in different order)
        result_by_id = {r.get("id", ""): r for r in results}
        for i, sample in enumerate(samples[:10]):
            result = result_by_id.get(sample.sample_id, {})
            f.write(f"\n[{i+1}] {sample.sample_id}\n")
            f.write(f"  Step: {sample.step_index}\n")
            f.write(f"  Observation: {sample.current_observation_text[:]}...\n")
            f.write(f"  Reflection: {sample.reflection_tokens[:]}...\n")
            f.write(f"  Action: {sample.action_tokens}\n")
            f.write(f"  Score: {result.get('score', 'N/A')}\n")

            # Show per-dimension scores if available
            extra = result.get('extra') or {}
            scalar_scores = extra.get('scalar_scores') or {}
            if scalar_scores:
                f.write(f"  Per-dim scores: grounding={scalar_scores.get('grounding', 'N/A')}, ")
                f.write(f"action={scalar_scores.get('action', 'N/A')}, ")
                f.write(f"temporal={scalar_scores.get('temporal', 'N/A')}\n")

            f.write(f"  Parse OK: {result.get('parse_success', 'N/A')}\n")
            if result.get('response'):
                f.write(f"  LLM Response: {result['response'][:]}...\n")

            # Include debug prompts if available
            if result.get('debug_system_prompt'):
                f.write(f"\n  [DEBUG] System Prompt:\n")
                f.write(f"  {'-' * 30}\n")
                for line in result['debug_system_prompt'].split('\n')[:]:
                    f.write(f"    {line}\n")
                f.write(f"  ... (truncated)\n")
            if result.get('debug_user_prompt'):
                f.write(f"\n  [DEBUG] User Prompt:\n")
                f.write(f"  {'-' * 30}\n")
                for line in result['debug_user_prompt'].split('\n')[:]:
                    f.write(f"    {line}\n")
                f.write(f"  ... (truncated)\n")

    print(f"Saved report to {report_file}")


# =============================================================================
# Plotting Functions for Academic-Quality Visualizations
# =============================================================================

def setup_academic_style():
    """Configure matplotlib for academic publication-quality plots."""
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    # Use a clean, academic style
    plt.style.use('seaborn-v0_8-whitegrid')

    # Academic color palette (colorblind-friendly)
    # Based on Okabe-Ito palette
    ACADEMIC_COLORS = {
        'grounding': '#0072B2',      # Blue
        'action': '#D55E00',          # Vermillion/Orange
        'temporal': '#009E73',        # Bluish Green
        'aggregate': '#CC79A7',       # Reddish Purple
    }

    # Font settings for academic papers
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
    """Extract trajectory ID from sample ID.

    Sample ID format: "{sample_idx}_{env_id}_step{step}"
    e.g., "0_val1_step1" -> "0_val1"
    """
    parts = sample_id.rsplit('_step', 1)
    return parts[0] if len(parts) > 1 else sample_id


def extract_step_number(sample_id: str) -> int:
    """Extract step number from sample ID.

    Sample ID format: "{sample_idx}_{env_id}_step{step}"
    e.g., "0_val1_step1" -> 1
    """
    match = re.search(r'_step(\d+)$', sample_id)
    return int(match.group(1)) if match else 0


def group_results_by_trajectory(results: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group evaluation results by trajectory ID."""
    grouped = defaultdict(list)
    for r in results:
        traj_id = extract_trajectory_id(r.get('id', ''))
        grouped[traj_id].append(r)

    # Sort each trajectory's results by step number
    for traj_id in grouped:
        grouped[traj_id].sort(key=lambda x: extract_step_number(x.get('id', '')))

    return dict(grouped)


def compute_per_turn_statistics(
    grouped_results: Dict[str, List[Dict[str, Any]]],
    dimensions: List[str] = ['grounding', 'action', 'temporal', 'aggregate'],
) -> Tuple[Dict[str, List[float]], Dict[str, List[float]], List[int], int]:
    """Compute per-turn mean and std error across trajectories.

    Handles trajectories with different lengths by only averaging over
    trajectories that have data for that turn.

    Returns:
        means: Dict mapping dimension -> list of mean values per turn
        stderrs: Dict mapping dimension -> list of std error values per turn
        turn_counts: List of how many trajectories contributed to each turn
        max_turns: Maximum number of turns across all trajectories
    """
    import numpy as np

    # Find max turns across all trajectories
    max_turns = max(len(results) for results in grouped_results.values())

    # Collect scores per turn per dimension
    scores_per_turn = {dim: [[] for _ in range(max_turns)] for dim in dimensions}

    for traj_id, results in grouped_results.items():
        for turn_idx, result in enumerate(results):
            extra = result.get('extra') or {}
            scalar_scores = extra.get('scalar_scores') or {}

            for dim in dimensions:
                if dim in scalar_scores and scalar_scores[dim] is not None:
                    scores_per_turn[dim][turn_idx].append(scalar_scores[dim])

    # Compute mean and stderr for each turn
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
) -> str:
    """Create academic-quality plots of per-dimension scores over turns.

    Args:
        results: List of evaluation results (from JSONL)
        output_dir: Directory to save plots
        config_name: Optional name for the configuration (for plot title/filename)
        timestamp: Optional timestamp string for filename
        show_individual: If True and multiple trajectories, also plot individual trajectories

    Returns:
        Path to the saved plot
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Setup academic style
    colors = setup_academic_style()

    # Group results by trajectory
    grouped = group_results_by_trajectory(results)
    n_trajectories = len(grouped)

    # Create output directory
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

    if n_trajectories == 1:
        # Single trajectory: plot raw scores
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

        # Add aggregate score as secondary y-axis annotation
        agg_scores = []
        for r in traj_results:
            extra = r.get('extra') or {}
            scalar_scores = extra.get('scalar_scores') or {}
            agg_scores.append(scalar_scores.get('aggregate', np.nan))
        avg_agg = np.nanmean(agg_scores)
        ax.axhline(y=avg_agg, color=colors['aggregate'], linestyle='--',
                  alpha=0.7, linewidth=1, label=f'Avg Aggregate ({avg_agg:.2f})')
        ax.legend(loc='lower left', framealpha=0.9)

        plot_path = plots_dir / f"scores_over_turns{config_str}_{timestamp}.pdf"
        plt.savefig(plot_path, format='pdf', bbox_inches='tight')
        plt.savefig(plot_path.with_suffix('.png'), format='png', bbox_inches='tight')
        plt.close()

    else:
        # Multiple trajectories: plot mean ± stderr
        means, stderrs, turn_counts, max_turns = compute_per_turn_statistics(grouped, dimensions)

        fig, ax = plt.subplots(figsize=(10, 6))

        turns = np.arange(1, max_turns + 1)

        for dim in dimensions:
            mean_vals = np.array(means[dim])
            stderr_vals = np.array(stderrs[dim])

            # Plot mean line
            line, = ax.plot(turns, mean_vals, 'o-', color=colors[dim],
                           label=dim_labels[dim], markersize=5, linewidth=1.5)

            # Plot error band (shaded region for ±1 stderr)
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

        # Add secondary x-axis showing trajectory count per turn
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(turns[::5] if max_turns > 10 else turns)  # Show every 5th if many turns
        ax2.set_xticklabels([f'n={turn_counts[i-1]}' for i in (turns[::5] if max_turns > 10 else turns)],
                           fontsize=8, color='gray')
        ax2.set_xlabel('Trajectories at turn', fontsize=9, color='gray')

        plot_path = plots_dir / f"scores_over_turns_avg{config_str}_{timestamp}.pdf"
        plt.savefig(plot_path, format='pdf', bbox_inches='tight')
        plt.savefig(plot_path.with_suffix('.png'), format='png', bbox_inches='tight')
        plt.close()

        # Optionally plot individual trajectories
        if show_individual and n_trajectories <= 6:
            fig, axes = plt.subplots(2, 3, figsize=(14, 8), squeeze=False)
            axes = axes.flatten()

            for idx, (traj_id, traj_results) in enumerate(grouped.items()):
                if idx >= 6:
                    break
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

            # Hide unused subplots
            for idx in range(n_trajectories, 6):
                axes[idx].set_visible(False)

            plt.suptitle('Individual Trajectory Scores', fontweight='bold', y=1.02)
            plt.tight_layout()

            indiv_path = plots_dir / f"scores_individual{config_str}_{timestamp}.pdf"
            plt.savefig(indiv_path, format='pdf', bbox_inches='tight')
            plt.savefig(indiv_path.with_suffix('.png'), format='png', bbox_inches='tight')
            plt.close()

    print(f"Saved plots to {plots_dir}")
    return str(plot_path)


def plot_dimension_comparison_bar(
    results: List[Dict[str, Any]],
    output_dir: str,
    config_name: Optional[str] = None,
    timestamp: Optional[str] = None,
) -> str:
    """Create a bar chart comparing average scores across dimensions.

    Useful for quick comparison of which dimensions are strongest/weakest.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    colors = setup_academic_style()

    # Compute average scores per dimension
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

    # Create bar chart
    fig, ax = plt.subplots(figsize=(6, 4))

    x = np.arange(len(dimensions))
    bar_colors = [colors[dim] for dim in dimensions]

    bars = ax.bar(x, means, yerr=stderrs, capsize=5, color=bar_colors,
                  edgecolor='black', linewidth=0.5, alpha=0.85)

    ax.set_ylabel('Score (Mean ± SE)', fontweight='medium')
    ax.set_title('Average Scores by Evaluation Dimension', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(dim_labels)
    ax.set_ylim(0, 1.1)
    ax.grid(True, axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, mean, stderr in zip(bars, means, stderrs):
        height = bar.get_height()
        ax.annotate(f'{mean:.2f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=10, fontweight='medium')

    # Create output directory
    plots_dir = Path(output_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    config_str = f"_{config_name}" if config_name else ""

    plot_path = plots_dir / f"dimension_comparison{config_str}_{timestamp}.pdf"
    plt.savefig(plot_path, format='pdf', bbox_inches='tight')
    plt.savefig(plot_path.with_suffix('.png'), format='png', bbox_inches='tight')
    plt.close()

    return str(plot_path)


def generate_all_plots(
    results: List[Dict[str, Any]],
    output_dir: str,
    config_name: Optional[str] = None,
    timestamp: Optional[str] = None,
    show_individual: bool = True,
) -> List[str]:
    """Generate all evaluation plots.

    Args:
        results: List of evaluation results
        output_dir: Output directory for plots
        config_name: Configuration name for plot titles/filenames
        timestamp: Timestamp for filenames
        show_individual: Whether to show individual trajectory plots

    Returns:
        List of paths to generated plots
    """
    plot_paths = []

    try:
        # Per-turn scores plot
        path = plot_scores_over_turns(
            results, output_dir, config_name, timestamp, show_individual
        )
        plot_paths.append(path)

        # Dimension comparison bar chart
        path = plot_dimension_comparison_bar(
            results, output_dir, config_name, timestamp
        )
        plot_paths.append(path)

    except ImportError as e:
        print(f"Warning: Could not generate plots. Missing dependency: {e}")
        print("Install matplotlib and numpy: pip install matplotlib numpy")
    except Exception as e:
        print(f"Warning: Error generating plots: {e}")
        import traceback
        traceback.print_exc()

    return plot_paths


async def main():
    parser = argparse.ArgumentParser(description="Evaluate SciWorld validation samples with LLM judge")
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
        help="Maximum number of trajectories (samples.jsonl entries) to load. Each trajectory has multiple steps.",
    )
    parser.add_argument(
        "--debug-prompts",
        type=int,
        default=0,
        help="Number of samples for which to save full prompts (system+user) for debugging. Default 0.",
    )
    parser.add_argument(
        "--rubric",
        type=str,
        choices=["universal", "grounding", "self_consistency"],
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
        help="Evaluation mode: 'api' uses FastAPI endpoint, 'direct' calls verifier directly",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:8000/batch_verify",
        help="FastAPI endpoint URL (for api mode)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate academic-quality plots of per-dimension scores over turns",
    )
    parser.add_argument(
        "--plot-individual",
        action="store_true",
        help="Also plot individual trajectories (only if --plot is enabled and <= 6 trajectories)",
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default=None,
        help="Configuration name for plot titles and filenames (e.g., 'baseline', 'finegrained_v1')",
    )
    parser.add_argument(
        "--mechanical-prefilter",
        action="store_true",
        help="Run mechanical pre-filter checks before LLM evaluation",
    )
    parser.add_argument(
        "--skip-llm-on-mechanical-failure",
        action="store_true",
        help="Skip LLM evaluation for samples with mechanical failures (saves tokens)",
    )
    parser.add_argument(
        "--skip-format-and-validity",
        action="store_true",
        help="[v2.2 mode] Skip samples with empty reflections OR invalid actions. Use with v2.2 prompt.",
    )
    parser.add_argument(
        "--posthoc-empty-no",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Post-hoc mark empty reflections as all-NO without LLM call (default: True). "
             "Use --no-posthoc-empty-no to disable.",
    )

    args = parser.parse_args()

    # Set output directory
    if args.output_dir is None:
        args.output_dir = str(Path(args.samples_dir) / "eval_results")

    print(f"Loading samples from: {args.samples_dir}")
    samples = load_samples(
        args.samples_dir,
        max_samples=args.max_samples,
        max_trajectories=args.max_trajectories,
    )

    if not samples:
        print("No samples found!")
        return

    # Run mechanical checks (needed for metadata + posthoc scoring + prefilter)
    mechanical_results = {}
    need_mechanical = args.mechanical_prefilter or args.skip_format_and_validity or args.posthoc_empty_no
    if need_mechanical:
        print("\n[Mechanical Checks] Running checks...")
        samples_after_mech, mechanical_results = run_mechanical_prefilter_on_samples(
            samples,
            skip_llm_on_failure=args.skip_llm_on_mechanical_failure,
            skip_format_and_validity=args.skip_format_and_validity,
        )
    else:
        samples_after_mech = samples

    # Post-hoc: score empty reflections as all-NO, skip LLM for them
    synthetic_results = []
    if args.posthoc_empty_no:
        samples_for_llm, empty_samples, synthetic_results = split_samples_for_eval(
            samples_after_mech, mechanical_results
        )
    else:
        samples_for_llm = samples_after_mech

    print(f"\nEvaluating {len(samples_for_llm)} samples via LLM using rubric '{args.rubric}'...")
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

    # Merge LLM results with synthetic results (empty reflections scored all-NO)
    results = llm_results + synthetic_results

    # Attach mechanical metadata (action_validity, etc.) to all results
    if mechanical_results:
        results = attach_mechanical_metadata(results, mechanical_results)

    # Compute metrics
    metrics = compute_aggregate_metrics(results)
    metrics["rubric"] = args.rubric
    metrics["model"] = args.model
    metrics["samples_dir"] = args.samples_dir

    # Add mechanical + posthoc stats to metrics
    if mechanical_results:
        metrics["mechanical_prefilter_enabled"] = True
        metrics["mechanical_total_samples"] = len(samples)
        metrics["mechanical_failures"] = sum(1 for r in mechanical_results.values() if r["any_failure"])
        metrics["mechanical_empty_reflections"] = sum(
            1 for r in mechanical_results.values() if r.get("empty_reflection", False)
        )
        metrics["mechanical_validity_failures"] = sum(
            1 for r in mechanical_results.values() if not r["action_validity_check"]["passed"]
        )
        metrics["mechanical_repetition_failures"] = sum(
            1 for r in mechanical_results.values() if not r["action_repetition_check"]["passed"]
        )
        metrics["mechanical_format_or_validity_skipped"] = sum(
            1 for r in mechanical_results.values() if r.get("has_format_or_validity_issue", False)
        )
    metrics["llm_evaluated_samples"] = len(samples_for_llm)
    metrics["posthoc_empty_no_samples"] = len(synthetic_results)
    metrics["total_results"] = len(results)

    # Count trajectories for metrics
    grouped = group_results_by_trajectory(results)
    metrics["num_trajectories"] = len(grouped)

    print("\n" + "=" * 40)
    print("Evaluation Results:")
    print("=" * 40)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_results(results, samples, args.output_dir, metrics)

    # Generate plots if requested
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

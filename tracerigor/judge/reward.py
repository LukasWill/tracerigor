"""
Reward mapping: judge output → bounded process reward.

Maps multi-rubric judge scores into a single scalar reward that is
added to the environment reward during RL training.

Design goals:
  - Bounded: r_proc ∈ [reward_range]
  - Interpretable: linear combination of rubric scores
  - Tunable: weights and bounds configurable
  - Compatible: works with both universal (3-rubric) and per-rubric modes

Shaping modes (ported from service_llm_verifier_wrapper):
  - Default:     r_proc = clamp(2 * r_raw - 1, lo, hi)
  - Symmetric:   delta = w_t * (2*score - 1)   — allows negative
  - Gate-mul:    big scores → multiplicative, small scores → additive/floor
  - Annealing:   w_t decays over [anneal_start, anneal_end]
"""
from __future__ import annotations

from typing import Dict

from tracerigor.judge.config import GatingConfig, RewardConfig
from tracerigor.judge.schema import JudgeResponse, RubricResult


# ---------------------------------------------------------------------------
# Annealing helper (ported from service_llm_verifier_wrapper._scale_piecewise)
# ---------------------------------------------------------------------------

def _scale_piecewise(
    t: int, start_step: int, end_step: int, hi: float, lo: float,
) -> float:
    """Linearly interpolate from *hi* → *lo* over [start_step, end_step]."""
    if end_step <= start_step:
        return lo
    if t <= start_step:
        return hi
    if t >= end_step:
        return lo
    alpha = (t - float(start_step)) / float(end_step - start_step)
    return hi + (lo - hi) * alpha


# ---------------------------------------------------------------------------
# Core reward computation
# ---------------------------------------------------------------------------

def compute_process_reward(
    judge_response: JudgeResponse,
    cfg: RewardConfig,
    gating_cfg: GatingConfig | None = None,
    train_step: int = 0,
) -> float:
    """
    Map a JudgeResponse to a bounded process reward scalar.

    Supports three shaping modes (selected via RewardConfig flags):

    1. **Default** (gate_mul=False, symmetric=False):
       r_raw = weighted_avg(rubric_scores)
       r_proc = clamp(2*r_raw - 1, lo, hi)

    2. **Symmetric** (symmetric_shaping=True, gate_mul=False):
       delta = w_t × (2*r_raw - 1)   — allows both positive and negative

    3. **Gate-mul** (gate_mul=True):
       Split big/small scores. Big → multiplicative with floor alpha_big.
       Small → additive beta_small. Mirrors service_llm_verifier_wrapper logic.

    Annealing:
       When gating_cfg.anneal_enabled, w_t decays from anneal_hi_scale → anneal_lo_scale
       over [anneal_start_step, anneal_end_step].
    """
    lo, hi = cfg.reward_range

    # Short-circuit: heuristics fired, use fixed penalty
    if judge_response.short_circuited:
        return max(lo, min(hi, -abs(cfg.insufficient_evidence_penalty)))

    rubrics = judge_response.rubrics
    weights = cfg.rubric_weights

    # Weighted average of rubric scores → r_raw in [0,1]
    total_weight = 0.0
    weighted_sum = 0.0
    for rubric_name, w in weights.items():
        result = rubrics.get(rubric_name)
        if result is None:
            continue
        weighted_sum += w * result.score
        total_weight += w

    if total_weight > 0:
        r_raw = weighted_sum / total_weight
    else:
        r_raw = 0.5  # no rubrics scored → neutral

    # Insufficient evidence penalty
    if judge_response.insufficient_evidence:
        r_raw -= cfg.insufficient_evidence_penalty

    # Compute time-varying weight w_t from annealing schedule
    w_t = 1.0
    if gating_cfg and gating_cfg.anneal_enabled:
        w_t = _scale_piecewise(
            train_step,
            gating_cfg.anneal_start_step,
            gating_cfg.anneal_end_step,
            gating_cfg.anneal_hi_scale,
            gating_cfg.anneal_lo_scale,
        )

    # --- Shaping mode selection ---
    if cfg.gate_mul:
        # Multiplicative gating (ported from service_llm_verifier_wrapper)
        if r_raw >= cfg.big_threshold:
            # Big reward: multiplicative with floor
            r_proc = max(cfg.alpha_big, w_t * r_raw)
        else:
            # Small reward: additive component
            r_proc = w_t * r_raw + cfg.beta_small
    elif cfg.symmetric_shaping:
        # Symmetric: delta = w_t * (2*score − 1)
        r_proc = w_t * (2.0 * r_raw - 1.0)
    else:
        # Default: map [0,1] → [-1,+1] and scale
        r_proc = w_t * (2.0 * r_raw - 1.0)

    # Clamp to configured range
    return max(lo, min(hi, r_proc))


def rubric_scores_dict(judge_response: JudgeResponse) -> Dict[str, float]:
    """Extract a flat dict of rubric scores for metric logging."""
    d = {}
    for name, result in judge_response.rubrics.items():
        d[f"judge_{name}_score"] = result.score
        d[f"judge_{name}_confidence"] = result.confidence
        d[f"judge_{name}_label"] = {"pass": 1.0, "fail": 0.0, "uncertain": 0.5}.get(
            result.label, 0.5
        )
    d["judge_overall_confidence"] = judge_response.overall_confidence
    d["judge_process_reward"] = judge_response.process_reward
    d["judge_short_circuited"] = float(judge_response.short_circuited)
    return d

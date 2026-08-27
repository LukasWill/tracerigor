"""
WandB logging for the judge stack.

Logs judge metrics into the *training* run's WandB context (no separate run).
Call `log_judge_batch` after each score_batch to record aggregate stats.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from tracerigor.judge.schema import JudgeResponse

logger = logging.getLogger(__name__)

try:
    import wandb

    _HAS_WANDB = True
except ImportError:
    _HAS_WANDB = False


def log_judge_batch(
    judge_results: Dict[str, JudgeResponse],
    train_step: int,
    duration_ms: float = 0.0,
    prefix: str = "judge",
) -> None:
    """Log aggregate judge metrics for one batch to WandB.

    Safe to call even when wandb is not initialized — will silently no-op.
    """
    if not _HAS_WANDB or wandb.run is None:
        return

    results = list(judge_results.values())
    n = len(results)
    if n == 0:
        return

    metrics: Dict[str, Any] = {
        f"{prefix}/batch_size": n,
        f"{prefix}/duration_ms": duration_ms,
    }

    # Short-circuit rate
    n_sc = sum(1 for r in results if r.short_circuited)
    metrics[f"{prefix}/short_circuit_rate"] = n_sc / n

    # Query / parse success (only for LLM-judged items)
    llm_results = [r for r in results if not r.short_circuited]
    if llm_results:
        metrics[f"{prefix}/query_success_rate"] = (
            sum(1 for r in llm_results if r.query_success) / len(llm_results)
        )
        metrics[f"{prefix}/parse_success_rate"] = (
            sum(1 for r in llm_results if r.parse_success) / len(llm_results)
        )

    # Avg process reward
    proc_rewards = [r.process_reward for r in results]
    metrics[f"{prefix}/avg_process_reward"] = sum(proc_rewards) / n

    # Avg confidence
    confs = [r.overall_confidence for r in results]
    metrics[f"{prefix}/avg_confidence"] = sum(confs) / n

    # Per-rubric averages
    rubric_scores: Dict[str, List[float]] = {}
    rubric_labels: Dict[str, List[str]] = {}
    for r in results:
        for name, rr in r.rubrics.items():
            rubric_scores.setdefault(name, []).append(rr.score)
            rubric_labels.setdefault(name, []).append(rr.label)

    for name, scores in rubric_scores.items():
        metrics[f"{prefix}/{name}_avg_score"] = sum(scores) / len(scores)
        labels = rubric_labels[name]
        metrics[f"{prefix}/{name}_pass_rate"] = (
            sum(1 for l in labels if l == "pass") / len(labels)
        )

    try:
        wandb.log(metrics, step=train_step)
    except Exception as e:
        logger.warning("[judge] wandb.log failed: %s", e)

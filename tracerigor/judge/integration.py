"""
Integration module — bridges the judge stack to the TraceRigor rollout manager.

This is the single entry point that the rollout manager calls after stepping
environments. It replaces the old decorator-based approach with an explicit
function call that has access to all rollout context.

Usage in rollout_manager_service.py:
    from tracerigor.judge.integration import JudgeIntegration

    # In __init__:
    self.judge = JudgeIntegration(judge_config)

    # In rollout_loop, after step_results:
    self.judge.process_step_batch(
        step_results=step_results,
        responses_str=ids2actions,
        recorder=self.recorder,
        env_states=self.env_states,
        env_configs=self.env_configs,
    )
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from tracerigor.judge.config import JudgeConfig
from tracerigor.judge.logging import log_judge_batch
from tracerigor.judge.packet_builder import build_packets_from_step_results
from tracerigor.judge.reward import compute_process_reward, rubric_scores_dict
from tracerigor.judge.router import JudgeRouter

logger = logging.getLogger(__name__)


class JudgeIntegration:
    """
    High-level judge interface for the rollout manager.

    Manages the judge router lifecycle and provides a clean API for
    processing step batches and injecting process rewards.
    """

    def __init__(self, cfg: Optional[JudgeConfig] = None):
        if cfg is None:
            cfg = JudgeConfig(enabled=False)
        self.cfg = cfg
        self._router = JudgeRouter(cfg) if cfg.enabled else None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def set_train_step(self, step: int) -> None:
        """Update the training step counter (called on reset)."""
        if self._router:
            self._router.set_train_step(step)

    def close(self) -> None:
        """Flush audit and release resources."""
        if self._router:
            self._router.close()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def process_step_batch(
        self,
        step_results: Dict[str, Tuple[Dict, float, bool, Dict]],
        responses_str: Dict[str, str],
        recorder: Dict[str, List[Dict[str, Any]]],
        env_states: Dict[str, Dict[str, Any]],
        env_configs: Optional[Dict[str, Any]] = None,
        default_admissible: Optional[List[str]] = None,
    ) -> Dict[str, Tuple[Dict, float, bool, Dict]]:
        """
        Process a batch of step results through the judge.

        This method:
        1. Builds judge packets from step results and recorder
        2. Runs the judge router (heuristics → LLM → audit)
        3. Injects judge scores into info["metrics"]["turn_metrics"]
        4. Adds process reward to the step reward

        Args:
            step_results:      {env_id: (obs, reward, done, info)}
            responses_str:     {env_id: decoded_response_string}
            recorder:          {env_id: [step_records]}
            env_states:        {env_id: {"step": int, ...}}
            env_configs:       {env_id: config_obj}
            default_admissible: Fallback admissible actions

        Returns:
            Modified step_results with judge scores and adjusted rewards.
        """
        if not self.cfg.enabled or self._router is None:
            return step_results

        # Build packets
        packets = build_packets_from_step_results(
            step_results=step_results,
            responses_str=responses_str,
            recorder=recorder,
            env_states=env_states,
            env_configs=env_configs or {},
            history_k=self.cfg.history_k,
            default_admissible=default_admissible,
        )

        if not packets:
            return step_results

        # Run judge
        t0 = time.monotonic()
        judge_results = self._router.score_batch(packets)
        elapsed_ms = (time.monotonic() - t0) * 1000

        # Log to WandB (no-op if wandb not initialized)
        log_judge_batch(judge_results, self._router._train_step, elapsed_ms)

        # Inject results back into step_results
        lambda_proc = self.cfg.reward.lambda_proc
        new_results = {}

        # Collect env_ids that got judge packets so we can zero-fill the rest
        judged_env_ids = {p.env_id for p in packets}

        for env_id, (obs, reward, done, info) in step_results.items():
            judge_resp = judge_results.get(env_id)

            # Zero-fill: envs with no judge packet keep metrics step-aligned
            if judge_resp is None:
                if env_id not in judged_env_ids:
                    tm = info.setdefault("metrics", {}).setdefault("turn_metrics", {})
                    for rubric_name in self.cfg.rubrics:
                        tm.setdefault(f"judge_{rubric_name}_score", 0.0)
                        tm.setdefault(f"judge_{rubric_name}_label", 0.5)
                        tm.setdefault(f"judge_{rubric_name}_confidence", 0.0)
                    tm.setdefault("judge_process_reward", 0.0)
                    tm.setdefault("judge_short_circuited", 0.0)
                    tm.setdefault("judge_overall_confidence", 0.0)
                new_results[env_id] = (obs, reward, done, info)
                continue

            # Write judge metrics into turn_metrics
            tm = info.setdefault("metrics", {}).setdefault("turn_metrics", {})
            scores = rubric_scores_dict(judge_resp)
            for k, v in scores.items():
                tm[k] = v

            # Store the raw judge response for debugging if needed
            info["judge_response"] = judge_resp

            # Add process reward to step reward
            proc_reward = judge_resp.process_reward
            adjusted_reward = reward + lambda_proc * proc_reward

            new_results[env_id] = (obs, adjusted_reward, done, info)

        return new_results

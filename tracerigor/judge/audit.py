"""
Async audit logger for the judge stack.

Non-blocking JSONL logger that records low-confidence or sampled turns
for offline analysis and calibration. Never blocks the RL rollout loop.

Features:
  - Writes to a JSONL file on disk (one line per audited turn)
  - Background thread for I/O (never blocks the caller)
  - Configurable routing: low-confidence, heuristic disagreement, uniform sample
  - Thread-safe queue with bounded size (drops oldest on overflow)
"""
from __future__ import annotations

import json
import logging
import os
import queue
import random
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from tracerigor.judge.config import AuditConfig
from tracerigor.judge.schema import JudgeResponse, TurnJudgePacket

logger = logging.getLogger(__name__)

# Max queue size before dropping oldest entries
_MAX_QUEUE_SIZE = 10_000


class AuditQueue:
    """Non-blocking audit logger backed by a JSONL file."""

    def __init__(self, cfg: AuditConfig, run_id: str = ""):
        self.cfg = cfg
        self._queue: queue.Queue = queue.Queue(maxsize=_MAX_QUEUE_SIZE)
        self._stopped = threading.Event()
        self._worker: Optional[threading.Thread] = None

        if not cfg.enabled:
            return

        log_dir = Path(cfg.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self._log_path = log_dir / f"audit_{run_id}_{timestamp}.jsonl"
        logger.info("[AuditQueue] Logging to %s", self._log_path)

        self._worker = threading.Thread(target=self._writer_loop, daemon=True)
        self._worker.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def maybe_enqueue(
        self,
        packet: TurnJudgePacket,
        response: JudgeResponse,
    ) -> bool:
        """
        Decide whether this turn should be audited, and if so, enqueue it.

        Returns True if the turn was enqueued.
        """
        if not self.cfg.enabled:
            return False

        should_audit = self._should_audit(response)
        if not should_audit:
            return False

        record = self._build_record(packet, response)
        try:
            self._queue.put_nowait(record)
            return True
        except queue.Full:
            logger.warning("[AuditQueue] Queue full, dropping oldest entry")
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(record)
                return True
            except queue.Full:
                return False

    def flush(self) -> None:
        """Block until the queue is drained."""
        if self._worker is not None:
            self._queue.join()

    def stop(self) -> None:
        """Signal the writer thread to stop and wait for it."""
        self._stopped.set()
        if self._worker is not None:
            self._worker.join(timeout=10.0)

    # ------------------------------------------------------------------
    # Routing policy
    # ------------------------------------------------------------------

    def _should_audit(self, response: JudgeResponse) -> bool:
        """Decide if a turn warrants audit logging."""
        cfg = self.cfg

        # Uniform random sample
        if random.random() < cfg.sample_rate:
            return True

        # Low overall confidence
        if response.overall_confidence < cfg.low_confidence_threshold:
            return True

        # Any rubric is "uncertain"
        for result in response.rubrics.values():
            if result.label == "uncertain":
                return True

        # Parse/query failure
        if not response.query_success or not response.parse_success:
            return True

        # Low grounding confidence on visual turns
        grounding = response.rubrics.get("observation_grounding") or response.rubrics.get("grounding")
        if grounding and grounding.confidence < cfg.visual_confidence_threshold:
            return True

        return False

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_record(
        self, packet: TurnJudgePacket, response: JudgeResponse
    ) -> Dict[str, Any]:
        """Build a serializable audit record (no images)."""
        return {
            "timestamp": time.time(),
            "env_id": packet.env_id,
            "episode_step": packet.episode_step,
            "task_name": packet.task_name,
            "agent_modality": packet.agent_modality,
            "current_observation_text": packet.current_observation_text[:500],
            "reasoning_tokens": packet.reasoning_tokens[:500],
            "action_tokens": packet.action_tokens[:200],
            "chosen_action": packet.chosen_action,
            "heuristic_flags": packet.heuristic_flags,
            "response": {
                "short_circuited": response.short_circuited,
                "short_circuit_reason": response.short_circuit_reason,
                "rubrics": {
                    k: {"label": v.label, "score": v.score, "confidence": v.confidence}
                    for k, v in response.rubrics.items()
                },
                "overall_confidence": response.overall_confidence,
                "insufficient_evidence": response.insufficient_evidence,
                "process_reward": response.process_reward,
                "model_used": response.model_used,
                "query_success": response.query_success,
                "parse_success": response.parse_success,
            },
        }

    def _writer_loop(self) -> None:
        """Background thread: drains queue and writes JSONL."""
        while not self._stopped.is_set():
            try:
                record = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                with open(self._log_path, "a") as f:
                    f.write(json.dumps(record, default=str) + "\n")
            except Exception as e:
                logger.error("[AuditQueue] Write failed: %s", e)
            finally:
                self._queue.task_done()

        # Drain remaining
        while not self._queue.empty():
            try:
                record = self._queue.get_nowait()
                with open(self._log_path, "a") as f:
                    f.write(json.dumps(record, default=str) + "\n")
                self._queue.task_done()
            except (queue.Empty, Exception):
                break

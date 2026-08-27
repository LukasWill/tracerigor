"""
Judge Router — the central orchestrator of the multi-fidelity judge stack.

Responsibilities:
  1. Accept batches of TurnJudgePackets
  2. Run Layer-0 heuristics → short-circuit obvious failures
  3. Build prompts and call the synchronous LLM judge (Layer-1)
  4. Parse structured JSON responses into JudgeResponse objects
  5. Compute process rewards
  6. Route low-confidence / sampled turns to the audit queue
  7. Return JudgeResponse objects keyed by env_id

This is the single entry point the rollout manager calls.
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

from tracerigor.judge.audit import AuditQueue
from tracerigor.judge.client import JudgeClient
from tracerigor.judge.config import JudgeConfig, GatingConfig
from tracerigor.judge.heuristics import run_heuristics
from tracerigor.judge.prompt import build_messages
from tracerigor.judge.reward import compute_process_reward, rubric_scores_dict
from tracerigor.judge.schema import (
    JUDGE_OUTPUT_JSON_SCHEMA,
    JudgeResponse,
    RubricResult,
    TurnJudgePacket,
)

# --- Reuse verifier parsers for robust response parsing ---
try:
    from tracerigor.verifier.utils.parsers import (
        _safe_json_loads as _verifier_json_loads,
        _universal_score as _verifier_universal_score,
        _yesno_score as _verifier_yesno_score,
    )
    _HAS_VERIFIER_PARSERS = True
except ImportError:
    _HAS_VERIFIER_PARSERS = False

logger = logging.getLogger(__name__)


class JudgeRouter:
    """
    Stateful judge router.

    Instantiated once per training run. Holds the LLM client, audit queue,
    and training-step counter for gating.
    """

    def __init__(self, cfg: JudgeConfig):
        self.cfg = cfg
        self._client = JudgeClient(cfg.provider)
        self._audit = AuditQueue(cfg.audit)
        self._train_step: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def set_train_step(self, step: int) -> None:
        """Update the current trainer step (called on each reset/rollout)."""
        self._train_step = step

    def close(self) -> None:
        """Flush audit queue and release resources."""
        self._audit.flush()
        self._audit.stop()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def score_batch(
        self,
        packets: List[TurnJudgePacket],
    ) -> Dict[str, JudgeResponse]:
        """
        Score a batch of turn packets.

        Returns:
            {env_id: JudgeResponse} for each packet.
        """
        if not self.cfg.enabled or not packets:
            return {p.env_id: _empty_response(p) for p in packets}

        # Gating: should we run this step?
        if not self._should_run():
            return {p.env_id: _empty_response(p) for p in packets}

        results: Dict[str, JudgeResponse] = {}

        # --- Layer 0: heuristics ---
        llm_packets: List[TurnJudgePacket] = []
        for packet in packets:
            heur = run_heuristics(packet, self.cfg.heuristics)
            packet.heuristic_flags = heur["checks"]

            if heur["should_short_circuit"]:
                resp = JudgeResponse(
                    env_id=packet.env_id,
                    episode_step=packet.episode_step,
                    short_circuited=True,
                    short_circuit_reason=heur["reason"],
                    query_success=True,
                    parse_success=True,
                )
                resp.process_reward = compute_process_reward(
                    resp, self.cfg.reward, self.cfg.gating, self._train_step,
                )
                results[packet.env_id] = resp
                self._audit.maybe_enqueue(packet, resp)
            else:
                llm_packets.append(packet)

        if not llm_packets:
            return results

        # --- Layer 1: LLM judge ---
        llm_results = self._call_llm_judge(llm_packets)
        for packet, resp in zip(llm_packets, llm_results):
            resp.process_reward = compute_process_reward(
                resp, self.cfg.reward, self.cfg.gating, self._train_step,
            )
            results[packet.env_id] = resp
            self._audit.maybe_enqueue(packet, resp)

        return results

    # ------------------------------------------------------------------
    # Gating
    # ------------------------------------------------------------------

    def _should_run(self) -> bool:
        """Check gating schedule against current train step."""
        g = self.cfg.gating
        t = self._train_step

        if g.enable_after_step >= 0 and t < g.enable_after_step:
            return False
        if g.disable_after_step >= 0 and t >= g.disable_after_step:
            return False

        if g.run_every_k_steps > 1:
            base = g.enable_after_step if g.enable_after_step >= 0 else 0
            if (t - base) % g.run_every_k_steps != 0:
                return False

        return True

    # ------------------------------------------------------------------
    # LLM judge call
    # ------------------------------------------------------------------

    def _call_llm_judge(
        self,
        packets: List[TurnJudgePacket],
    ) -> List[JudgeResponse]:
        """Call the LLM judge for a batch of packets."""

        use_images = self.cfg.use_images and self.cfg.provider.is_multimodal
        json_schema = JUDGE_OUTPUT_JSON_SCHEMA if self.cfg.provider.use_structured_output else None

        # Build message batches
        messages_batch = []
        for packet in packets:
            rubric = "universal" if self.cfg.mode == "universal" else self.cfg.rubrics[0]
            msgs = build_messages(
                packet=packet,
                rubric=rubric,
                use_images=use_images,
                json_schema_hint=_schema_hint() if not json_schema else "",
            )
            messages_batch.append(msgs)

        # Call LLM
        t0 = time.monotonic()
        raw_results = self._client.judge_batch_sync(messages_batch, json_schema)
        elapsed = (time.monotonic() - t0) * 1000

        if len(packets) > 0:
            logger.info(
                "[JudgeRouter] LLM judge batch: %d items, %.0fms total, %.0fms/item avg",
                len(packets), elapsed, elapsed / len(packets),
            )

        # Parse results
        responses = []
        for packet, raw in zip(packets, raw_results):
            resp = self._parse_llm_response(packet, raw)
            responses.append(resp)

        return responses

    def _parse_llm_response(
        self,
        packet: TurnJudgePacket,
        raw: Dict[str, Any],
    ) -> JudgeResponse:
        """Parse a single LLM result into a JudgeResponse."""
        resp = JudgeResponse(
            env_id=packet.env_id,
            episode_step=packet.episode_step,
            raw_llm_response=raw.get("response", ""),
            query_success=raw.get("success", False),
            model_used=raw.get("model", ""),
        )

        if not raw.get("success"):
            resp.parse_success = False
            return resp

        text = raw.get("response", "")

        # Try to parse as JSON
        parsed = _try_parse_json(text)
        if parsed is None:
            resp.parse_success = False
            # Fallback: try to extract from <think>…</think><answer>…</answer> format
            resp = self._fallback_parse(resp, text)
            return resp

        resp.parse_success = True

        # The universal prompt returns:
        #   {"observation_grounding": {"yes_no": "YES|NO", "evidence": "..."},
        #    "action_coherence":      {"yes_no": "YES|NO", "evidence": "..."},
        #    "temporal_consistency":   {"yes_no": "YES|NO", "evidence": "..."}}
        #
        # Map these to rubric results.  The keys from the LLM match the
        # rubric names in the config, or we try a few common aliases.

        _RUBRIC_ALIASES = {
            "grounding": ["observation_grounding", "factual_grounding", "grounding"],
            "action_coherence": ["action_coherence", "action_reasoning_consistency", "behavioral"],
            "temporal_consistency": ["temporal_consistency", "history_consistency", "history"],
            # legacy aliases
            "observation_grounding": ["observation_grounding", "factual_grounding", "grounding"],
            "action_reasoning_consistency": ["action_reasoning_consistency", "action_coherence", "behavioral"],
            "history_consistency": ["history_consistency", "temporal_consistency", "history"],
        }

        for rubric_name in self.cfg.rubrics:
            # Try direct key first, then aliases
            rdata = parsed.get(rubric_name) or {}
            if not rdata:
                for alias in _RUBRIC_ALIASES.get(rubric_name, []):
                    rdata = parsed.get(alias) or {}
                    if rdata:
                        break

            if isinstance(rdata, dict):
                yes_no = rdata.get("yes_no", "").upper()
                if yes_no in ("YES", "NO"):
                    score = 1.0 if yes_no == "YES" else 0.0
                    label = "pass" if yes_no == "YES" else "fail"
                else:
                    score = float(rdata.get("score", 0.5))
                    label = rdata.get("label", "uncertain")
                evidence = rdata.get("evidence", [])
                if isinstance(evidence, str):
                    evidence = [evidence]
                # Confidence is deterministic: 1.0 for clear YES/NO, 0.5 for other
                conf = 1.0 if yes_no in ("YES", "NO") else 0.5
                resp.rubrics[rubric_name] = RubricResult(
                    label=label,
                    score=score,
                    confidence=conf,
                    evidence=evidence,
                )
            else:
                resp.rubrics[rubric_name] = RubricResult()

        # Compute overall confidence: high if all rubrics agree, lower otherwise
        if resp.rubrics:
            confs = [r.confidence for r in resp.rubrics.values()]
            labels = [r.label for r in resp.rubrics.values()]
            # Base: mean of individual confidences
            base_conf = sum(confs) / len(confs)
            # Penalty: if rubrics disagree (mix of pass/fail), lower confidence
            unique_labels = set(labels) - {"uncertain"}
            if len(unique_labels) > 1:
                base_conf *= 0.8  # disagreement discount
            resp.overall_confidence = base_conf
        resp.insufficient_evidence = bool(parsed.get("insufficient_evidence", False))

        return resp

    def _fallback_parse(
        self,
        resp: JudgeResponse,
        text: str,
    ) -> JudgeResponse:
        """
        Fallback parsing for non-JSON responses.
        Handles <think>…</think><answer>YES|NO</answer> format used by
        per-rubric binary verifiers.
        """
        answer_match = re.search(
            r"<answer>\s*(YES|NO)\s*</answer>", text, re.I
        )
        if answer_match:
            verdict = answer_match.group(1).upper()
            score = 1.0 if verdict == "YES" else 0.0

            # Apply to all configured rubrics (single-rubric binary mode)
            for rubric_name in self.cfg.rubrics:
                resp.rubrics[rubric_name] = RubricResult(
                    label="pass" if verdict == "YES" else "fail",
                    score=score,
                    confidence=0.7,  # default for binary
                    evidence=[],
                )
            resp.overall_confidence = 0.7
            resp.parse_success = True  # partial success
        else:
            # Truly unparseable — fill defaults
            for rubric_name in self.cfg.rubrics:
                resp.rubrics[rubric_name] = RubricResult()
            resp.overall_confidence = 0.0

        return resp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _empty_response(packet: TurnJudgePacket) -> JudgeResponse:
    """Return a neutral JudgeResponse (judge not run)."""
    return JudgeResponse(
        env_id=packet.env_id,
        episode_step=packet.episode_step,
        overall_confidence=0.0,
    )


def _try_parse_json(text: str) -> Optional[Dict[str, Any]]:
    """Attempt to parse JSON from text, handling common issues.

    Delegates to verifier's _safe_json_loads (handles code fences, whitespace)
    when available, with additional extraction fallbacks for the judge.
    """
    text = text.strip()

    # Try verifier parser first (handles code fences, markdown wrapping)
    if _HAS_VERIFIER_PARSERS:
        try:
            return _verifier_json_loads(text)
        except Exception:
            pass

    # Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Extract JSON from markdown code blocks
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # Extract first {...} block
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    return None


def _schema_hint() -> str:
    """Short inline schema hint for prompts (when not using structured output).

    Note: this is only used when the Sokoban/SciWorld env templates don't already
    embed the expected JSON format (they do). The default template uses this.
    """
    return (
        'Output JSON with exactly these keys:\n'
        '{"observation_grounding": {"yes_no": "YES|NO", "evidence": "<=2 short bullets"}, '
        '"action_coherence": {"yes_no": "YES|NO", "evidence": "<=2 short bullets"}, '
        '"temporal_consistency": {"yes_no": "YES|NO", "evidence": "<=2 short bullets"}}'
    )

# tracerigor/judge — Multi-fidelity routing judge for process-level supervision
#
# Architecture:
#   schema.py        – TurnJudgePacket, RubricResult, JudgeResponse dataclasses + JSON schema
#   config.py        – JudgeConfig dataclass tree
#   client.py        – Provider-agnostic async LLM client (vLLM / OpenAI / Together / OpenRouter)
#   heuristics.py    – Cheap deterministic checks (Layer 0)
#   prompt.py        – Env-agnostic prompt builder + template registry
#   env_templates.py – Env-specific prompt templates (sokoban, sciworld, …)
#   router.py        – Orchestrates Layer 0 → Layer 1 → audit routing
#   reward.py        – Maps judge outputs to bounded process rewards
#   audit.py         – Non-blocking JSONL audit logger
#   packet_builder.py – Bridges rollout manager data → TurnJudgePacket
#   integration.py   – Single entry point for rollout manager

from tracerigor.judge.config import JudgeConfig
from tracerigor.judge.integration import JudgeIntegration
from tracerigor.judge.env_templates import register_all_templates

__all__ = ["JudgeConfig", "JudgeIntegration", "register_all_templates"]

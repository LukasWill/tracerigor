"""
Judge stack configuration.

Single source of truth for all judge-related settings: provider selection,
model parameters, rubric weights, heuristic thresholds, gating schedule,
and audit policy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ProviderConfig:
    """One LLM endpoint (self-hosted vLLM, hosted API, or proprietary)."""

    name: str = "vllm-local"                   # human-readable label
    base_url: str = "http://127.0.0.1:8001/v1" # OpenAI-compatible endpoint
    api_key: str = "EMPTY"                      # vLLM doesn't need a real key
    model: str = "gpt-oss-120b"                 # model name as served
    is_multimodal: bool = False                 # whether to send images
    max_completion_tokens: int = 256
    temperature: float = 0.0
    timeout_s: float = 45.0
    max_retries: int = 1
    max_concurrency: int = 64                   # semaphore limit
    use_structured_output: bool = True          # JSON schema guided decoding
    use_responses_api: bool = False             # True for OpenAI gpt-5 models


@dataclass
class HeuristicConfig:
    """Thresholds for Layer-0 deterministic checks."""

    min_trace_tokens: int = 8                   # traces shorter than this → fail
    repeat_action_window: int = 3               # flag if same action k times in last k steps
    short_circuit_penalty: float = -0.20        # r_proc when heuristic fires


@dataclass
class AuditConfig:
    """Controls for the async audit logger."""

    enabled: bool = True
    log_dir: str = "judge_audit"                # relative to working dir
    sample_rate: float = 0.02                   # uniform random sample
    low_confidence_threshold: float = 0.65      # below this → audit
    visual_confidence_threshold: float = 0.75   # for grounding on vision turns


@dataclass
class RewardConfig:
    """How rubric scores map to a bounded process reward.

    Shaping modes (inherited from service_llm_verifier_wrapper):
      - symmetric: delta = w_t * (2*score - 1), allows negative rewards
      - asymmetric (default): only positive bonus for score > 0.5
      - gate_mul: multiplicative gating with big/small reward splitting
    """

    rubric_weights: Dict[str, float] = field(default_factory=lambda: {
        "observation_grounding": 0.35,
        "action_coherence": 0.40,
        "temporal_consistency": 0.25,
    })
    insufficient_evidence_penalty: float = 0.15
    reward_range: tuple = (-0.25, 0.25)         # clamp bounds
    lambda_proc: float = 0.10                   # multiplier before adding to env reward

    # --- Shaping mode (ported from service_llm_verifier_wrapper) ---
    symmetric_shaping: bool = False             # True => allow negative deltas
    gate_mul: bool = False                      # True => multiplicative big/small gating
    big_threshold: float = 0.5                  # score above this is "big"
    alpha_big: float = 0.0                      # floor for big rewards in multiplicative mode
    beta_small: float = 0.0                     # additive component for small rewards
    alpha_small: float = 0.0                    # floor for small rewards after hardening


@dataclass
class GatingConfig:
    """Controls when and how often the judge runs during training."""

    enable_after_step: int = 0                  # don't run before this trainer step
    disable_after_step: int = -1                # -1 = never disable
    run_every_k_steps: int = 1                  # 1 = every step, 5 = every 5th
    anneal_enabled: bool = False
    anneal_start_step: int = 35
    anneal_end_step: int = 100
    anneal_hi_scale: float = 1.0
    anneal_lo_scale: float = 0.5


@dataclass
class JudgeConfig:
    """Top-level config for the multi-fidelity routing judge."""

    enabled: bool = True

    # Rubrics to evaluate (must match the keys in the universal JSON output)
    rubrics: List[str] = field(default_factory=lambda: [
        "observation_grounding",
        "action_coherence",
        "temporal_consistency",
    ])

    # Judge mode: "universal" (one LLM call, all rubrics) or "per_rubric" (one call each)
    mode: str = "universal"

    # Primary judge provider (synchronous, in training path)
    provider: ProviderConfig = field(default_factory=ProviderConfig)

    # Optional secondary providers (e.g., for ensemble or fallback — NOT in sync path)
    secondary_providers: Dict[str, ProviderConfig] = field(default_factory=dict)

    # Layer-0 heuristics
    heuristics: HeuristicConfig = field(default_factory=HeuristicConfig)

    # Reward mapping
    reward: RewardConfig = field(default_factory=RewardConfig)

    # Training gating schedule
    gating: GatingConfig = field(default_factory=GatingConfig)

    # Audit queue
    audit: AuditConfig = field(default_factory=AuditConfig)

    # History depth: how many past turns to include in the judge prompt
    history_k: int = 3

    # Whether to include images in the judge prompt (requires multimodal provider)
    use_images: bool = True

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "JudgeConfig":
        """Build JudgeConfig from a flat/nested dict (e.g. from YAML/OmegaConf)."""
        cfg = cls()
        if not d:
            return cfg

        for k, v in d.items():
            if k == "provider" and isinstance(v, dict):
                cfg.provider = ProviderConfig(**v)
            elif k == "secondary_providers" and isinstance(v, dict):
                cfg.secondary_providers = {
                    name: ProviderConfig(**pcfg) for name, pcfg in v.items()
                }
            elif k == "heuristics" and isinstance(v, dict):
                cfg.heuristics = HeuristicConfig(**v)
            elif k == "reward" and isinstance(v, dict):
                cfg.reward = RewardConfig(**v)
            elif k == "gating" and isinstance(v, dict):
                cfg.gating = GatingConfig(**v)
            elif k == "audit" and isinstance(v, dict):
                cfg.audit = AuditConfig(**v)
            elif hasattr(cfg, k):
                setattr(cfg, k, v)

        return cfg

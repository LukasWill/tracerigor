# verifier/common/config.py
from dataclasses import dataclass, field
from typing import Dict, List, Any

@dataclass
class AnnealConfig:
    decay_start_step: int = 35
    decay_end_step: int = 100
    hi_scale: float = 1.0
    lo_scale: float = 0.5

@dataclass
class VerifierConfig:
    enabled: bool = True
    rubrics: List[str] = field(default_factory=lambda: ["self_consistency"])
    # rubrics: List[str] = field(default_factory=lambda: ["self_consistency", "history", "grounding"])
    model_name: str = "gpt-5-nano-2025-08-07"
    model_params: Dict[str, Any] = field(default_factory=lambda: {"temperature": 1.0})
    # optional reward shaping
    reward_weights: Dict[str, float] = field(default_factory=lambda: {
        "self_consistency": 1.0,
        "history_consistency": 1.0,
        "grounding": 1.0,
    })
    keep_raw: bool = False
    use_images: bool = True
    use_observation_text: bool = False
    use_responses_api: bool = False   # lets you flip Chat ↔ Responses without touching env
    history_k: int = 3                # how many past turns to send in history
    max_history_len: int = 5          # how many past turns to keep in memory for verifier context
    anneal: AnnealConfig = field(default_factory=AnnealConfig)
    symmetric_shaping: bool = False   # true => allow negative deltas
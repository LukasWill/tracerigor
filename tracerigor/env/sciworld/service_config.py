"""
SciWorld Service Configuration for TraceRigor.

This module defines the service-level configuration for batch processing
of SciWorld environments.
"""
from tracerigor.env.base.base_service_config import BaseServiceConfig
from dataclasses import dataclass


@dataclass
class SciWorldServiceConfig(BaseServiceConfig):
    """
    Configuration for SciWorldService with support for state reward functionality.

    Attributes:
        max_workers: Maximum number of worker threads for parallel processing
        use_state_reward: Whether to enable state-based reward computation
        top_strings_m: Maximum number of strings to track for repetition detection
        top_strings_k: Top-k strings for repetition penalty
    """

    # Inherited from BaseServiceConfig
    max_workers: int = 10

    # State reward configuration
    use_state_reward: bool = False
    top_strings_m: int = 1000  # Maximum number of strings to track for repetition detection
    top_strings_k: int = 5     # Top-k strings for repetition penalty


if __name__ == "__main__":
    config = SciWorldServiceConfig()
    print(f"max_workers: {config.max_workers}")
    print(f"use_state_reward: {config.use_state_reward}")

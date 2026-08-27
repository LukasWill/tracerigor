import re
from typing import List, Dict, Any, Optional

import asyncio
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
from ollama import AsyncClient, ResponseError
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Verifier base class and specific aspects
# -----------------------------------------------------------------------------
class BaseVerifier:
    def __init__(self, config: DictConfig, model_name: str, client: Optional[AsyncClient], model_params: Dict[str, Any]):
        # TODO: swap client with model_params if needed
        self.config = config
        self.model_name = model_name
        self.client = client
        self.model_params = model_params

    def get_template(self, item: Dict[str, Any]) -> str:
        raise NotImplementedError(
            f"{self.__class__.__name__}.get_template() not implemented. "
            "Subclass must return a prompt string."
        )

    def parse_response(self, text: str) -> tuple[float, bool]:
        # default YES/NO parse with no extra metric
        m = re.search(r'<answer>(YES|NO)</answer>', text, re.IGNORECASE)
        score = 1.0 if (m and m.group(1).upper() == "YES") else 0.0
        parse_success = bool(m)
        return score, parse_success

    async def evaluate_batch(self, input_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        raise NotImplementedError(
            f"{self.__class__.__name__}.evaluate_batch() not implemented. "
            "Subclass must implement this method to process a batch of inputs."
        )
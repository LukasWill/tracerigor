# utils/load_verifier_cfg.py
import yaml
from tracerigor.verifier.verifier.common.config import VerifierConfig

def load_verifier_cfg(path: str) -> VerifierConfig:
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    return VerifierConfig(**(data.get("verifier", {}) or {}))


if __name__ == "__main__":
    cfg = load_verifier_cfg("tracerigor/verifier/verifier/common/verifier_config.yaml")
    print(cfg)
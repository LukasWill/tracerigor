# Trace verifier layer

This package contains the environment-aware prompt and verification machinery
used to evaluate visible multi-turn reasoning traces.

## Components

- `prompt/`: rubric prompts for Sokoban, SciWorld, Navigation, and ALFWorld;
- `verifier/`: OpenAI-compatible and Ollama verifier clients, shared state,
  replay, and mechanical checks;
- `utils/`: response parsing, multimodal normalization, registration, and
  optional W&B logging;
- `scripts/`: offline evaluators and summaries for externally obtained records;
- `config/`: public non-secret provider and rubric defaults.

The main judge integration in `tracerigor.judge` builds typed turn packets and
reuses these prompts. The standard universal rubric returns observation
grounding, action coherence, and temporal consistency judgments.

Example imports:

```python
from tracerigor.verifier.prompt.sciworld import SciWorldUniversalTemplate
from tracerigor.verifier.verifier.common.verifier_memory import VerifierMemory
from tracerigor.verifier.verifier.sciworld_mechanical_checks import run_mechanical_prefilter
```

The lightweight `tracerigor.judge` client requires the `judge` extra and a
runtime provider. Legacy verifier servers and domain-specific evaluation
scripts additionally import their direct optional stack, including
Hydra/OmegaConf, FastAPI/Pydantic, W&B, and the selected environment/provider
packages. Keep API credentials in environment variables; the tracked YAML
files contain no working secrets. Offline mechanical checks and parsers do not
require a remote judge.

Evaluation scripts accept experiment records supplied by the user. Raw project
runs and judge outputs are intentionally not bundled with the public release.

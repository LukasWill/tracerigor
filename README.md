# TraceRigor

**Train, verify, and analyze multi-turn agent trajectories.**

TraceRigor is a research toolkit for studying whether reasoning traces generated
during reinforcement learning with verifiable rewards (RLVR) remain reliable.
It connects service-backed multi-turn rollouts and RL training with a
judge/verifier layer and schema-tolerant analysis of externally obtained
experiment records.

The project focuses on three observable trace properties:

- observation grounding;
- action coherence; and
- temporal consistency across turns.

It does not treat a high environment reward as proof that a trajectory's
reasoning is faithful or reliable.

## Public workflows

TraceRigor supports five related workflows:

1. configure and register an interactive environment;
2. generate lightweight training and evaluation seed datasets;
3. run service-backed multi-turn rollout and RL training;
4. evaluate turns and trajectories through the judge/verifier layer; and
5. summarize JSON or JSONL experiment records with a generic analysis command.

The analysis command and lightweight tests run on a CPU. Full rollout and
training workflows require external model weights, environment dependencies,
GPU resources, and a compatible VERL installation.

## Compute and hardware scope

Hardware requirements are workflow-dependent; the repository does not claim a
single universal GPU or VRAM minimum.

| Workflow | Expected compute | External requirements |
| --- | --- | --- |
| CLI, record analysis, and unit tests | CPU | Python 3.10 or 3.11 |
| Lightweight seed-dataset generation | CPU for the bundled example | Dataset and selected environment dependencies |
| Hosted judge evaluation | CPU client | Network access, provider account, and runtime API key |
| Self-hosted judge evaluation | Model-dependent GPU capacity | An OpenAI-compatible vLLM/Ollama service and its model weights |
| Multi-turn rollout and RL training | Normally one or more CUDA GPUs | Compatible VERL runtime, model weights, environment services, and storage for checkpoints |

For GPU-backed workflows, size capacity from the selected model, precision,
sequence length, batch size, optimizer state, and sharding strategy. Start with
the two-step training smoke configuration before increasing any of those
dimensions. CPU-only users can still install the base package, analyze records,
inspect environment availability, and run the test suite.

## Installation

Use Python 3.10 or 3.11 in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Install only the extras needed for a workflow:

```bash
python -m pip install -e '.[analysis]'
python -m pip install -e '.[data,envs]'
python -m pip install -e '.[judge]'
python -m pip install -e '.[train]'
python -m pip install -e '.[dev]'
```

Some environments have additional upstream prerequisites; `tracerigor envs
--verbose` reports integrations that are unavailable in the current
environment without preventing the base package from importing.

## Quick start: analyze trace reliability

The included records are synthetic and small:

```bash
tracerigor analyze examples/analysis/sample_records.jsonl \
  --group-by training_step
```

The command reports rubric coverage, pass and uncertainty rates, mean trace
reliability, and parse/query success rates. Input may be JSON or JSONL and can
use direct rubric fields or the nested judge-response schemas used by the
project.

## Environments and seed datasets

List registered integrations:

```bash
tracerigor envs --verbose
```

Generate a small Blackjack seed dataset:

```bash
python -m pip install -e '.[data,envs]'
tracerigor data examples/data/blackjack.yaml \
  --train-path data/blackjack/train.parquet \
  --test-path data/blackjack/test.parquet
```

The example defaults to text observations. Blackjack vision observations are
rendered in code and do not rely on redistributed card artwork.

## Judge and verifier layer

`tracerigor.judge` provides typed judge packets, configurable providers,
heuristic gating, structured responses, audit support, and process-reward
integration. `tracerigor.verifier` contains environment-specific prompts,
parsers, mechanical checks, and OpenAI-compatible verifier clients.

Provider secrets must be supplied at runtime through the standard environment
variables used by the selected client, such as `OPENAI_API_KEY`,
`ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, or `TOGETHER_API_KEY`. Never put working
credentials in tracked configuration files. A local OpenAI-compatible judge
server can use the non-secret placeholder `EMPTY`; see
`scripts/launch_judge_server.sh`.

## Training

Training is intentionally not a zero-compute quick start. It depends on a
compatible external VERL runtime, model weights, environment services, and
normally one or more GPUs. After generating train and validation parquet files:

```bash
export MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct
export TRAIN_FILE=data/blackjack/train.parquet
export VAL_FILE=data/blackjack/test.parquet
bash examples/train/run.sh
```

The example defaults to two training steps and console-only logging. It is a
configuration smoke example, not a performance reproduction recipe. Review
model licences and remote-code settings before using third-party checkpoints.

## Repository layout

```text
tracerigor/          installable package
  analysis/          schema-tolerant reliability summaries
  env/               environment and service registry
  judge/             typed judge, routing, reward, and audit layer
  verifier/          prompts, parsers, mechanical checks, clients
  rollout/           service-backed multi-turn rollout managers
  trainer/           RL training entry point and credit assignment
examples/             small public analysis, data, and training examples
scripts/              generic setup and service utilities
tests/                public unit and integration checks
```

Raw run directories, model checkpoints, W&B downloads, generated plots, and
private cluster launchers are intentionally excluded. Research-working
directories and manuscript sources are also kept outside this software release.

## Reproducibility scope

The repository provides reusable implementation, prompt definitions, public
configuration examples, and lightweight fixtures. It does not contain the
original raw experiment runs, proprietary judge outputs, research notebooks,
manuscript sources, third-party datasets, or model checkpoints. The included
synthetic analysis example is fully reproducible; paper-level numeric
reproduction is outside the scope of this software release.

## Relationship to VAGEN and VERL

TraceRigor is derived from the public
[VAGEN](https://github.com/RAGEN-AI/VAGEN) codebase and retains substantial
environment, rollout, and training infrastructure from that project. The
current project extends that foundation with trace-reliability judging,
verifier prompts and mechanical checks, RLVR-oriented reliability analysis,
additional environment integrations, and public workflow consolidation.

The original MIT licence and copyright notice are retained in `LICENSE`.
VERL and environment packages remain external dependencies governed by their
own licences. See `THIRD_PARTY_NOTICES.md` for provenance and review notes.

The TraceRigor manuscript is currently anonymized; project-specific citation
metadata will be added when the author list and archival identifier are public.

## Licence

The repository is released under the MIT License in `LICENSE`. Retained
third-party components and dependencies may carry separate terms.

# TraceRigor

[![CI](https://github.com/LukasWill/tracerigor/actions/workflows/ci.yml/badge.svg)](https://github.com/LukasWill/tracerigor/actions/workflows/ci.yml)
![Python 3.10 | 3.11](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**Code and results for [RLVR Makes Agents Win More, but Not Necessarily Reason Better](https://openreview.net/forum?id=jsTJu11CIh)**<br>
Wenhao Lu, Jae Hee Lee, Josua Spisak, Xufeng Zhao, Stefan Wermter<br>
NeurIPS 2026 [Workshop on Interpreting Agent Behavior](https://openreview.net/forum?id=jsTJu11CIh) ·
[Workshop on Evaluation of Interactive Agents](https://openreview.net/forum?id=xpTzSAVfAF)

Reinforcement learning from verifiable rewards (RLVR) makes LLM and VLM agents
succeed more often. TraceRigor asks whether the reasoning they show along the
way gets more reliable too. It trains multi-turn agents with RLVR and judges
every turn of their visible reasoning on three rubrics: is it **grounded** in
the current observation, is the action **coherent** with it, and is it
**temporally consistent** with what earlier turns revealed?

<p align="center">
  <img src="docs/assets/figures/overview.png" alt="Evaluation framework: multi-turn RLVR produces trajectories; a trace reliability judge scores grounding, action coherence, and temporal consistency per turn; internal signals are logged; analyses run across checkpoints, within a checkpoint, and across turns." width="100%">
</p>

## Key findings

Five agents (Qwen2.5-3B/7B and Qwen2.5-VL-3B) trained with PPO on ALFWorld,
ScienceWorld, Sokoban, and AI2-THOR Navigation. Turns from evaluation rollouts
at each checkpoint are judged post hoc; the judge is never part of the
training reward.

**1. Performance and reliability rise together across training...**
Mean cumulative reward climbs from −8.0 to +60.5 for SciWorld-3B, and
ALFWorld success from 8.6% to 85.2%. Mean rubric scores also tend to rise, but
non-monotonically, and the weakest rubric depends on modality: temporal
consistency for long-horizon text agents, grounding for visual agents.

<p align="center">
  <img src="docs/assets/figures/performance_reliability_across_training.png" alt="Per-checkpoint mean rubric scores and task performance across training for five agents, with OLS fits of composite reliability against performance." width="100%">
</p>

**2. ...but within a checkpoint, the trajectories that win are barely more reliable.**
Splitting each checkpoint's trajectories into high and low performers, the
length-controlled gap in composite reliability is small (≈0.1 on a 0–1
scale). The two groups' reliability distributions overlap by 49–82%, and a
randomly drawn high performer is more reliable than a low performer only
55–70% of the time (50% is chance). Controlling for trajectory length
collapses apparent couplings, for example action coherence in ALFWorld from
+0.40 to +0.08 and grounding in Sokoban from +0.45 to +0.15 (Spearman ρ).

<p align="center">
  <img src="docs/assets/figures/high_vs_low_performers.png" alt="Length-controlled composite reliability of high- and low-performing trajectories per checkpoint for five agents, with 95% confidence ribbons and the pooled gap." width="100%">
</p>

**3. Reliability decays over the turns of an episode, even in successful ones.**
Temporal consistency leads the decline in long-horizon text agents, with the
strongest decline in ALFWorld (mean within-trajectory ρ = −0.27); Sokoban's
decline is led by grounding (−0.18) and Navigation's by action coherence
(−0.21). The early- vs late-turn gap in temporal consistency peaks at +0.42
(SciWorld-3B, step 50); it often narrows as agents master the task and can
widen again when performance regresses.

<p align="center">
  <img src="docs/assets/figures/reliability_by_turn_position.png" alt="Per-rubric reliability against normalised turn position for SciWorld-3B, SciWorld-7B, and ALFWorld-3B." width="49%">
  <img src="docs/assets/figures/early_vs_late_turns.png" alt="Mean rubric scores over the first five and last five turns of each trajectory across training." width="49%">
</p>

**4. Agents act coherently on stale or false beliefs.** Auditing 34,436
text-agent and 3,524 visual-agent turns: in ScienceWorld, 25.2% of turns contain
unsupported progress or outcome claims and 20.1% repeat failures without
adapting the plan; in Sokoban, 56.9% follow a false premise coherently and
72.6% reverse a player/box/target relation relative to the image.

<p align="center">
  <img src="docs/assets/figures/sokoban_false_premise.png" alt="Three Sokoban turns where the agent misstates the spatial relation between itself, the box, and the target, yet chooses actions coherent with that false premise." width="520">
</p>

**5. Internal signals don't certify reliable reasoning.** Token entropy and
attention allocation shift broadly during training but rarely separate more-
from less-reliable traces, and more attention to the image does not by itself
mean grounded reasoning.

Judge labels (`gpt-5.4-nano-2026-03-17`) agree with independent human labels
on 465 doubly-annotated turns with raw agreement of at least 0.79 in every
modality × rubric cell. All main-text figures, tables, and run settings are in
**[docs/results.md](docs/results.md)**.

## What's in this repository

| Rubric | Question asked of each turn |
| --- | --- |
| **Observation grounding** | Do the reasoning's claims about the current state match the observation? |
| **Action coherence** | Does the chosen action follow from the stated reasoning? |
| **Temporal consistency** | Does the reasoning update its beliefs after feedback from earlier turns? |

- **Multi-fidelity trace judge.** Deterministic Layer-0 checks (format,
  out-of-action-space, repetition, empty reasoning) resolve mechanically
  detectable failures *before* any LLM call. Only the remaining turns go to a
  Layer-1 LLM judge on any OpenAI-compatible endpoint (vLLM, OpenAI, Together,
  OpenRouter), with JSON-schema guided decoding, rate limiting, retries, and
  sampled audit logs of low-confidence verdicts
  ([`tracerigor/judge/`](tracerigor/judge/)).
- **Environment-aware prompts with ground truth.** Sokoban prompts include
  simulator-derived spatial relations, and ScienceWorld prompts encode the
  game's observation types, so the judge checks claims against state rather
  than plausibility. Non-LLM ScienceWorld checks and trajectory replay provide
  ground truth for validating the judge itself
  ([`tracerigor/verifier/`](tracerigor/verifier/)).
- **Multi-turn RLVR stack.** Service-backed environments, a multi-turn rollout
  manager with turn-level loss/GAE masks, and PPO/GRPO with `bi_level_gae`,
  `turn_wise_gae`, and `masked_gae` advantage estimators for turn-level credit
  assignment. Rubric-based process rewards can optionally shape training, with
  bounded mapping, gating, and annealing schedules (not used in the paper,
  which judges post hoc).
- **Nine environment integrations.** Sokoban, FrozenLake, and Blackjack run on
  CPU from `pip`; ALFWorld, ScienceWorld, BabyAI-Text, AI2-THOR Navigation,
  ManiSkill PrimitiveSkill, and SVG reconstruction are opt-in. Missing
  dependencies are isolated per environment and reported by `tracerigor envs`.
- **Reliability analysis.** `tracerigor analyze` summarizes judged JSON/JSONL
  records by checkpoint or any other field: rubric pass/uncertain rates, mean
  trace reliability, parse and query success, and heuristic short-circuit rate.

<details>
<summary><b>Toolkit architecture</b></summary>

```mermaid
flowchart LR
    subgraph train["Multi-turn RLVR (VERL + Ray, GPU)"]
        direction TB
        data["Seed datasets<br/>tracerigor data"] --> rollout["Rollout manager<br/>multi-turn, turn masks"]
        envs["Environment services<br/>Sokoban · ALFWorld · ScienceWorld · …"] <--> rollout
        rollout --> trainer["PPO / GRPO<br/>bi-level & turn-wise GAE"]
        trainer -- "updated policy" --> rollout
        envs -. "optional rubric process reward" .-> rollout
    end
    rollout -- "agent turns" --> L0
    subgraph judge["Trace judge (CPU client, tracerigor judge)"]
        direction TB
        L0["Layer 0: deterministic checks<br/>format · action space · repetition"] -- "pass" --> L1["Layer 1: LLM judge<br/>grounding · coherence · consistency"]
        L0 -- "fail" --> SC["short-circuit verdict<br/>(no LLM call)"]
    end
    L1 --> analyze["tracerigor analyze<br/>reliability by checkpoint"]
    SC --> analyze
```

</details>

## Quick start (CPU, about two minutes)

Requires Python 3.10 or 3.11. No GPU, model weights, or API key needed.

```bash
git clone https://github.com/LukasWill/tracerigor.git
cd tracerigor
python -m venv .venv && source .venv/bin/activate
python -m pip install -e '.[envs]'
```

**1. Run the judge's deterministic layer on example turns.**
[`examples/judge/sample_turns.jsonl`](examples/judge/sample_turns.jsonl)
contains six Sokoban turns. The observations come from the seeded simulator,
and the agent responses are hand-written to show distinct failure modes.

```bash
tracerigor judge examples/judge/sample_turns.jsonl --dry-run --show-prompt -o routed.jsonl
# 6 turns: 2 short-circuited by heuristics, 4 would be sent to the LLM judge (dry run, no requests made)
```

| Turn | Failure mode | Route |
| --- | --- | --- |
| `invalid_action` | `Jump` is outside the action space | Layer 0, rejected without an LLM call |
| `format_violation` | no parseable action | Layer 0, rejected without an LLM call |
| `hallucinated_state` | reasoning misdescribes the grid | Layer 1 (grounding) |
| `incoherent_action` | pushes `Left` after planning to push right | Layer 1 (coherence) |
| `stale_belief` | claims last turn solved the puzzle | Layer 1 (temporal consistency) |
| `grounded` | consistent control case | Layer 1 |

`routed.jsonl` contains each turn's heuristic checks and the exact prompt the
LLM judge would receive, including the ground-truth spatial state.

**2. Judge them with an LLM.** Point the judge at any OpenAI-compatible server
(for example one started with [`scripts/launch_judge_server.sh`](scripts/launch_judge_server.sh))
or a hosted provider, then summarize:

```bash
python -m pip install -e '.[judge]'
tracerigor judge examples/judge/sample_turns.jsonl \
  --base-url http://127.0.0.1:8001/v1 --model <served-model-name> -o judged.jsonl
# hosted provider: --base-url https://api.openai.com/v1 --model <model> --api-key-env OPENAI_API_KEY
tracerigor analyze judged.jsonl
```

Verdicts depend on the judge model you choose. CI checks this path end to end
against a local stub endpoint. If a provider rejects JSON-schema decoding, add
`--no-structured-output`.

**3. Step an environment directly.**

```python
from tracerigor.env import get_environment

spec = get_environment("sokoban")
env = spec["env_cls"](spec["config_cls"](render_mode="text"))
obs, info = env.reset(seed=7)
print(obs["obs_str"])
obs, reward, done, info = env.step("<think>The box is to my right.</think><action>Right</action>")
print(reward, info["metrics"]["turn_metrics"])  # 0.4 {'action_is_valid': True, 'action_is_effective': True}
```

## Workflows and requirements

| Workflow | Command | Compute | Install |
| --- | --- | --- | --- |
| Analyze judged records | `tracerigor analyze` | CPU | base |
| Heuristic routing and prompt inspection | `tracerigor judge --dry-run` | CPU | base |
| LLM judging | `tracerigor judge` | CPU client + an endpoint | `.[judge]` |
| Inspect environments | `tracerigor envs --verbose` | CPU | `.[envs]` |
| Generate seed datasets | `tracerigor data` | CPU (Blackjack example) | `.[data,envs]` |
| Self-hosted judge model | `scripts/launch_judge_server.sh` | GPU sized to the model | vLLM (separate env) |
| Multi-turn RL training | `tracerigor train` / `examples/train/run.sh` | One or more CUDA GPUs | `.[train]` + verl-tracerigor |

There is no single VRAM figure: requirements follow from model size,
precision, context length, batch size, and sharding. See
[Installation and workflows](docs/run-exp.md) for details on each workflow and
[The trace judge](docs/judge.md) for the input schema, configuration, Python
API, and reward mapping.

### Seed datasets

```bash
python -m pip install -e '.[data,envs]'
tracerigor data examples/data/blackjack.yaml \
  --train-path data/blackjack/train.parquet \
  --test-path data/blackjack/test.parquet
```

### Training

The paper's agents were trained with
[**verl-tracerigor**](https://github.com/LukasWill/verl-tracerigor), a VERL
0.2.0.dev build whose PPO policy loss uses asymmetric clipping (0.2 / 0.28) with
dual-clip (3.0). The tag `tracerigor-neurips2026` marks the exact revision; its
[`TRACERIGOR_CHANGES.md`](https://github.com/LukasWill/verl-tracerigor/blob/main/TRACERIGOR_CHANGES.md)
lists every change to upstream VERL.

```bash
python -m pip install -e '.[train]'
python -m pip install --no-deps \
  "verl @ git+https://github.com/LukasWill/verl-tracerigor.git@tracerigor-neurips2026"
```

VERL's runtime stack (vLLM, flash-attention, and a matching PyTorch/CUDA build)
is installed separately; the exact package versions used for the paper are not
recorded yet. Training is not a zero-compute quick start. The launcher checks
its required variables and defaults to a two-step smoke configuration with
console-only logging:

```bash
export MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct
export TRAIN_FILE=data/blackjack/train.parquet
export VAL_FILE=data/blackjack/test.parquet
bash examples/train/run.sh
```

The smoke configuration is not the paper's recipe. The paper used bi-level GAE
with turn-level discount 0.95, actor/critic learning rates 1e-6/1e-5, 128
environments per step, a K = 5 turn history window, and 300 PPO steps; see
[Run configuration](docs/results.md#run-configuration) for all per-agent
settings.

## Where to look in the code

| If you want to see… | Start here |
| --- | --- |
| Layer-0/Layer-1 routing, response parsing, confidence | [`judge/router.py`](tracerigor/judge/router.py), [`judge/heuristics.py`](tracerigor/judge/heuristics.py) |
| Async provider client (rate limits, retries, event-loop handling) | [`judge/client.py`](tracerigor/judge/client.py) |
| Rubric scores → bounded process reward | [`judge/reward.py`](tracerigor/judge/reward.py) |
| Environment-specific judge prompts | [`judge/env_templates.py`](tracerigor/judge/env_templates.py), [`verifier/prompt/`](tracerigor/verifier/prompt/) |
| Mechanical ScienceWorld checks and replay | [`verifier/verifier/sciworld_mechanical_checks.py`](tracerigor/verifier/verifier/sciworld_mechanical_checks.py), [`sciworld_trajectory_replay.py`](tracerigor/verifier/verifier/sciworld_trajectory_replay.py) |
| Turn-level credit assignment | `compute_advantage` in [`trainer/ppo/ray_trainer.py`](tracerigor/trainer/ppo/ray_trainer.py) |
| Multi-turn rollout and masks | [`rollout/qwen_rollout/`](tracerigor/rollout/qwen_rollout/), [`rollout/utils/`](tracerigor/rollout/utils/) |
| A compact environment implementation | [`env/blackjack/`](tracerigor/env/blackjack/), [`env/sokoban/`](tracerigor/env/sokoban/) |

```text
tracerigor/
  analysis/   schema-tolerant reliability summaries (tracerigor analyze)
  judge/      typed packets, heuristics, router, client, reward, audit, batch CLI
  verifier/   environment prompts, parsers, mechanical checks, verifier clients
  env/        environment + service registry and nine integrations
  rollout/    service-backed multi-turn rollout managers
  trainer/    Hydra/VERL PPO entry point and advantage estimators
  inference/  provider adapters for evaluation rollouts
  server/     legacy environment and judge servers
examples/     judge, analysis, dataset, and training examples
docs/         paper results, workflow, configuration, judge, and environment guides
```

## Reproducibility scope

This repository is the software release accompanying the paper. It contains
the implementation, prompt definitions, public configuration, and small
fixtures. It does not contain raw experiment runs, model checkpoints, judge
outputs from the original experiments, research notebooks, manuscript sources,
or third-party datasets, so the paper's numbers cannot be regenerated from it
alone. The paper's settings, the judge model, and the pinned VERL build are
documented in [docs/results.md](docs/results.md). Everything in the quick
start is reproducible on CPU and runs in CI. Open issues that affect
reproducibility are tracked in [docs/known-issues.md](docs/known-issues.md).

## Relationship to VAGEN and VERL

TraceRigor is derived from the
[VAGEN](https://github.com/RAGEN-AI/VAGEN) codebase and retains
its rollout mechanism and training infrastructure. The
current project extends that foundation with trace-reliability (online) judging,
verifier prompts and mechanical (e.g., attention) checks, turn-wise reliability relationship analysis,
and supports additional environment integrations.

## Citation

If you use TraceRigor, please cite the paper. It appears at two NeurIPS 2026
workshops; cite the venue you refer to. Both entries share the key
`lu2026rlvr`, so rename one if you include both in the same `.bib` file.

```bibtex
@inproceedings{
lu2026rlvr,
title={{RLVR} Makes Agents Win More, but Not Necessarily Reason Better},
author={Wenhao Lu and Jae Hee Lee and Josua Spisak and Xufeng Zhao and Stefan Wermter},
booktitle={The 1st Workshop on Interpreting Agent Behavior (IAB) at NeurIPS 2026},
year={2026},
url={https://openreview.net/forum?id=jsTJu11CIh}
}
```

```bibtex
@inproceedings{
lu2026rlvr,
title={{RLVR} Makes Agents Win More, but Not Necessarily Reason Better},
author={Wenhao Lu and Jae Hee Lee and Josua Spisak and Xufeng Zhao and Stefan Wermter},
booktitle={Workshop on Evaluation of Interactive Agents @ NeurIPS 2026},
year={2026},
url={https://openreview.net/forum?id=xpTzSAVfAF}
}
```

GitHub's "Cite this repository" button uses [`CITATION.cff`](CITATION.cff).

## Licence

Except where individual files state otherwise, TraceRigor is distributed under
the MIT License in [`LICENSE`](LICENSE), which retains the upstream VAGEN
copyright notice. Four retained files are under Apache License 2.0; see
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and
[`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt). Optional dependencies,
datasets, and models keep their own terms.

Contributions are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md).

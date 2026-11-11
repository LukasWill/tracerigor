# TraceRigor

TraceRigor is a toolkit for training multi-turn LLM/VLM agents with RLVR and
auditing whether their visible reasoning traces stay reliable: grounded in the
observation, coherent with the chosen action, and consistent across turns.

It accompanies the paper
[RLVR Makes Agents Win More, but Not Necessarily Reason Better](https://openreview.net/forum?id=jsTJu11CIh)
(NeurIPS 2026 Workshop on Interpreting Agent Behavior; also at the
[Workshop on Evaluation of Interactive Agents](https://openreview.net/forum?id=xpTzSAVfAF)).

The paper's figures, tables, and run settings are collected in
[Paper results](results.md).

Use the documentation to:

- install only the dependencies needed for your workflow;
- inspect and register optional environments;
- generate seed datasets;
- configure multi-turn rollout, training, and judge-backed evaluation;
- score recorded turns with the multi-fidelity [trace judge](judge.md); and
- analyze judged trajectory records without a training runtime.

Start with [Installation and workflows](run-exp.md). The smallest executable
workflow is:

```bash
python -m pip install -e .
tracerigor judge examples/judge/sample_turns.jsonl --dry-run
tracerigor analyze examples/analysis/sample_records.jsonl
```

Issues that affect reproducibility are listed under [Known issues](known-issues.md).

TraceRigor derives from [VAGEN](https://github.com/RAGEN-AI/VAGEN). See the
repository README and `THIRD_PARTY_NOTICES.md` for provenance and attribution.

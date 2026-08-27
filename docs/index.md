# TraceRigor

TraceRigor is a toolkit for training, verifying, and analyzing multi-turn agent
trajectories, with an emphasis on the reliability of visible reasoning traces
during RLVR.

Use the documentation to:

- install only the dependencies needed for your workflow;
- inspect and register optional environments;
- generate seed datasets;
- configure multi-turn rollout, training, and judge-backed evaluation; and
- analyze judged trajectory records without a training runtime.

Start with [Installation and workflows](run-exp.md). The smallest executable
workflow is:

```bash
python -m pip install -e .
tracerigor analyze examples/analysis/sample_records.jsonl
```

TraceRigor derives from [VAGEN](https://github.com/RAGEN-AI/VAGEN). See the
repository README and `THIRD_PARTY_NOTICES.md` for provenance and attribution.

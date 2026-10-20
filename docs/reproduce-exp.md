# Reproducibility scope

The public release retains the software paths needed to configure environments,
generate seed datasets, run rollout and RL training, judge trace reliability,
and analyze exported records.

It intentionally excludes raw W&B runs, checkpoints, media exports, judge-case
datasets, generated plots, cluster-specific launchers, and local caches. Those
materials are either large, operational, privacy-sensitive, or not suitable for
redistribution. Research notebooks and manuscript sources are also outside this
software release, so paper-level results cannot be regenerated from this
repository alone.

What can be checked locally:

```bash
tracerigor analyze examples/analysis/sample_records.jsonl
tracerigor judge examples/judge/sample_turns.jsonl --dry-run
python -m pytest
bash -n examples/train/run.sh
```

The analysis fixture is synthetic. The judge example uses observations from
the seeded Sokoban simulator with hand-written agent responses; regenerate it
with `python examples/judge/make_sample_turns.py`. The test suite also runs
the LLM-judge path end to end against a local stub endpoint.

Open reproducibility issues, including the unpinned VERL revision, are listed
in [Known issues](known-issues.md).

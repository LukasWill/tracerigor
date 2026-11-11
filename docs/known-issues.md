# Known issues and implementation notes

This page lists issues found during code review that affect reproducibility
or could change numerical behaviour. Items marked **open** were left unchanged
on purpose, because fixing them would alter rewards, prompts, or training
dynamics relative to existing runs. Each needs an explicit decision from the
maintainers.

## Reproducibility

**Runtime package versions are not recorded (open).** The VERL build used for
the paper is published as
[verl-tracerigor](https://github.com/LukasWill/verl-tracerigor) at tag
`tracerigor-neurips2026` (commit `a06492b`, based on JamesKrW/verl `f0c9812`).
Its policy loss differs from upstream VERL (asymmetric clip 0.2 / 0.28 with
dual-clip 3.0, fixed in code), so upstream VERL does not reproduce the paper's
training dynamics. The `torch`, `transformers`, `vllm`, `flash-attn`, and `ray`
versions used for the paper are not yet recorded. *Proposed:* export them from
the training environment (`pip freeze`) and add a lock file.

**Paper settings differ from shipped defaults (by design).** The paper's judge
model is `gpt-5.4-nano-2026-03-17`, while the shipped verifier configuration
defaults to `gpt-5-nano-2025-08-07`. `examples/train/run.sh` is a two-step
smoke configuration with `algorithm.high_level_gamma=1.0`; the paper used a
turn-level discount of 0.95. [Paper results](results.md#run-configuration)
lists the settings used for each run.

**The judge package is not wired into training (open, by design in this
release).** `tracerigor.judge.JudgeIntegration` documents how to call it from
the rollout manager, but the shipped rollout loop does not. Training-time
process rewards come from
`tracerigor/env/utils/service_llm_verifier_wrapper.py`, which calls the
`tracerigor.verifier` clients. Use `tracerigor.judge` for offline scoring
(`tracerigor judge`) or integrate it explicitly.

## Reward and judge semantics

**`heuristics.short_circuit_penalty` is unused (open).** The config documents
it as "r_proc when heuristic fires" (default −0.20), but
`compute_process_reward` assigns short-circuited turns
`-|insufficient_evidence_penalty|` (default −0.15). *Proposed:* use
`short_circuit_penalty` for short-circuited turns, or remove the field.

**Default shaping mode equals `symmetric_shaping` (open).** The
`RewardConfig` docstring describes the default as asymmetric ("only positive
bonus for score > 0.5"), but the default and symmetric branches both compute
`w_t · (2·r_raw − 1)` and allow negative rewards. `alpha_small` is
not used by `compute_process_reward`. *Proposed:* implement the documented
asymmetric default or correct the docstring.

**Rollout history order (open).** `build_packets_from_step_results` passes
recorder entries in chronological order, while the prompt header says "most
recent first", the formatter labels index 0 as `t-1`, and the `repetition`
heuristic compares against `history[0]`. With `history_k > 1`, the repetition
check therefore compares against the oldest turn in the window. `tracerigor
judge` input is documented as most-recent-first and is not affected.
*Proposed:* reverse the recorder window in `packet_builder.py`.

**Blocking-call helper in the training-path verifier wrapper (open).**
`_run_coro_blocking` in `service_llm_verifier_wrapper.py` uses the same
`try/except RuntimeError` pattern that was fixed in the judge client: when
called inside a running event loop, a `RuntimeError` raised by the coroutine
falls through to `asyncio.run()`, which fails with a different error and hides
the original one. Behaviour is unchanged outside a running loop.

## Fixed in this release

- **Judge client and event loops.** `JudgeClient.judge_batch_sync` runs each
  batch under a new `asyncio.run` loop, but its concurrency semaphore stayed
  bound to the loop in which a request first had to wait for a free slot.
  After any batch larger than `max_concurrency`, requests in later batches
  that also had to wait failed and were recorded as `query_success=False`,
  which maps to a neutral process reward. Loop-bound resources are now
  recreated per loop; covered by `tests/test_judge_batch.py`. This affected
  only direct users of `JudgeRouter`/`JudgeIntegration`. The paper's judging
  used the `tracerigor.verifier` client, which creates its rate limiter per
  batch and is unaffected.
- **Sokoban import with the `envs` extra.** Sokoban imported the legacy judge
  server (Hydra, OmegaConf, W&B, Together) at module load, so `pip install
  '.[envs]'` did not make it usable. Those backends are now imported on first
  use.
- **Environment availability.** `tracerigor envs` reported ALFWorld and
  ScienceWorld as available without their simulator packages installed; it
  now checks for `alfworld`/`textworld` and `scienceworld`.
- **Test isolation.** Two integration tests replaced the installed `openai`,
  `together`, and `gym_sokoban` packages with stubs for the whole pytest
  session; stubs now apply only when a package is missing.

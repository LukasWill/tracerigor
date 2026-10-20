# Known issues and implementation notes

This page lists issues found during code review that affect reproducibility
or could change numerical behaviour. Items marked **open** were left unchanged
on purpose, because fixing them would alter rewards, prompts, or training
dynamics relative to existing runs. Each needs an explicit decision from the
maintainers.

## Reproducibility

**VERL revision is not pinned (open).** The trainer imports VERL APIs
(`verl.workers.fsdp_workers`, `verl.workers.reward_manager`,
`verl.utils.dataset.rl_dataset`, `verl.single_controller.ray`) but the release
does not record which VERL revision the experiments used. Upstream VAGEN
vendors VERL as a git submodule. *Proposed:* record the exact VERL commit (or
add it as a submodule) together with the tested `torch`, `transformers`,
`vllm`, and `ray` versions.

**The judge package is not wired into training (open, by design in this
release).** `tracerigor.judge.JudgeIntegration` documents how to call it from
the rollout manager, but the shipped rollout loop does not. Training-time
process rewards come from
`tracerigor/env/utils/service_llm_verifier_wrapper.py`, which calls the
`tracerigor.verifier` clients. Use `tracerigor.judge` for offline scoring
(`tracerigor judge`) or integrate it explicitly.

**Bundled aggregate statistics have no generating code.**
`examples/analysis/aggregate_step_controlled_partials/*.json` contain
per-rubric cross-step and step-controlled partial correlations for several
runs. The code and the raw judged records that produced them are not part of
this release, so treat them as schema examples, not reproducible results.

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
  bound to the first loop. In batches larger than `max_concurrency`, every
  later batch silently returned `query_success=False` for most turns, which
  map to a neutral process reward. Loop-bound resources are now recreated per
  loop; covered by `tests/test_judge_batch.py`. Only direct users of
  `JudgeRouter`/`JudgeIntegration` were affected; the training-path verifier
  uses a different client.
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

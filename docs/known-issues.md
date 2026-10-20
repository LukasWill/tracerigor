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

**Repetition check never fires on rollout history (open).** History built by
`build_packets_from_step_results` stores the observation under `obs`, while
the `repetition` heuristic reads `observation_text` or
`post_action_observation`, and stores `action_tokens` only when the recorder
entry has `think_content`. The check therefore never fires for packets built
from the rollout recorder; `tracerigor judge` input uses `observation_text`
and is not affected. The check is informational and does not change rewards.
*Proposed:* emit `observation_text` and `action_tokens` in
`_format_recorder_history`.

**Blocking-call helper in the training-path verifier wrapper (open).**
`_run_coro_blocking` in `service_llm_verifier_wrapper.py` uses the same
`try/except RuntimeError` pattern that was fixed in the judge client: when
called inside a running event loop, a `RuntimeError` raised by the coroutine
falls through to `asyncio.run()`, which fails with a different error and hides
the original one. Behaviour is unchanged outside a running loop.

## Fixed in this release

- **Short-circuit penalty.** Turns short-circuited by a Layer-0 heuristic now
  receive `heuristics.short_circuit_penalty` (default −0.20), which
  `JudgeRouter` passes to `compute_process_reward`. They previously received
  `-|reward.insufficient_evidence_penalty|` (−0.15), and
  `short_circuit_penalty` was ignored. Covered by
  `tests/test_judge_semantics.py`.
- **Default shaping is asymmetric.** With `symmetric_shaping: false` and
  `gate_mul: false`, the process reward is now `w_t · max(0, 2·r_raw − 1)`, as
  the `RewardConfig` docstring described: a bonus only when `r_raw > 0.5`,
  never negative. It previously computed `w_t · (2·r_raw − 1)`, identical to
  `symmetric_shaping: true`; set that flag to keep the old numbers.
  `alpha_small` is still unused by the judge and is now documented as such.
  Covered by `tests/test_judge_semantics.py`.
- **Rollout history order.** `build_packets_from_step_results` now passes the
  recorder window most recent first, matching the prompt header ("most recent
  first"), the `[t-1]` labels, and the `repetition` heuristic, which compares
  against `history[0]`. It previously passed the window oldest first, so with
  `history_k > 1` the prompt mislabelled turns. `tracerigor judge` input was
  already most recent first and is unchanged. Covered by
  `tests/test_judge_semantics.py`.
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

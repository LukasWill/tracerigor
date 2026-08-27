# Third-party notices and provenance

TraceRigor is derived from the public
[VAGEN](https://github.com/RAGEN-AI/VAGEN) project. Substantial environment,
rollout, trainer, and utility code originated there and has since been modified.
The upstream MIT copyright and permission notice remain in `LICENSE`.

The training stack integrates with VERL as an external dependency. Supported
environments can additionally require Gym/Gymnasium, gym-sokoban, ALFWorld,
ScienceWorld, BabyAI/Verlog, or ManiSkill and their associated assets. Model
providers and checkpoints are not distributed here and retain their own terms.

The following retained source files carry their own Apache License 2.0 headers
and copyright notices, which remain authoritative for those files:

- `tracerigor/trainer/main_ppo.py` and
  `tracerigor/trainer/ppo/ray_trainer.py` (ByteDance Ltd. and/or affiliates);
- `tracerigor/trainer/gigpo/core_gigpo.py` and
  `tracerigor/env/sokoban/prompt_verl_agent.py` (Nanyang Technological
  University and the verl-agent/GiGPO team).

These files have been modified for TraceRigor and carry change notices in their
headers. Redistribution remains subject to Apache License 2.0; the complete
licence text is provided in `LICENSES/Apache-2.0.txt`.

The public cleanup excludes card-image artwork whose provenance was not
established; Blackjack cards are now rendered programmatically. Raw judge-case
exports, experiment media, W&B downloads, and model outputs are also excluded
because their redistribution status or privacy properties are not uniformly
known.

This notice is a provenance record, not a legal opinion; downstream
redistributors should review the licences of the optional dependencies, data,
and models they choose to install.

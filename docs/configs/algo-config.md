# Training algorithms

The retained trainer supports ordinary outcome-level estimators and multi-turn
credit-assignment variants implemented in `tracerigor/trainer/ppo/ray_trainer.py`.
The public example selects `bi_level_gae`, which can use turn boundaries and
loss/GAE masks produced by the rollout manager.

Key Hydra settings include:

```text
algorithm.adv_estimator=bi_level_gae
algorithm.high_level_gamma=1.0
rollout_manager.use_multi_turn_reward=true
rollout_manager.use_loss_mask=true
rollout_manager.use_gae_mask=true
```

These settings affect credit assignment; they do not make environment reward a
direct supervision signal for the semantic reliability of the emitted trace.
Use the judge/verifier layer and checkpoint-conditioned analysis for that
question.

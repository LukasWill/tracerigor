# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Modified by the TraceRigor contributors, 2026.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Type, Dict, List, Any
from copy import deepcopy
from collections import defaultdict
import numpy as np
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from torch.utils.data import RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader

from tracerigor.rollout.qwen_rollout.rollout_manager import QwenVLRolloutManager
from tracerigor.rollout.qwen_rollout.rollout_manager_service import QwenVLRolloutManagerService
WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """
    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


class AdvantageEstimator(str, Enum):
    """
    Using an enumeration class to avoid spelling errors in adv_estimator
    """
    GAE = 'gae'
    MASKED_GAE = 'masked_gae'
    BI_LEVEL_GAE = 'bi_level_gae'
    TURN_WISE_GAE = 'turn_wise_gae'
    GRPO = 'grpo'
    REINFORCE_PLUS_PLUS = 'reinforce_plus_plus'
    REMAX = 'remax'
    RLOO = 'rloo'
    MULTI_TURN_GRPO = 'multi_turn_grpo'


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    Mapping
    """
    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes,
                                            use_gpu=True,
                                            max_colocate_count=1,
                                            name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]


import torch
from verl.utils.torch_functional import masked_mean


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty='kl'):
    responses = data.batch['responses']
    response_length = responses.size(1)
    token_level_scores = data.batch['token_level_scores']
    batch_size = data.batch.batch_size[0]
    attention_mask = data.batch['attention_mask']
    response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    if 'ref_log_prob' in data.batch.keys():
        kld = core_algos.kl_penalty(data.batch['old_log_probs'], data.batch['ref_log_prob'],
                                    kl_penalty=kl_penalty)  # (batch_size, response_length)
        kld = kld * response_mask
        beta = kl_ctrl.value
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch['token_level_rewards'] = token_level_rewards

    metrics = {'critic/kl': current_kl, 'critic/kl_coeff': beta}

    return data, metrics


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1,high_level_gamma=1.0,):
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == AdvantageEstimator.GAE:
        values = data.batch['values']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        token_level_rewards = data.batch['token_level_rewards']
        advantages, returns = core_algos.compute_gae_advantage_return(token_level_rewards=token_level_rewards,
                                                                    values=values,
                                                                    eos_mask=response_mask,
                                                                    gamma=gamma,
                                                                    lam=lam)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.MASKED_GAE:
        values = data.batch['values']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        token_level_rewards = data.batch['token_level_rewards']
        gae_mask = data.batch['gae_mask'][:, -response_length:]
        advantages, returns =core_algos.compute_gae_advantage_return_with_loss_mask(token_level_rewards=token_level_rewards,
                                                                values=values,
                                                                loss_mask=gae_mask,
                                                                gamma=gamma,
                                                                lam=lam)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.BI_LEVEL_GAE:
        values = data.batch['values']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        #assert "multi_turn_token_level_rewards" in data.batch.keys()
        # assert "loss_mask" in data.batch.keys()
        # loss_mask = data.batch['loss_mask'][:, -response_length:]
        if "loss_mask" in data.batch.keys():
            loss_mask = data.batch['loss_mask'][:, -response_length:]
        else:
            loss_mask=data.batch['attention_mask'][:, -response_length:]
        advantages, returns = core_algos.compute_bi_level_gae_advantage_return(token_level_rewards=data.batch['token_level_rewards'],
                                                                        values=values,
                                                                        loss_mask=loss_mask,
                                                                        gamma=gamma,
                                                                        lam=lam,
                                                                        high_level_gamma=high_level_gamma,
                                                                        reward_mask=data.batch['end_of_response_position_mask'][:, -response_length:])

        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.TURN_WISE_GAE:
        values = data.batch['values']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        #assert "multi_turn_token_level_rewards" in data.batch.keys()
        # assert "loss_mask" in data.batch.keys()
        # loss_mask = data.batch['loss_mask'][:, -response_length:]
        if "loss_mask" in data.batch.keys():
            loss_mask = data.batch['loss_mask'][:, -response_length:]
        else:
            loss_mask=data.batch['attention_mask'][:, -response_length:]
        advantages, returns = core_algos.compute_turn_wise_gae_advantage_return(token_level_rewards=data.batch['token_level_rewards'],
                                                                        values=values,
                                                                        loss_mask=loss_mask,
                                                                        reward_mask=data.batch['end_of_response_position_mask'][:, -response_length:],
                                                                        lam=lam,
                                                                        high_level_gamma=high_level_gamma,
                                                                        )

        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.GRPO:
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]



        if "loss_mask" in data.batch.keys():
            loss_mask = data.batch['loss_mask'][:, -response_length:]

            # valid_token_level_rewards_positions = token_level_rewards[0].nonzero(as_tuple=True)[0]
            # valid_loss_positions = loss_mask[0].nonzero(as_tuple=True)[0]
            # print(f"[DEBUG]valid_token_level_rewards_positions={valid_token_level_rewards_positions}")
            # print(f"[DEBUG]valid_loss_positions={valid_loss_positions}")
            # seems here only need to replace eos_mask with loss_mask
            advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=loss_mask,
                                                                        index=index)
        else:
            advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                            eos_mask=response_mask,
                                                                            index=index)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    # elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
    #     token_level_rewards = data.batch['token_level_rewards']
    #     responses = data.batch['responses']
    #     response_length = responses.size(-1)
    #     attention_mask = data.batch['attention_mask']
    #     response_mask = attention_mask[:, -response_length:]
    #     advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
    #         token_level_rewards=token_level_rewards, eos_mask=response_mask, gamma=gamma)
    #     data.batch['advantages'] = advantages
    #     data.batch['returns'] = returns
    # elif adv_estimator == AdvantageEstimator.REMAX:
    #     token_level_rewards = data.batch['token_level_rewards']
    #     index = data.non_tensor_batch['uid']
    #     responses = data.batch['responses']
    #     response_length = responses.size(-1)
    #     attention_mask = data.batch['attention_mask']
    #     response_mask = attention_mask[:, -response_length:]

    #     reward_baselines = data.batch['reward_baselines']

    #     advantages, returns = core_algos.compute_remax_outcome_advantage(token_level_rewards=token_level_rewards,
    #                                                                      reward_baselines=reward_baselines,
    #                                                                      eos_mask=response_mask)

    #     data.batch['advantages'] = advantages
    #     data.batch['returns'] = returns
    # elif adv_estimator == AdvantageEstimator.RLOO:
    #     token_level_rewards = data.batch['token_level_rewards']
    #     index = data.non_tensor_batch['uid']
    #     responses = data.batch['responses']
    #     response_length = responses.size(-1)
    #     attention_mask = data.batch['attention_mask']
    #     response_mask = attention_mask[:, -response_length:]
    #     advantages, returns = core_algos.compute_rloo_outcome_advantage(token_level_rewards=token_level_rewards,
    #                                                                     eos_mask=response_mask,
    #                                                                     index=index)
    #     data.batch['advantages'] = advantages
    #     data.batch['returns'] = returns

    else:
        raise NotImplementedError

    # --- NEW (Option B): aux LM-like boost on <think> tokens ---
    # If config.algorithm.think_aux_advantage > 0 and think_mask is present,
    # we add a constant positive advantage on those tokens. This approximates
    # an LM loss on the corrected reasoning tokens within the PPO framework.
    think_aux = 0.0
    #if config is not None:
    #    # AlgoConfig is OmegaConf-like, so .get works
    #    think_aux = float(config.get("think_aux_advantage", 0.0))
    if think_aux > 0.0 and "think_mask" in data.batch:
        # advantages has shape [B, T_resp]; think_mask is [B, T_resp]
        adv = data.batch["advantages"]
        think_mask = data.batch["think_mask"].to(adv.dtype)
        data.batch["advantages"] = adv + think_aux * think_mask
    # --- END NEW ---
    return data


def reduce_metrics(metrics: dict):
    for key, val in metrics.items():
        metrics[key] = np.mean(val)
    return metrics


def _compute_response_info(batch):
    if "loss_mask" in batch.batch.keys():
        # end_of_response_position_mask=batch.batch["end_of_response_position_mask"]
        response_length = batch.batch['loss_mask'].sum(-1).float()
        prompt_length = (batch.batch['attention_mask'].sum(-1)-batch.batch['loss_mask'].sum(-1)).float()
        response_part_length = batch.batch['responses'].shape[-1]
        response_mask = batch.batch['loss_mask'][:, -response_part_length:]
    else:
        response_length = batch.batch['responses'].shape[-1]

        prompt_mask = batch.batch['attention_mask'][:, :-response_length]
        response_mask = batch.batch['attention_mask'][:, -response_length:]

        prompt_length = prompt_mask.sum(-1).float()
        response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


def compute_data_metrics(batch, use_critic=True):
    # TODO: add response length
    sequence_score = batch.batch['token_level_scores'].sum(-1)
    sequence_reward = batch.batch['token_level_rewards'].sum(-1)

    advantages = batch.batch['advantages']
    returns = batch.batch['returns']

    max_response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-max_response_length].bool()
    response_mask = batch.batch['attention_mask'][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)


    response_info = _compute_response_info(batch)
    prompt_length = response_info['prompt_length']
    response_length = response_info['response_length']

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch['values']
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    metrics = {
        # score
        'critic/score/mean':
            torch.mean(sequence_score).detach().item(),
        'critic/score/max':
            torch.max(sequence_score).detach().item(),
        'critic/score/min':
            torch.min(sequence_score).detach().item(),
        # reward
        'critic/rewards/mean':
            torch.mean(sequence_reward).detach().item(),
        'critic/rewards/max':
            torch.max(sequence_reward).detach().item(),
        'critic/rewards/min':
            torch.min(sequence_reward).detach().item(),
        # adv
        'critic/advantages/mean':
            torch.mean(valid_adv).detach().item(),
        'critic/advantages/max':
            torch.max(valid_adv).detach().item(),
        'critic/advantages/min':
            torch.min(valid_adv).detach().item(),
        # returns
        'critic/returns/mean':
            torch.mean(valid_returns).detach().item(),
        'critic/returns/max':
            torch.max(valid_returns).detach().item(),
        'critic/returns/min':
            torch.min(valid_returns).detach().item(),
        **({
            # values
            'critic/values/mean': torch.mean(valid_values).detach().item(),
            'critic/values/max': torch.max(valid_values).detach().item(),
            'critic/values/min': torch.min(valid_values).detach().item(),
            # vf explained var
            'critic/vf_explained_var': (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
        } if use_critic else {}),

        # response length
        'response_length/mean':
            torch.mean(response_length).detach().item(),
        'response_length/max':
            torch.max(response_length).detach().item(),
        'response_length/min':
            torch.min(response_length).detach().item(),
        'response_length/clip_ratio':
            torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        # prompt length
        'prompt_length/mean':
            torch.mean(prompt_length).detach().item(),
        'prompt_length/max':
            torch.max(prompt_length).detach().item(),
        'prompt_length/min':
            torch.min(prompt_length).detach().item(),
        'prompt_length/clip_ratio':
            torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }
    return metrics


def compute_timing_metrics(batch, timing_raw):
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info['prompt_length']).item()
    num_response_tokens = torch.sum(response_info['response_length']).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        'gen': num_response_tokens,
        **{
            name: num_overall_tokens for name in ['ref', 'values', 'adv', 'update_critic', 'update_actor']
        },
    }

    return {
        **{
            f'timing_s/{name}': value for name, value in timing_raw.items()
        },
        **{
            f'timing_per_token_ms/{name}': timing_raw[name] * 1000 / num_tokens_of_section[name] for name in set(num_tokens_of_section.keys(
            )) & set(timing_raw.keys())
        },
    }


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timer.last


class RayPPOTrainer(object):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 processor=None,
                 reward_fn=None,
                 val_reward_fn=None):

        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if self.use_reference_policy:
            if config.algorithm.kl_ctrl.type == 'fixed':
                self.kl_ctrl = core_algos.FixedKLController(kl_coef=config.algorithm.kl_ctrl.kl_coef)
            elif config.algorithm.kl_ctrl.type == 'adaptive':
                assert config.algorithm.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
                self.kl_ctrl = core_algos.AdaptiveKLController(init_kl_coef=config.algorithm.kl_ctrl.kl_coef,
                                                               target_kl=config.algorithm.kl_ctrl.target_kl,
                                                               horizon=config.algorithm.kl_ctrl.horizon)
            else:
                raise NotImplementedError
        else:
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=0.)

        if self.config.algorithm.adv_estimator in [AdvantageEstimator.GAE, AdvantageEstimator.BI_LEVEL_GAE,
                                                   AdvantageEstimator.MASKED_GAE,AdvantageEstimator.TURN_WISE_GAE]:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
                AdvantageEstimator.GRPO, AdvantageEstimator.REINFORCE_PLUS_PLUS, AdvantageEstimator.REMAX,
                AdvantageEstimator.RLOO, AdvantageEstimator.MULTI_TURN_GRPO
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError

        self._validate_config()
        self._create_dataloader()
        self.test_rollout_config=None
        self.test_rollout_manager=None


    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % n_gpus == 0, \
            f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus})."

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            if mbs is None and mbs_per_gpu is None:
                raise ValueError(f"[{name}] Please set at least one of '{name}.micro_batch_size' or "
                                 f"'{name}.micro_batch_size_per_gpu'.")

            if mbs is not None and mbs_per_gpu is not None:
                raise ValueError(f"[{name}] You have set both '{name}.micro_batch_size' AND "
                                 f"'{name}.micro_batch_size_per_gpu'. Please remove '{name}.micro_batch_size' "
                                 f"because only '*_micro_batch_size_per_gpu' is supported (the former is deprecated).")

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.actor.ppo_micro_batch_size,
                                     config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.actor")

            # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                                     config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.ref")

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                                     config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.rollout")

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu,
                                     "critic")

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu,
                                     "reward_model")

        # Actor
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            sp_size = config.actor_rollout_ref.actor.get('ulysses_sequence_parallel_size', 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size == 0
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            sp_size = config.critic.get('ulysses_sequence_parallel_size', 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus
            if config.algorithm.adv_estimator == AdvantageEstimator.TURN_WISE_GAE:
                assert config.critic.get('use_reward_mask', False), \
                    "TURN_WISE_GAE needs reward mask"

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == 'fsdp':
            if config.actor_rollout_ref.actor.get('ulysses_sequence_parallel_size', 1) > 1 or \
                    config.actor_rollout_ref.ref.get('ulysses_sequence_parallel_size', 1) > 1:
                assert config.actor_rollout_ref.model.use_remove_padding, \
                    "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."

        if self.use_critic and config.critic.strategy == 'fsdp':
            if config.critic.get('ulysses_sequence_parallel_size', 1) > 1:
                assert config.critic.model.use_remove_padding, \
                    "When using sequence parallelism for critic, you must enable `use_remove_padding`."

        if config.data.get('val_batch_size', None) is not None:
            print(
                f"WARNING: val_batch_size is deprecated. Validation datasets are sent to inference engines as a whole batch, which will schedule the memory themselves."
            )

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self):
        # TODO: we have to make sure the batch size is divisible by the dp size
        self.train_dataset = RLHFDataset(parquet_files=self.config.data.train_files,
                                         tokenizer=self.tokenizer,
                                         processor=self.processor,
                                         prompt_key=self.config.data.prompt_key,
                                         image_key=self.config.data.get('image_key', 'images'),
                                         max_prompt_length=self.config.data.max_prompt_length,
                                         filter_prompts=True,
                                         return_raw_chat=self.config.data.get('return_raw_chat', False),
                                         truncation='error')
        # use sampler for better ckpt resume
        if self.config.data.shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(self.config.data.get('seed', 1))
            sampler = RandomSampler(data_source=self.train_dataset, generator=train_dataloader_generator)
        else:
            sampler = SequentialSampler(data_source=self.train_dataset)

        self.train_dataloader = StatefulDataLoader(dataset=self.train_dataset,
                                                   batch_size=self.config.data.train_batch_size,
                                                   num_workers=8,
                                                   drop_last=True,
                                                   collate_fn=collate_fn,
                                                   sampler=sampler)

        self.val_dataset = RLHFDataset(parquet_files=self.config.data.val_files,
                                       tokenizer=self.tokenizer,
                                       processor=self.processor,
                                       prompt_key=self.config.data.prompt_key,
                                       image_key=self.config.data.get('image_key', 'images'),
                                       max_prompt_length=self.config.data.max_prompt_length,
                                       filter_prompts=True,
                                       return_raw_chat=self.config.data.get('return_raw_chat', False),
                                       truncation='error')
        if self.config.data.val_batch_size is None:
            val_batch_size=len(self.val_dataset)
        else:
            val_batch_size=self.config.data.val_batch_size
        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            # Validation datasets are sent to inference engines as a whole batch,
            # which will schedule the memory themselves.
            batch_size=val_batch_size,
            num_workers=8,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn)

        assert len(self.train_dataloader) >= 1
        # assert len(
        #     self.val_dataloader
        # ) == 1, "Validation dataloader must have a single batch, which inference engines will schedule the memory themselves." # for agent training we still use val batch size

        print(f'Size of train dataloader: {len(self.train_dataloader)}')
        print(f"Size of val dataloader: {len(self.val_dataloader)}")

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

    def _maybe_log_val_generations_to_wandb(self, log_rst: List[Dict[str, Any]]):
        """Log a table of validation samples with multiple images per sample to wandb"""

        generations_to_log = self.config.trainer.val_generations_to_log_to_wandb

        if generations_to_log == 0:
            return

        if generations_to_log > 0 and 'wandb' not in self.config.trainer.logger:
            print('WARNING: `val_generations_to_log_to_wandb` is set to a positive value, but no wandb logger is found. ')
            return

        import wandb
        import numpy as np
        import json
        inputs=[]
        outputs=[]
        scores=[]
        images=[]
        # --- NEW: holders ---
        rewards_per_turn, verifier_scores_per_turn, think_len_per_turn = [], [], []

        for item in log_rst:
            inputs.append(item['config_id'])
            outputs.append(item['output_str'])
            scores.append(item['metrics']['score'])
            images.append(item['image_data'])
            rewards_per_turn.append(item.get('turn_rewards', []))
            verifier_scores_per_turn.append(item.get('turn_verifier_scores', {}))
            think_len_per_turn.append(item.get('turn_reason_len', []))

        ## right before building max_images_per_sample
        #for idx, img_list in enumerate(images):
        #    n = (len(img_list) if isinstance(img_list, (list, tuple)) else (0 if img_list is None else 1))
        #    print(f"[VALLOG] sample={idx} image_count={n}")

        # Handle the case where images might not be provided
        if images is None:
            samples = list(zip(inputs, outputs, scores))
            has_images = False
        else:
            # Here, images is expected to be a list of lists, where each inner list contains images for one sample
            samples = list(zip(inputs, outputs, scores, images,
                           rewards_per_turn, verifier_scores_per_turn, think_len_per_turn))
            has_images = True

        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState()
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Compute max_images_per_sample from *subsetted* samples only, so
        # the column count is stable regardless of the full batch content.
        if has_images:
            # Use a constant derived from max_turns so the table schema never
            # changes between validation steps.  Each trajectory produces at
            # most (max_turns + 1) images: 1 initial observation + max_turns
            # step observations.  Trajectories that finish early are None-padded.
            max_images_per_sample = self.config.rollout_manager.get('max_turns', 5) + 1
        else:
            max_images_per_sample = 0

        # Create column names for all samples
        if has_images:
            columns = ["step"]
            for i in range(len(samples)):
                columns.extend([f"input_{i+1}", f"output_{i+1}", f"score_{i+1}"])
                # --- NEW columns (store raw lists/dict) ---
                columns.extend([f"turn_rewards_{i+1}",
                                f"turn_verifier_scores_{i+1}",
                                f"turn_reason_len_{i+1}"])
                columns.extend([f"image_{i+1}_{j+1}" for j in range(max_images_per_sample)])
        else:
            columns = ["step"] + sum([[f"input_{i+1}", f"output_{i+1}", f"score_{i+1}"] for i in range(len(samples))], [])

        if not hasattr(self, 'validation_table'):
            # Initialize the table on first call
            self.validation_table = wandb.Table(columns=columns)

        # Create a new table with same columns and existing data
        new_table = wandb.Table(columns=columns, data=self.validation_table.data)

        # Add new row with all data
        row_data = []
        row_data.append(self.global_steps)

        for sample in samples:
            if has_images:
                (input_text, output_text, score, sample_images,
                sample_turn_rewards, sample_verifier_scores, sample_think_len) = sample
                row_data.extend([input_text, output_text, score])
                row_data.append(json.dumps(sample_turn_rewards or []))
                row_data.append(json.dumps(sample_verifier_scores or {}))
                row_data.append(json.dumps(sample_think_len or []))

                # Handle if sample_images is a single image or list of images
                if not isinstance(sample_images, (list, tuple)):
                    sample_images = [sample_images]

                # Convert each image to wandb.Image
                wandb_images = []
                for img in sample_images:
                    if not isinstance(img, wandb.Image):
                        img = wandb.Image(img)
                    wandb_images.append(img)

                # Pad with None if there are fewer images than max_images_per_sample
                wandb_images.extend([None] * (max_images_per_sample - len(wandb_images)))
                row_data.extend(wandb_images)
            else:
                row_data.extend(sample)

        new_table.add_data(*row_data)

        # Update reference and log
        wandb.log({"val/generations": new_table}, step=self.global_steps)
        self.validation_table = new_table

    def _save_val_generations_to_local(self, log_rst: List[Dict[str, Any]]):
        """
        Save ALL validation samples to a local folder for later multi-turn CoT analysis.

        Creates a folder structure:
            val_generations/{experiment_name}/step_{global_steps}/
                samples.jsonl  - all text data (one JSON per line)
                images/        - folder containing images (if multimodal)
                                 (may be a symlink to an external storage path)

        Each sample in samples.jsonl contains:
            - env_id, config_id, output_str, score, done, step
            - turn_rewards, turn_verifier_scores, turn_reason_len
            - image_paths (list of relative paths to images if present)
            - all other metrics

        When ``trainer.val_images_external_dir`` is set (for example, a shared
        high-capacity path), images are written there and a symlink is
        created at the local ``images/`` path so that ``samples.jsonl`` references
        remain valid.  This avoids filling the user-node home directory with
        large image data while keeping the metadata locally accessible.
        """
        import json
        import os
        from pathlib import Path

        # Get experiment name from config
        experiment_name = self.config.trainer.get('experiment_name', 'default_experiment')
        project_name = self.config.trainer.get('project_name', 'tracerigor')

        # Create output directory
        base_dir = Path(f"val_generations/{project_name}/{experiment_name}")
        step_dir = base_dir / f"step_{self.global_steps}"
        local_images_dir = step_dir / "images"

        step_dir.mkdir(parents=True, exist_ok=True)

        # Check if any sample has images
        has_any_images = any(
            item.get('image_data') is not None and
            (isinstance(item.get('image_data'), (list, tuple)) and len(item.get('image_data')) > 0 or
             not isinstance(item.get('image_data'), (list, tuple)) and item.get('image_data') is not None)
            for item in log_rst
        )

        # Determine where to write images.  If an external dir is configured
        # and this run has images, write there and symlink locally.
        external_base = self.config.trainer.get('val_images_external_dir', None)
        if has_any_images and external_base:
            external_images_dir = Path(external_base) / project_name / experiment_name / f"step_{self.global_steps}" / "images"
            external_images_dir.mkdir(parents=True, exist_ok=True)
            images_dir = external_images_dir
            # Create a local symlink so relative paths in samples.jsonl still work
            if local_images_dir.is_symlink() or local_images_dir.exists():
                # Remove stale symlink or dir from a previous (interrupted) run
                if local_images_dir.is_symlink():
                    local_images_dir.unlink()
                # (If it's a real dir, leave it — the user may have manually
                # moved data back; we just won't overwrite.)
            if not local_images_dir.exists():
                try:
                    local_images_dir.symlink_to(external_images_dir)
                except OSError as e:
                    print(f"[WARNING] Could not create symlink {local_images_dir} -> {external_images_dir}: {e}")
                    # Fall back to writing locally
                    local_images_dir.mkdir(parents=True, exist_ok=True)
                    images_dir = local_images_dir
        elif has_any_images:
            local_images_dir.mkdir(parents=True, exist_ok=True)
            images_dir = local_images_dir
        else:
            images_dir = local_images_dir  # won't actually be used

        samples_path = step_dir / "samples.jsonl"

        with open(samples_path, 'w', encoding='utf-8') as f:
            for idx, item in enumerate(log_rst):
                sample_data = {
                    "sample_idx": idx,
                    "env_id": item.get('env_id', f'env_{idx}'),
                    "config_id": item.get('config_id', ''),
                    "output_str": item.get('output_str', ''),
                    "metrics": item.get('metrics', {}),
                    "turn_rewards": item.get('turn_rewards', []),
                    "turn_verifier_scores": item.get('turn_verifier_scores', {}),
                    "turn_reason_len": item.get('turn_reason_len', []),
                    "image_paths": [],
                }

                # Handle images
                image_data = item.get('image_data')
                if image_data is not None:
                    # Normalize to list
                    if not isinstance(image_data, (list, tuple)):
                        image_data = [image_data]

                    image_paths = []
                    for img_idx, img in enumerate(image_data):
                        if img is None:
                            continue

                        # Generate image filename
                        img_filename = f"sample_{idx}_img_{img_idx}.png"
                        img_path = images_dir / img_filename

                        try:
                            # Handle different image types
                            from PIL import Image
                            import numpy as np

                            if hasattr(img, 'save'):
                                # PIL Image
                                img.save(img_path)
                            elif isinstance(img, np.ndarray):
                                # NumPy array
                                Image.fromarray(img).save(img_path)
                            elif hasattr(img, 'image'):
                                # wandb.Image object
                                if hasattr(img.image, 'save'):
                                    img.image.save(img_path)
                            else:
                                # Try converting to PIL Image
                                Image.fromarray(np.array(img)).save(img_path)

                            # Store relative path
                            image_paths.append(f"images/{img_filename}")
                        except Exception as e:
                            print(f"[WARNING] Failed to save image {img_idx} for sample {idx}: {e}")

                    sample_data["image_paths"] = image_paths

                # Write as JSON line
                f.write(json.dumps(sample_data, ensure_ascii=False) + '\n')

        # Also save a summary metadata file
        from datetime import datetime
        metadata = {
            "global_step": self.global_steps,
            "experiment_name": experiment_name,
            "project_name": project_name,
            "num_samples": len(log_rst),
            "has_images": has_any_images,
            "timestamp": datetime.now().isoformat(),
        }
        if has_any_images and external_base:
            metadata["images_external_dir"] = str(images_dir)

        metadata_path = step_dir / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"[INFO] Saved {len(log_rst)} validation samples to {step_dir}")


    def _compute_and_save_entropy(self, validation_rst, traj_batches):
        """
        After validation rollout, run a forward pass to compute per-token entropy
        for each trajectory, then segment by turns and save to disk.

        Args:
            validation_rst: list of per-trajectory log dicts from recording_to_log()
            traj_batches: list of DataProto batches from generate_batch_for_update(),
                          one per val_dataloader micro-batch

        Saved artifacts per step:
            entropy/entropy_summary.jsonl  - per-trajectory aggregated entropy metrics
            entropy/token_entropies.npz    - raw per-token entropy arrays
        """
        import json
        import numpy as np
        from pathlib import Path

        try:
            entropy_summaries = []
            token_entropy_dict = {}
            token_logprob_dict = {}
            rst_offset = 0  # track position in validation_rst across batches

            for batch_idx, traj_batch in enumerate(traj_batches):
                batch_size = traj_batch.batch.batch_size[0]
                print(f"[ENTROPY] Running forward pass for batch {batch_idx+1}/{len(traj_batches)} "
                      f"({batch_size} trajectories)...") # 128

                # Run forward pass to get entropy and log_probs
                log_prob_result = self.actor_rollout_wg.compute_log_prob(traj_batch)

                if 'entropys' not in log_prob_result.batch.keys():
                    print("[ENTROPY] WARNING: entropys not found in compute_log_prob output. Skipping batch.")
                    rst_offset += batch_size
                    continue

                entropys = log_prob_result.batch['entropys']  # (bs, response_length)
                # Token probability mass: log P(chosen token | context)
                logprobs = log_prob_result.batch['old_log_probs']  # (bs, response_length)

                # Get masks for segmentation
                response_length = traj_batch.batch['responses'].shape[-1]
                if 'loss_mask' in traj_batch.batch.keys():
                    loss_mask = traj_batch.batch['loss_mask'][:, -response_length:]
                else:
                    loss_mask = traj_batch.batch['attention_mask'][:, -response_length:]

                # end_of_response_position_mask marks the end of each turn's response
                if 'end_of_response_position_mask' in traj_batch.batch.keys():
                    eor_mask = traj_batch.batch['end_of_response_position_mask'][:, -response_length:]
                else:
                    eor_mask = None

                # think_mask if available (marks <think>...</think> tokens)
                if 'think_mask' in traj_batch.batch.keys():
                    think_mask = traj_batch.batch['think_mask']  # (bs, response_length)
                else:
                    think_mask = None

                for i in range(batch_size):
                    global_idx = rst_offset + i
                    ent_i = entropys[i]           # (response_length,)
                    lp_i = logprobs[i]            # (response_length,)
                    mask_i = loss_mask[i].float()  # (response_length,)

                    # Get truncation offset for correct turn numbering
                    turn_offset = 0
                    if 'truncated_turns' in traj_batch.batch.keys():
                        turn_offset = int(traj_batch.batch['truncated_turns'][i].item())

                    # Get env_id from the batch itself (robust to dict-order mismatches)
                    if 'env_id' in traj_batch.non_tensor_batch:
                        env_id = str(traj_batch.non_tensor_batch['env_id'][i])
                        config_id = str(traj_batch.non_tensor_batch['config_id'][i]) if 'config_id' in traj_batch.non_tensor_batch else ''
                    elif global_idx < len(validation_rst):
                        env_id = validation_rst[global_idx].get('env_id', f'env_{global_idx}')
                        config_id = validation_rst[global_idx].get('config_id', '')
                    else:
                        env_id = f'env_{global_idx}'
                        config_id = ''

                    # Overall trajectory entropy + probability mass (masked)
                    valid_mask = mask_i.bool()
                    valid_ent = ent_i[valid_mask]
                    valid_lp = lp_i[valid_mask]

                    if valid_ent.numel() == 0:
                        continue

                    valid_prob = valid_lp.exp()  # P(chosen token)
                    summary = {
                        'env_id': env_id,
                        'config_id': config_id,
                        'truncated_turns': turn_offset,  # num early turns removed by left truncation
                        # Entropy statistics
                        'traj_mean_entropy': valid_ent.mean().item(),
                        'traj_std_entropy': valid_ent.std().item() if valid_ent.numel() > 1 else 0.0,
                        'traj_max_entropy': valid_ent.max().item(),
                        'traj_min_entropy': valid_ent.min().item(),
                        'traj_median_entropy': valid_ent.median().item(),
                        'num_valid_tokens': valid_ent.numel(),
                        # Token probability mass (P_base in entropy-vs-prob literature)
                        'traj_mean_logprob': valid_lp.mean().item(),
                        'traj_mean_prob': valid_prob.mean().item(),
                        'traj_median_prob': valid_prob.median().item(),
                        'traj_max_prob': valid_prob.max().item(),
                        'traj_min_prob': valid_prob.min().item(),
                    }

                    # Think-only vs action-only entropy + probability mass
                    if think_mask is not None:
                        think_i = think_mask[i].bool() & valid_mask
                        action_i = (~think_mask[i].bool()) & valid_mask

                        if think_i.any():
                            think_ent = ent_i[think_i]
                            think_lp = lp_i[think_i]
                            summary['think_mean_entropy'] = think_ent.mean().item()
                            summary['think_std_entropy'] = think_ent.std().item() if think_ent.numel() > 1 else 0.0
                            summary['think_mean_logprob'] = think_lp.mean().item()
                            summary['think_mean_prob'] = think_lp.exp().mean().item()
                            summary['num_think_tokens'] = think_ent.numel()

                        if action_i.any():
                            action_ent = ent_i[action_i]
                            action_lp = lp_i[action_i]
                            summary['action_mean_entropy'] = action_ent.mean().item()
                            summary['action_std_entropy'] = action_ent.std().item() if action_ent.numel() > 1 else 0.0
                            summary['action_mean_logprob'] = action_lp.mean().item()
                            summary['action_mean_prob'] = action_lp.exp().mean().item()
                            summary['num_action_tokens'] = action_ent.numel()

                    # Per-turn entropy (segment using end_of_response_position_mask)
                    if eor_mask is not None:
                        eor_positions = eor_mask[i].nonzero(as_tuple=True)[0]
                        turn_entropies = []
                        prev_pos = 0

                        for turn_idx, eor_pos in enumerate(eor_positions):
                            eor_pos_val = eor_pos.item()
                            turn_mask = mask_i[prev_pos:eor_pos_val + 1]
                            turn_ent = ent_i[prev_pos:eor_pos_val + 1]
                            turn_valid = turn_mask.bool()

                            if turn_valid.any():
                                t_ent = turn_ent[turn_valid]
                                t_lp = lp_i[prev_pos:eor_pos_val + 1][turn_valid]
                                turn_entropies.append({
                                    'turn': turn_idx + turn_offset,  # actual turn in full trajectory
                                    'turn_local': turn_idx,  # turn index in truncated view
                                    'mean_entropy': t_ent.mean().item(),
                                    'std_entropy': t_ent.std().item() if t_ent.numel() > 1 else 0.0,
                                    'mean_logprob': t_lp.mean().item(),
                                    'mean_prob': t_lp.exp().mean().item(),
                                    'num_tokens': t_ent.numel(),
                                })

                            prev_pos = eor_pos_val + 1

                        summary['turn_entropies'] = turn_entropies

                        # Entropy trend: slope of mean entropy across turns
                        if len(turn_entropies) >= 2:
                            means = [t['mean_entropy'] for t in turn_entropies]
                            x = np.arange(len(means), dtype=np.float64)
                            y = np.array(means, dtype=np.float64)
                            if np.std(x) > 0:
                                slope = float(np.corrcoef(x, y)[0, 1] * np.std(y) / np.std(x))
                            else:
                                slope = 0.0
                            summary['entropy_trend_slope'] = slope

                    entropy_summaries.append(summary)
                    token_entropy_dict[str(env_id)] = valid_ent.cpu().numpy().astype(np.float16)
                    token_logprob_dict[str(env_id)] = valid_lp.cpu().numpy().astype(np.float16)

                rst_offset += batch_size

            # Save to disk
            experiment_name = self.config.trainer.get('experiment_name', 'default_experiment')
            project_name = self.config.trainer.get('project_name', 'tracerigor')
            base_dir = Path(f"val_generations/{project_name}/{experiment_name}")
            entropy_dir = base_dir / f"step_{self.global_steps}" / "entropy"
            entropy_dir.mkdir(parents=True, exist_ok=True)

            # Save summary JSONL
            summary_path = entropy_dir / "entropy_summary.jsonl"
            with open(summary_path, 'w', encoding='utf-8') as f:
                for s in entropy_summaries:
                    f.write(json.dumps(s, ensure_ascii=False) + '\n')

            # Save raw token entropies as .npz (compact: float16)
            npz_path = entropy_dir / "token_entropies.npz"
            np.savez_compressed(str(npz_path), **token_entropy_dict)

            # Save raw token log-probabilities as .npz (compact: float16)
            logprob_path = entropy_dir / "token_logprobs.npz"
            np.savez_compressed(str(logprob_path), **token_logprob_dict)

            # Log aggregate entropy + prob mass metrics
            if entropy_summaries:
                all_means = [s['traj_mean_entropy'] for s in entropy_summaries]
                all_probs = [s['traj_mean_prob'] for s in entropy_summaries]
                print(f"[ENTROPY] Step {self.global_steps}: "
                      f"entropy_mean={np.mean(all_means):.4f}, "
                      f"entropy_std={np.std(all_means):.4f}, "
                      f"prob_mass_mean={np.mean(all_probs):.4f}, "
                      f"n={len(entropy_summaries)} trajectories")
                print(f"[ENTROPY] Saved to {entropy_dir}")

        except Exception as e:
            print(f"[ENTROPY] ERROR during entropy computation: {e}")
            import traceback
            traceback.print_exc()
            print("[ENTROPY] Continuing without entropy data.")

    def _compute_and_save_attention_mass(self, validation_rst, traj_batches):
        """
        After validation rollout, run a probe forward pass on a subset of
        trajectories to compute cross-segment attention mass for mechanistic
        analysis.

        For LLM:  Segments P (Prompt), R (Reasoning), A (Action)       → 3×3
        For VLM:  Segments P, R, A, V (Vision)                         → 4×4

        R→V attention mass tracks how much reasoning tokens attend to
        vision tokens (inspired by Frankenstein paper, arXiv 2602.12395).

        Saved artifacts per step:
            attention/attention_mass.jsonl  — per-trajectory per-layer N×N mass
        """
        import json
        import numpy as np
        from pathlib import Path

        try:
            max_probe_samples = self.config.trainer.get('attn_probe_samples', 128)
            layer_stride = self.config.trainer.get('attn_layer_stride', 4)
            chunk_size = self.config.trainer.get('attn_chunk_size', 128)

            # Collect samples across batches up to max_probe_samples
            probe_records = []   # list of (traj_batch_idx, in_batch_idx, global_idx)
            global_offset = 0
            for batch_idx, traj_batch in enumerate(traj_batches):
                bs = traj_batch.batch.batch_size[0]
                for j in range(bs):
                    if len(probe_records) >= max_probe_samples:
                        break
                    probe_records.append((batch_idx, j, global_offset + j))
                global_offset += bs
                if len(probe_records) >= max_probe_samples:
                    break

            if not probe_records:
                return

            print(f"[ATTN_PROBE] Probing {len(probe_records)} trajectories "
                  f"(layer_stride={layer_stride}, chunk_size={chunk_size})")

            # Group probe_records by batch for efficient processing
            from collections import defaultdict
            records_by_batch = defaultdict(list)
            for rec in probe_records:
                records_by_batch[rec[0]].append(rec)

            all_summaries = []

            for batch_idx, records in records_by_batch.items():
                traj_batch = traj_batches[batch_idx]
                in_batch_indices = [r[1] for r in records]

                # Select the subset from this batch
                # Build a sub-DataProto with only the selected samples
                select_keys = ['input_ids', 'attention_mask', 'position_ids', 'responses']
                if 'loss_mask' in traj_batch.batch.keys():
                    select_keys.append('loss_mask')
                if 'think_mask' in traj_batch.batch.keys():
                    select_keys.append('think_mask')
                if 'end_of_response_position_mask' in traj_batch.batch.keys():
                    select_keys.append('end_of_response_position_mask')
                if 'truncated_turns' in traj_batch.batch.keys():
                    select_keys.append('truncated_turns')

                subset_tensors = {}
                for key in select_keys:
                    subset_tensors[key] = traj_batch.batch[key][in_batch_indices]

                subset_non_tensor = {}
                if 'multi_modal_inputs' in traj_batch.non_tensor_batch.keys():
                    subset_non_tensor['multi_modal_inputs'] = [
                        traj_batch.non_tensor_batch['multi_modal_inputs'][i]
                        for i in in_batch_indices
                    ]

                from verl import DataProto
                subset_data = DataProto.from_dict(
                    tensors=subset_tensors,
                    non_tensors=subset_non_tensor,
                )
                subset_data.meta_info['micro_batch_size'] = self.config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu
                subset_data.meta_info['temperature'] = self.config.actor_rollout_ref.rollout.temperature
                subset_data.meta_info['attn_layer_stride'] = layer_stride
                subset_data.meta_info['attn_chunk_size'] = chunk_size
                subset_data.meta_info['attn_per_turn'] = True  # enable per-turn mass

                # Dispatch to GPU workers
                attn_result = self.actor_rollout_wg.compute_attention_mass(subset_data)

                if attn_result is None or attn_result.batch['attention_mass'].numel() == 0:
                    continue

                am_flat = attn_result.batch['attention_mass']  # (n, num_layers*N*N)
                probed_layers = attn_result.meta_info.get('probed_layers', [])
                attn_shape = attn_result.meta_info.get('attn_shape', [])

                if attn_shape:
                    attention_mass = am_flat.reshape(-1, *attn_shape)  # (n, num_layers, N, N)
                else:
                    continue

                segment_names = attn_result.meta_info.get('segment_names', ['P', 'R', 'A'])
                seg_idx = {name: i for i, name in enumerate(segment_names)}
                per_turn_mass_all = attn_result.meta_info.get('per_turn_mass', None)

                for local_idx, (_, in_batch_idx, global_idx) in enumerate(records):
                    if local_idx >= attention_mass.shape[0]:
                        break

                    # Get env_id from the batch itself (robust to dict-order mismatches)
                    if 'env_id' in traj_batch.non_tensor_batch:
                        env_id = str(traj_batch.non_tensor_batch['env_id'][in_batch_idx])
                        config_id = str(traj_batch.non_tensor_batch['config_id'][in_batch_idx]) if 'config_id' in traj_batch.non_tensor_batch else ''
                    elif global_idx < len(validation_rst):
                        env_id = validation_rst[global_idx].get('env_id', f'env_{global_idx}')
                        config_id = validation_rst[global_idx].get('config_id', '')
                    else:
                        env_id = f'env_{global_idx}'
                        config_id = ''

                    # Get truncation offset for correct turn numbering
                    turn_offset = 0
                    if 'truncated_turns' in traj_batch.batch.keys():
                        turn_offset = int(traj_batch.batch['truncated_turns'][in_batch_idx].item())

                    mass_i = attention_mass[local_idx]  # (num_layers, N, N)

                    entry = {
                        'env_id': env_id,
                        'config_id': config_id,
                        'probed_layers': probed_layers,
                        'segment_names': segment_names,
                        'truncated_turns': turn_offset,
                        'mass_by_layer': {},
                    }
                    for layer_pos, layer_idx in enumerate(probed_layers):
                        entry['mass_by_layer'][str(layer_idx)] = mass_i[layer_pos].tolist()

                    # Key derived metrics (using segment index lookup)
                    avg_mass = mass_i.mean(dim=0)  # (N, N) averaged over layers
                    entry['avg_mass'] = avg_mass.tolist()
                    entry['A_to_R'] = avg_mass[seg_idx['A'], seg_idx['R']].item()
                    entry['R_to_P'] = avg_mass[seg_idx['R'], seg_idx['P']].item()
                    entry['A_to_P'] = avg_mass[seg_idx['A'], seg_idx['P']].item()
                    entry['R_to_R'] = avg_mass[seg_idx['R'], seg_idx['R']].item()

                    # Vision-specific metrics (Frankenstein paper: R→V attention)
                    if 'V' in seg_idx:
                        entry['R_to_V'] = avg_mass[seg_idx['R'], seg_idx['V']].item()
                        entry['A_to_V'] = avg_mass[seg_idx['A'], seg_idx['V']].item()
                        entry['P_to_V'] = avg_mass[seg_idx['P'], seg_idx['V']].item()

                    # Per-turn attention mass
                    if per_turn_mass_all and local_idx < len(per_turn_mass_all):
                        turn_masses = per_turn_mass_all[local_idx]
                        turn_entries = []
                        for td in turn_masses:
                            # td['mass'] is (num_layers, 2, N) as nested list
                            # rows: [R_t, A_t], cols: target segments [P, R, A, (V)]
                            import torch
                            mass_t = torch.tensor(td['mass'])  # (num_layers, 2, N)
                            avg_t = mass_t.mean(dim=0)  # (2, N) averaged over layers
                            te = {
                                'turn': td['turn'] + turn_offset,  # actual turn in trajectory
                                'turn_local': td['turn'],  # turn index in truncated view
                                'r_count': td['r_count'],
                                'a_count': td['a_count'],
                            }
                            # R_t → target segments
                            for ti, tname in enumerate(segment_names):
                                te[f'R_to_{tname}'] = avg_t[0, ti].item()
                            # A_t → target segments
                            for ti, tname in enumerate(segment_names):
                                te[f'A_to_{tname}'] = avg_t[1, ti].item()
                            turn_entries.append(te)
                        entry['turn_masses'] = turn_entries

                    all_summaries.append(entry)

            # Save to disk
            if all_summaries:
                experiment_name = self.config.trainer.get('experiment_name', 'default_experiment')
                project_name = self.config.trainer.get('project_name', 'tracerigor')
                base_dir = Path(f"val_generations/{project_name}/{experiment_name}")
                attn_dir = base_dir / f"step_{self.global_steps}" / "attention"
                attn_dir.mkdir(parents=True, exist_ok=True)

                attn_path = attn_dir / "attention_mass.jsonl"
                with open(attn_path, 'w', encoding='utf-8') as f:
                    for entry in all_summaries:
                        f.write(json.dumps(entry, ensure_ascii=False) + '\n')

                # Print summary
                avg_a2r = np.mean([s['A_to_R'] for s in all_summaries])
                avg_r2p = np.mean([s['R_to_P'] for s in all_summaries])
                has_v = 'R_to_V' in all_summaries[0]
                summary_msg = (f"[ATTN_PROBE] Step {self.global_steps}: "
                               f"A→R={avg_a2r:.4f}, R→P={avg_r2p:.4f}")
                if has_v:
                    avg_r2v = np.mean([s['R_to_V'] for s in all_summaries])
                    avg_a2v = np.mean([s['A_to_V'] for s in all_summaries])
                    summary_msg += f", R→V={avg_r2v:.4f}, A→V={avg_a2v:.4f}"
                summary_msg += f", n={len(all_summaries)} trajectories"
                print(summary_msg)
                print(f"[ATTN_PROBE] Saved to {attn_dir}")

        except Exception as e:
            print(f"[ATTN_PROBE] ERROR during attention mass computation: {e}")
            import traceback
            traceback.print_exc()
            print("[ATTN_PROBE] Continuing without attention data.")

    def _compute_and_save_rollout_entropy(self, rollout_batches):
        """
        Compute per-turn entropy/log-prob on exact rollout-time conditioning windows.

        rollout_batches is a list of DataProto, each potentially containing
        multiple samples (all per-turn replays from one val_dataloader
        micro-batch, left-padded and collated).  We dispatch one
        compute_log_prob call per batch (amortising RPC overhead), then
        iterate over samples to build per-turn summaries.
        """
        import json
        import numpy as np
        from pathlib import Path

        try:
            summaries = []
            token_entropy_dict = {}
            token_logprob_dict = {}

            for batch in rollout_batches:
                batch_size = batch.batch.batch_size[0]
                if batch_size == 0:
                    continue

                # Pad to world-size divisor so DP_COMPUTE_PROTO chunk() succeeds
                n_gpus = self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes
                batch_padded, pad_size = pad_dataproto_to_divisor(batch, n_gpus)

                print(f"[ROLLOUT_ENTROPY] Running forward pass ({batch_size} turns, "
                      f"padded {pad_size})...")

                log_prob_result = self.actor_rollout_wg.compute_log_prob(batch_padded)
                log_prob_result = unpad_dataproto(log_prob_result, pad_size)

                if 'entropys' not in log_prob_result.batch.keys():
                    continue

                entropys = log_prob_result.batch['entropys']      # (bs, resp_len)
                logprobs = log_prob_result.batch['old_log_probs']  # (bs, resp_len)
                response_length = batch.batch['responses'].shape[-1]

                meta = batch.non_tensor_batch
                has_think = 'think_mask' in batch.batch.keys()

                for i in range(batch_size):
                    env_id = str(meta['env_id'][i])
                    config_id = str(meta.get('config_id', np.full(batch_size, '', dtype=object))[i])
                    turn = int(meta['turn'][i])
                    window_size = meta.get('window_size', np.full(batch_size, None, dtype=object))[i]
                    window_start_turn = int(meta.get('window_start_turn', np.zeros(batch_size, dtype=object))[i])

                    ent_i = entropys[i]
                    lp_i = logprobs[i]
                    loss_mask = batch.batch['loss_mask'][i, -response_length:].bool()
                    think_mask = batch.batch['think_mask'][i].bool() if has_think else None

                    valid_ent = ent_i[loss_mask]
                    valid_lp = lp_i[loss_mask]
                    if valid_ent.numel() == 0:
                        continue

                    valid_prob = valid_lp.exp()
                    response_tokens = int(loss_mask.sum().item())
                    prompt_tokens = int(batch.batch['attention_mask'][i].sum().item()) - response_tokens

                    summary = {
                        'env_id': env_id,
                        'config_id': config_id,
                        'turn': turn,
                        'window_size': None if window_size is None else int(window_size),
                        'window_start_turn': window_start_turn,
                        'prompt_tokens': prompt_tokens,
                        'response_tokens': response_tokens,
                        'mean_entropy': valid_ent.mean().item(),
                        'std_entropy': valid_ent.std().item() if valid_ent.numel() > 1 else 0.0,
                        'max_entropy': valid_ent.max().item(),
                        'min_entropy': valid_ent.min().item(),
                        'mean_logprob': valid_lp.mean().item(),
                        'mean_prob': valid_prob.mean().item(),
                        'median_prob': valid_prob.median().item(),
                    }

                    if think_mask is not None:
                        think_i = think_mask & loss_mask
                        action_i = (~think_mask) & loss_mask
                        if think_i.any():
                            think_ent = ent_i[think_i]
                            think_lp = lp_i[think_i]
                            summary['think_mean_entropy'] = think_ent.mean().item()
                            summary['think_mean_logprob'] = think_lp.mean().item()
                            summary['think_mean_prob'] = think_lp.exp().mean().item()
                            summary['num_think_tokens'] = int(think_ent.numel())
                        if action_i.any():
                            action_ent = ent_i[action_i]
                            action_lp = lp_i[action_i]
                            summary['action_mean_entropy'] = action_ent.mean().item()
                            summary['action_mean_logprob'] = action_lp.mean().item()
                            summary['action_mean_prob'] = action_lp.exp().mean().item()
                            summary['num_action_tokens'] = int(action_ent.numel())

                    summaries.append(summary)
                    cache_key = f"{env_id}__turn_{turn}"
                    token_entropy_dict[cache_key] = valid_ent.cpu().numpy().astype(np.float16)
                    token_logprob_dict[cache_key] = valid_lp.cpu().numpy().astype(np.float16)

            experiment_name = self.config.trainer.get('experiment_name', 'default_experiment')
            project_name = self.config.trainer.get('project_name', 'tracerigor')
            base_dir = Path(f"val_generations/{project_name}/{experiment_name}")
            entropy_dir = base_dir / f"step_{self.global_steps}" / "rollout_replay_entropy"
            entropy_dir.mkdir(parents=True, exist_ok=True)

            summary_path = entropy_dir / "entropy_summary.jsonl"
            with open(summary_path, 'w', encoding='utf-8') as f:
                for item in summaries:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')

            np.savez_compressed(str(entropy_dir / "token_entropies.npz"), **token_entropy_dict)
            np.savez_compressed(str(entropy_dir / "token_logprobs.npz"), **token_logprob_dict)

            if summaries:
                all_means = [item['mean_entropy'] for item in summaries]
                print(f"[ROLLOUT_ENTROPY] Step {self.global_steps}: "
                      f"entropy_mean={np.mean(all_means):.4f}, n={len(summaries)} turns")
                print(f"[ROLLOUT_ENTROPY] Saved to {entropy_dir}")

        except Exception as e:
            print(f"[ROLLOUT_ENTROPY] ERROR during rollout entropy computation: {e}")
            import traceback
            traceback.print_exc()
            print("[ROLLOUT_ENTROPY] Continuing without rollout entropy data.")

    def _compute_and_save_rollout_attention_mass(self, rollout_batches):
        """
        Compute cross-segment attention mass on exact rollout-time conditioning
        windows.  Each element of rollout_batches is a multi-sample DataProto
        (one per val_dataloader micro-batch).  We dispatch one
        compute_attention_mass call per batch, then iterate over samples to
        produce per-turn summaries.
        """
        import json
        import numpy as np
        from pathlib import Path

        try:
            layer_stride = self.config.trainer.get('attn_layer_stride', 4)
            chunk_size = self.config.trainer.get('attn_chunk_size', 128)
            summaries = []

            for batch in rollout_batches:
                batch_size = batch.batch.batch_size[0]
                if batch_size == 0:
                    continue

                batch.meta_info['attn_layer_stride'] = layer_stride
                batch.meta_info['attn_chunk_size'] = chunk_size
                batch.meta_info['attn_per_turn'] = True
                batch.meta_info['split_prompt_segments'] = True

                # Pad to world-size divisor
                n_gpus = self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes
                batch_padded, pad_size = pad_dataproto_to_divisor(batch, n_gpus)

                print(f"[ROLLOUT_ATTN] Running attention probe ({batch_size} turns, "
                      f"padded {pad_size})...")

                attn_result = self.actor_rollout_wg.compute_attention_mass(batch_padded)
                if attn_result is None or attn_result.batch['attention_mass'].numel() == 0:
                    continue

                attn_shape = attn_result.meta_info.get('attn_shape', [])
                if not attn_shape:
                    continue

                # Unpad and reshape
                attn_result = unpad_dataproto(attn_result, pad_size)
                n_result = attn_result.batch['attention_mass'].shape[0]
                all_mass = attn_result.batch['attention_mass'].reshape(n_result, *attn_shape)
                segment_names = attn_result.meta_info.get('segment_names', ['P', 'R', 'A'])
                seg_idx = {name: idx for idx, name in enumerate(segment_names)}
                probed_layers = attn_result.meta_info.get('probed_layers', [])

                meta = batch.non_tensor_batch
                response_length = batch.batch['responses'].shape[-1]

                per_turn_mass_all = attn_result.meta_info.get('per_turn_mass', None)

                for i in range(min(batch_size, n_result)):
                    env_id = str(meta['env_id'][i])
                    config_id = str(meta.get('config_id', np.full(batch_size, '', dtype=object))[i])
                    turn = int(meta['turn'][i])
                    window_size = meta.get('window_size', np.full(batch_size, None, dtype=object))[i]
                    window_start_turn = int(meta.get('window_start_turn', np.zeros(batch_size, dtype=object))[i])
                    response_tokens = int(batch.batch['loss_mask'][i].sum().item())
                    prompt_tokens = int(batch.batch['attention_mask'][i].sum().item()) - response_tokens

                    mass_i = all_mass[i]  # (num_layers, N, N)

                    entry = {
                        'env_id': env_id,
                        'config_id': config_id,
                        'turn': turn,
                        'window_size': None if window_size is None else int(window_size),
                        'window_start_turn': window_start_turn,
                        'prompt_tokens': prompt_tokens,
                        'response_tokens': response_tokens,
                        'probed_layers': probed_layers,
                        'segment_names': segment_names,
                        'mass_by_layer': {},
                    }

                    for layer_pos, layer_idx in enumerate(probed_layers):
                        entry['mass_by_layer'][str(layer_idx)] = mass_i[layer_pos].tolist()

                    avg_mass = mass_i.mean(dim=0)
                    entry['avg_mass'] = avg_mass.tolist()
                    if 'A' in seg_idx and 'R' in seg_idx:
                        entry['A_to_R'] = avg_mass[seg_idx['A'], seg_idx['R']].item()
                    if 'R' in seg_idx:
                        entry['R_to_R'] = avg_mass[seg_idx['R'], seg_idx['R']].item()

                    # Prompt sub-segment metrics (P_hist / P_curr split)
                    if 'P_hist' in seg_idx:
                        entry['R_to_P_hist'] = avg_mass[seg_idx['R'], seg_idx['P_hist']].item()
                        entry['R_to_P_curr'] = avg_mass[seg_idx['R'], seg_idx['P_curr']].item()
                        entry['A_to_P_hist'] = avg_mass[seg_idx['A'], seg_idx['P_hist']].item()
                        entry['A_to_P_curr'] = avg_mass[seg_idx['A'], seg_idx['P_curr']].item()
                        # Combined P for backward compatibility
                        entry['R_to_P'] = entry['R_to_P_hist'] + entry['R_to_P_curr']
                        entry['A_to_P'] = entry['A_to_P_hist'] + entry['A_to_P_curr']
                    elif 'P' in seg_idx:
                        entry['R_to_P'] = avg_mass[seg_idx['R'], seg_idx['P']].item()
                        entry['A_to_P'] = avg_mass[seg_idx['A'], seg_idx['P']].item()

                    if 'V' in seg_idx and 'R' in seg_idx:
                        entry['R_to_V'] = avg_mass[seg_idx['R'], seg_idx['V']].item()
                        entry['A_to_V'] = avg_mass[seg_idx['A'], seg_idx['V']].item() if 'A' in seg_idx else 0.0

                    if per_turn_mass_all and i < len(per_turn_mass_all):
                        turn_entries = []
                        for td in per_turn_mass_all[i]:
                            mass_t = torch.tensor(td['mass'])
                            avg_t = mass_t.mean(dim=0)
                            turn_entry = {
                                'turn_local': td['turn'],
                                'r_count': td['r_count'],
                                'a_count': td['a_count'],
                            }
                            for ti, tname in enumerate(segment_names):
                                turn_entry[f'R_to_{tname}'] = avg_t[0, ti].item()
                                turn_entry[f'A_to_{tname}'] = avg_t[1, ti].item()
                            # Backward-compatible combined P
                            if 'P_hist' in seg_idx:
                                turn_entry['R_to_P'] = turn_entry.get('R_to_P_hist', 0) + turn_entry.get('R_to_P_curr', 0)
                                turn_entry['A_to_P'] = turn_entry.get('A_to_P_hist', 0) + turn_entry.get('A_to_P_curr', 0)
                            turn_entries.append(turn_entry)
                        entry['turn_masses'] = turn_entries

                    summaries.append(entry)

            experiment_name = self.config.trainer.get('experiment_name', 'default_experiment')
            project_name = self.config.trainer.get('project_name', 'tracerigor')
            base_dir = Path(f"val_generations/{project_name}/{experiment_name}")
            attn_dir = base_dir / f"step_{self.global_steps}" / "rollout_replay_attention"
            attn_dir.mkdir(parents=True, exist_ok=True)

            attn_path = attn_dir / "attention_mass.jsonl"
            with open(attn_path, 'w', encoding='utf-8') as f:
                for item in summaries:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')

            if summaries:
                avg_a2r = np.mean([item.get('A_to_R', 0.0) for item in summaries])
                avg_r2p = np.mean([item.get('R_to_P', 0.0) for item in summaries])
                summary_msg = (f"[ROLLOUT_ATTN] Step {self.global_steps}: "
                               f"A→R={avg_a2r:.4f}, R→P={avg_r2p:.4f}")
                # Log history/current split if available
                if summaries[0].get('R_to_P_hist') is not None:
                    avg_r2ph = np.mean([s.get('R_to_P_hist', 0.0) for s in summaries])
                    avg_r2pc = np.mean([s.get('R_to_P_curr', 0.0) for s in summaries])
                    summary_msg += f", R→P_hist={avg_r2ph:.4f}, R→P_curr={avg_r2pc:.4f}"
                summary_msg += f", n={len(summaries)} turns"
                print(summary_msg)
                print(f"[ROLLOUT_ATTN] Saved to {attn_dir}")

        except Exception as e:
            print(f"[ROLLOUT_ATTN] ERROR during rollout attention computation: {e}")
            import traceback
            traceback.print_exc()
            print("[ROLLOUT_ATTN] Continuing without rollout attention data.")

    def _validate(self):
        print(f"[DEBUG] validation at global step {self.global_steps} begins")
        # Lists to collect samples for the table

        if self.test_rollout_manager==None:
            if self.config.rollout_manager.get("use_service",False):
                self.test_rollout_manager =QwenVLRolloutManagerService(
                    actor_rollout_wg=self.actor_rollout_wg,
                    config=self.config.rollout_manager,
                    tokenizer=self.tokenizer,
                    processor=self.processor,
                    split="val",
                )
            else:
                self.test_rollout_manager =QwenVLRolloutManager(
                    actor_rollout_wg=self.actor_rollout_wg,
                    config=self.config.rollout_manager,
                    tokenizer=self.tokenizer,
                    processor=self.processor,
                )

        validation_rst=[]
        traj_batches=[]  # collect DataProto batches for entropy computation
        rollout_analysis_batches=[]  # exact per-turn rollout-time replay batches


        for batch_dict in self.val_dataloader:

            batch: DataProto = DataProto.from_single_dict(batch_dict)
            # pop these keys so it will not cause error when rollout
            if 'multi_modal_inputs' in batch.non_tensor_batch.keys():
                batch.pop(
                    batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                    non_tensor_batch_keys=['raw_prompt_ids', 'multi_modal_data', 'multi_modal_inputs'],
                )
            else:
                batch.pop(
                    batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                    non_tensor_batch_keys=['raw_prompt_ids'],
                )

            env_configs = [
                    batch.non_tensor_batch['extra_info'][i]
                    for i in range(len(batch))
                ]

            self.test_rollout_manager.reset(env_configs)
            self.test_rollout_manager.rollout_loop()
            micro_validation_rst = self.test_rollout_manager.recording_to_log() # data source == inputs in our current setting, outputs=whole trjecotry
            validation_rst.extend(micro_validation_rst)

            # Build trajectory batch for entropy forward pass (before state is reset)
            try:
                micro_traj_batch = self.test_rollout_manager.generate_batch_for_update()
                # Set meta_info needed by compute_log_prob
                micro_traj_batch.meta_info['micro_batch_size'] = self.config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu
                micro_traj_batch.meta_info['max_token_len'] = self.config.actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu
                micro_traj_batch.meta_info['use_dynamic_bsz'] = self.config.actor_rollout_ref.rollout.log_prob_use_dynamic_bsz
                micro_traj_batch.meta_info['temperature'] = self.config.actor_rollout_ref.rollout.temperature
                traj_batches.append(micro_traj_batch)
            except Exception as e:
                print(f"[ENTROPY] WARNING: Failed to build traj_batch: {e}")

            try:
                micro_rollout_batches = self.test_rollout_manager.generate_rollout_analysis_batches()
                for rollout_batch in micro_rollout_batches:
                    rollout_batch.meta_info['micro_batch_size'] = self.config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu
                    rollout_batch.meta_info['max_token_len'] = self.config.actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu
                    rollout_batch.meta_info['use_dynamic_bsz'] = self.config.actor_rollout_ref.rollout.log_prob_use_dynamic_bsz
                    rollout_batch.meta_info['temperature'] = self.config.actor_rollout_ref.rollout.temperature
                rollout_analysis_batches.extend(micro_rollout_batches)
            except Exception as e:
                print(f"[ROLLOUT_REPLAY] WARNING: Failed to build rollout analysis batches: {e}")

        # Save ALL validation samples to local folder for later analysis
        self._save_val_generations_to_local(validation_rst)

        # Compute and save per-token entropy via forward pass
        if traj_batches:
            try:
                self._compute_and_save_entropy(validation_rst, traj_batches)
            except Exception as e:
                print(f"[ENTROPY] Failed to compute entropy: {e}")

        # Compute and save cross-segment attention mass via probe forward pass
        if traj_batches:
            try:
                self._compute_and_save_attention_mass(validation_rst, traj_batches)
            except Exception as e:
                print(f"[ATTN_PROBE] Failed to compute attention mass: {e}")

        if rollout_analysis_batches:
            try:
                self._compute_and_save_rollout_entropy(rollout_analysis_batches)
            except Exception as e:
                print(f"[ROLLOUT_ENTROPY] Failed to compute rollout replay entropy: {e}")

        if rollout_analysis_batches:
            try:
                self._compute_and_save_rollout_attention_mass(rollout_analysis_batches)
            except Exception as e:
                print(f"[ROLLOUT_ATTN] Failed to compute rollout replay attention: {e}")

        # Log limited samples to wandb table
        self._maybe_log_val_generations_to_wandb(validation_rst)
        metric_dict = self.log_rst_to_metrics_dict(validation_rst,mode='val')
        print(f"[DEBUG] validation at global step {self.global_steps} ends")
        return metric_dict

    def log_rst_to_metrics_dict(self,rst,mode='train'):
        metric_dict = {}


        metrics_by_config_id = defaultdict(dict)  # a dict of dict of list

        #for item in rst:
        #    for k,v in item["metrics"].items():
        #        if k not in metrics_by_config_id[item["config_id"]]:
        #            metrics_by_config_id[item["config_id"]][k] = []
        #        metrics_by_config_id[item["config_id"]][k].append(v)

        for item in rst:
            cfg = item["config_id"]

            # existing scalar metrics
            for k, v in item["metrics"].items():
                # Handle nested violation_metrics dict by flattening
                if k == "violation_metrics" and isinstance(v, dict):
                    for vk, vv in v.items():
                        if isinstance(vv, (int, float)):
                            metrics_by_config_id[cfg].setdefault(f"violation/{vk}", []).append(vv)
                    continue

                # Skip non-numeric values (strings like termination_reason, booleans, dicts)
                if not isinstance(v, (int, float)):
                    continue

                metrics_by_config_id[cfg].setdefault(k, []).append(v)


            # --- NEW: per-turn env rewards ---
            tr = item.get("turn_rewards")
            if tr:
                for ti, val in enumerate(tr, start=1):
                    key = f"turn_reward/t{ti}"
                    metrics_by_config_id[cfg].setdefault(key, []).append(val)

            # --- NEW: per-turn verifier scores (each rubric separately) ---
            tvs = item.get("turn_verifier_scores")
            if tvs:
                for rubric, seq in tvs.items():
                    for ti, val in enumerate(seq, start=1):
                        key = f"{rubric}/t{ti}"
                        metrics_by_config_id[cfg].setdefault(key, []).append(val)

            # --- NEW: per-turn reasoning token length ---
            trl = item.get("turn_reason_len")
            if trl:
                for ti, val in enumerate(trl, start=1):
                    key = f"turn_reason_len/t{ti}"
                    metrics_by_config_id[cfg].setdefault(key, []).append(val)

        for config_id, metrics in metrics_by_config_id.items():
            for k,v in metrics.items():
                metric_dict[f'{mode}/{k}/{config_id}'] = np.mean(v)

        return metric_dict


    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRollout],
                                                     config=self.config.actor_rollout_ref,
                                                     role='actor_rollout')
            self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]['critic'] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy],
                                                  config=self.config.actor_rollout_ref,
                                                  role='ref')
            self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]['rm'] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg['critic']
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg['ref']
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg['rm']
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg['actor_rollout']
        self.actor_rollout_wg.init_model()

    def _save_checkpoint(self):
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir,
                                                f'global_step_{self.global_steps}')
        actor_local_path = os.path.join(local_global_step_folder, 'actor')

        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
            self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', 'actor')
        self.actor_rollout_wg.save_checkpoint(actor_local_path,
                                              actor_remote_path,
                                              self.global_steps,
                                              remove_previous_ckpt=self.config.trainer.remove_previous_ckpt_in_save)

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, 'critic')
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
                self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', 'critic')
            self.critic_wg.save_checkpoint(critic_local_path,
                                           critic_remote_path,
                                           self.global_steps,
                                           remove_previous_ckpt=self.config.trainer.remove_previous_ckpt_in_save)

        # save dataloader
        dataloader_local_path = os.path.join(local_global_step_folder, 'data.pt')
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir,
                                                           'latest_checkpointed_iteration.txt')
        with open(local_latest_checkpointed_iteration, 'w') as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == 'disable':
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError('load from hdfs is not implemented yet')
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == 'auto':
            if global_step_folder is None:
                print('Training from scratch')
                return 0
        else:
            if not (self.config.trainer.resume_from_path and global_step_folder is not None):
                assert isinstance(self.config.trainer.resume_mode, str), "resume ckpt must be str type"
                assert 'global_step_' in self.config.trainer.resume_mode, "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_mode
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f'Load from checkpoint folder: {global_step_folder}')
        # set global step
        self.global_steps = int(global_step_folder.split('global_step_')[-1])

        print(f'Setting global step to {self.global_steps}')
        print(f'Resuming from {global_step_folder}')

        actor_path = os.path.join(global_step_folder, 'actor')
        critic_path = os.path.join(global_step_folder, 'critic')
        # load actor
        self.actor_rollout_wg.load_checkpoint(actor_path,
                                              del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(critic_path,
                                           del_local_after_load=self.config.trainer.del_local_ckpt_after_load)

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, 'data.pt')
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix='global_seqlen'):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch['attention_mask']
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch['attention_mask'].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst,
                                                              k_partitions=world_size,
                                                              equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst,
                                                    partitions=global_partition_lst,
                                                    prefix=logging_prefix)
        metrics.update(global_balance_stats)

    def _process_in_mini_batches(self,batch, rollout_manager, mini_batch_size):
        """
        Process the batch in mini-batches.

        Args:
            batch: DataProto containing the data
            rollout_manager: Manager for rollout operations
            mini_batch_size: Size of each mini-batch to process

        Returns:
            Tuple of (final_combined_batch_output, combined_rst)
        """
        batch_size = len(batch)
        num_mini_batches = (batch_size + mini_batch_size - 1) // mini_batch_size  # Ceiling division

        all_final_gen_batch_outputs = []
        all_rst = []

        for i in range(num_mini_batches):
            start_idx = i * mini_batch_size
            end_idx = min((i + 1) * mini_batch_size, batch_size)
            actual_mini_batch_size = end_idx - start_idx
            print(f"Processing mini-batch {i+1}/{num_mini_batches}, size: {actual_mini_batch_size}")

            # Extract env_configs for this mini-batch
            mini_batch_env_configs = [
                batch.non_tensor_batch['extra_info'][j]
                for j in range(start_idx, end_idx)
            ]

            # Reset and process this mini-batch
            rollout_manager.reset(mini_batch_env_configs)
            rollout_manager.rollout_loop()
            mini_batch_output = rollout_manager.generate_batch_for_update()
            mini_batch_rst = rollout_manager.recording_to_log()

            # Store results
            all_final_gen_batch_outputs.append(mini_batch_output)
            all_rst.extend(mini_batch_rst)  # Extend the list since rst is a list


            combined_output = DataProto.concat(all_final_gen_batch_outputs)


        return combined_output, all_rst

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf

        logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            val_metrics = self._validate()
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return

        # we start from step 1
        self.global_steps += 1

        if self.config.rollout_manager.get("use_service",False):
            rollout_manager = QwenVLRolloutManagerService(
                actor_rollout_wg=self.actor_rollout_wg,
                config=self.config.rollout_manager,
                tokenizer=self.tokenizer,
                processor=self.processor,
                split="train",
            )
        else:
            rollout_manager = QwenVLRolloutManager(
                actor_rollout_wg=self.actor_rollout_wg,
                config=self.config.rollout_manager,
                tokenizer=self.tokenizer,
                processor=self.processor,
            )

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                print(f'global_steps: {self.global_steps}')
                metrics = {}
                timing_raw = {}

                batch: DataProto = DataProto.from_single_dict(batch_dict)
                # pop these keys so it will not cause error when rollout
                if 'multi_modal_inputs' in batch.non_tensor_batch.keys():
                    batch.pop(
                        batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                        non_tensor_batch_keys=['raw_prompt_ids', 'multi_modal_data', 'multi_modal_inputs'],
                    )
                else:
                    batch.pop(
                        batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                        non_tensor_batch_keys=['raw_prompt_ids'],
                    )


                # We control vanilla-grpo sampling param here (start from init state s0, sample n_trajectory)
                batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],dtype=object)
                batch = batch.repeat(repeat_times=self.config.rollout_manager.n_trajectory, interleave=True)




                with _timer('step', timing_raw):
                    # generate a batch
                    with _timer('gen', timing_raw):

                        mini_batch_size=self.config.rollout_manager.get('mini_batch_size',len(batch))
                        final_gen_batch_output, rst=self._process_in_mini_batches(batch, rollout_manager, mini_batch_size)
                        train_metrics=self.log_rst_to_metrics_dict(rst=rst,mode='train')
                        metrics.update(train_metrics)
                    print(f"[DEBUG] step {self.global_steps} rollout ends")
                    batch = batch.union(final_gen_batch_output)

                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

                    # recompute old_log_probs
                    with _timer('old_log_prob', timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        # Extract entropy metric before merging
                        if 'entropys' in old_log_prob.batch.keys():
                            import verl.utils.torch_functional as verl_F
                            entropys = old_log_prob.batch['entropys']
                            if 'response_mask' not in batch.batch.keys():
                                resp_len = batch.batch['responses'].shape[-1]
                                response_mask = batch.batch['attention_mask'][:, -resp_len:]
                            else:
                                response_mask = batch.batch['response_mask']
                            entropy_agg = verl_F.masked_mean(entropys, response_mask)
                            metrics.update({'actor/entropy': entropy_agg.detach().item()})
                            old_log_prob.batch.pop('entropys')
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer('ref', timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer('values', timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer('adv', timing_raw):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)


                        if self.config.rollout_manager.use_multi_turn_reward:
                        #TraceRigor: TODO: use a new reward_fn to combine the results from reward model and rule-based multi-turn token reward.
                            response_len=batch.batch['responses'].shape[1]
                            batch.batch['token_level_scores'] = batch.batch['multi_turn_token_level_rewards'][:,-response_len:]
                        else:
                            # we combine with rule-based rm
                            reward_tensor = self.reward_fn(batch)
                            batch.batch['token_level_scores'] = reward_tensor

                        ## --- NEW (Option A): add verifier-shaped bonus on <think> tokens ---
                        ## token_level_scores has shape [B, T_resp]; think_mask is [B, T_resp]
                        ## think_verifier_score is [B]; we broadcast it along T_resp.
                        #think_lambda = float(self.config.algorithm.get("think_reward_weight", 1.0))
                        #if (
                        #    think_lambda > 0.0
                        #    and "think_mask" in batch.batch
                        #    and "think_verifier_score" in batch.batch
                        #):
                        #    think_mask = batch.batch["think_mask"].to(batch.batch["token_level_scores"].dtype)
                        #    num_think_tokens = think_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
                        #    print(f"[DEBUG] think_mask.sum(dim=1, keepdim=True) => {think_mask.sum(dim=1, keepdim=True)}")
                        #    print(f"[DEBUG] num_think_tokens => {num_think_tokens}")
                        #    qc = batch.batch["think_verifier_score"].view(-1, 1).to(
                        #        batch.batch["token_level_scores"].dtype
                        #    )
                        #    # distribute fixed total bonus alpha * qc per episode
                        #    bonus_per_token = think_lambda * qc / num_think_tokens
                        #    bonus = bonus_per_token * think_mask
                        #    #bonus = think_lambda * qc * think_mask
                        #    batch.batch["token_level_scores"] = batch.batch["token_level_scores"] + bonus
                        ## --- END NEW ---

                        # --- NEW: verifier-shaped bonus ---
                        think_lambda = float(self.config.algorithm.get("think_reward_weight", 0.0))
                        if think_lambda > 0.0:
                            is_tvtr_in_batch = "think_verifier_token_rewards" in batch.batch
                            print(f"[DEBUG] is_tvtr_in_batch is {is_tvtr_in_batch}")
                            is_qc_in_batch = "think_verifier_score" in batch.batch
                            print(f"[DEBUG] is_qc_in_batch is {is_qc_in_batch}")
                            if "think_verifier_token_rewards" in batch.batch:
                                # preferred: fine-grained per-token rewards coming from rollout manager
                                tvtr = batch.batch["think_verifier_token_rewards"]  # [B, T_full] or [B, T_resp]
                                tvtr_sum_1d = tvtr.sum(dim=1, keepdim=True)
                                print(f"[DEBUG] tvtr.sum(dim=1, keepdim=True) is {tvtr_sum_1d}")
                                tvtr_non_zero_1d = tvtr_sum_1d[tvtr_sum_1d.nonzero(as_tuple=True)]
                                print(f"[DEBUG] tvtr_non_zero_1d is {tvtr_non_zero_1d} and non zero index is {tvtr_sum_1d.nonzero(as_tuple=True)}")
                                if tvtr.abs().sum() > 0:
                                    resp_len = batch.batch["responses"].shape[1]
                                    bonus = tvtr[:, -resp_len:]  # align to response tokens
                                    batch.batch["token_level_scores"] = batch.batch["token_level_scores"] + think_lambda * bonus

                            elif "think_mask" in batch.batch and "think_verifier_score" in batch.batch:
                                # fallback: old Option A (global scalar per trajectory, normalized by #think tokens)
                                qc = batch.batch["think_verifier_score"]  # [B]
                                if qc.abs().sum() > 0:
                                    token_level_scores = batch.batch["token_level_scores"]
                                    think_mask = batch.batch["think_mask"].to(token_level_scores.dtype)

                                    # number of <think> tokens per sample
                                    num_think_tokens = think_mask.sum(dim=1, keepdim=True).clamp_min(1.0)  # [B, 1]
                                    print(f"[DEBUG] think_mask.sum(dim=1, keepdim=True) => {num_think_tokens}")
                                    has_think = num_think_tokens > 0
                                    print(f"[DEBUG] has_think is {has_think}")

                                    qc = qc.view(-1, 1).to(token_level_scores.dtype)  # [B, 1]

                                    # Each trajectory gets total bonus think_lambda * qc, split over think tokens
                                    bonus_per_token = think_lambda * qc / num_think_tokens            # [B, 1]
                                    bonus = bonus_per_token * think_mask                              # [B, T]
                                    batch.batch["token_level_scores"] = token_level_scores + bonus

                                #if has_think.any():
                                #    qc = batch.batch["think_verifier_score"].view(-1, 1).to(token_level_scores.dtype)
                                #    bonus = torch.zeros_like(token_level_scores)

                                #    # only normalize where we actually have think tokens
                                #    bonus_per_token = torch.zeros_like(token_level_scores)
                                #    bonus_per_token[has_think] = think_lambda * qc[has_think] / num_think_tokens[has_think]

                                #    bonus = bonus_per_token * think_mask
                                #    batch.batch["token_level_scores"] = token_level_scores + bonus
                        # --- END NEW ---

                        # compute rewards. apply_kl_penalty if available
                        if not self.config.actor_rollout_ref.actor.get('use_kl_loss', False):
                            batch, kl_metrics = apply_kl_penalty(batch,
                                                                 kl_ctrl=self.kl_ctrl,
                                                                 kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

                        # compute advantages, executed on the driver process
                        batch = compute_advantage(batch,
                                                  adv_estimator=self.config.algorithm.adv_estimator,
                                                  gamma=self.config.algorithm.gamma,
                                                  lam=self.config.algorithm.lam,
                                                  num_repeat=self.config.actor_rollout_ref.rollout.n,
                                                  high_level_gamma=self.config.algorithm.high_level_gamma,)

                    # update critic
                    if self.use_critic:
                        with _timer('update_critic', timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer('update_actor', timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                        self.global_steps % self.config.trainer.test_freq == 0:
                        with _timer('testing', timing_raw):
                            val_metrics: dict = self._validate()
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and \
                            self.global_steps % self.config.trainer.save_freq == 0:
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                self.global_steps += 1

                if self.global_steps >= self.total_training_steps:

                    # perform validation after training
                    if self.val_reward_fn is not None:
                        val_metrics = self._validate()
                        pprint(f'Final validation metrics: {val_metrics}')
                        logger.log(data=val_metrics, step=self.global_steps)
                    if self.config.trainer.save_freq > 0 and \
                            (self.global_steps - 1) % self.config.trainer.save_freq != 0:
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()
                    return

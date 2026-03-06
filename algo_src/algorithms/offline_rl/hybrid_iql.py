# source: https://github.com/gwthomas/IQL-PyTorch
# https://arxiv.org/pdf/2110.06169.pdf
import copy
from os import getenv

from os.path import join
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from dotenv import load_dotenv
from tensordict import TensorDict
from torch.distributions import Normal, OneHotCategorical
from torch.nn.functional import one_hot
from torch.optim.lr_scheduler import CosineAnnealingLR

from actions.discrete_actions import get_possible_actions, get_discrete_action_size, get_continuous_action_size
from actions.space import get_discrete_actions_list
from agents.base import RLAgent
from agents.configs import TrainerExperimentConfig
from agents.train import AgentTrainer
from dataset.buffer import ReplayBuffer
from dataset.config import load_dataset_config
from dataset.load import load_dataset_to_buffer
from dataset.pre_processing_configs import get_pre_processing_configs

from network.mlp import create_mlp
from network.update import soft_update
from utils.files import save_json, load_yaml

from utils.wandb import wandb_init

TensorBatch = List[torch.Tensor]

EXP_ADV_MAX = 100.0
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


class TrainConfig(TrainerExperimentConfig):
    hidden_dims: List[int]

    batch_size: int = 256  # Batch size for all networks
    discount: float = 0.99  # Discount factor
    tau: float = 0.005  # Target network update rate
    beta: float = 3.0  # Inverse temperature. Small beta -> BC, big beta -> maximizing Q
    iql_tau: float = 0.7  # Coefficient for asymmetric loss

    vf_lr: float = 3e-4  # V function learning rate
    qf_lr: float = 3e-4  # Critic learning rate
    actor_lr: float = 3e-4  # Actor learning rate

    max_action: float = 1.0


def asymmetric_l2_loss(u: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.mean(torch.abs(tau - (u < 0).float()) * u ** 2)


class Squeeze(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(dim=self.dim)


class HybridPolicy(nn.Module):
    def __init__(
            self,
            state_dim: int,
            continuous_act_dim: int,
            discrete_act_dim: int,
            max_action: float,
            hidden_dims: List[int]
    ):
        super().__init__()

        self.policy_mlp = create_mlp(
            input_dim=state_dim,
            hidden_dims=hidden_dims
        )

        self.cont_mean_out = create_mlp(
            input_dim=hidden_dims[-1],
            hidden_dims=[],
            output_dim=continuous_act_dim,
            output_activation=nn.Tanh
        )

        self.discrete_out = nn.Linear(in_features=hidden_dims[-1], out_features=discrete_act_dim)

        self.log_std = nn.Parameter(torch.zeros(continuous_act_dim, dtype=torch.float32))
        self.max_action = max_action

    def forward(self, obs: torch.Tensor) -> tuple[Normal, OneHotCategorical]:
        mlp_out = self.policy_mlp(obs)

        # Get continuous action distribution
        mean = self.cont_mean_out(mlp_out)
        std = torch.exp(self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX))
        continuous_action_dist = Normal(mean, std)

        # Get discrete actions distribution
        logits = self.discrete_out(mlp_out)

        discrete_action_dist = OneHotCategorical(logits=logits)
        return continuous_action_dist, discrete_action_dist

    @torch.no_grad()
    def get_action(self, state: torch.Tensor) -> TensorDict:
        cont_dist, disc_dist = self(state)

        cont_action = cont_dist.mean if not self.training else cont_dist.sample()
        cont_action = cont_action * self.max_action
        cont_action = torch.clamp(cont_action, min=-self.max_action, max=self.max_action)

        disc_action = one_hot(torch.argmax(disc_dist.logits, dim=-1), num_classes=disc_dist.logits.shape[-1]) if not self.training else disc_dist.sample()

        return TensorDict(
            {
                'discrete_actions': disc_action,
                'continuous_actions': cont_action},
            batch_size=state.shape[0]
        )


class TwinQ(nn.Module):
    def __init__(
            self, state_dim: int, action_dim: int, hidden_dims: List[int]
    ):
        super().__init__()

        self.q1 = create_mlp(
            input_dim=state_dim + action_dim,
            hidden_dims=hidden_dims,
            output_dim=1).append(Squeeze(-1))

        self.q2 = create_mlp(
            input_dim=state_dim + action_dim,
            hidden_dims=hidden_dims,
            output_dim=1
        ).append(Squeeze(-1))

    def both(
            self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], 1)
        return self.q1(sa), self.q2(sa)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return torch.min(*self.both(state, action))


class ValueFunction(nn.Module):
    def __init__(self, state_dim: int, hidden_dims: List[int]):
        super().__init__()

        self.v = create_mlp(
            input_dim=state_dim,
            hidden_dims=hidden_dims,
            output_dim=1
        ).append(Squeeze(-1))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.v(state)


class ImplicitQLearning(RLAgent):

    def __init__(
            self,
            max_action: float,
            actor: nn.Module,
            actor_optimizer: torch.optim.Optimizer,
            q_network: nn.Module,
            q_optimizer: torch.optim.Optimizer,
            v_network: nn.Module,
            v_optimizer: torch.optim.Optimizer,
            iql_tau: float,
            beta: float,
            max_steps: int,
            discount: float,
            tau: float,
            batch_size: int,
            device: str
    ):
        self.max_action = max_action
        self.qf = q_network
        self.q_target = copy.deepcopy(self.qf).requires_grad_(False).to(device)
        self.vf = v_network
        self.actor = actor
        self.v_optimizer = v_optimizer
        self.q_optimizer = q_optimizer
        self.actor_optimizer = actor_optimizer
        self.actor_lr_schedule = CosineAnnealingLR(self.actor_optimizer, max_steps)
        self.iql_tau = iql_tau
        self.beta = beta
        self.discount = discount
        self.tau = tau
        self.batch_size = batch_size
        self.total_it = 0
        self.device = device

    def get_action(self, state, deterministic: bool, **kwargs) -> torch.Tensor:

        if deterministic:
            self.actor.training = False
        else:
            self.actor.training = True
        return self.actor.get_action(state=state)

    def learn(self, buffer: ReplayBuffer, **kwargs) -> Dict[str, float]:
        batch_size = self.batch_size
        batch = buffer.sample(batch_size)

        self.total_it += 1

        observations = batch.observations
        actions = batch.actions
        rewards = batch.rewards
        next_observations = batch.next_observations
        dones = batch.terminals

        discrete_actions = actions['discrete_actions']
        continuous_actions = actions['continuous_actions']
        combined_actions = torch.cat([continuous_actions, discrete_actions], dim=-1)

        log_dict = {}

        with torch.no_grad():
            next_v = self.vf(next_observations)
        # Update value function
        adv = self._update_v(observations, combined_actions, log_dict)
        rewards = rewards.squeeze(dim=-1)
        dones = dones.squeeze(dim=-1)
        # Update Q function
        self._update_q(next_v, observations, combined_actions, rewards, dones, log_dict)
        # Update actor
        self._update_policy(
            adv=adv,
            observations=observations,
            discrete_actions=discrete_actions,
            continuous_actions=continuous_actions,
            log_dict=log_dict
        )

        self.clear_all_grads()

        return log_dict

    def _update_v(self, observations, actions, log_dict) -> torch.Tensor:
        # Update value function
        with torch.no_grad():
            target_q = self.q_target(observations, actions)

        v = self.vf(observations)
        adv = target_q - v
        v_loss = asymmetric_l2_loss(adv, self.iql_tau)
        log_dict["value_loss"] = v_loss.item()
        log_dict["adv"] = adv.mean().item()
        log_dict["v"] = v.mean().item()
        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()
        return adv

    def _update_q(
            self,
            next_v: torch.Tensor,
            observations: torch.Tensor,
            actions: torch.Tensor,
            rewards: torch.Tensor,
            terminals: torch.Tensor,
            log_dict: Dict,
    ):
        targets = rewards + (1.0 - terminals.float()) * self.discount * next_v.detach()
        qs = self.qf.both(observations, actions)
        q_loss = sum(F.mse_loss(q, targets) for q in qs) / len(qs)
        log_dict["q_loss"] = q_loss.item()
        log_dict['Q'] = qs[0].mean().item()
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # Update target Q network
        soft_update(target=self.q_target, source=self.qf, tau=self.tau)

    def _update_policy(
            self,
            adv: torch.Tensor,
            observations: torch.Tensor,
            discrete_actions: torch.Tensor,
            continuous_actions: torch.Tensor,
            log_dict: Dict,
    ):
        exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=EXP_ADV_MAX)
        cont_dist, disc_dist = self.actor(observations)

        log_prob_cont = cont_dist.log_prob(continuous_actions)
        log_prob_disc = disc_dist.log_prob(discrete_actions)
        log_prob_disc = log_prob_disc.unsqueeze(-1) if log_prob_disc.dim() == 1 else log_prob_disc

        log_dict['log_prob_disc'] = log_prob_disc.mean().item()
        log_dict['log_prob_cont'] = log_prob_cont.mean().item()

        log_prob = torch.cat([log_prob_cont, log_prob_disc], dim=-1)
        bc_losses = -(log_prob.sum(-1, keepdim=False))

        policy_loss = torch.mean(exp_adv * bc_losses)
        log_dict["actor_loss"] = policy_loss.item()
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        self.actor_lr_schedule.step()

    def eval(self, eval_buffer: ReplayBuffer, **kwargs):
        batch = eval_buffer.sample_all()

        observations = batch.observations
        actions = batch.actions
        rewards = batch.rewards
        next_observations = batch.next_observations
        dones = batch.terminals

        discrete_actions = actions['discrete_actions']
        continuous_actions = actions['continuous_actions']
        combined_actions = torch.cat([continuous_actions, discrete_actions], dim=-1)

        log_dict = {}

        mini_batch_size = self.batch_size
        batch_range = range(0, observations.shape[0], mini_batch_size)

        # Initialize accumulators for losses and metrics
        total_v_loss = 0.0
        total_q_loss = 0.0
        total_actor_loss = 0.0
        total_q = 0.0
        total_adv = 0.0
        total_v = 0.0
        num_batches = 0

        for i in batch_range:
            states_mini_batch = batch.observations[i: i + mini_batch_size]
            next_states_mini_batch = next_observations[i: i + mini_batch_size]
            disc_actions_mini_batch = discrete_actions[i: i + mini_batch_size]
            cont_actions_mini_batch = continuous_actions[i: i + mini_batch_size]
            actions_mini_batch = combined_actions[i: i + mini_batch_size]
            dones_mini_batch = dones[i: i + mini_batch_size]
            rewards_mini_batch = rewards[i: i + mini_batch_size]

            with torch.no_grad():
                v = self.vf(states_mini_batch)
                next_v = self.vf(next_states_mini_batch)
                target_q = self.q_target(states_mini_batch, actions_mini_batch)
                adv = target_q - v
                v_loss = asymmetric_l2_loss(adv, self.iql_tau)
                total_v_loss += v_loss.item()

                targets = rewards_mini_batch + (1.0 - dones_mini_batch.float()) * self.discount * next_v.detach()
                qs = self.qf.both(states_mini_batch, actions_mini_batch)

                # Accumulate metrics
                total_q += qs[0].mean().item()  # Use first Q-network for logging
                total_adv += adv.mean().item()
                total_v += v.mean().item()

                q_loss = sum(F.mse_loss(q, targets) for q in qs) / len(qs)
                total_q_loss += q_loss.item()

                exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=EXP_ADV_MAX)
                cont_dist, disc_dist = self.actor(states_mini_batch)

                log_prob_cont = cont_dist.log_prob(cont_actions_mini_batch)
                log_prob_disc = disc_dist.log_prob(disc_actions_mini_batch)
                log_prob_disc = log_prob_disc.unsqueeze(-1) if log_prob_disc.dim() == 1 else log_prob_disc

                log_prob = torch.cat([log_prob_cont, log_prob_disc], dim=-1)
                bc_losses = -(log_prob.sum(-1, keepdim=False))

                policy_loss = torch.mean(exp_adv * bc_losses)
                total_actor_loss += policy_loss.item()

            num_batches += 1

        # Average the losses and metrics across all mini-batches
        log_dict["value_loss"] = total_v_loss / num_batches
        log_dict["q_loss"] = total_q_loss / num_batches
        log_dict["actor_loss"] = total_actor_loss / num_batches
        log_dict["Q"] = total_q / num_batches
        log_dict["adv"] = total_adv / num_batches
        log_dict["v"] = total_v / num_batches

        return log_dict

    def clear_all_grads(self):
        self.v_optimizer.zero_grad()
        self.q_optimizer.zero_grad()
        self.actor_optimizer.zero_grad()


def train(dataset_config_path, config, device):
    config = TrainConfig(**config)
    # Load and save configs
    dataset_config = load_dataset_config(dataset_config_path=dataset_config_path,
                                         experiment_path=config.experiment_path)
    dataset_type = dataset_config.dataset_type
    dataset_config_save_path = join(config.experiment_path, 'dataset_config.json')

    pre_processing_config_save_path = join(config.experiment_path, 'pre_processing_metadata.json')
    pre_processing_config = get_pre_processing_configs(configs_id=dataset_type)
    action_space = pre_processing_config.action_space
    assert len(get_discrete_actions_list(
        action_space)) == 1, 'Currently only supported for single discrete action in hybrid action space'

    save_json(data=dataset_config.model_dump(), path=dataset_config_save_path)
    save_json(data=pre_processing_config.model_dump(), path=pre_processing_config_save_path,
              default_serialization=lambda x: x.__dict__)

    # Load dataset to dataloaders
    train_buffer = load_dataset_to_buffer(dataset=dataset_config.train_dataset_split,
                                          dataset_configs=dataset_config,
                                          pre_process_configs=pre_processing_config, device=device)
    eval_buffer = load_dataset_to_buffer(dataset=dataset_config.test_dataset_split,
                                         dataset_configs=dataset_config,
                                         pre_process_configs=pre_processing_config, device=device)

    state_dim = train_buffer.observations.size(dim=1)
    discrete_action_dim = get_discrete_action_size(
        action_space=action_space,
        discrete_actions_ranges=dataset_config.discrete_action_bin_ranges,
        factored_actions=True,
        vent_mode_action_masking=pre_processing_config.vent_mode_action_masking
    )
    continuous_action_dim = get_continuous_action_size(
        action_space=action_space
    )

    action_dim = discrete_action_dim + continuous_action_dim

    q_network = TwinQ(state_dim, action_dim, hidden_dims=config.hidden_dims).to(device)
    v_network = ValueFunction(state_dim, hidden_dims=config.hidden_dims).to(device)
    actor = HybridPolicy(
        state_dim=state_dim,
        continuous_act_dim=continuous_action_dim,
        discrete_act_dim=discrete_action_dim,
        max_action=config.max_action,
        hidden_dims=config.hidden_dims,
    ).to(device)

    v_optimizer = torch.optim.Adam(v_network.parameters(), lr=config.vf_lr)
    q_optimizer = torch.optim.Adam(q_network.parameters(), lr=config.qf_lr)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_lr)

    kwargs = {
        "max_action": config.max_action,
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "q_network": q_network,
        "q_optimizer": q_optimizer,
        "v_network": v_network,
        "v_optimizer": v_optimizer,
        "discount": config.discount,
        "tau": config.tau,
        "device": device,
        # IQL
        "beta": config.beta,
        "iql_tau": config.iql_tau,
        "max_steps": config.steps,
        "batch_size": config.batch_size
    }
    wandb_init(
        config=config.model_dump(),
        dataset_config=dataset_config.model_dump(),
        pre_processing_metadata=pre_processing_config.model_dump()
    )

    agent = ImplicitQLearning(**kwargs)

    trainer = AgentTrainer(agent=agent, train_config=config)
    trainer.train(buffer=train_buffer, eval_buffer=eval_buffer)

    wandb.finish()
    return config.experiment_path


if __name__ == "__main__":
    load_dotenv()
    train(
        dataset_config_path=getenv('DATASET_CONFIG_PATH'),
        config=load_yaml(getenv('HYBRID_IQL_CONFIG_PATH')),
        device=getenv('DEVICE', default='cpu')
    )
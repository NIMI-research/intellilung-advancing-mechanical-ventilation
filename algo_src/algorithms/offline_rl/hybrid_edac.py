# Inspired by:
# 1. paper for SAC-N: https://arxiv.org/abs/2110.01548
# 2. implementation: https://github.com/snu-mllab/EDAC
import math
import os
import uuid
from copy import deepcopy
from dataclasses import field
from os.path import join
from typing import Any, Dict, Optional, Tuple, List

import numpy as np

import torch
import torch.nn as nn
import wandb
from dotenv import load_dotenv
from tensordict import TensorDict
from torch.distributions import Normal, OneHotCategorical
from torch.nn.functional import one_hot, gumbel_softmax, softmax

from actions.discrete_actions import get_discrete_action_size, get_continuous_action_size, get_possible_actions
from actions.hybrid import flatten_action_dict_to_tensor, get_discrete_action, get_continuous_action
from actions.space import get_discrete_actions_list
from agents.base import RLAgent
from agents.configs import TrainerExperimentConfig
from agents.train import AgentTrainer
from dataset.buffer import ReplayBuffer
from dataset.config import load_dataset_config
from dataset.load import load_dataset_to_buffer
from dataset.pre_processing_configs import get_pre_processing_configs

from network.mlp import create_vectorized_ensemble_mlp, create_mlp
from network.update import soft_update
from utils.files import save_json, load_yaml
from utils.wandb import wandb_init


class TrainConfig(TrainerExperimentConfig):
    # model params
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256, 256])
    num_critics: int = 10
    gamma: float = 0.99
    tau: float = 5e-3
    eta: float = 1.0
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    alpha_learning_rate: float = 3e-4
    max_action: float = 1.0
    targ_entropy_scale_cont: float = 1.0
    targ_entropy_scale_disc: float = 1.0
    log_std_max: float
    log_std_min: float

    batch_size: int = 256


# SAC Actor & Critic implementation
class Actor(nn.Module):
    def __init__(
            self,
            state_dim: int,
            cont_action_dim: int,
            discrete_act_dim: int,
            hidden_dims: List[int],
            log_std_max: float,
            log_std_min: float,
            max_action: float = 1.0
    ):
        super().__init__()
        self.log_std_max = log_std_max
        self.log_std_min = log_std_min
        self.trunk = create_mlp(input_dim=state_dim, hidden_dims=hidden_dims)
        # Using separate layers for mu and log_sigma as in EDAC paper
        self.mu = nn.Linear(hidden_dims[-1], cont_action_dim)
        self.log_sigma = nn.Linear(hidden_dims[-1], cont_action_dim)
        self.discrete_out = nn.Linear(hidden_dims[-1], discrete_act_dim)

        # Initialization as in the EDAC paper
        for layer in self.trunk[::2]:
            torch.nn.init.constant_(layer.bias, 0.1)

        torch.nn.init.uniform_(self.mu.weight, -1e-3, 1e-3)
        torch.nn.init.uniform_(self.mu.bias, -1e-3, 1e-3)
        torch.nn.init.uniform_(self.log_sigma.weight, -1e-3, 1e-3)
        torch.nn.init.uniform_(self.log_sigma.bias, -1e-3, 1e-3)

        self.cont_action_dim = cont_action_dim
        self.disc_action_dim = discrete_act_dim
        self.max_action = max_action

    def forward(
            self,
            state: torch.Tensor,
            deterministic: bool = False,
    ):
        hidden = self.trunk(state)
        mu, log_sigma = self.mu(hidden), self.log_sigma(hidden)
        # Clip log_sigma as in EDAC paper (range differs from SAC)
        log_sigma = torch.clip(log_sigma, self.log_std_min, self.log_std_max)
        cont_policy_dist = Normal(mu, torch.exp(log_sigma))

        # Compute logits for the discrete branch
        logits = self.discrete_out(hidden)
        # Also compute the softmax probabilities (used for log-probs and alpha updates)
        prob_disc = softmax(logits, dim=-1)
        log_prob_disc = torch.log(prob_disc + 1e-8)

        # For discrete actions, choose between a hard sample and a differentiable (soft) approximation.
        if deterministic:
            cont_action = cont_policy_dist.mean
            disc_action = OneHotCategorical(logits=logits).mode
        else:
            cont_action = cont_policy_dist.rsample()
            disc_action = OneHotCategorical(logits=logits).sample()

        # Apply tanh squashing to continuous actions
        tanh_action = torch.tanh(cont_action)

        # Compute log probability for continuous actions with tanh correction.
        all_log_prob_c = cont_policy_dist.log_prob(cont_action)
        all_log_prob_c -= torch.log(1.0 - tanh_action.pow(2) + 1e-6)
        log_prob_cont = all_log_prob_c.sum(dim=-1, keepdim=True)

        # Package actions into a TensorDict-like structure.
        actions = TensorDict(
            {
                'discrete_actions': disc_action,
                'continuous_actions': tanh_action
            },
            batch_size=state.shape[0]
        )

        return actions, log_prob_cont, log_prob_disc, prob_disc

    @torch.no_grad()
    def get_action(self, state: torch.Tensor) -> torch.Tensor:
        deterministic = not self.training
        action = self(state, deterministic=True)[0]
        cont_action = action['continuous_actions'] * self.max_action
        cont_action = torch.clamp(cont_action, min=-self.max_action, max=self.max_action)
        action['continuous_actions'] = cont_action
        return action


class VectorizedCritic(nn.Module):
    def __init__(
            self,
            state_dim: int,
            action_dim_cont: int,
            action_dim_disc: int,
            hidden_dims: List[int],
            num_critics: int
    ):
        super().__init__()
        self.critic = create_vectorized_ensemble_mlp(
            input_dim=state_dim + action_dim_cont + action_dim_disc,
            output_dim=1,
            n_ensembles=num_critics,
            hidden_dims=hidden_dims
        )

        # Initialization as in EDAC paper
        for layer in self.critic[::2]:
            torch.nn.init.constant_(layer.bias, 0.1)

        torch.nn.init.uniform_(self.critic[-1].weight, -3e-3, 3e-3)
        torch.nn.init.uniform_(self.critic[-1].bias, -3e-3, 3e-3)

        self.num_critics = num_critics

    def forward(self, state: torch.Tensor, cont_action: torch.Tensor, disc_action: torch.Tensor) -> torch.Tensor:
        # Concatenate state and continuous action along the last dimension.
        state_action = torch.cat([state, cont_action, disc_action], dim=-1)
        if state_action.dim() != 3:
            # If state_action is [batch_size, ...], unsqueeze to [1, batch_size, ...] and replicate.
            assert state_action.dim() == 2
            state_action = state_action.unsqueeze(0).repeat_interleave(self.num_critics, dim=0)
        assert state_action.dim() == 3 and state_action.shape[0] == self.num_critics
        # The critic network outputs Q-values per discrete action.
        q_values = self.critic(state_action).squeeze(-1)  # shape: [num_critics, batch_size, ?]
        # Combine Q-values across the discrete branch using the provided mapping.
        return q_values


class EDAC(RLAgent):  # Assuming RLAgent is defined elsewhere and imported.
    def __init__(
            self,
            actor: Actor,
            actor_optimizer: torch.optim.Optimizer,
            critic: VectorizedCritic,
            critic_optimizer: torch.optim.Optimizer,
            batch_size: int,
            targ_entropy_scale_cont: float,
            targ_entropy_scale_disc: float,
            gamma: float = 0.99,
            tau: float = 0.005,
            eta: float = 1.0,
            alpha_learning_rate: float = 1e-4,
            device: str = "cpu",
    ):
        self.device = device

        self.actor = actor
        self.critic = critic
        with torch.no_grad():
            self.target_critic = deepcopy(self.critic)

        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer

        self.tau = tau
        self.gamma = gamma
        self.eta = eta

        # Adaptive alpha setup
        self.target_entropy_scale_cont = targ_entropy_scale_cont
        self.target_entropy_scale_disc = targ_entropy_scale_disc
        self.target_entropy_cont = targ_entropy_scale_cont#-targ_entropy_scale_cont * float(self.actor.cont_action_dim)
        self.target_entropy_disc = targ_entropy_scale_disc#targ_entropy_scale_disc * float(math.log(self.actor.disc_action_dim))
        print(self.target_entropy_cont)
        print(self.target_entropy_disc)
        self.log_alpha_cont = torch.tensor([0.0], dtype=torch.float32, device=self.device, requires_grad=True)
        self.alpha_optimizer_cont = torch.optim.Adam([self.log_alpha_cont], lr=alpha_learning_rate)
        self.alpha_cont = self.log_alpha_cont.exp().detach()

        self.log_alpha_disc = torch.tensor(
            [0.0], dtype=torch.float32, device=self.device, requires_grad=True
        )
        self.alpha_optimizer_disc = torch.optim.Adam([self.log_alpha_disc], lr=alpha_learning_rate)
        self.alpha_disc = self.log_alpha_disc.exp().detach()
        self.batch_size = batch_size

    def get_action(self, state, deterministic: bool, **kwargs) -> torch.Tensor:

        if deterministic:
            self.actor.training = False
        else:
            self.actor.training = True
        return self.actor.get_action(state=state)

    def _alpha_loss(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            _, action_log_prob_cont, action_log_prob_disc, prob_disc = self.actor(state)

        loss_cont = (-self.log_alpha_cont * prob_disc * (
                prob_disc * action_log_prob_cont + self.target_entropy_cont)).sum(-1).mean()
        loss_disc = (-self.log_alpha_disc * prob_disc * (action_log_prob_disc + self.target_entropy_disc)).sum(
            -1).mean()
        return loss_cont, loss_disc

    def _actor_loss(self, state: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
        action, action_log_prob_cont, action_log_prob_disc, prob_d = self.actor(state)
        cont_action = get_continuous_action(action)
        disc_action = get_discrete_action(action)
        q_value_dist = self.critic(state, cont_action, disc_action)
        q_value_min = q_value_dist.min(0).values.unsqueeze(-1)
        q_value_std = q_value_dist.std(0).mean().item()
        batch_entropy = -action_log_prob_cont.mean().item() - action_log_prob_disc.mean().item()
        # Note: here the loss is weighted by the discrete probability.
        loss_disc = (prob_d * (self.alpha_disc * action_log_prob_disc - q_value_min)).sum(-1).mean()
        loss_cont = (prob_d * (self.alpha_cont * prob_d * action_log_prob_cont - q_value_min)).sum(-1).mean()
        loss = loss_cont + loss_disc
        return loss, batch_entropy, q_value_std

    def _critic_diversity_loss(
            self, state: torch.Tensor, discrete_action: torch.Tensor, cont_action: torch.Tensor,
    ) -> torch.Tensor:
        num_critics = self.critic.num_critics
        # Expand state and actions along a new dimension for critics.
        state_exp = state.unsqueeze(0).repeat_interleave(num_critics, dim=0)
        cont_action_exp = cont_action.unsqueeze(0).repeat_interleave(num_critics, dim=0)
        # For diversity loss, we want the actions to be differentiable. Make sure to require grad.
        cont_action_exp = cont_action_exp.detach().clone().requires_grad_(True)
        discrete_action_exp = discrete_action.unsqueeze(0).repeat_interleave(num_critics, dim=0)
        discrete_action_exp = discrete_action_exp.detach().clone().requires_grad_(True)
        # Get Q-values from the critic (each critic outputs a vector over discrete actions).
        q_ensemble = self.critic(state_exp, cont_action_exp, discrete_action_exp).unsqueeze(-1)  # shape: [num_critics, batch_size, disc_dim]
        # Combine Q-values using the soft (differentiable) discrete action.
        #q_ensemble = (q_ensemble * discrete_action_exp).sum(-1)  # shape: [num_critics, batch_size]

        # Compute gradients with respect to both continuous and discrete parts.
        grads = torch.autograd.grad(
            q_ensemble.sum(), [cont_action_exp, discrete_action_exp],
            retain_graph=True, create_graph=True
        )
        grad_cont = grads[0]  # shape: [num_critics, batch_size, cont_dim]
        grad_disc = grads[1]  # shape: [num_critics, batch_size, disc_dim]
        # Concatenate gradients along the last dimension.
        combined_grad = torch.cat([grad_cont, grad_disc], dim=-1)  # shape: [num_critics, batch_size, cont_dim+disc_dim]
        # Normalize each gradient vector.
        norm = torch.norm(combined_grad, p=2, dim=-1, keepdim=True) + 1e-10
        combined_grad = combined_grad / norm
        # Rearrange to shape [batch_size, num_critics, combined_dim]
        combined_grad = combined_grad.transpose(0, 1)
        # Compute pairwise dot products between critics for each sample.
        masks = torch.eye(num_critics, device=self.device).unsqueeze(0).repeat(combined_grad.shape[0], 1, 1)
        dot_products = combined_grad @ combined_grad.transpose(1, 2)
        dot_products = (1 - masks) * dot_products
        grad_loss = dot_products.sum(dim=(1, 2)).mean() / (num_critics - 1)
        return grad_loss

    def _critic_loss(
            self,
            state: torch.Tensor,
            discrete_action: torch.Tensor,
            cont_action: torch.Tensor,
            reward: torch.Tensor,
            next_state: torch.Tensor,
            done: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            next_action, next_action_log_prob_cont, next_action_log_prob_disc, next_action_prob_disc = self.actor(
                next_state)
            next_action_cont = get_continuous_action(next_action)
            next_action_disc = get_discrete_action(next_action)
            q_next = self.target_critic(next_state, next_action_cont, next_action_disc).min(0).values.unsqueeze(-1)
            q_next = next_action_prob_disc * (
                    q_next - self.alpha_cont * next_action_prob_disc * next_action_log_prob_cont - self.alpha_disc * next_action_log_prob_disc)

            v_next = q_next.sum(-1).unsqueeze(-1)
            q_target = reward + self.gamma * (1 - done) * v_next

        q_values = self.critic(state, cont_action, discrete_action)

        critic_loss = ((q_values - q_target.view(1, -1)) ** 2).mean(dim=1).sum(dim=0)

        diversity_loss = self._critic_diversity_loss(
            state=state,
            discrete_action=discrete_action,
            cont_action=cont_action
        )

        loss = critic_loss + self.eta * diversity_loss
        return loss

    def learn(self, buffer, **kwargs) -> Dict[str, float]:
        # Sample a batch from the replay buffer.
        batch = buffer.sample(self.batch_size)
        state = batch.observations
        hybrid_action = batch.actions
        reward = batch.rewards
        next_state = batch.next_observations
        done = batch.terminals

        discrete_actions = get_discrete_action(actions=hybrid_action)  # expected as one-hot vectors (hard)
        continuous_actions = get_continuous_action(actions=hybrid_action)

        # Update alpha parameters
        alpha_loss_cont, alpha_loss_disc = self._alpha_loss(state)
        self.alpha_optimizer_cont.zero_grad()
        alpha_loss_cont.backward()
        self.alpha_optimizer_disc.step()

        self.alpha_optimizer_disc.zero_grad()
        alpha_loss_disc.backward()
        self.alpha_optimizer_cont.step()

        self.alpha_cont = self.log_alpha_cont.exp().detach()
        self.alpha_disc = self.log_alpha_disc.exp().detach()

        # Actor update
        actor_loss, actor_batch_entropy, q_policy_std = self._actor_loss(state)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Critic update
        critic_loss = self._critic_loss(
            state=state,
            discrete_action=discrete_actions,
            cont_action=continuous_actions,
            reward=reward,
            next_state=next_state,
            done=done
        )
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Soft-update of target critic network
        with torch.no_grad():
            soft_update(self.target_critic, self.critic, tau=self.tau)

        update_info = {
            "alpha_loss_cont": alpha_loss_cont.item(),
            "alpha_loss_disc": alpha_loss_disc.item(),
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "batch_entropy": actor_batch_entropy,
            "alpha_cont": self.alpha_cont.item(),
            "alpha_disc": self.alpha_disc.item(),
            "q_policy_std": q_policy_std,
        }
        return update_info

    def eval(self, buffer: ReplayBuffer, **kwargs):
        return {
        }


def train(dataset_config_path, config, device):
    config = TrainConfig(**config)
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

    # Actor & Critic setup
    actor = Actor(
        state_dim=state_dim,
        cont_action_dim=continuous_action_dim,
        discrete_act_dim=discrete_action_dim,
        hidden_dims=config.hidden_dims,
        max_action=config.max_action,
        log_std_max=config.log_std_max,
        log_std_min=config.log_std_min
    )
    actor.to(device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_learning_rate)

    critic = VectorizedCritic(
        state_dim=state_dim,
        action_dim_cont=continuous_action_dim,
        action_dim_disc=discrete_action_dim,
        hidden_dims=config.hidden_dims,
        num_critics=config.num_critics
    )
    critic.to(device)
    critic_optimizer = torch.optim.Adam(
        critic.parameters(), lr=config.critic_learning_rate
    )

    wandb_init(
        config=config.model_dump(),
        dataset_config=dataset_config.model_dump(),
        pre_processing_metadata=pre_processing_config.model_dump()
    )

    agent = EDAC(
        actor=actor,
        actor_optimizer=actor_optimizer,
        critic=critic,
        critic_optimizer=critic_optimizer,
        gamma=config.gamma,
        tau=config.tau,
        eta=config.eta,
        alpha_learning_rate=config.alpha_learning_rate,
        device=device,
        batch_size=config.batch_size,
        targ_entropy_scale_cont=config.targ_entropy_scale_cont,
        targ_entropy_scale_disc=config.targ_entropy_scale_disc

    )
    trainer = AgentTrainer(agent=agent, train_config=config)
    trainer.train(buffer=train_buffer, eval_buffer=eval_buffer)

    wandb.finish()
    return config.experiment_path


if __name__ == "__main__":
    load_dotenv()
    train(
        dataset_config_path=os.getenv('DATASET_CONFIG_PATH'),
        config=load_yaml(os.getenv('HYBRID_EDAC_CONFIG_PATH')),
        device=os.getenv('DEVICE', default='cpu')
    )

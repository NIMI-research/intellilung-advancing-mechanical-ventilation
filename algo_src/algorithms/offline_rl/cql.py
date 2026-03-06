import itertools
from copy import deepcopy
from dataclasses import field
from os import getenv
from os.path import join
from random import random, choices
import numpy as np
import torch
import wandb
from dotenv import load_dotenv
from torch import Tensor
from torch.nn.utils import clip_grad_norm_

from actions.discrete_actions import get_action_size, get_possible_actions, get_bins_per_action_dim
from agents.base import RLAgent
from agents.configs import TrainerExperimentConfig
from agents.train import AgentTrainer
from dataset.buffer import ReplayBuffer
import torch.optim as optim
import torch.nn.functional as F

from dataset.config import load_dataset_config
from dataset.load import load_dataset_to_buffer
from dataset.pre_processing_configs import get_pre_processing_configs
from network.mlp import create_mlp
from network.update import soft_update
from utils.files import save_json, load_yaml
from utils.wandb import wandb_init


class CQLConfig(TrainerExperimentConfig):
    lr: float = 5e-5
    gamma: float = 0.996
    batch_size: int = 128
    eval_batch_size: int = 10_000
    alpha: float = 0.1
    hidden_layers: list = field(default_factory=lambda: [256, 256, 256, 256])
    use_dense_net: bool = False
    clip_grad_max_norm: float = 1.
    polyak_coef: float = 5e-3


class CQLAgent(RLAgent):
    def __init__(self, state_size, action_size, config: CQLConfig, index_to_hot_encoded_factorized_action,
                 device="cpu"):
        self.config = config
        self.batch_size = config.batch_size
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.gamma = config.gamma
        self.cql_alpha = config.alpha
        self.polyak_coef = config.polyak_coef
        self.use_dense_net = config.use_dense_net
        self.clip_grad_max_norm = config.clip_grad_max_norm
        self.index_to_hot_encoded_factorized_action = index_to_hot_encoded_factorized_action.to(self.device)

        self.critic_1 = create_mlp(
            input_dim=state_size,
            hidden_dims=config.hidden_layers,
            output_dim=action_size
        ).to(self.device)
        self.target_critic_1 = deepcopy(self.critic_1).to(device)

        self.critic_2 = create_mlp(
            input_dim=state_size,
            hidden_dims=config.hidden_layers,
            output_dim=action_size
        ).to(self.device)
        self.target_critic_2 = deepcopy(self.critic_2).to(device)

        params = list(self.critic_1.parameters()) + list(self.critic_2.parameters())
        self.optimizer = optim.Adam(params=params, lr=config.lr)
        print(self.critic_1)

    def get_action(self, state, **kwargs):
        state = Tensor(state).reshape(-1, self.state_size).to(self.device)
        self.critic_1.eval()
        with torch.no_grad():
            action_values = self.get_q_values(critic=self.critic_1, states=state)
            action = action_values.argmax(dim=1)
            action = self.index_to_hot_encoded_factorized_action[action]
        self.critic_1.train()

        return action

    def learn(self, buffer: ReplayBuffer, **kwargs):
        self.critic_1.train()
        self.critic_2.train()
        batch = buffer.sample(batch_size=self.batch_size)
        states = batch.observations
        actions = batch.actions
        next_states = batch.next_observations
        rewards = batch.rewards
        dones = batch.terminals

        with torch.no_grad():
            target_q_1 = self.get_q_values(critic=self.target_critic_1, states=next_states)
            target_q_2 = self.get_q_values(critic=self.target_critic_2, states=next_states)

            target_q_values = torch.max(
                torch.min(
                    target_q_1,
                    target_q_2
                ),
                dim=1
            ).values

            target_q_values = target_q_values.unsqueeze(1)
            td_target = rewards + (self.gamma * target_q_values * (1 - dones))

        q1_values = self.get_q_values(critic=self.critic_1, states=states)
        q1_predicted = q1_values.gather(1, actions)
        q2_values = self.get_q_values(critic=self.critic_2, states=states)
        q2_predicted = q2_values.gather(1, actions)

        conservative_loss_1 = (torch.logsumexp(q1_values, dim=1, keepdim=True) - q1_predicted.detach()).mean()
        conservative_loss_2 = (torch.logsumexp(q2_values, dim=1, keepdim=True) - q2_predicted.detach()).mean()

        conservative_loss_1 = self.cql_alpha * conservative_loss_1
        conservative_loss_2 = self.cql_alpha * conservative_loss_2

        critic_1_loss = F.mse_loss(q1_predicted, td_target)
        critic_2_loss = F.mse_loss(q2_predicted, td_target)

        cql_loss = conservative_loss_1 + conservative_loss_2 + 0.5 * (critic_1_loss + critic_2_loss)

        self.optimizer.zero_grad()
        cql_loss.backward()
        clip_grad_norm_(self.critic_1.parameters(), self.clip_grad_max_norm)
        clip_grad_norm_(self.critic_2.parameters(), self.clip_grad_max_norm)
        self.optimizer.step()

        self.target_update()

        loss_dict = dict(
            cql_loss=cql_loss.detach().item(),
            conservative_loss_1=conservative_loss_1.item(),
            critic_1_loss=critic_1_loss.item(),
            conservative_loss_2=conservative_loss_2.item(),
            critic_2_loss=critic_2_loss.item()
        )

        return loss_dict

    def eval(self, buffer: ReplayBuffer, **kwargs):
        self.critic_1.eval()
        self.critic_2.eval()
        batch = buffer.sample_all()

        batch_size = batch.observations.shape[0]
        total_cql_loss = 0
        total_critic_1_loss = 0
        total_conservative_loss_1 = 0
        total_critic_2_loss = 0
        total_conservative_loss_2 = 0
        total_q_value = 0

        mini_batch_size = self.config.eval_batch_size
        batch_range = range(0, batch_size, mini_batch_size)
        for i in batch_range:
            states_mini_batch = batch.observations[i: i + mini_batch_size]
            next_states_mini_batch = batch.next_observations[i: i + mini_batch_size]
            actions_mini_batch = batch.actions[i: i + mini_batch_size]
            dones_mini_batch = batch.terminals[i: i + mini_batch_size]
            rewards_mini_batch = batch.rewards[i: i + mini_batch_size]
            with torch.no_grad():
                q1_values = self.get_q_values(critic=self.critic_1, states=states_mini_batch)
                q2_values = self.get_q_values(critic=self.critic_2, states=states_mini_batch)
                q1_predicted = q1_values.gather(1, actions_mini_batch)
                q2_predicted = q2_values.gather(1, actions_mini_batch)

                target_q_values, _ = torch.max(
                    torch.min(
                        self.get_q_values(critic=self.target_critic_1, states=next_states_mini_batch),
                        self.get_q_values(critic=self.target_critic_2, states=next_states_mini_batch)
                    ),
                    dim=1
                )
                target_q_values = target_q_values.unsqueeze(1)
                td_target = rewards_mini_batch + (self.gamma * target_q_values * (1 - dones_mini_batch))

                critic_1_loss = F.mse_loss(q1_predicted, td_target)
                critic_2_loss = F.mse_loss(q2_predicted, td_target)

                conservative_loss_1 = self.cql_alpha * (
                        torch.logsumexp(q1_values, dim=1, keepdim=True) - q1_predicted.detach()).mean()
                conservative_loss_2 = self.cql_alpha * (
                        torch.logsumexp(q2_values, dim=1, keepdim=True) - q2_predicted.detach()).mean()

                cql_loss = conservative_loss_1 + conservative_loss_2 + 0.5 * (critic_1_loss + critic_2_loss)

                total_q_value += q1_predicted.sum().item()

                total_conservative_loss_1 += conservative_loss_1.item()
                total_conservative_loss_2 += conservative_loss_2.item()
                total_critic_1_loss += critic_1_loss.item()
                total_critic_2_loss += critic_2_loss.item()
                total_cql_loss += cql_loss.item()

        len_batch_range = len(batch_range)
        loss_dict = dict(
            cql_loss=total_cql_loss / len_batch_range,
            conservative_loss_1=total_conservative_loss_1 / len_batch_range,
            conservative_loss_2=total_conservative_loss_2 / len_batch_range,
            critic_1_loss=total_critic_1_loss / len_batch_range,
            critic_2_loss=total_critic_2_loss / len_batch_range,
            q_value=total_q_value / batch_size
        )

        return loss_dict

    def get_q_values(self, critic, states):
        return critic(states)

    def target_update(self):
        soft_update(target=self.target_critic_1, source=self.critic_1, tau=self.polyak_coef)
        soft_update(target=self.target_critic_2, source=self.critic_2, tau=self.polyak_coef)


def get_specific_action_q_value(q_value, actions):
    values = q_value * actions
    value = values.sum(dim=1).unsqueeze(-1)
    return value


def project_factored_q_to_actions(q_factored, index_to_hot_factored_action):
    return q_factored @ index_to_hot_factored_action.T


class FactoredCQLAgent(CQLAgent):

    def get_action(self, state, **kwargs):
        state = torch.tensor(state, device=self.device).reshape(-1, self.state_size)
        self.critic_1.eval()
        with torch.no_grad():
            action_values = self.get_q_values(critic=self.critic_1, states=state)
            action_index = action_values.argmax(dim=1)
            action = self.index_to_hot_encoded_factorized_action[action_index]
        self.critic_1.train()

        return action

    def save_weights(self, weights_save_path):
        policy_components_dict = {
            'critic_1': self.critic_1.state_dict(),
            'index_to_hot_encoded_factorized_action': self.index_to_hot_encoded_factorized_action
        }
        torch.save(policy_components_dict, weights_save_path)

    def learn(self, buffer: ReplayBuffer, **kwargs):
        self.critic_1.train()
        self.critic_2.train()
        batch = buffer.sample(batch_size=self.batch_size)
        states = batch.observations
        actions = batch.actions
        next_states = batch.next_observations
        rewards = batch.rewards
        dones = batch.terminals

        with torch.no_grad():
            target_q_1_factored = self.target_critic_1(next_states)
            target_q_2_factored = self.target_critic_2(next_states)
            target_q_1 = project_factored_q_to_actions(target_q_1_factored, self.index_to_hot_encoded_factorized_action)
            target_q_2 = project_factored_q_to_actions(target_q_2_factored, self.index_to_hot_encoded_factorized_action)

            target_q_values = torch.max(
                torch.min(
                    target_q_1,
                    target_q_2
                ),
                dim=1
            ).values

            target_q_values = target_q_values.unsqueeze(1)
            td_target = rewards + (self.gamma * target_q_values * (1 - dones))

        q1_values_factored = self.critic_1(states)
        q1_values = project_factored_q_to_actions(q1_values_factored, self.index_to_hot_encoded_factorized_action)
        q1_predicted = get_specific_action_q_value(q1_values_factored, actions)

        q2_values_factored = self.critic_2(states)
        q2_values = project_factored_q_to_actions(q2_values_factored, self.index_to_hot_encoded_factorized_action)
        q2_predicted = get_specific_action_q_value(q2_values_factored, actions)

        conservative_loss_1 = (torch.logsumexp(q1_values, dim=1, keepdim=True) - q1_predicted.detach()).mean()
        conservative_loss_2 = (torch.logsumexp(q2_values, dim=1, keepdim=True) - q2_predicted.detach()).mean()

        conservative_loss_1 = self.cql_alpha * conservative_loss_1
        conservative_loss_2 = self.cql_alpha * conservative_loss_2

        critic_1_loss = F.mse_loss(q1_predicted, td_target)
        critic_2_loss = F.mse_loss(q2_predicted, td_target)

        cql_loss = conservative_loss_1 + conservative_loss_2 + 0.5 * (critic_1_loss + critic_2_loss)

        self.optimizer.zero_grad()
        cql_loss.backward()
        clip_grad_norm_(self.critic_1.parameters(), self.clip_grad_max_norm)
        clip_grad_norm_(self.critic_2.parameters(), self.clip_grad_max_norm)
        self.optimizer.step()

        self.target_update()

        loss_dict = dict(
            cql_loss=cql_loss.detach().item(),
            conservative_loss_1=conservative_loss_1.item(),
            critic_1_loss=critic_1_loss.item(),
            conservative_loss_2=conservative_loss_2.item(),
            critic_2_loss=critic_2_loss.item()
        )

        return loss_dict

    def eval(self, buffer: ReplayBuffer, **kwargs):
        self.critic_1.eval()
        self.critic_2.eval()
        batch = buffer.sample_all()

        batch_size = batch.observations.shape[0]
        total_cql_loss = 0
        total_critic_1_loss = 0
        total_conservative_loss_1 = 0
        total_critic_2_loss = 0
        total_conservative_loss_2 = 0
        total_q_value = 0

        mini_batch_size = self.config.eval_batch_size
        batch_range = range(0, batch_size, mini_batch_size)
        for i in batch_range:
            states_mini_batch = batch.observations[i: i + mini_batch_size]
            next_states_mini_batch = batch.next_observations[i: i + mini_batch_size]
            actions_mini_batch = batch.actions[i: i + mini_batch_size]
            dones_mini_batch = batch.terminals[i: i + mini_batch_size]
            rewards_mini_batch = batch.rewards[i: i + mini_batch_size]
            with torch.no_grad():
                q1_factored = self.critic_1(states_mini_batch)
                q1_values = project_factored_q_to_actions(q1_factored, self.index_to_hot_encoded_factorized_action)
                q1_predicted = get_specific_action_q_value(q1_factored, actions_mini_batch)

                q2_factored = self.critic_2(states_mini_batch)
                q2_values = project_factored_q_to_actions(q2_factored, self.index_to_hot_encoded_factorized_action)

                q2_predicted = get_specific_action_q_value(q2_factored, actions_mini_batch)

                target_q_values, _ = torch.max(
                    torch.min(
                        self.get_q_values(critic=self.target_critic_1, states=next_states_mini_batch),
                        self.get_q_values(critic=self.target_critic_2, states=next_states_mini_batch)
                    ),
                    dim=1
                )
                target_q_values = target_q_values.unsqueeze(1)
                td_target = rewards_mini_batch + (self.gamma * target_q_values * (1 - dones_mini_batch))

                critic_1_loss = F.mse_loss(q1_predicted, td_target)
                critic_2_loss = F.mse_loss(q2_predicted, td_target)

                conservative_loss_1 = self.cql_alpha * (
                        torch.logsumexp(q1_values, dim=1, keepdim=True) - q1_predicted).mean()
                conservative_loss_2 = self.cql_alpha * (
                        torch.logsumexp(q2_values, dim=1, keepdim=True) - q2_predicted).mean()

                cql_loss = conservative_loss_1 + conservative_loss_2 + 0.5 * (critic_1_loss + critic_2_loss)

                total_q_value += q1_predicted.sum().item()

                total_conservative_loss_1 += conservative_loss_1.item()
                total_conservative_loss_2 += conservative_loss_2.item()
                total_critic_1_loss += critic_1_loss.item()
                total_critic_2_loss += critic_2_loss.item()
                total_cql_loss += cql_loss.item()

        len_batch_range = len(batch_range)
        loss_dict = dict(
            cql_loss=total_cql_loss / len_batch_range,
            conservative_loss_1=total_conservative_loss_1 / len_batch_range,
            conservative_loss_2=total_conservative_loss_2 / len_batch_range,
            critic_1_loss=total_critic_1_loss / len_batch_range,
            critic_2_loss=total_critic_2_loss / len_batch_range,
            q_value=total_q_value / batch_size
        )

        return loss_dict

    def get_q_values(self, critic, states):
        values = critic(states)
        return values @ self.index_to_hot_encoded_factorized_action.T


def train(dataset_config_path, config, device):
    config = CQLConfig(**config)
    dataset_config = load_dataset_config(dataset_config_path=dataset_config_path,
                                         experiment_path=config.experiment_path)

    dataset_config_save_path = join(config.experiment_path, 'dataset_config.json')
    discrete_actions_file_path = dataset_config.discrete_actions_file_path
    pre_processing_metadata_save_path = join(config.experiment_path, 'pre_processing_metadata.json')
    pre_processing_config = get_pre_processing_configs(configs_id=dataset_config.dataset_type)
    save_json(data=dataset_config.model_dump(), path=dataset_config_save_path)
    save_json(data=pre_processing_config.model_dump(), path=pre_processing_metadata_save_path)

    buffer = load_dataset_to_buffer(dataset=dataset_config.train_dataset_split,
                                    dataset_configs=dataset_config,
                                    pre_process_configs=pre_processing_config, device=device)
    full_data_buffer = load_dataset_to_buffer(dataset=dataset_config.dataset,
                                              dataset_configs=dataset_config,
                                              pre_process_configs=pre_processing_config, device=device)
    state_dimension = buffer.observations.size(dim=1)
    action_size = get_action_size(
        pre_process_config=pre_processing_config,
        discrete_actions_file_path=discrete_actions_file_path,
        factored_actions=True
    )

    index_to_hot_encoded_factorized_action = get_possible_actions(dataset_actions=full_data_buffer.actions)
    print(index_to_hot_encoded_factorized_action.shape)
    agent = FactoredCQLAgent(state_size=state_dimension, action_size=action_size, config=config, device=device,
                             index_to_hot_encoded_factorized_action=index_to_hot_encoded_factorized_action)
    wandb_init(
        config=config.model_dump(),
        dataset_config=dataset_config.model_dump(),
        pre_processing_metadata=pre_processing_config.model_dump()
    )
    trainer = AgentTrainer(agent=agent, train_config=config)
    trainer.train(buffer=buffer)
    wandb.finish()
    return config.experiment_path


def train_factored_only(dataset_config_path, config, device):
    config = CQLConfig(**config)
    dataset_config = load_dataset_config(dataset_config_path=dataset_config_path,
                                         experiment_path=config.experiment_path)

    dataset_config_save_path = join(config.experiment_path, 'dataset_config.json')
    discrete_actions_file_path = dataset_config.discrete_actions_file_path
    pre_processing_metadata_save_path = join(config.experiment_path, 'pre_processing_metadata.json')
    pre_processing_config = get_pre_processing_configs(configs_id=dataset_config.dataset_type)
    save_json(data=dataset_config.model_dump(), path=dataset_config_save_path)
    save_json(data=pre_processing_config.model_dump(), path=pre_processing_metadata_save_path)

    buffer = load_dataset_to_buffer(dataset=dataset_config.train_dataset_split,
                                    dataset_configs=dataset_config,
                                    pre_process_configs=pre_processing_config, device=device)
    full_data_buffer = load_dataset_to_buffer(dataset=dataset_config.dataset,
                                              dataset_configs=dataset_config,
                                              pre_process_configs=pre_processing_config, device=device)
    state_dimension = buffer.observations.size(dim=1)
    action_size = get_action_size(
        pre_process_config=pre_processing_config,
        discrete_actions_file_path=discrete_actions_file_path,
        factored_actions=True
    )

    index_to_hot_encoded_factorized_action = get_all_possible_actions(
        pre_processing_config=pre_processing_config,
        discrete_action_bin_ranges=dataset_config.discrete_action_bin_ranges,
        device=device
    )
    print(index_to_hot_encoded_factorized_action.shape)
    agent = FactoredCQLAgent(state_size=state_dimension, action_size=action_size, config=config, device=device,
                             index_to_hot_encoded_factorized_action=index_to_hot_encoded_factorized_action)
    wandb_init(
        config=config.model_dump(),
        dataset_config=dataset_config.model_dump(),
        pre_processing_metadata=pre_processing_config.model_dump()
    )
    trainer = AgentTrainer(agent=agent, train_config=config)
    trainer.train(buffer=buffer)
    wandb.finish()
    return config.experiment_path


def train_cql(dataset_config_path, config, device):
    config = CQLConfig(**config)
    dataset_config = load_dataset_config(dataset_config_path=dataset_config_path,
                                         experiment_path=config.experiment_path)

    dataset_config_save_path = join(config.experiment_path, 'dataset_config.json')
    discrete_actions_file_path = dataset_config.discrete_actions_file_path
    pre_processing_metadata_save_path = join(config.experiment_path, 'pre_processing_metadata.json')
    pre_processing_config = get_pre_processing_configs(configs_id=dataset_config.dataset_type)
    save_json(data=dataset_config.model_dump(), path=dataset_config_save_path)
    save_json(data=pre_processing_config.model_dump(), path=pre_processing_metadata_save_path)

    buffer = load_dataset_to_buffer(dataset=dataset_config.train_dataset_split,
                                    dataset_configs=dataset_config,
                                    pre_process_configs=pre_processing_config, device=device)
    full_data_buffer = load_dataset_to_buffer(dataset=dataset_config.dataset,
                                              dataset_configs=dataset_config,
                                              pre_process_configs=pre_processing_config, device=device)
    state_dimension = buffer.observations.size(dim=1)

    index_to_hot_encoded_factorized_action = get_possible_actions(dataset_actions=full_data_buffer.actions)

    factored_action_to_index = {tuple(value.cpu().tolist()): idx for idx, value in
                                enumerate(index_to_hot_encoded_factorized_action)}
    action_size = index_to_hot_encoded_factorized_action.shape[0]

    actions = []
    for row in buffer.actions:
        row_tuple = tuple(row.cpu().tolist())
        actions.append(factored_action_to_index[row_tuple])

    buffer.actions = torch.tensor(actions, dtype=torch.long, device=device).reshape(-1, 1)

    print(index_to_hot_encoded_factorized_action.shape)
    agent = CQLAgent(state_size=state_dimension, action_size=action_size, config=config, device=device,
                     index_to_hot_encoded_factorized_action=index_to_hot_encoded_factorized_action)
    wandb_init(
        config=config.model_dump(),
        dataset_config=dataset_config.model_dump(),
        pre_processing_metadata=pre_processing_config.model_dump()
    )
    trainer = AgentTrainer(agent=agent, train_config=config)
    trainer.train(buffer=buffer)
    wandb.finish()
    return config.experiment_path


def get_all_possible_actions(pre_processing_config, discrete_action_bin_ranges, device):
    list_of_actions = pre_processing_config.get_list_of_actions()
    bins_per_action_dim = get_bins_per_action_dim(
        actions_ranges=discrete_action_bin_ranges,
        list_of_actions=list_of_actions,
        vent_mode_conditional_null_bins=pre_processing_config.vent_mode_action_masking
    )
    print(bins_per_action_dim)
    total_dims = sum(bins_per_action_dim)
    all_combinations = list(itertools.product(*[range(b) for b in bins_per_action_dim]))
    n_combinations = len(all_combinations)

    # Initialize one-hot tensor
    one_hot = torch.zeros((n_combinations, total_dims), device=device)

    # Fill one-hot tensor
    for idx, combo in enumerate(all_combinations):
        offset = 0
        for dim, bin_idx in enumerate(combo):
            one_hot[idx, offset + bin_idx] = 1
            offset += bins_per_action_dim[dim]

    return one_hot


def train_cql_no_addons(dataset_config_path, config, device):
    config = CQLConfig(**config)
    dataset_config = load_dataset_config(dataset_config_path=dataset_config_path,
                                         experiment_path=config.experiment_path)

    dataset_config_save_path = join(config.experiment_path, 'dataset_config.json')
    pre_processing_metadata_save_path = join(config.experiment_path, 'pre_processing_metadata.json')
    pre_processing_config = get_pre_processing_configs(configs_id=dataset_config.dataset_type)
    save_json(data=dataset_config.model_dump(), path=dataset_config_save_path)
    save_json(data=pre_processing_config.model_dump(), path=pre_processing_metadata_save_path)

    buffer = load_dataset_to_buffer(dataset=dataset_config.train_dataset_split,
                                    dataset_configs=dataset_config,
                                    pre_process_configs=pre_processing_config, device=device)

    state_dimension = buffer.observations.size(dim=1)

    index_to_hot_encoded_factorized_action = get_all_possible_actions(
        pre_processing_config=pre_processing_config,
        discrete_action_bin_ranges=dataset_config.discrete_action_bin_ranges,
        device=device
    )

    print('Action Space Shape:', index_to_hot_encoded_factorized_action.shape)

    factored_action_to_index = {tuple(value.cpu().tolist()): idx for idx, value in
                                enumerate(index_to_hot_encoded_factorized_action)}
    action_size = index_to_hot_encoded_factorized_action.shape[0]

    actions = []
    for row in buffer.actions:
        row_tuple = tuple(row.cpu().tolist())
        actions.append(factored_action_to_index[row_tuple])

    buffer.actions = torch.tensor(actions, dtype=torch.long, device=device).reshape(-1, 1)

    print(index_to_hot_encoded_factorized_action.shape)
    agent = CQLAgent(state_size=state_dimension, action_size=action_size, config=config, device=device,
                     index_to_hot_encoded_factorized_action=index_to_hot_encoded_factorized_action)
    wandb_init(
        config=config.model_dump(),
        dataset_config=dataset_config.model_dump(),
        pre_processing_metadata=pre_processing_config.model_dump()
    )
    trainer = AgentTrainer(agent=agent, train_config=config)
    trainer.train(buffer=buffer)
    wandb.finish()
    return config.experiment_path


if __name__ == "__main__":
    load_dotenv()
    train(
        dataset_config_path=getenv('DATASET_CONFIG_PATH'),
        config=load_yaml(getenv('CQL_CONFIG_PATH')),
        device=getenv('DEVICE', default='cpu')
    )

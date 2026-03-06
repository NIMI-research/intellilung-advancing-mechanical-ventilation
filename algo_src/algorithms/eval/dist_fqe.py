import copy
import glob
import os
from dataclasses import field
from os import getenv
from os.path import join, getmtime
from time import time
from typing import List, Dict

import torch
import wandb
from dotenv import load_dotenv
from torch import Tensor, cat, nn, optim, no_grad
from torch.nn.utils import clip_grad_norm_

from actions.discrete_actions import get_action_size
from agents.base import Agent, load_state_encoder
from agents.configs import EvalExpConfig, TrainerConfig, load_eval_configs
from agents.eval import AgentEvaluator
from agents.train import AgentTrainer
from dataset.buffer import ReplayBuffer
from dataset.config import load_dataset_config
from dataset.load import load_dataset_to_buffer, load_dataset_to_eval_buffer
from dataset.pre_processing_configs import load_pre_processing_configs
from network.mlp import create_mlp
from network.update import soft_update
from policy.base import BasePolicy, AgentPolicyWrapper
from utils.files import load_json, load_yaml
from utils.wandb import wandb_init

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"  # This fixes memory bug with pytorch for this script


class TrainConfig(EvalExpConfig, TrainerConfig):
    discount: float = 0.996
    lr: float = 5e-4
    hidden_dim: list = field(default_factory=lambda: [256, 256, 256, 256])
    batch_size: int = 1280
    tau: float = 5e-3
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    # model params
    hidden_dims: List[int]

    n_atoms: int = 10


class DistCritic(nn.Module):
    def __init__(
            self, state_dim: int, action_dim: int, hidden_dims: List[int], n_atoms: int
    ):
        super().__init__()
        self.critic = create_mlp(input_dim=state_dim + action_dim, output_dim=n_atoms,
                                 hidden_dims=hidden_dims)
        self.n_atoms = n_atoms

        self.register_buffer(
            "_quantiles", torch.arange(0, self.n_atoms + 1) / self.n_atoms
        )
        self.register_buffer(
            "_quantile_midpoints",
            ((self._quantiles[1:] + self._quantiles[:-1]) / 2)
            .unsqueeze(0)
            .unsqueeze(0),
        )

        print(self.critic)

    @property
    def quantiles(self) -> Tensor:
        return self._quantiles

    @property
    def quantile_midpoints(self) -> Tensor:
        return self._quantile_midpoints

    def forward(self, state: Tensor, action: Tensor) -> Tensor:
        state_action = cat([state, action], dim=1)
        return self.critic(state_action)


def compute_elementwise_huber_loss(input_errors: Tensor, kappa: float = 1.0) -> Tensor:
    huber_loss = torch.where(
        torch.abs(input_errors) <= kappa,
        0.5 * (input_errors.pow(2)),
        kappa * (torch.abs(input_errors) - (0.5 * kappa)),
    )
    return huber_loss


class DistFQE(Agent):

    def __init__(
            self,
            critic: DistCritic,
            critic_optimizer: optim.Optimizer,
            batch_size: int,
            grad_clip: float,
            discount: float,
            tau: float,
            device: str,
            state_encoder=None,
            policy: BasePolicy = None
    ):
        self.policy = policy

        self.critic = critic
        self.critic_target = copy.deepcopy(critic)
        self.critic_target.eval()
        self.critic_optimizer = critic_optimizer
        self.batch_size = batch_size
        self.discount = discount
        self.tau = tau
        self.grad_clip = grad_clip
        self.device = device
        self.state_encoder = state_encoder

    def learn(self, buffer: ReplayBuffer, **kwargs) -> Dict[str, float]:
        self.critic.train()
        batch = buffer.sample(self.batch_size)

        if self.state_encoder is not None:
            with torch.no_grad():
                state = self.state_encoder.encode(batch.observations)
                next_state = self.state_encoder.encode(batch.next_observations)
        else:
            state = batch.observations
            next_state = batch.next_observations

        action = batch.actions
        reward = batch.rewards
        done = batch.terminals
        next_action = batch.next_actions
        not_done = 1 - done

        with no_grad():
            # Compute the target Q value
            target_q_quantiles = reward + not_done * self.discount * self.critic_target(next_state, next_action)

        current_q_quantiles = self.critic(state, action)

        pairwise_quantile_loss = target_q_quantiles.unsqueeze(2) - current_q_quantiles.unsqueeze(1)

        with torch.no_grad():
            asymmetric_weight = torch.abs(
                self.critic.quantile_midpoints - (pairwise_quantile_loss < 0).float()
            )

        huber_loss = compute_elementwise_huber_loss(pairwise_quantile_loss)
        quantile_huber_loss = asymmetric_weight * huber_loss
        quantile_bellman_loss = quantile_huber_loss.sum(dim=1).mean()

        self.critic_optimizer.zero_grad()
        quantile_bellman_loss.backward()
        if self.grad_clip:
            clip_grad_norm_(parameters=self.critic.parameters(), max_norm=self.grad_clip)
        self.critic_optimizer.step()

        soft_update(target=self.critic_target, source=self.critic, tau=self.tau)

        return {
            "loss": torch.abs(target_q_quantiles - current_q_quantiles).mean().item()
        }

    def eval(self, eval_buffer: ReplayBuffer, **kwargs):
        self.critic.eval()
        with torch.no_grad():
            batch = eval_buffer.sample_initial()

            state = batch.observations
            action = batch.actions

            batch_size = state.shape[0]
            mini_batch_size = self.batch_size

            q_values = torch.zeros(batch_size, device=self.device)
            for i in range(0, batch_size, mini_batch_size):
                if self.state_encoder is not None:
                    with torch.no_grad():
                        mb_state = self.state_encoder.encode(state[i: i + mini_batch_size])
                else:
                    mb_state = state[i: i + mini_batch_size]

                if self.policy:
                    policy_action = self.policy.select_action(obs=state[i: i + mini_batch_size], deterministic=True)
                else:
                    policy_action = action[i: i + mini_batch_size]
                q_values[i: i + mini_batch_size] = self.critic(state=mb_state, action=policy_action).mean(
                    dim=-1).flatten()

        return {
            'value_mean': q_values.mean().item(),
            'value_std': q_values.std().item()
        }

    def get_q_value(self, state: Tensor, action: Tensor) -> Tensor:
        """
        Used to get q values post training
        """

        if self.state_encoder is not None:
            with torch.no_grad():
                state = self.state_encoder.encode(state)

        return self.critic(state=state, action=action).mean(dim=-1).flatten()


def train_fqe_agent(dataset_config, pre_processing_configs, device, checkpoint, fqe_config, policy_id, state_encoder, concat_raw_states_with_embeddings):
    discrete_actions_file_path = dataset_config.discrete_actions_file_path
    buffer = load_dataset_to_buffer(dataset=dataset_config.train_dataset_split,
                                    dataset_configs=dataset_config,
                                    pre_process_configs=pre_processing_configs, device=device,
                                    flatten_hybrid_actions=True)
    test_buffer = load_dataset_to_buffer(dataset=dataset_config.test_dataset_split,
                                         dataset_configs=dataset_config,
                                         pre_process_configs=pre_processing_configs, device=device,
                                         flatten_hybrid_actions=True)

    state_dimension = buffer.observations.size(dim=1)

    if state_encoder is not None:
        if concat_raw_states_with_embeddings:
            state_dimension += state_encoder.state_emb_size()
        else:
            state_dimension = state_encoder.state_emb_size()


    action_size = get_action_size(
        pre_process_config=pre_processing_configs,
        discrete_actions_file_path=discrete_actions_file_path,
        factored_actions=True
    )

    policy = AgentPolicyWrapper(agent_path=checkpoint)

    train_data_buffer = load_dataset_to_eval_buffer(buffer=buffer, policy=policy, device=device)

    test_data_buffer = load_dataset_to_eval_buffer(buffer=test_buffer, policy=policy, device=device)

    critic = DistCritic(
        state_dim=state_dimension,
        action_dim=action_size,
        hidden_dims=fqe_config.hidden_dim,
        n_atoms=fqe_config.n_atoms
    ).to(device)
    agent = DistFQE(
        critic=critic,
        critic_optimizer=optim.AdamW(params=critic.parameters(), lr=fqe_config.lr,
                                     weight_decay=fqe_config.weight_decay),
        policy=policy,
        batch_size=fqe_config.batch_size,
        grad_clip=fqe_config.grad_clip,
        discount=fqe_config.discount,
        tau=fqe_config.tau,
        device=device,
        state_encoder=state_encoder
    )

    trainer = AgentTrainer(agent=agent, train_config=fqe_config)
    trainer.train(
        panel_name_infix=policy_id,
        agent=agent,
        buffer=train_data_buffer,
        eval_buffer=test_data_buffer
    )
    evaluator = AgentEvaluator(experiment_path=fqe_config.experiment_path)
    evaluator.eval(checkpoint_id=policy_id, agent=trainer.agent, panel_name='FQE_eval', eval_buffer=test_data_buffer)


def train(fqe_config_path, experiment_path, device):
    trainer_config = load_json(join(experiment_path, 'config.json'))
    fqe_config = load_eval_configs(
        trainer_config=trainer_config,
        eval_config=load_yaml(fqe_config_path),
        config_class=TrainConfig
    )

    dataset_config_path = join(experiment_path, 'dataset_config.json')
    dataset_config = load_dataset_config(dataset_config_path=dataset_config_path, experiment_path=experiment_path)
    pre_processing_config_path = join(experiment_path, 'pre_processing_metadata.json')
    pre_processing_configs = load_pre_processing_configs(pre_process_configs_path=pre_processing_config_path)
    wandb_init(
        config=fqe_config.model_dump(),
        dataset_config=dataset_config.model_dump(),
        pre_processing_metadata=pre_processing_configs.model_dump()
    )

    checkpoints_path = join(experiment_path, 'checkpoints')

    checkpoints = glob.glob(f"{checkpoints_path}/*.pkl")
    checkpoints.sort(key=getmtime, reverse=True)
    iterations = [int(checkpoint.split('/')[-1].split('.')[0]) for checkpoint in checkpoints]

    state_encoder_path = None#trainer_config.get('state_encoder_path', None) # disabled for results comparison compatibility
    concat_raw_states_with_embeddings = trainer_config['concat_raw_states_with_embeddings']
    if state_encoder_path is not None:
        state_encoder = load_state_encoder(state_encoder_path, concat_raw_states_with_embeddings)
    else:
        state_encoder = None

    for i, (checkpoint, iteration) in enumerate(zip(checkpoints, iterations)):
        start = time()

        train_fqe_agent(
            dataset_config=dataset_config,
            pre_processing_configs=pre_processing_configs,
            device=device,
            checkpoint=checkpoint,
            fqe_config=fqe_config,
            policy_id=iteration,
            state_encoder=state_encoder,
            concat_raw_states_with_embeddings=concat_raw_states_with_embeddings
        )

        print("Trainings Completed: ", i + 1, "/", len(checkpoints))
        print(time() - start)
    wandb.finish()


if __name__ == "__main__":
    load_dotenv()

    train(
        fqe_config_path=os.getenv('DIST_FQE_CONFIG_PATH'),
        experiment_path=getenv('EXPERIMENT_PATH'),
        device=getenv('DEVICE', default='cpu')
    )

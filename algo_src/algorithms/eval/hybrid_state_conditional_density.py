import math
from os import getenv
from os.path import join
from typing import List, Dict
import torch
import wandb
from dotenv import load_dotenv
from torch import nn

from torch.nn import Module
from torch.nn.functional import cross_entropy
from torch.utils.data import Dataset

from actions.discrete_actions import get_discrete_action_size, get_continuous_action_size
from actions.hybrid import get_continuous_action, get_discrete_action
from agents.base import Agent
from agents.configs import TrainerExperimentConfig
from agents.train import AgentTrainer
from dataset.buffer import ReplayBuffer
from dataset.config import load_dataset_config
from dataset.load import load_dataset_to_buffer

from dataset.pre_processing_configs import get_pre_processing_configs
from network.mlp import create_mlp
from utils.files import save_json, load_yaml
from utils.wandb import wandb_init


class AETrainConfig(TrainerExperimentConfig):
    name: str = "Hybrid-Autencoder"
    job_type: str = "OOD_Algo_Train"
    lr: float
    batch_size: int
    weight_decay: float = 1e-2
    layers: List[int]


class Policy(nn.Module):

    def __init__(self, state_dim, cont_action_dim, disc_action_dimension,
                 layers):
        super().__init__()

        self.trunk = create_mlp(input_dim=state_dim, hidden_dims=layers)

        self.cont_action_mean = nn.Linear(layers[-1], cont_action_dim)
        self.cont_action_log_var = nn.Linear(layers[-1], cont_action_dim)

        self.disc_action_out = nn.Linear(layers[-1], disc_action_dimension)

        # Clamp in log-variance space. With std = exp(0.5 * log_var),
        # log_var_min=-6 => min std ≈ exp(-3) ≈ 0.05; log_var_max=2 => max std ≈ exp(1) ≈ 2.72
        self.log_var_min = float(-math.log(2 * math.pi))  # ≈ -1.837877
        self.log_var_max = 2.0
        self.eps = 1e-6

    def forward(self, state):
        h = self.trunk(state)
        cont_action_mean = self.cont_action_mean(h)
        cont_action_log_var = torch.clamp(self.cont_action_log_var(h),
                                          self.log_var_min, self.log_var_max)
        cont_action_std = torch.exp(0.5 * cont_action_log_var) + self.eps

        cont_action_dist = torch.distributions.Normal(cont_action_mean, cont_action_std)
        disc_action_dist = torch.distributions.OneHotCategorical(
            logits=self.disc_action_out(h), validate_args=True
        )
        return cont_action_dist, disc_action_dist, cont_action_log_var


class DatasetPolicyAgent(Agent):
    def __init__(self, model: Module, optimizer, batch_size: int, gnll_eps: float = 1e-6, var_reg_weight=1e-3):
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        # Include constant term to match -Normal.log_prob numerics
        self._gnll = nn.GaussianNLLLoss(full=True, reduction='none', eps=gnll_eps)
        self.var_reg_weight = var_reg_weight

    @staticmethod
    def _per_example_reduce(x: torch.Tensor) -> torch.Tensor:
        """Reduce (B, D, ...) -> (B,) by averaging all non-batch dims."""
        if x.dim() == 1:
            return x
        return x.flatten(start_dim=1).mean(dim=-1)

    def learn(self, buffer: ReplayBuffer, **kwargs) -> Dict[str, float]:
        self.model.train()
        batch = buffer.sample(self.batch_size)
        log = {}

        state = batch.observations
        action = batch.actions
        cont_action = get_continuous_action(action)
        disc_action = get_discrete_action(action)

        cont_dist, disc_dist, cont_log_var = self.model(state=state)

        cont_var = (cont_dist.scale ** 2).clamp_min(self._gnll.eps)
        cont_nll = self._gnll(input=cont_dist.mean, target=cont_action, var=cont_var)
        cont_lp = self._per_example_reduce(cont_nll)

        logits = disc_dist.logits
        target_idx = disc_action.argmax(dim=-1)

        disc_lp = cross_entropy(logits, target_idx, reduction="none", label_smoothing=0.05)

        if disc_lp.dim() > 1:
            disc_lp = disc_lp.mean(dim=-1)

        # variance regularizer (encourage log_var near 0)
        var_reg = self.var_reg_weight * (cont_log_var ** 2).mean()

        loss = (cont_lp + disc_lp).mean() + var_reg

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        log["cont_action_loss"] = cont_lp.mean().item()
        log["disc_action_loss"] = disc_lp.mean().item()
        log["loss"] = loss.item()
        return log

    def eval(self, eval_buffer: ReplayBuffer, **kwargs) -> Dict[str, float]:
        self.model.eval()
        batch = eval_buffer.sample_all()
        state = batch.observations
        action = batch.actions

        losses, cont_losses, disc_losses, cont_abs_errs = [], [], [], []
        disc_accs = []  # <-- collect accuracy

        with torch.no_grad():
            for i in range(0, state.shape[0], self.batch_size):
                mb_state = state[i:i + self.batch_size]
                mb_action = action[i:i + self.batch_size]
                mb_cont = get_continuous_action(mb_action)
                mb_disc = get_discrete_action(mb_action)

                cont_dist, disc_dist, cont_log_var = self.model(state=mb_state)

                cont_var = (cont_dist.scale ** 2).clamp_min(self._gnll.eps)
                cont_nll = self._gnll(cont_dist.mean, mb_cont, cont_var)
                cont_lp = self._per_example_reduce(cont_nll)

                logits = disc_dist.logits
                target_idx = mb_disc.argmax(dim=-1)  # Long indices
                disc_lp = cross_entropy(logits, target_idx, reduction="none", label_smoothing=0.05)

                # ---- accuracy ----
                pred_idx = logits.argmax(dim=-1)
                acc = (pred_idx == target_idx).float()  # (B,)
                disc_accs.append(acc)

                batch_loss = cont_lp + disc_lp

                cont_abs = (mb_cont - cont_dist.mean).abs().mean(dim=-1)

                cont_losses.append(cont_lp)
                disc_losses.append(disc_lp)
                losses.append(batch_loss)
                cont_abs_errs.append(cont_abs)

        return {
            "loss": torch.cat(losses).mean().item(),
            "cont_action_loss": torch.cat(cont_losses).mean().item(),
            "disc_action_loss": torch.cat(disc_losses).mean().item(),
            "disc_accuracy": torch.cat(disc_accs).mean().item(),  # <-- new
            "absolute_cont_action_error": torch.cat(cont_abs_errs).mean().item(),
        }


def train(dataset_config_path, config, device):
    # Load and save configs
    dataset_config = load_dataset_config(dataset_config_path=dataset_config_path,
                                         experiment_path=config.experiment_path)
    dataset_type = dataset_config.dataset_type
    dataset_config_save_path = join(config.experiment_path, 'dataset_config.json')

    pre_processing_config_save_path = join(config.experiment_path, 'pre_processing_metadata.json')
    pre_process_configs = get_pre_processing_configs(configs_id=dataset_type)
    save_json(data=dataset_config.model_dump(), path=dataset_config_save_path)
    save_json(data=pre_process_configs.model_dump(), path=pre_processing_config_save_path,
              default_serialization=lambda x: x.__dict__)

    # Load dataset to dataloaders
    train_buffer = load_dataset_to_buffer(dataset=dataset_config.train_dataset_split,
                                          pre_process_configs=pre_process_configs, device=device,
                                          dataset_configs=dataset_config)
    eval_buffer = load_dataset_to_buffer(dataset=dataset_config.test_dataset_split,
                                         pre_process_configs=pre_process_configs, device=device,
                                         dataset_configs=dataset_config)

    action_space = pre_process_configs.action_space

    state_dim = train_buffer.observations.size(dim=1)
    discrete_action_dim = get_discrete_action_size(
        action_space=action_space,
        discrete_actions_ranges=dataset_config.discrete_action_bin_ranges,
        factored_actions=True,
        vent_mode_action_masking=pre_process_configs.vent_mode_action_masking
    )
    continuous_action_dim = get_continuous_action_size(
        action_space=action_space
    )

    model = Policy(
        state_dim=state_dim,
        cont_action_dim=continuous_action_dim,
        disc_action_dimension=discrete_action_dim,
        layers=config.layers
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    wandb_init(config=config.model_dump(),
               dataset_config=dataset_config.model_dump(),
               pre_processing_metadata=pre_process_configs.model_dump()
               )
    agent = DatasetPolicyAgent(
        model=model,
        optimizer=optimizer,
        batch_size=config.batch_size
    )
    trainer = AgentTrainer(agent=agent, train_config=config)
    trainer.train(buffer=train_buffer, eval_buffer=eval_buffer)
    wandb.finish()


if __name__ == '__main__':
    load_dotenv()

    train(
        dataset_config_path=getenv('DATASET_CONFIG_PATH'),
        config=AETrainConfig(**load_yaml(getenv('STATE_CONDITIONAL_DENSITY_CONFIG_PATH'))),
        device=getenv('DEVICE')
    )

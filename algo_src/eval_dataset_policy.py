from os import makedirs, getenv
from os.path import join

from torch import optim

import wandb
from dotenv import load_dotenv

from actions.discrete_actions import get_action_size
from agents.train import AgentTrainer
from algorithms.eval.dist_fqe import DistCritic, TrainConfig, DistFQE
from dataset.config import load_dataset_config
from dataset.load import load_behavior_policy_eval_dataset
from dataset.pre_processing_configs import get_pre_processing_configs

from utils.files import save_json, load_yaml
from utils.wandb import wandb_init


def train_fqe_on_dataset_policy(dataset_config_path, fqe_config_path, device, root_path=None):
    fqe_config = TrainConfig(**load_yaml(fqe_config_path))

    root_path = root_path if root_path is not None else 'logs'

    fqe_config.job_type = 'behavior_policy_eval'
    fqe_config.experiment_path = join(root_path, f'{fqe_config.name}-{fqe_config.group_id}')
    makedirs(fqe_config.experiment_path, exist_ok=True)
    dataset_config = load_dataset_config(dataset_config_path=dataset_config_path,
                                         experiment_path=fqe_config.experiment_path)

    dataset_type = dataset_config.dataset_type
    discrete_actions_file_path = dataset_config.discrete_actions_file_path

    dataset_config_save_path = join(fqe_config.experiment_path, 'dataset_config.json')
    pre_processing_config_save_path = join(fqe_config.experiment_path, 'pre_processing_metadata.json')
    pre_process_configs = get_pre_processing_configs(configs_id=dataset_type)
    save_json(data=dataset_config.model_dump(), path=dataset_config_save_path)
    save_json(data=pre_process_configs.model_dump(), path=pre_processing_config_save_path,
              default_serialization=lambda x: x.__dict__)

    wandb_init(
        config=fqe_config.model_dump(),
        dataset_config=dataset_config.model_dump(),
        pre_processing_metadata=pre_process_configs.model_dump()
    )

    buffer = load_behavior_policy_eval_dataset(dataset=dataset_config.train_dataset_split,
                                               dataset_configs=dataset_config,
                                               pre_process_configs=pre_process_configs,
                                               device=device, flatten_hybrid_actions=True)

    test_buffer = load_behavior_policy_eval_dataset(dataset=dataset_config.test_dataset_split,
                                               dataset_configs=dataset_config,
                                               pre_process_configs=pre_process_configs,
                                               device=device, flatten_hybrid_actions=True)

    state_dimension = buffer.observations.size(dim=1)

    action_size = get_action_size(
        pre_process_config=pre_process_configs,
        discrete_actions_file_path=discrete_actions_file_path,
        factored_actions=True
    )

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
        batch_size=fqe_config.batch_size,
        grad_clip=fqe_config.grad_clip,
        discount=fqe_config.discount,
        tau=fqe_config.tau,
        device=device
    )

    trainer = AgentTrainer(agent=agent, train_config=fqe_config)
    trainer.train(
        agent=agent,
        buffer=buffer,
        eval_buffer=test_buffer
    )

    wandb.finish()
    return fqe_config.experiment_path


if __name__ == "__main__":
    load_dotenv()

    train_fqe_on_dataset_policy(
        dataset_config_path=getenv('DATASET_CONFIG_PATH'),
        fqe_config_path=getenv('DIST_FQE_CONFIG_PATH'),
        device=getenv('DEVICE', default='cpu')
    )

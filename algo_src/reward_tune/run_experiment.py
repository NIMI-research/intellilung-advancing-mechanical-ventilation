import argparse
from os.path import join

import torch
import wandb

from actions.discrete_actions import get_action_size, get_possible_actions, get_continuous_action_size, \
    get_discrete_action_size
from actions.space import get_discrete_actions_list
from agents.train import AgentTrainer
from algorithms.offline_rl.cql import FactoredCQLAgent, CQLConfig
from algorithms.offline_rl.hybrid_iql import ImplicitQLearning, HybridPolicy, ValueFunction, TwinQ
from dataset.config import load_dataset_config
from dataset.load import load_dataset_to_buffer
from dataset.pre_processing_configs import get_pre_processing_configs, PreProcessingConfigs
from reward_tune.create_experiment import configs_classes
from utils.files import save_json, load_json
from utils.wandb import wandb_init
from algorithms.eval.dist_fqe import train as train_dist_fqe


def train_cql(dataset_config, config, pre_processing_config, device):
    dataset_config_save_path = join(config.experiment_path, 'dataset_config.json')
    discrete_actions_file_path = dataset_config.discrete_actions_file_path
    pre_processing_metadata_save_path = join(config.experiment_path, 'pre_processing_metadata.json')

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


def train_hybrid_iql(dataset_config, config, pre_processing_config, device):
    dataset_config_save_path = join(config.experiment_path, 'dataset_config.json')
    pre_processing_config_save_path = join(config.experiment_path, 'pre_processing_metadata.json')
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


def main(task_path):
    experiment_config = load_json(join(task_path, 'experiment_config.json'))
    if not experiment_config['finished']:
        algo = experiment_config['algo_config']['name']
        config_class = configs_classes[algo]
        algo_config = config_class(**experiment_config['algo_config'])
        dataset_config = load_dataset_config(dataset_config_path=experiment_config['dataset_config_path'],
                                             experiment_path=algo_config.experiment_path)
        pre_processing_config = get_pre_processing_configs(configs_id=dataset_config.dataset_type)
        pre_processing_config_dict = pre_processing_config.model_dump()
        pre_processing_config_dict['reward_function'] = experiment_config['reward_fn']
        pre_processing_config = PreProcessingConfigs(**pre_processing_config_dict)
        device = experiment_config['device']

        print("\n============================================================")
        print(f"\n========== Step 1: Training Policy ==========")
        print("\n============================================================")

        if algo == 'factored-CQL':

            experiment_path = train_cql(
                dataset_config=dataset_config,
                config=algo_config,
                pre_processing_config=pre_processing_config,
                device=device
            )
        else:
            experiment_path = train_hybrid_iql(
                dataset_config=dataset_config,
                config=algo_config,
                pre_processing_config=pre_processing_config,
                device=device
            )

        print("\n========================================================================")
        print("\n========== Step 2: Policy Evaluation using Distributional FQE ==========")
        print("\n========================================================================")

        train_dist_fqe(
            fqe_config_path=experiment_config['fqe_config_path'],
            experiment_path=experiment_path,
            device=device
        )

        experiment_config['finished'] = True
        save_json(data=experiment_config, path=join(task_path, 'experiment_config.json'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_root_path', type=str)
    parser.add_argument('--task_id')
    args = parser.parse_args()
    main(
        task_path=join(args.experiment_root_path, 'tasks', str(args.task_id))
    )

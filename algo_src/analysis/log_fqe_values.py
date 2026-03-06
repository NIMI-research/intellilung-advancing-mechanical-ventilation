import glob
from os import getenv
from os.path import join, getmtime

import pandas as pd
import torch
from dotenv import load_dotenv

from agents.base import Agent
from dataset.config import load_dataset_config
from dataset.load import load_dataset_to_buffer, get_policy_actions
from dataset.pre_processing_configs import load_pre_processing_configs
from policy.base import AgentPolicyWrapper
from utils.files import load_json


def get_fqe_values(states, actions, fqe_model_path, mini_batch_size):
    agent = Agent.load(fqe_model_path)
    batch_size = states.shape[0]
    q_values = torch.empty(batch_size, device=states.device)

    with torch.no_grad():
        for i in range(0, batch_size, mini_batch_size):
            mb_state = states[i: i + mini_batch_size]
            mb_policy_actions = actions[i: i + mini_batch_size]
            mb_q_values = agent.get_q_value(state=mb_state, action=mb_policy_actions).squeeze()
            q_values[i: i + mini_batch_size] = mb_q_values
    return q_values.detach().cpu().numpy()


def get_latest_checkpoint(checkpoint_folder):
    latest_checkpoint_paths = glob.glob(join(checkpoint_folder, '*.pkl'))
    latest_checkpoint_paths.sort(key=getmtime, reverse=True)
    return latest_checkpoint_paths[0]


def get_latest_fqe_path(experiment_path):
    fqe_paths = glob.glob(join(experiment_path, 'eval', '*fqe', '*'))
    fqe_paths.sort(key=getmtime, reverse=True)
    fqe_path = fqe_paths[0]
    return fqe_path


def load_fqe_model_paths(fqe_path):
    checkpoints_folders = glob.glob(join(fqe_path, '*', 'checkpoints'))

    fqe_model_paths = {}
    for checkpoint_folder in checkpoints_folders:
        checkpoint_number = checkpoint_folder.split('/')[-2]
        fqe_model_paths[checkpoint_number] = get_latest_checkpoint(checkpoint_folder=checkpoint_folder)

    return fqe_model_paths


def main(experiment_path, device, behaviour_policy_eval_path):
    dataset_config_path = join(experiment_path, 'dataset_config.json')
    dataset_config = load_dataset_config(dataset_config_path=dataset_config_path, experiment_path=experiment_path)

    pre_processing_config_path = join(experiment_path, 'pre_processing_metadata.json')
    pre_processing_configs = load_pre_processing_configs(pre_process_configs_path=pre_processing_config_path)

    behaviour_policy_pre_proces_config_path = join(behaviour_policy_eval_path, 'pre_processing_metadata.json')
    behaviour_policy_pre_proces_config = load_pre_processing_configs(behaviour_policy_pre_proces_config_path)

    buffer = load_dataset_to_buffer(dataset=dataset_config.test_dataset_split,
                                    dataset_configs=dataset_config,
                                    pre_process_configs=pre_processing_configs,
                                    device=device, flatten_hybrid_actions=True)

    checkpoints_path = join(experiment_path, 'checkpoints')
    policy_actions_save_path = join(experiment_path, 'policy_actions')
    checkpoints = glob.glob(f"{checkpoints_path}/*.pkl")
    checkpoints.sort(key=getmtime, reverse=True)
    iterations = [int(checkpoint.split('/')[-1].split('.')[0]) for checkpoint in checkpoints]

    behaviour_policy_fqe_model = get_latest_checkpoint(join(behaviour_policy_eval_path, 'checkpoints'))

    fqe_path = get_latest_fqe_path(experiment_path=experiment_path)
    fqe_model_paths = load_fqe_model_paths(fqe_path=fqe_path)

    behaviour_policy_eval_configs = load_json(join(behaviour_policy_eval_path, 'config.json'))
    fqe_configs = load_json(join(fqe_path, 'config.json'))

    assert behaviour_policy_eval_configs['discount'] == fqe_configs[
        'discount'], 'Both behavior policy and trained policy should have been evaluated using same discount'

    assert behaviour_policy_pre_proces_config.get_list_of_states() == pre_processing_configs.get_list_of_states(), "behavior policy and trained policy should have been evaluated using same states"
    assert behaviour_policy_pre_proces_config.get_list_of_actions() == pre_processing_configs.get_list_of_actions(), "behavior policy and trained policy should have been evaluated using same actions"
    assert behaviour_policy_pre_proces_config.action_space == pre_processing_configs.action_space, "Behavior policy and trained policy should have been evaluated using same action type (either continuous or discrete)"

    batch = buffer.sample_all()
    states = batch.observations
    dataset_actions = batch.actions
    mini_batch_size = 1024

    for i, (checkpoint, iteration) in enumerate(zip(checkpoints, iterations)):
        iteration_path = join(policy_actions_save_path, str(iteration))
        policy = AgentPolicyWrapper(agent_path=checkpoint)
        policy_actions = get_policy_actions(
            states=states,
            policy=policy,
            batch_size=mini_batch_size,
            device=device
        )

        policy_action_logs_df = pd.read_csv(join(iteration_path, 'policy_action_logs.csv'), index_col=[0])
        dataset_action_logs_df = pd.read_csv(join(iteration_path, 'dataset_action_logs.csv'), index_col=[0])

        policy_action_logs_df['q_value'] = get_fqe_values(
            states=states,
            actions=policy_actions,
            fqe_model_path=fqe_model_paths[str(iteration)],
            mini_batch_size=mini_batch_size
        )
        policy_action_logs_df.to_csv(join(iteration_path, 'policy_q_values.csv'))
        dataset_action_logs_df['q_value'] = get_fqe_values(
            states=states,
            actions=dataset_actions,
            fqe_model_path=behaviour_policy_fqe_model,
            mini_batch_size=mini_batch_size
        )
        dataset_action_logs_df.to_csv(join(iteration_path, 'dataset_q_values.csv'))


if __name__ == "__main__":
    load_dotenv()

    main(
        experiment_path=getenv('EXPERIMENT_PATH'),
        behaviour_policy_eval_path=getenv('BEHAVIOUR_POLICY_EVAL_PATH'),
        device=getenv('DEVICE', default='cpu')
    )

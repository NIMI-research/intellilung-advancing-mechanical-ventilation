import glob
from os import makedirs, getenv
from os.path import join, getmtime
import pandas as pd
import torch
from dotenv import load_dotenv

from actions.discrete_actions import get_bins_per_action_dim, one_hot_to_discrete_actions
from dataset.buffer import ReplayBuffer
from dataset.config import load_dataset_config
from dataset.load import get_policy_actions, load_dataset_to_buffer
from dataset.pre_processing_configs import load_pre_processing_configs
from actions.space import is_only_discrete, is_only_continuous, get_discrete_actions_list, get_continuous_actions_list
from policy.base import BasePolicy, AgentPolicyWrapper
from utils.files import save_json, load_json


def hot_encoded_actions_to_factorized_df(actions, list_of_actions, bins_per_action_dimension):
    actions = one_hot_to_discrete_actions(one_hot_actions=actions,
                                          bins_per_action_dimension=bins_per_action_dimension)
    actions_split = torch.split(actions, split_size_or_sections=1, dim=1)
    factorized_actions_dict = {action_label: actions.squeeze().cpu().numpy() for action_label, actions in
                               zip(list_of_actions, actions_split)}
    return pd.DataFrame(factorized_actions_dict)


class LogDiscretePolicyActions:

    def __init__(self, policy: BasePolicy, buffer: ReplayBuffer, list_of_actions, discrete_actions_file_path,
                 vent_mode_action_masking):
        self.policy: BasePolicy = policy
        self.buffer: ReplayBuffer = buffer
        self.list_of_actions = list_of_actions
        self.dataset_size = buffer.observations.size(dim=0)
        self.actions_ranges = load_json(path=discrete_actions_file_path)
        self.vent_mode_action_masking = vent_mode_action_masking

    def __call__(self, batch_size=1000, device='cpu'):
        batch = self.buffer.sample_all()
        policy_actions = get_policy_actions(
            states=batch.observations,
            policy=self.policy,
            batch_size=batch_size,
            device=device
        )
        bins_per_action_dimension = get_bins_per_action_dim(
            actions_ranges=self.actions_ranges,
            list_of_actions=self.list_of_actions,
            vent_mode_conditional_null_bins=self.vent_mode_action_masking
        )

        policy_df = hot_encoded_actions_to_factorized_df(
            actions=policy_actions,
            list_of_actions=self.list_of_actions,
            bins_per_action_dimension=bins_per_action_dimension
        )

        dataset_df = hot_encoded_actions_to_factorized_df(
            actions=batch.actions,
            list_of_actions=self.list_of_actions,
            bins_per_action_dimension=bins_per_action_dimension
        )

        return policy_df, dataset_df


class LogPolicyActions:

    def __init__(self, policy: BasePolicy, buffer: ReplayBuffer, list_of_actions):
        self.policy: BasePolicy = policy
        self.buffer: ReplayBuffer = buffer
        self.list_of_actions = list_of_actions
        self.dataset_size = buffer.observations.size(dim=0)

    def __call__(self, batch_size=1000, device='cpu'):
        batch = self.buffer.sample_all()
        feature_size = len(self.list_of_actions)
        policy_actions = get_policy_actions(
            states=batch.observations,
            policy=self.policy,
            batch_size=batch_size,
            device=device
        ).reshape(self.dataset_size, feature_size).cpu().numpy()

        dataset_actions = batch.actions.reshape(self.dataset_size, feature_size).cpu().numpy()
        policy_actions = pd.DataFrame(columns=self.list_of_actions, data=policy_actions)
        dataset_actions = pd.DataFrame(columns=self.list_of_actions, data=dataset_actions)

        return policy_actions, dataset_actions


class LogHybridPolicyActions:

    def __init__(
            self,
            policy: BasePolicy,
            buffer: ReplayBuffer,
            discrete_actions_list,
            continuous_actions_list,
            actions_ranges,
            vent_mode_action_masking
    ):
        self.policy: BasePolicy = policy
        self.buffer: ReplayBuffer = buffer
        self.discrete_actions_list = discrete_actions_list
        self.continuous_actions_list = continuous_actions_list
        self.dataset_size = buffer.observations.size(dim=0)
        self.actions_ranges = actions_ranges
        self.vent_mode_action_masking = vent_mode_action_masking

    def __call__(self, batch_size=1000, device='cpu'):
        batch = self.buffer.sample_all()

        bins_per_action_dimension = get_bins_per_action_dim(
            actions_ranges=self.actions_ranges,
            list_of_actions=self.discrete_actions_list,
            vent_mode_conditional_null_bins=self.vent_mode_action_masking
        )

        policy_actions = get_policy_actions(
            states=batch.observations,
            policy=self.policy,
            batch_size=batch_size,
            device=device
        )
        policy_actions_df = hot_encoded_actions_to_factorized_df(
            actions=policy_actions['discrete_actions'],
            list_of_actions=self.discrete_actions_list,
            bins_per_action_dimension=bins_per_action_dimension
        )
        policy_actions_df[self.continuous_actions_list] = policy_actions['continuous_actions'].squeeze().cpu().numpy()

        dataset_actions_df = hot_encoded_actions_to_factorized_df(
            actions=batch.actions['discrete_actions'],
            list_of_actions=self.discrete_actions_list,
            bins_per_action_dimension=bins_per_action_dimension
        )
        dataset_actions_df[self.continuous_actions_list] = batch.actions['continuous_actions'].squeeze().cpu().numpy()

        return policy_actions_df, dataset_actions_df


def main(experiment_path, device):
    dataset_config_path = join(experiment_path, 'dataset_config.json')
    dataset_config = load_dataset_config(dataset_config_path=dataset_config_path, experiment_path=experiment_path)

    pre_processing_config_path = join(experiment_path, 'pre_processing_metadata.json')
    pre_processing_configs = load_pre_processing_configs(pre_process_configs_path=pre_processing_config_path)

    checkpoints_path = join(experiment_path, 'checkpoints')

    policy_actions_save_path = join(experiment_path, 'policy_actions')

    checkpoints = glob.glob(f"{checkpoints_path}/*.pkl")
    checkpoints.sort(key=getmtime, reverse=True)
    iterations = [int(checkpoint.split('/')[-1].split('.')[0]) for checkpoint in checkpoints]

    test_dataset = dataset_config.test_dataset_split
    test_buffer = load_dataset_to_buffer(dataset=test_dataset,
                                         dataset_configs=dataset_config,
                                         pre_process_configs=pre_processing_configs, device=device)

    ep_id_col = pre_processing_configs.episode_id_column
    stay_id = test_dataset[ep_id_col].values
    time_step = test_buffer.time_step.cpu()

    policy_vs_dataset_action_metrics = {}

    for i, (checkpoint, iteration) in enumerate(zip(checkpoints, iterations)):
        policy = AgentPolicyWrapper(agent_path=checkpoint, keep_dict_hybrid_action=True)
        action_space = pre_processing_configs.action_space

        if is_only_discrete(action_space=action_space):
            log_actions = LogDiscretePolicyActions(
                policy=policy,
                buffer=test_buffer,
                list_of_actions=pre_processing_configs.get_list_of_actions(),
                discrete_actions_file_path=dataset_config.discrete_actions_file_path,
                vent_mode_action_masking=pre_processing_configs.vent_mode_action_masking
            )
            policy_actions, dataset_actions = log_actions(device=device)
        elif is_only_continuous(action_space=action_space):
            log_actions = LogPolicyActions(
                policy=policy,
                buffer=test_buffer,
                list_of_actions=pre_processing_configs.get_list_of_actions()
            )
            policy_actions, dataset_actions = log_actions(device=device)
        else:
            log_actions = LogHybridPolicyActions(
                policy=policy,
                buffer=test_buffer,
                discrete_actions_list=get_discrete_actions_list(action_space=action_space),
                continuous_actions_list=get_continuous_actions_list(action_space=action_space),
                actions_ranges=dataset_config.discrete_action_bin_ranges,
                vent_mode_action_masking=pre_processing_configs.vent_mode_action_masking
            )
            policy_actions, dataset_actions = log_actions(device=device)

        policy_actions[ep_id_col] = stay_id
        policy_actions['time_step'] = time_step
        policy_actions['time_step'] = policy_actions.groupby(ep_id_col)['time_step'].cumcount()

        dataset_actions[ep_id_col] = stay_id
        dataset_actions['time_step'] = time_step
        dataset_actions['time_step'] = dataset_actions.groupby(ep_id_col)['time_step'].cumcount()

        iteration_path = join(policy_actions_save_path, str(iteration))
        makedirs(iteration_path, exist_ok=True)

        policy_actions.reset_index(drop=True).to_csv(join(iteration_path, 'policy_action_logs.csv'))
        dataset_actions.reset_index(drop=True).to_csv(join(iteration_path, 'dataset_action_logs.csv'))

        save_json(data=policy_vs_dataset_action_metrics,
                  path=join(experiment_path, 'policy_vs_dataset_action_metrics.json'))


if __name__ == "__main__":
    load_dotenv()

    main(
        experiment_path=getenv('EXPERIMENT_PATH'),
        device=getenv('DEVICE', default='cpu')
    )

import pandas as pd
import torch
from tensordict import TensorDict
from torch import tensor
from tqdm import tqdm

from actions.discrete_actions import continuous_to_discrete_actions, discrete_actions_to_one_hot, \
    get_bins_per_action_dim
from actions.hybrid import create_hybrid_action_tensor_dict, flatten_action_dict_to_tensor
from actions.masking import mask_actions_with_empty_bin
from actions.space import get_list_of_actions, get_discrete_actions_list, get_continuous_actions_list
from dataset.base import RLEvalBatch
from dataset.buffer import ReplayBuffer, EvalBuffer
from dataset.config import DatasetLoadConfig
from dataset.pre_processing import dataset_to_rl_sequences, normalize_data

from dataset.pre_processing_configs import PreProcessingConfigs, ActionType
from dataset.stacking import stack_history_rl_batch
from policy.base import BasePolicy


def get_policy_actions(states, policy: BasePolicy, batch_size, device):
    total_samples = states.shape[0]

    actions = []
    for i in tqdm(range(0, total_samples, batch_size)):
        states_mini_batch = states[i: i + batch_size]
        actions.append(policy.select_action(obs=states_mini_batch, deterministic=True))
    actions = torch.cat(actions).to(device=device)
    if type(actions) is TensorDict:
        actions = actions.reshape(total_samples)
    else:
        actions = actions.reshape(total_samples, -1)
    return actions


def load_dataset_to_buffer(dataset: pd.DataFrame, dataset_configs: DatasetLoadConfig,
                           pre_process_configs: PreProcessingConfigs,
                           device='cpu',
                           action_dtype=torch.float32, flatten_hybrid_actions=False):
    normalization_params = dataset_configs.normalization_params
    discrete_action_bin_ranges = dataset_configs.discrete_action_bin_ranges

    action_space = pre_process_configs.action_space
    list_of_discrete_actions = get_discrete_actions_list(action_space=action_space)
    list_of_continuous_actions = get_continuous_actions_list(action_space=action_space)

    continuous_actions = None
    one_hot_discrete_actions = None

    if list_of_discrete_actions:

        discrete_actions = continuous_to_discrete_actions(
            data=dataset,
            list_of_actions=list_of_discrete_actions,
            discrete_action_bin_ranges=discrete_action_bin_ranges
        )
        if pre_process_configs.vent_mode_action_masking:
            discrete_actions = mask_actions_with_empty_bin(
                actions=discrete_actions,
                list_of_actions=list_of_discrete_actions,
                action_ranges=discrete_action_bin_ranges
            )
        one_hot_discrete_actions = discrete_actions_to_one_hot(
            dataset_actions=discrete_actions,
            list_of_actions=list_of_discrete_actions,
            bins_per_action_dimension=get_bins_per_action_dim(
                actions_ranges=discrete_action_bin_ranges,
                list_of_actions=list_of_discrete_actions,
                vent_mode_conditional_null_bins=pre_process_configs.vent_mode_action_masking
            )

        )

    if list_of_continuous_actions:
        continuous_actions_unnormalized = dataset[list_of_continuous_actions]
        continuous_actions = normalize_data(
            dataset=continuous_actions_unnormalized,
            columns=list_of_continuous_actions,
            norm_dict=normalization_params
        ).values
        continuous_actions = tensor(continuous_actions, device=device, dtype=torch.float32)

    if list_of_discrete_actions and list_of_continuous_actions:
        actions = create_hybrid_action_tensor_dict(
            continuous_actions=continuous_actions,
            discrete_actions=one_hot_discrete_actions
        ).to(device=device)
        if flatten_hybrid_actions:
            actions = flatten_action_dict_to_tensor(actions=actions)
    else:
        actions = [act for act in [continuous_actions, one_hot_discrete_actions] if act is not None]
        actions = torch.cat(actions, dim=1).to(device=device)

    rl_dataset = dataset_to_rl_sequences(
        dataset=dataset,
        data_pre_process_configs=pre_process_configs,
        actions=actions,
        normalization_params=normalization_params
    )

    if pre_process_configs.history_len > 0:
        rl_dataset = stack_history_rl_batch(
            batch=rl_dataset,
            history_len=pre_process_configs.history_len,
            device=device
        )

    buffer = ReplayBuffer(
        dataset=rl_dataset,
        action_dtype=action_dtype,
        device=device
    )

    return buffer


def load_dataset_to_eval_buffer(
        buffer: ReplayBuffer,
        policy: BasePolicy, device,
        batch_size=10000,
        action_dtype=torch.float32
) -> EvalBuffer:
    rl_dataset = buffer.sample_all()

    next_actions = get_policy_actions(
        states=rl_dataset.next_observations,
        policy=policy,
        batch_size=batch_size,
        device=device
    )

    eval_batch = RLEvalBatch(
        **rl_dataset.__dict__,
        next_actions=next_actions
    )
    buffer = EvalBuffer(
        dataset=eval_batch,
        action_dtype=action_dtype,
        device=device
    )
    return buffer


def get_next_actions(actions, ep_id):
    actions = actions.cpu().numpy()
    action_size = actions.shape[1]
    temp_action_names = [f'col {idx}' for idx in range(action_size)]
    ep_id = ep_id.squeeze().cpu().numpy()

    actions_df = pd.DataFrame(columns=temp_action_names, data=actions)
    actions_df['ep_id'] = ep_id

    next_actions = actions_df.groupby('ep_id')[temp_action_names].shift(-1)
    next_actions = next_actions.fillna(actions_df[temp_action_names])

    return torch.tensor(next_actions[temp_action_names].values).reshape(-1, action_size)


def load_behavior_policy_eval_dataset(dataset, dataset_configs, pre_process_configs, flatten_hybrid_actions,
                                      device, action_dtype=torch.float32) -> EvalBuffer:
    buffer = load_dataset_to_buffer(
        dataset=dataset,
        dataset_configs=dataset_configs,
        pre_process_configs=pre_process_configs,
        device=device,
        flatten_hybrid_actions=flatten_hybrid_actions
    )

    next_actions = get_next_actions(
        actions=buffer.actions,
        ep_id=buffer.epi_id
    ).to(device)

    rl_dataset = buffer.sample_all()

    eval_dataset = RLEvalBatch(
        **rl_dataset.__dict__,
        next_actions=next_actions
    )
    return EvalBuffer(action_dtype=action_dtype, device=device, dataset=eval_dataset)

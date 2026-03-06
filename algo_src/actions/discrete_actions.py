import pandas as pd
import torch
from torch import tensor, float32, Tensor
from torch.nn.functional import one_hot

from actions.space import ActionSpace, get_discrete_actions_list, get_continuous_actions_list
from dataset.pre_processing_configs import PreProcessingConfigs
from utils.files import load_json


def get_discrete_action_size(
        action_space: ActionSpace,
        discrete_actions_ranges,
        factored_actions,
        vent_mode_action_masking
):
    discrete_actions_list = get_discrete_actions_list(action_space=action_space)

    if factored_actions:
        discrete_action_size = get_factored_action_size(
            list_of_actions=discrete_actions_list,
            discrete_actions_ranges=discrete_actions_ranges,
            vent_mode_conditional_null_bins=vent_mode_action_masking
        )
    else:
        discrete_action_size = get_action_size_discrete(
            list_of_actions=discrete_actions_list,
            discrete_actions_ranges=discrete_actions_ranges,
            vent_mode_conditional_null_bins=vent_mode_action_masking
        )
    return discrete_action_size


def get_continuous_action_size(action_space: ActionSpace):
    return len(get_continuous_actions_list(action_space=action_space))


def get_action_size(pre_process_config: PreProcessingConfigs, factored_actions=True,
                    discrete_actions_file_path=None) -> int:
    """

    @param pre_process_config:
    @param discrete_actions_file_path: if discrete action spaces specify the path to the file containing bin ranges
    @type factored_actions: If discrete actions specify whether normal or factored space
    """
    discrete_actions_list = get_discrete_actions_list(action_space=pre_process_config.action_space)
    continuous_actions_list = get_continuous_actions_list(action_space=pre_process_config.action_space)
    contains_discrete_actions = False if discrete_actions_list == [] else True
    contains_continuous_actions = False if continuous_actions_list == [] else True
    discrete_action_size = 0
    continuous_action_size = 0
    if contains_discrete_actions:
        assert discrete_actions_file_path is not None, "Specify discrete_actions_file_path"
        discrete_actions_ranges = load_json(discrete_actions_file_path)
        discrete_action_size = get_discrete_action_size(
            action_space=pre_process_config.action_space,
            discrete_actions_ranges=discrete_actions_ranges,
            factored_actions=factored_actions,
            vent_mode_action_masking=pre_process_config.vent_mode_action_masking
        )

    if contains_continuous_actions:
        continuous_action_size = get_continuous_action_size(action_space=pre_process_config.action_space)

    return discrete_action_size + continuous_action_size


def get_action_size_discrete(list_of_actions, discrete_actions_ranges, vent_mode_conditional_null_bins=False):
    bins_per_action_dim = get_bins_per_action_dim(
        actions_ranges=discrete_actions_ranges,
        list_of_actions=list_of_actions,
        vent_mode_conditional_null_bins=vent_mode_conditional_null_bins
    )
    total_actions = 1
    for bins in bins_per_action_dim:
        total_actions *= bins
    return total_actions


def get_bins_per_action_dim(actions_ranges, list_of_actions, vent_mode_conditional_null_bins=False):
    bins_per_action_dimension = []
    for action in list_of_actions:
        if action in ['vent_vt_action', 'vent_pinsp-peep'] and vent_mode_conditional_null_bins:
            bins_per_action_dimension.append(len(actions_ranges[action]))
        else:
            bins_per_action_dimension.append(len(actions_ranges[action]) - 1)
    return bins_per_action_dimension


def get_factored_action_size(list_of_actions, discrete_actions_ranges, vent_mode_conditional_null_bins=False):
    """
    Get total number of action dimension required by neural networks when using factored discrete action spaces
    @param discrete_actions_ranges:
    @param list_of_actions: list of actions being used for current experiment
    @return: total number of action dimensions required by neural networks
    """

    return sum(get_bins_per_action_dim(actions_ranges=discrete_actions_ranges, list_of_actions=list_of_actions,
                                       vent_mode_conditional_null_bins=vent_mode_conditional_null_bins))


def get_possible_actions(dataset_actions):
    dataset_actions = tensor(dataset_actions)
    return torch.unique(dataset_actions, dim=0)


def continuous_to_discrete_actions(data, list_of_actions, discrete_action_bin_ranges):
    """
    Converts continuous action variables to multi-dimensional discrete actions then create action combinations
     (across action dimension) and encode these action combination by using their index instead of value tuple.
    @param data: dataset containing the action variables
    @param list_of_actions: ordered list of actions to discretize
    @param discrete_action_bin_ranges: dict containing action bin ranges
    @return: discrete action array
    """
    continuous_actions = data[list_of_actions]
    discrete_actions = pd.DataFrame().reindex_like(continuous_actions)

    for action in list_of_actions:
        bins = discrete_action_bin_ranges[action]
        labels = range(len(discrete_action_bin_ranges[action]) - 1)
        discrete_actions[action] = pd.cut(x=continuous_actions[action], bins=bins, labels=labels, include_lowest=True)

    return discrete_actions[list_of_actions].values


def discrete_actions_to_one_hot(dataset_actions, list_of_actions, bins_per_action_dimension):
    dataset_actions = tensor(dataset_actions)

    encoded_action_dimensions = []

    for i, action in enumerate(list_of_actions):
        encoded_action_dimensions.append(
            one_hot(dataset_actions[:, i].squeeze(), num_classes=bins_per_action_dimension[i]).reshape(-1,
                                                                                                       bins_per_action_dimension[
                                                                                                           i])
        )
    return torch.cat(encoded_action_dimensions, dim=1).to(dtype=float32)


def one_hot_to_discrete_actions(one_hot_actions: Tensor, bins_per_action_dimension):
    actions_split = torch.split(one_hot_actions, split_size_or_sections=bins_per_action_dimension, dim=1)

    action_index_split = [action_dim_logits.argmax(dim=1).unsqueeze(-1) for
                          action_dim_logits in actions_split]

    return torch.cat(action_index_split, dim=1)

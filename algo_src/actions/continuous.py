import torch

from actions.discrete_actions import one_hot_to_discrete_actions
from utils.files import load_json


def disc_to_cont_using_mode(actions, list_of_actions, bins_per_action_dim, discrete_action_bin_ranges):
    device = actions.device
    one_hot_split = torch.split(actions, bins_per_action_dim, dim=-1)
    discrete_actions = one_hot_to_discrete_actions(one_hot_actions=actions,
                                                   bins_per_action_dimension=bins_per_action_dim)
    discrete_actions = torch.split(discrete_actions, split_size_or_sections=1, dim=1)
    converted_actions = {}
    actions_bin_values = load_json('configs/action_bin_index_to_value.json')
    for action_label, action, one_hot_action in zip(list_of_actions, discrete_actions, one_hot_split):
        #bin_edges = torch.tensor(discrete_action_bin_ranges[action_label], device=device)
        #bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_centers = torch.tensor(actions_bin_values[action_label], device=device)

        action = action.squeeze()
        if not action_label == 'vent_mode':

            cont_action = bin_centers[action]
            converted_actions[action_label] = cont_action.squeeze()
        else:
            converted_actions[action_label] = one_hot_action.squeeze()
    return converted_actions


def disc_to_cont_using_uniform(actions, list_of_actions, bins_per_action_dim, discrete_action_bin_ranges):
    device = actions.device
    one_hot_split = torch.split(actions, bins_per_action_dim, dim=-1)
    discrete_actions = one_hot_to_discrete_actions(one_hot_actions=actions,
                                                   bins_per_action_dimension=bins_per_action_dim)
    discrete_actions = torch.split(discrete_actions, split_size_or_sections=1, dim=1)
    converted_actions = {}
    for action_label, action, one_hot_action in zip(list_of_actions, discrete_actions, one_hot_split):
        bin_edges = torch.tensor(discrete_action_bin_ranges[action_label], device=device)
        action = action.squeeze()
        if action_label != 'vent_mode':
            # Sample uniformly between the lower and upper bin edges
            lower_edge = bin_edges[action]
            upper_edge = bin_edges[action + 1]
            # torch.rand produces a value in [0, 1); scale it to the bin width
            cont_action = lower_edge + (upper_edge - lower_edge) * torch.rand(1, device=device)
            converted_actions[action_label] = cont_action.squeeze()
        else:
            converted_actions[action_label] = one_hot_action.squeeze()
    return converted_actions


def disc_to_cont_using_gauss(actions, list_of_actions, bins_per_action_dim, discrete_action_bin_ranges,
                             std_scale=1 / 3):
    device = actions.device
    one_hot_split = torch.split(actions, bins_per_action_dim, dim=-1)
    discrete_actions = one_hot_to_discrete_actions(one_hot_actions=actions,
                                                   bins_per_action_dimension=bins_per_action_dim)
    discrete_actions = torch.split(discrete_actions, split_size_or_sections=1, dim=1)
    converted_actions = {}
    actions_bin_values = load_json('configs/action_bin_index_to_value.json')
    for action_label, action, one_hot_action in zip(list_of_actions, discrete_actions, one_hot_split):
        action = action.squeeze()
        if not action_label == 'vent_mode':
            bin_edges = torch.tensor(discrete_action_bin_ranges[action_label], device=device)

            lower_edge = bin_edges[action]
            upper_edge = bin_edges[action + 1]
            bin_centers = torch.tensor(actions_bin_values[action_label], device=device)
            std = upper_edge - lower_edge
            samples = torch.normal(bin_centers[action], std * std_scale)
            cont_action = torch.clamp(samples, min=lower_edge, max=upper_edge)

            converted_actions[action_label] = cont_action.squeeze()
        else:
            converted_actions[action_label] = one_hot_action.squeeze()
    return converted_actions


def disc_to_cont_using_mid(actions, list_of_actions, bins_per_action_dim, discrete_action_bin_ranges):
    device = actions.device
    one_hot_split = torch.split(actions, bins_per_action_dim, dim=-1)
    discrete_actions = one_hot_to_discrete_actions(one_hot_actions=actions,
                                                   bins_per_action_dimension=bins_per_action_dim)
    discrete_actions = torch.split(discrete_actions, split_size_or_sections=1, dim=1)
    converted_actions = {}
    for action_label, action, one_hot_action in zip(list_of_actions, discrete_actions, one_hot_split):
        bin_edges = torch.tensor(discrete_action_bin_ranges[action_label], device=device)
        action = action.squeeze()
        if action_label != 'vent_mode':
            # Sample uniformly between the lower and upper bin edges
            lower_edge = bin_edges[action]
            upper_edge = bin_edges[action + 1]
            # torch.rand produces a value in [0, 1); scale it to the bin width
            cont_action = (upper_edge + lower_edge) / 2
            converted_actions[action_label] = cont_action.squeeze()
        else:
            converted_actions[action_label] = one_hot_action.squeeze()
    return converted_actions

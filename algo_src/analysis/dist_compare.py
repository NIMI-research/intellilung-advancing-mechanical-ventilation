import glob
from os import getenv
from os.path import join, getmtime
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv

from actions.discrete_actions import get_bins_per_action_dim
from dataset.config import load_dataset_config
from dataset.pre_processing import un_normalize_data
from dataset.pre_processing_configs import load_pre_processing_configs
from actions.space import is_only_discrete, is_only_continuous, get_discrete_actions_list, get_continuous_actions_list
from utils.files import load_json


def range_to_labels(ranges):
    labels = []
    for i in range(len(ranges) - 1):
        labels.append(f'{ranges[i]:.2f}-{ranges[i + 1]:.2f}')
    return labels


def plot_dist_compare(policy_actions, dataset_actions, save_path, discrete_action_bin_ranges, pre_process_configs,
                      fig_name_postfix=''):
    bins_per_action_dim = get_bins_per_action_dim(
        actions_ranges=discrete_action_bin_ranges,
        list_of_actions=pre_process_configs.get_list_of_actions(),
        vent_mode_conditional_null_bins=pre_process_configs.vent_mode_action_masking
    )
    list_of_actions = pre_process_configs.get_list_of_actions()

    actions_tick_labels = {}
    exclude_x_tick_range_label = ['vent_mode']
    for key, value in discrete_action_bin_ranges.items():
        if key not in exclude_x_tick_range_label:
            actions_tick_labels[key] = range_to_labels(value)

    x_label = {
        'vent_rrtot': 'bpm',
        'vent_peep': 'cmH2O',
        'vent_fio2': '%',
        'vent_inspexp': 'I:E',
        'vent_vt_action': 'ml/kg',
        'vent_pinsp-peep': 'cmH2O'
    }

    merged_actions_df = pd.merge(policy_actions, dataset_actions, left_index=True, right_index=True,
                                 suffixes=(' AI', ' Clinician'))
    for i, action_label in enumerate(list_of_actions):

        plot_df = merged_actions_df[[action_label + " Clinician", action_label + " AI"]]
        sns.histplot(plot_df.melt(), x='value', hue='variable',
                     multiple='dodge', stat='percent', shrink=3, linewidth=0)
        fig_name = f'{action_label}_{fig_name_postfix}.png' if fig_name_postfix else f'{action_label}.png'
        fig_path = join(save_path, fig_name)
        x_ticks_label = actions_tick_labels.get(action_label, None)

        if x_ticks_label:
            total_bins = bins_per_action_dim[i]
            ticks = range(total_bins)
            x_ticks_label.append('-') if total_bins > len(x_ticks_label) else x_ticks_label

            plt.xticks(ticks=ticks, labels=x_ticks_label, fontsize=6)
        plt.xlabel(x_label.get(action_label, ''))
        plt.title(f'Physician vs AI - {action_label}')
        plt.gcf().set_dpi(300)
        plt.savefig(fig_path)
        plt.close()


def plot_dist_compare_cont(policy_actions, dataset_actions, save_path, list_of_actions, normalization_dict):
    x_label = {
        'vent_rrtot': 'bpm',
        'vent_peep': 'cmH2O',
        'vent_fio2': '%',
        'vent_inspexp': 'I:E',
        'vent_vtnorm': 'ml/kg',
        'vent_pinsp-peep': 'cmH2O'
    }

    policy_actions[list_of_actions] = un_normalize_data(dataset=policy_actions, columns=list_of_actions,
                                                     norm_dict=normalization_dict)
    dataset_actions[list_of_actions] = un_normalize_data(dataset=dataset_actions, columns=list_of_actions,
                                                      norm_dict=normalization_dict)
    merged_actions_df = pd.merge(policy_actions, dataset_actions, left_index=True, right_index=True,
                                 suffixes=(' AI', ' Clinician'))
    for action_label in list_of_actions:
        plot_df = merged_actions_df[[action_label + " Clinician", action_label + " AI"]]
        sns.histplot(plot_df.melt(), x='value', hue='variable', kde=True)
        fig_name = f'{action_label}.png'
        fig_path = join(save_path, fig_name)

        plt.xlabel(x_label.get(action_label, ''))
        plt.title(f'Physician vs AI - {action_label}')
        plt.savefig(fig_path)
        plt.close()


def plot_dist_compare_hybrid(
        policy_actions,
        dataset_actions,
        save_path,
        discrete_actions_list,
        continuous_actions_list,
        normalization_dict
):
    x_label = {
        'vent_rrtot': 'bpm',
        'vent_peep': 'cmH2O',
        'vent_fio2': '%',
        'vent_inspexp': 'I:E',
        'vent_vtnorm': 'ml/kg',
        'vent_pinsp-peep': 'cmH2O'
    }

    list_of_actions = [*discrete_actions_list, *continuous_actions_list]

    policy_actions[continuous_actions_list] = un_normalize_data(dataset=policy_actions, columns=continuous_actions_list,
                                                        norm_dict=normalization_dict)
    dataset_actions[continuous_actions_list] = un_normalize_data(dataset=dataset_actions, columns=continuous_actions_list,
                                                         norm_dict=normalization_dict)
    merged_actions_df = pd.merge(policy_actions, dataset_actions, left_index=True, right_index=True,
                                 suffixes=(' AI', ' Clinician'))

    for action_label in list_of_actions:

        plot_df = merged_actions_df[[action_label + " Clinician", action_label + " AI"]]
        if action_label in continuous_actions_list:
            sns.histplot(plot_df.melt(), x='value', hue='variable', kde=True)
        else:
            sns.histplot(plot_df.melt(), x='value', hue='variable',
                         multiple='dodge', stat='percent', shrink=3, linewidth=0)
        fig_name = f'{action_label}.png'
        fig_path = join(save_path, fig_name)

        plt.xlabel(x_label.get(action_label, ''))
        plt.title(f'Physician vs AI - {action_label}')
        plt.savefig(fig_path)
        plt.close()


def main(experiment_path):
    dataset_config_path = join(experiment_path, 'dataset_config.json')
    dataset_config = load_dataset_config(dataset_config_path=dataset_config_path, experiment_path=experiment_path)
    checkpoints_path = join(experiment_path, 'checkpoints')
    policy_actions_save_path = join(experiment_path, 'policy_actions')

    pre_processing_config_path = join(experiment_path, 'pre_processing_metadata.json')
    pre_processing_config = load_pre_processing_configs(pre_process_configs_path=pre_processing_config_path)

    checkpoints = glob.glob(f"{checkpoints_path}/*.pkl")
    checkpoints.sort(key=getmtime, reverse=True)
    iterations = [int(checkpoint.split('/')[-1].split('.')[0]) for checkpoint in checkpoints]

    for i, (checkpoint, iteration) in enumerate(zip(checkpoints, iterations)):
        iteration_path = join(policy_actions_save_path, str(iteration))
        policy_action_logs = pd.read_csv(join(iteration_path, 'policy_action_logs.csv'))
        dataset_action_logs = pd.read_csv(join(iteration_path, 'dataset_action_logs.csv'))
        action_space = pre_processing_config.action_space
        if is_only_discrete(action_space=action_space):
            plot_dist_compare(
                policy_actions=policy_action_logs,
                dataset_actions=dataset_action_logs,
                save_path=iteration_path,
                discrete_action_bin_ranges=dataset_config.discrete_action_bin_ranges,
                pre_process_configs=pre_processing_config
            )
        elif is_only_continuous(action_space=action_space):
            plot_dist_compare_cont(
                policy_actions=policy_action_logs,
                dataset_actions=dataset_action_logs,
                save_path=iteration_path,
                list_of_actions=pre_processing_config.get_list_of_actions(),
                normalization_dict=dataset_config.normalization_params
            )
        else:
            plot_dist_compare_hybrid(
                policy_actions=policy_action_logs,
                dataset_actions=dataset_action_logs,
                save_path=iteration_path,
                discrete_actions_list=get_discrete_actions_list(action_space=action_space),
                continuous_actions_list=get_continuous_actions_list(action_space=action_space),
                normalization_dict=dataset_config.normalization_params
            )


if __name__ == "__main__":
    load_dotenv()
    main(
        experiment_path=getenv('EXPERIMENT_PATH')
    )

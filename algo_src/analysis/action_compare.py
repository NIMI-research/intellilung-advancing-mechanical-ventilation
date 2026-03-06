import glob
from os import getenv
from os.path import join, getmtime

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from dataset.pre_processing_configs import load_pre_processing_configs
from actions.space import is_only_discrete, is_only_continuous, get_discrete_actions_list, get_continuous_actions_list


def calculate_changes(actions: pd.DataFrame, discrete_actions_list, continuous_actions_list, group_col):
    changes_per_action = []

    action_cols = discrete_actions_list + continuous_actions_list
    for col in action_cols:
        if col in discrete_actions_list:
            changes_per_group = actions.groupby([group_col])[col].apply(
                lambda x: (x.diff().fillna(0) != 0).sum()).to_frame().reset_index()
        elif col in continuous_actions_list:
            changes_per_group = actions.groupby([group_col])[col].apply(
                lambda x: x.diff(-1).abs().dropna() / 2
            ).to_frame().reset_index()
        else:
            raise Exception(f'{col} is not in discrete list or continuous actions list')

        changes_per_stay_id = changes_per_group[[group_col, col]].groupby(group_col).sum()

        changes_per_action.append(changes_per_stay_id)

    changes_per_action = pd.concat(changes_per_action, axis=1)

    return changes_per_action


def confusion_matrix_scorer(x):
    action_filter = x['ai_action']
    y_pred = x[action_filter].drop(['ai_action'], axis=1)
    y_true = x[~action_filter].drop(['ai_action'], axis=1)
    cm = confusion_matrix(y_true=y_true.values.squeeze(), y_pred=y_pred.values.squeeze())
    tp_per_class = cm.diagonal()
    tn_per_class = np.sum(cm) - np.sum(cm, axis=0) - np.sum(cm, axis=1) + tp_per_class
    fp_per_class = np.sum(cm, axis=0) - tp_per_class
    fn_per_class = np.sum(cm, axis=1) - tp_per_class
    cm = {
        'tn': np.sum(tn_per_class),
        'fp': np.sum(fp_per_class),
        'fn': np.sum(fn_per_class),
        'tp': np.sum(tp_per_class)
    }
    metrics = {
        'accuracy': (cm['tp'] + cm['tn']) / (cm['tp'] + cm['tn'] + cm['fp'] + cm['fn']),
        'precision': cm['tp'] / (cm['tp'] + cm['fp'])
    }

    return pd.Series(metrics,
                     index=['accuracy', 'precision']
                     )


def calculate_confusion_matrix_per_stay_id(policy_actions: pd.DataFrame, dataset_actions: pd.DataFrame, action_cols,
                                           group_col):
    policy_actions['ai_action'] = True
    dataset_actions['ai_action'] = False
    actions_df = pd.concat([policy_actions, dataset_actions]).reset_index()
    actions_metrics_dfs = []
    for action in action_cols:
        metrics_df = actions_df.groupby([group_col])[['ai_action', action]].apply(confusion_matrix_scorer)
        metrics_df['action_type'] = action
        actions_metrics_dfs.append(metrics_df)

    actions_metrics_df = pd.concat(actions_metrics_dfs)

    return actions_metrics_df


def calculate_action_error_per_stay_id(policy_actions: pd.DataFrame, dataset_actions: pd.DataFrame, action_cols,
                                       group_col):
    action_diffs = {group_col: policy_actions[group_col].values}
    for action in action_cols:
        diff = policy_actions[action] - dataset_actions[action]
        diff = diff.abs() / 2

        action_diffs[action] = diff.values

    diff_df = pd.DataFrame(action_diffs)
    diff_df_per_stay = diff_df.groupby([group_col]).mean()
    return diff_df_per_stay


def main(experiment_path):
    pre_processing_config_path = join(experiment_path, 'pre_processing_metadata.json')
    pre_processing_config = load_pre_processing_configs(pre_process_configs_path=pre_processing_config_path)

    checkpoints_path = join(experiment_path, 'checkpoints')
    policy_actions_save_path = join(experiment_path, 'policy_actions')
    checkpoints = glob.glob(f"{checkpoints_path}/*.pkl")
    checkpoints.sort(key=getmtime, reverse=True)
    iterations = [int(checkpoint.split('/')[-1].split('.')[0]) for checkpoint in checkpoints]

    ep_id_col = pre_processing_config.episode_id_column

    for i, (checkpoint, iteration) in enumerate(zip(checkpoints, iterations)):
        iteration_path = join(policy_actions_save_path, str(iteration))
        policy_action_logs = pd.read_csv(join(iteration_path, 'policy_action_logs.csv'))
        dataset_action_logs = pd.read_csv(join(iteration_path, 'dataset_action_logs.csv'))

        action_space = pre_processing_config.action_space

        discrete_actions_list = get_discrete_actions_list(action_space=action_space)
        continuous_actions_list = get_continuous_actions_list(action_space=action_space)

        policy_action_changes = calculate_changes(
            policy_action_logs,
            discrete_actions_list=discrete_actions_list,
            continuous_actions_list=continuous_actions_list,
            group_col=ep_id_col
        )
        dataset_action_changes = calculate_changes(
            dataset_action_logs,
            discrete_actions_list=discrete_actions_list,
            continuous_actions_list=continuous_actions_list,
            group_col=ep_id_col
        )

        if discrete_actions_list:
            discrete_action_metrics = calculate_confusion_matrix_per_stay_id(
                policy_actions=policy_action_logs,
                dataset_actions=dataset_action_logs,
                action_cols=discrete_actions_list,
                group_col=ep_id_col
            )
            discrete_action_metrics.to_csv(join(iteration_path, 'discrete_action_metrics.csv'))

        if continuous_actions_list:
            continuous_action_metrics = calculate_action_error_per_stay_id(
                policy_actions=policy_action_logs,
                dataset_actions=dataset_action_logs,
                action_cols=continuous_actions_list,
                group_col=ep_id_col
            )
            continuous_action_metrics.to_csv(join(iteration_path, 'continuous_action_metrics.csv'))

        policy_action_changes.to_csv(join(iteration_path, 'policy_action_changes.csv'))
        dataset_action_changes.to_csv(join(iteration_path, 'dataset_action_changes.csv'))

        policy_action_changes = policy_action_changes.melt(var_name='action_type', value_name='action_changes')
        dataset_action_changes = dataset_action_changes.melt(var_name='action_type', value_name='action_changes')
        policy_action_changes['type'] = 'AI'
        dataset_action_changes['type'] = 'Clinician'

        action_changes_df = pd.concat([policy_action_changes.reset_index(), dataset_action_changes.reset_index()])
        sns.barplot(action_changes_df, x='action_type', y='action_changes', hue='type')

        plt.xticks(rotation=90)
        plt.subplots_adjust(bottom=0.35)
        plt.savefig(join(iteration_path, 'clinician_vs_ai_action_changes.jpg'))
        plt.close()


if __name__ == "__main__":
    load_dotenv()

    main(
        experiment_path=getenv('EXPERIMENT_PATH')
    )

import copy
import glob
import os
from os import getenv
from os.path import join, getmtime

import pandas as pd
import torch
from dotenv import load_dotenv

import wandb
from torch import tensor
from torch.nn.functional import cross_entropy
import torch
from actions.continuous import disc_to_cont_using_mode, disc_to_cont_using_uniform, disc_to_cont_using_gauss, \
    disc_to_cont_using_mid
from actions.discrete_actions import get_bins_per_action_dim, continuous_to_discrete_actions, \
    discrete_actions_to_one_hot
from actions.hybrid import get_continuous_action, get_discrete_action, create_hybrid_action_tensor_dict
from agents.base import Agent
from agents.configs import load_eval_configs, EvalExpConfig
from agents.eval import AgentEvaluator
from dataset.buffer import ReplayBuffer
from dataset.config import load_dataset_config
from dataset.load import load_dataset_to_buffer, get_policy_actions
from dataset.pre_processing import un_normalize_data, normalize_data

from dataset.pre_processing_configs import load_pre_processing_configs
from actions.space import is_only_discrete, get_discrete_actions_list, get_continuous_actions_list

from policy.base import AgentPolicyWrapper
from utils.files import load_json, save_json
from utils.wandb import wandb_init


class AEEvalConfigs(EvalExpConfig):
    name: str = "Hybrid-Autencoder"
    job_type: str = "OOD_Algo_Eval"
    original_experiment_group_id: str


def convert_buffer_actions_to_discrete_and_back(buffer, cont_actions_list, dataset_config, ae_dataset_config,
                                                device, disc_to_cont_fn):
    buffer = copy.deepcopy(buffer)
    policy_actions_cont = pd.DataFrame(columns=cont_actions_list,
                                       data=get_continuous_action(buffer.actions).cpu().numpy())
    dataset_norm_dict = dataset_config.normalization_params
    un_normalized_policy_actions_cont = un_normalize_data(dataset=policy_actions_cont,
                                                          columns=cont_actions_list,
                                                          norm_dict=dataset_norm_dict)
    discrete_actions = continuous_to_discrete_actions(
        data=un_normalized_policy_actions_cont,
        list_of_actions=cont_actions_list,
        discrete_action_bin_ranges=dataset_config.discrete_action_bin_ranges
    )

    one_hot_discrete_actions = discrete_actions_to_one_hot(
        dataset_actions=discrete_actions,
        list_of_actions=cont_actions_list,
        bins_per_action_dimension=get_bins_per_action_dim(
            actions_ranges=dataset_config.discrete_action_bin_ranges,
            list_of_actions=cont_actions_list,
            vent_mode_conditional_null_bins=False
        )
    )
    converted_actions = disc_to_cont_fn(
        actions=one_hot_discrete_actions,
        list_of_actions=cont_actions_list,
        bins_per_action_dim=get_bins_per_action_dim(
            actions_ranges=dataset_config.discrete_action_bin_ranges,
            list_of_actions=cont_actions_list,
            vent_mode_conditional_null_bins=False
        ),
        discrete_action_bin_ranges=dataset_config.discrete_action_bin_ranges
    )
    converted_actions_normalized = normalize_data(
        dataset=pd.DataFrame(converted_actions),
        columns=cont_actions_list,
        norm_dict=ae_dataset_config.normalization_params
    )
    converted_actions_normalized = converted_actions_normalized[cont_actions_list].values
    buffer.actions['continuous_actions'] = torch.tensor(converted_actions_normalized, device=device)
    return buffer


def eval_actions(model, eval_buffer, cont_action_list, batch_size):
    model.eval()
    batch = eval_buffer.sample_all()
    state = batch.observations
    action = batch.actions
    cont_actions_losses = []

    total_data_samples = state.shape[0]
    with torch.no_grad():
        batch_range = range(0, total_data_samples, batch_size)
        for i in batch_range:
            mb_state = state[i:i + batch_size]
            mb_action = action[i:i + batch_size]
            mb_continuous_action = get_continuous_action(mb_action)
            mb_discrete_action = get_discrete_action(mb_action)

            cont_action_dist, disc_action_dist, _ = model(
                state=mb_state
            )

            cont_actions_loss = -cont_action_dist.log_prob(mb_continuous_action)

            cont_actions_losses.append(cont_actions_loss)

    each_action_loss = torch.cat(cont_actions_losses).mean(dim=0)
    each_action_loss = {action: each_action_loss[i].mean().item() for i, action in
                        enumerate(cont_action_list)}

    return each_action_loss


def eval_get_all_losses(agent, eval_buffer: ReplayBuffer):
    agent.model.eval()
    batch = eval_buffer.sample_all()
    state = batch.observations
    action = batch.actions

    losses = []

    with torch.no_grad():
        for i in range(0, state.shape[0], agent.batch_size):
            mb_state = state[i:i + agent.batch_size]
            mb_action = action[i:i + agent.batch_size]
            mb_cont = get_continuous_action(mb_action)
            mb_disc = get_discrete_action(mb_action)

            cont_dist, disc_dist, cont_log_var = agent.model(state=mb_state)

            cont_var = (cont_dist.scale ** 2).clamp_min(agent._gnll.eps)
            cont_nll = agent._gnll(cont_dist.mean, mb_cont, cont_var)
            cont_lp = agent._per_example_reduce(cont_nll)

            logits = disc_dist.logits
            target_idx = mb_disc.argmax(dim=-1)  # Long indices
            disc_lp = cross_entropy(logits, target_idx, reduction="none", label_smoothing=0.05)

            batch_loss = cont_lp + disc_lp

            losses.append(batch_loss)

    return pd.DataFrame({'loss': -torch.cat(losses).cpu().numpy()})


def eval_checkpoint_ood(dataset_config, pre_processing_configs, ae_dataset_config, device, checkpoint, ae_checkpoint,
                        policy_id, ae_configs, ae_pre_processing_config, extra_info_save_path,
                        batch_size=1000):
    buffer = load_dataset_to_buffer(dataset=dataset_config.test_dataset_split,
                                    dataset_configs=dataset_config,
                                    pre_process_configs=pre_processing_configs,
                                    device=device)
    policy_actions = buffer.actions

    discrete_actions_list = get_discrete_actions_list(action_space=pre_processing_configs.action_space)
    continuous_actions_list = get_continuous_actions_list(action_space=pre_processing_configs.action_space)

    continuous_actions_dict = {}
    buffer_states_df = pd.DataFrame(columns=pre_processing_configs.get_list_of_states(),
                                    data=buffer.observations.cpu().numpy())
    un_normalized_states = un_normalize_data(
        dataset=buffer_states_df,
        columns=pre_processing_configs.get_list_of_states(),
        norm_dict=dataset_config.normalization_params
    )
    ae_normalized_states = normalize_data(
        dataset=un_normalized_states,
        columns=pre_processing_configs.get_list_of_states(),
        norm_dict=ae_dataset_config.normalization_params
    )
    if continuous_actions_list:
        policy_actions_cont = get_continuous_action(policy_actions)
        policy_actions_cont = pd.DataFrame(columns=continuous_actions_list, data=policy_actions_cont.cpu().numpy())
        dataset_norm_dict = dataset_config.normalization_params
        un_normalized_policy_actions_cont = un_normalize_data(dataset=policy_actions_cont,
                                                              columns=continuous_actions_list,
                                                              norm_dict=dataset_norm_dict)

        ae_normalized_policy_actions_cont = normalize_data(dataset=un_normalized_policy_actions_cont,
                                                           columns=continuous_actions_list,
                                                           norm_dict=ae_dataset_config.normalization_params)
        ae_normalized_policy_actions_cont = ae_normalized_policy_actions_cont[continuous_actions_list].values
        ae_normalized_policy_actions_cont = torch.tensor(ae_normalized_policy_actions_cont, device=device)

        continuous_actions_dict = {
            key: ae_normalized_policy_actions_cont[:, i] for i, key in enumerate(continuous_actions_list)
        }

    bins_per_action_dim = get_bins_per_action_dim(
        actions_ranges=dataset_config.discrete_action_bin_ranges,
        list_of_actions=discrete_actions_list,
        vent_mode_conditional_null_bins=pre_processing_configs.vent_mode_action_masking
    )
    if is_only_discrete(pre_processing_configs.action_space):
        policy_actions_disc = policy_actions
    else:
        policy_actions_disc = get_discrete_action(actions=policy_actions)

    converted_actions = disc_to_cont_using_mode(
        actions=policy_actions_disc,
        list_of_actions=discrete_actions_list,
        bins_per_action_dim=bins_per_action_dim,
        discrete_action_bin_ranges=dataset_config.discrete_action_bin_ranges
    )

    ae_cont_actions_list = get_continuous_actions_list(action_space=ae_pre_processing_config.action_space)
    norm_var = list(set(discrete_actions_list).intersection(ae_cont_actions_list))
    converted_actions_red = {name: actions.cpu() for name, actions in converted_actions.items() if name in norm_var}
    if norm_var:
        converted_actions_normalized = normalize_data(
            dataset=pd.DataFrame(converted_actions_red),
            columns=norm_var,
            norm_dict=ae_dataset_config.normalization_params
        )
        converted_actions_normalized = converted_actions_normalized.to_dict('list')

        combined_actions_dict = {**converted_actions_normalized, **continuous_actions_dict}
    else:
        combined_actions_dict = continuous_actions_dict

    combined_cont_actions_list = [tensor(combined_actions_dict[action], device=device).reshape(-1, 1) for action in
                                  ae_cont_actions_list]
    combined_cont_actions = torch.cat(combined_cont_actions_list, dim=-1).to(device=device)

    buffer.actions = create_hybrid_action_tensor_dict(
        continuous_actions=combined_cont_actions,
        discrete_actions=converted_actions['vent_mode'].to(device=device)
    )

    print(buffer.actions['continuous_actions'].shape)
    buffer.observations = torch.tensor(ae_normalized_states[pre_processing_configs.get_list_of_states()].values,
                                       device=device)

    agent = Agent.load(save_path=ae_checkpoint)

    out = agent.eval(checkpoint_id=policy_id, agent=agent, eval_buffer=buffer)
    print(out)


def main(ae_configs, experiment_path, ae_experiment_path, device):
    dataset_config_path = join(experiment_path, 'dataset_config.json')
    dataset_config = load_dataset_config(dataset_config_path=dataset_config_path, experiment_path=experiment_path)
    ae_dataset_config_path = join(ae_experiment_path, 'dataset_config.json')
    ae_dataset_config = load_dataset_config(dataset_config_path=ae_dataset_config_path,
                                            experiment_path=ae_experiment_path)
    pre_processing_config_path = join(experiment_path, 'pre_processing_metadata.json')
    pre_processing_configs = load_pre_processing_configs(pre_process_configs_path=pre_processing_config_path)
    ae_pre_processing_config_path = join(ae_experiment_path, 'pre_processing_metadata.json')
    ae_pre_processing_config = load_pre_processing_configs(pre_process_configs_path=ae_pre_processing_config_path)

    checkpoints_path = join(experiment_path, 'checkpoints')

    checkpoints = glob.glob(f"{checkpoints_path}/*.pkl")
    checkpoints.sort(key=getmtime, reverse=True)
    iterations = [int(checkpoint.split('/')[-1].split('.')[0]) for checkpoint in checkpoints]

    ae_checkpoints_path = join(ae_experiment_path, 'checkpoints')
    ae_checkpoints = glob.glob(f"{ae_checkpoints_path}/*.pkl")
    ae_checkpoints.sort(key=getmtime, reverse=True)
    print('Using AE Checkpoint: ', ae_checkpoints[0].split('/')[-1].split('.')[0])

    original_ae_configs = load_json(join(ae_experiment_path, 'config.json'))

    # assert dataset_config.dataset_type == ae_dataset_config.dataset_type, f"Trying to evaluate policy trained using {dataset_config.dataset_type} dataset on autoencoder trained used {ae_dataset_config.dataset_type} dataset"

    wandb_init(
        config=ae_configs.model_dump(),
        dataset_config=dataset_config.model_dump(),
        pre_processing_metadata=pre_processing_configs.model_dump(),
        autoencoder_config=original_ae_configs
    )
    for i, (checkpoint, iteration) in enumerate(zip(checkpoints, iterations)):
        policy_action_path = join(experiment_path, 'policy_actions', f'{iteration}')
        eval_checkpoint_ood(
            dataset_config=dataset_config,
            pre_processing_configs=pre_processing_configs,
            ae_dataset_config=ae_dataset_config,
            device=device,
            checkpoint=checkpoint,
            policy_id=iteration,
            ae_checkpoint=ae_checkpoints[0],
            ae_configs=ae_configs,
            ae_pre_processing_config=ae_pre_processing_config,
            extra_info_save_path=policy_action_path
        )
    wandb.finish()


if __name__ == "__main__":
    load_dotenv()

    main_experiment_path = getenv('EXPERIMENT_PATH')
    trainer_config = load_json(join(main_experiment_path, 'config.json'))
    ae_experiment_path = getenv('AE_EXPERIMENT_PATH')
    ae_config_path = join(ae_experiment_path, 'config.json')

    main(
        ae_configs=load_eval_configs(
            trainer_config=trainer_config,
            eval_config={'original_experiment_group_id': trainer_config['group_id']},
            config_class=AEEvalConfigs
        ),
        experiment_path=main_experiment_path,
        ae_experiment_path=ae_experiment_path,
        device=getenv('DEVICE', default='cpu')
    )

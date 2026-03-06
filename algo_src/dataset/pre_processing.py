import torch
from dataset.base import RLBatch
from dataset.pre_processing_configs import PreProcessingConfigs


def get_terminated(dataset, id_column):
    dataset["terminated"] = 0
    last_indices = dataset.groupby(id_column, group_keys=False).apply(lambda x: x.index[-1])
    dataset.loc[last_indices, "terminated"] = 1
    return dataset["terminated"].values


def normalize_data(dataset, columns, norm_dict):
    data = dataset.copy(deep=True)
    for col in columns:
        norm_param = norm_dict[col]
        if "mean" in norm_param:
            data[col] = (data[col] - norm_dict[col]["mean"]) / norm_dict[col]["std"]
        else:
            range_max = norm_param['range_max']
            range_min = norm_param['range_min']
            data_min = norm_param['min']
            data_max = norm_param['max']
            normalized_value = ((range_max - range_min) * (data[col] - data_min)) / (data_max - data_min)
            data[col] = normalized_value - 1

    return data[columns]


def un_normalize_data(dataset, columns, norm_dict):
    data = dataset.copy(deep=True)
    for col in columns:
        norm_param = norm_dict[col]
        col_data = data[col]
        if "mean" in norm_param:
            data[col] = col_data * norm_param["std"] + norm_param["mean"]
        else:
            range_max = norm_param['range_max']
            range_min = norm_param['range_min']
            data_min = norm_param['min']
            data_max = norm_param['max']
            un_norm_data = (col_data - range_min) * (data_max - data_min) / (range_max - range_min)
            data[col] = un_norm_data + data_min
    return data[columns]


def get_next_states(dataset, state_vector_columns, id_column):
    return dataset.groupby(id_column)[state_vector_columns].shift(-1).fillna(dataset[state_vector_columns])


def dataset_to_rl_sequences(
        dataset,
        data_pre_process_configs: PreProcessingConfigs,
        actions: torch.Tensor,
        normalization_params,
        device="cpu"
):
    terminated = get_terminated(
        dataset=dataset,
        id_column=data_pre_process_configs.episode_id_column
    )

    next_states = get_next_states(
        dataset=dataset,
        state_vector_columns=data_pre_process_configs.state_vector_columns,
        id_column=data_pre_process_configs.episode_id_column
    )

    normalized_states = normalize_data(
        dataset=dataset,
        columns=data_pre_process_configs.state_vector_columns,
        norm_dict=normalization_params
    )

    normalized_next_states = normalize_data(
        dataset=next_states,
        columns=data_pre_process_configs.state_vector_columns,
        norm_dict=normalization_params
    )

    reward = data_pre_process_configs.reward_function(dataset=dataset, terminated=terminated, pre_process_configs=data_pre_process_configs)

    state_dimension = len(data_pre_process_configs.state_vector_columns)
    rl_data = RLBatch(
        observations=torch.Tensor(normalized_states.values).to(device).reshape(-1, state_dimension),
        actions=actions.to(device),
        next_observations=torch.Tensor(normalized_next_states.values).to(device).reshape(-1, state_dimension),
        rewards=torch.Tensor(reward).to(device).reshape(-1, 1),
        terminals=torch.Tensor(terminated).to(device).reshape(-1, 1),
        ep_id=torch.Tensor(dataset[data_pre_process_configs.episode_id_column].values).to(device).reshape(-1, 1),
        time_step=torch.Tensor(dataset[data_pre_process_configs.timestep_column].values).to(device).reshape(-1, 1)
    )

    return rl_data

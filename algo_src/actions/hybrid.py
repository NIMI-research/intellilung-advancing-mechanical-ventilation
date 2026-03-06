import torch
from tensordict import TensorDict
from torch import Tensor


def create_hybrid_action_tensor_dict(continuous_actions, discrete_actions) -> TensorDict:
    return TensorDict({'discrete_actions': discrete_actions, 'continuous_actions': continuous_actions},
                      batch_size=discrete_actions.shape[0])


def get_discrete_action(actions: TensorDict) -> Tensor:
    return actions['discrete_actions']


def get_continuous_action(actions: TensorDict) -> Tensor:
    return actions['continuous_actions']


def flatten_action_dict_to_tensor(actions: TensorDict) -> Tensor:
    discrete_actions = actions['discrete_actions']
    continuous_actions = actions['continuous_actions']
    batch_size = actions.shape[0]
    return torch.cat(
        [continuous_actions.view(batch_size, -1), discrete_actions.view(batch_size, -1)],
        dim=-1
    )

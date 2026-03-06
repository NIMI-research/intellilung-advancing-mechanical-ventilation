from abc import abstractmethod

import numpy as np
import torch
from tensordict import TensorDict
from torch import Tensor, randint

from actions.hybrid import flatten_action_dict_to_tensor
from agents.base import RLAgent


class BasePolicy:
    @abstractmethod
    def select_action(self, obs: Tensor, deterministic: bool, **kwargs) -> Tensor:
        pass


class AgentPolicyWrapper(BasePolicy):

    def __init__(self, agent_path, keep_dict_hybrid_action=False):
        self.agent = RLAgent.load(save_path=agent_path)
        self.keep_dict_hybrid_action = keep_dict_hybrid_action

    def select_action(self, obs: Tensor, deterministic: bool, **kwargs) -> Tensor:
        with torch.no_grad():
            action = self.agent.get_action(state=obs, deterministic=deterministic, **kwargs).detach()
            if not self.keep_dict_hybrid_action and type(action) is TensorDict:
                action = flatten_action_dict_to_tensor(actions=action)
        return action

from typing import List, Union

import numpy as np
from numpy.typing import NDArray

from reward.base import RewardFunction
from reward.mortality import MortalityReward
from reward.range import RangeReward
from reward.ventilator_free_days import VentilatorFreeReward, VFDEachStep


class AddRewards(RewardFunction):
    reward_fns: List[Union[VentilatorFreeReward, RangeReward, VFDEachStep, MortalityReward]]

    def __call__(self, **kwargs) -> NDArray:
        rewards = []
        for fn in self.reward_fns:
            reward = fn(**kwargs)
            rewards.append(reward.reshape(1, -1))

        rewards = np.concatenate(rewards)
        print('rewards shape:', rewards.shape)
        final_reward = rewards.sum(axis=0)
        print('final reward shape:', final_reward.shape)

        return final_reward

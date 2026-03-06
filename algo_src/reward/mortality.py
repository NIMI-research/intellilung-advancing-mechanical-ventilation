import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame
from pydantic import Field

from reward.base import RewardFunction


class MortalityReward(RewardFunction):
    patient_alive_col: str = Field(default='daemo_discharge')
    morta_reward_scale: float

    def __call__(self, dataset: DataFrame, terminated: NDArray, **kwargs) -> NDArray:
        patient_alive = dataset[self.patient_alive_col].values
        # Map patient_alive from {0, 1} to {-1, 1}
        alive_multiplier = 2 * patient_alive - 1
        # If terminated is True, apply the reward; otherwise, reward remains 0.
        mv_reward = terminated * self.morta_reward_scale * alive_multiplier
        return mv_reward

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame
from pydantic import Field

from reward.base import RewardFunction


class VentilatorFreeReward(RewardFunction):
    patient_alive_col: str = Field(default='daemo_discharge')
    pause_until_next_mv_col: str = Field(default='pause_until_next')
    mv_duration_col: str = Field(default='mv_duration')
    post_extubation_interval_col: str = Field(default='post_extubation_interval')
    scale: float

    def __call__(self, dataset: DataFrame, terminated: NDArray, **kwargs) -> NDArray:
        pause_until_next_mv = dataset[self.pause_until_next_mv_col].values
        mv_duration = dataset[self.mv_duration_col].values
        # post_extubation_interval = dataset[self.post_extubation_interval_col].values
        patient_alive = dataset[self.patient_alive_col].values
        patient_alive = np.array(patient_alive, dtype=bool)

        next_mv_after_30 = (pause_until_next_mv + mv_duration) > 30
        # death_after_30 = (post_extubation_interval + mv_duration) > 30

        # if pause until next mv > 30 use 30 - mv duration for reward else pause until next mv
        # if post_extubation_interval > 30 use 30 - mv duration for reward else post extubation interval

        post_episode_vfd = ((30 - mv_duration) * next_mv_after_30) + (
                pause_until_next_mv * ~next_mv_after_30)
        # post_extubation_vfd = ((30 - mv_duration) * death_after_30) + (
        #         post_extubation_interval * ~death_after_30)

        # select post_episode_vfd for alive patients and post_extubation_vfd for others
        ventilator_free_days_30 = patient_alive * post_episode_vfd #+ (~patient_alive * post_extubation_vfd)

        # clip reward between (0, 30)
        ventilator_free_days_30 = ventilator_free_days_30.clip(0, 30)

        # normalize the reward
        mv_reward = ventilator_free_days_30 / 30

        # scale reward

        mv_reward = terminated * self.scale * mv_reward
        return mv_reward


class VFDEachStep(RewardFunction):
    patient_alive_col: str = Field(default='daemo_discharge')
    pause_until_next_mv_col: str = Field(default='pause_until_next')
    mv_duration_col: str = Field(default='mv_duration')
    post_extubation_interval_col: str = Field(default='post_extubation_interval')
    min_reward: float
    max_reward: float

    def __call__(self, dataset: DataFrame, terminated: NDArray, **kwargs) -> NDArray:
        pause_until_next_mv = dataset[self.pause_until_next_mv_col].values
        mv_duration = dataset[self.mv_duration_col].values
        # post_extubation_interval = dataset[self.post_extubation_interval_col].values
        patient_alive = dataset[self.patient_alive_col].values
        patient_alive = np.array(patient_alive, dtype=bool)

        next_mv_after_30 = (pause_until_next_mv + mv_duration) > 30
        # death_after_30 = (post_extubation_interval + mv_duration) > 30

        # if pause until next mv > 30 use 30 - mv duration for reward else pause until next mv
        # if post_extubation_interval > 30 use 30 - mv duration for reward else post extubation interval

        post_episode_vfd = ((30 - mv_duration) * next_mv_after_30) + (
                pause_until_next_mv * ~next_mv_after_30)
        # post_extubation_vfd = ((30 - mv_duration) * death_after_30) + (
        #         post_extubation_interval * ~death_after_30)

        # select post_episode_vfd for alive patients and post_extubation_vfd for others
        ventilator_free_days_30 = patient_alive * post_episode_vfd #+ (~patient_alive * post_extubation_vfd)

        # clip reward between (0, 30)
        ventilator_free_days_30 = ventilator_free_days_30.clip(0, 30)

        # normalize the reward
        mv_reward = ventilator_free_days_30 / 30
        mv_reward = mv_reward * (self.max_reward - self.min_reward)
        mv_reward += self.min_reward

        return mv_reward

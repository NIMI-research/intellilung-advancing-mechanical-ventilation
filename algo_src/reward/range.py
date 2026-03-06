import numpy as np
import pandas as pd
from numpy import zeros
from numpy.typing import NDArray
from pandas import DataFrame
from pydantic import computed_field
from reward.base import RewardFunction
from utils.files import load_json


class RangeReward(RewardFunction):
    normalize: bool = False
    time_penalty: bool = False
    ranges_file_path: str

    @computed_field
    @property
    def state_vector_ranges(self) -> dict:
        return load_json(self.ranges_file_path)

    def __call__(self, dataset: DataFrame, terminated: NDArray, pre_process_configs, **kwargs) -> NDArray:
        print('ranges normalized: ', self.normalize)
        ranges_keys = list(self.state_vector_ranges.keys())

        selected_states_for_experiment = pre_process_configs.get_list_of_states()

        filtered_ranges_keys = [key for key in ranges_keys if key in selected_states_for_experiment]
        print('State Variables Available for Reward Calculation: ', filtered_ranges_keys)

        total_rows = len(dataset)
        reward = zeros(total_rows, dtype=np.float32)
        scale = 0
        id_column = pre_process_configs.episode_id_column
        next_states = dataset.groupby(id_column)[filtered_ranges_keys].shift(-1).fillna(dataset[filtered_ranges_keys])
        for key in filtered_ranges_keys:
            low = self.state_vector_ranges[key]['range'][0]
            high = self.state_vector_ranges[key]['range'][1]
            priority = self.state_vector_ranges[key]['priority']
            column_values = next_states[key].values
            reward += (np.logical_and(low <= column_values, column_values <= high)) * priority
            scale += priority

        if self.normalize:
            reward /= scale

        df_describe = pd.DataFrame(reward)
        print(df_describe.describe())
        assert not np.isnan(
            reward).any(), f'Reward is NaN possibly due to NaN value in the dataset for variables: {filtered_ranges_keys}'
        if self.time_penalty:
            reward = reward - 1
        return reward

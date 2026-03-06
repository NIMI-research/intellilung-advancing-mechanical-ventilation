from numpy.typing import NDArray
from pandas import DataFrame
from pydantic import BaseModel


class RewardFunction(BaseModel):
    def __call__(self, dataset: DataFrame, terminated: NDArray, pre_process_configs, **kwargs) -> NDArray:
        raise NotImplementedError

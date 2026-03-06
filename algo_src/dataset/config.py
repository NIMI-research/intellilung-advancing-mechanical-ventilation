import pathlib
import warnings
from functools import cached_property
from os.path import join, dirname
from typing import Annotated

import pandas as pd
from pydantic import BaseModel
from pydantic import PlainSerializer

from dataset.pre_processing_configs import PreProcessingConfigs, get_pre_processing_configs
from dataset.test_train_split import create_stratified_splits
from utils.files import save_json, load_yaml, load_json

Path = Annotated[
    pathlib.Path, PlainSerializer(lambda x: str(x))
]


def read_to_df(file_path: str | Path) -> pd.DataFrame:
    """
    Look for a .json file with the same stem.
    If found, read it with pd.read_json; otherwise read the CSV.
    """
    file_path = Path(file_path)
    json_candidate = file_path.with_suffix(".json")
    json_gzip_candidate = file_path.with_suffix(".gz")

    if json_candidate.exists():
        return pd.read_json(json_candidate)
    elif json_gzip_candidate.exists():
        return pd.read_json(
            json_gzip_candidate,
            compression="gzip"
        )
    else:
        return pd.read_csv(file_path)


class BaseDatasetConfig(BaseModel):
    dataset_type: str
    dataset_path: Path
    train_split_path: Path = None
    test_split_path: Path = None
    test_split_size: float = 0.2
    seed: int | None = None

    @property
    def dataset(self) -> pd.DataFrame:
        return read_to_df(self.dataset_path)

    @property
    def train_dataset_split(self) -> pd.DataFrame:
        return read_to_df(self.train_split_path)

    @property
    def test_dataset_split(self) -> pd.DataFrame:
        return read_to_df(self.test_split_path)


class DatasetLoadConfig(BaseDatasetConfig):
    dataset_type: str
    dataset_path: Path
    train_split_path: Path = None
    test_split_path: Path = None
    dataset_normalization_param_path: Path
    test_split_size: float = 0.2
    seed: int | None = None
    discrete_actions_file_path: Path = join('configs', 'discrete_action_ranges.json')

    @cached_property
    def discrete_action_bin_ranges(self):
        return load_json(self.discrete_actions_file_path)

    @cached_property
    def normalization_params(self):
        return load_json(self.dataset_normalization_param_path)


class TextDatasetConfig(BaseDatasetConfig):
    encoder_tokenizer_path: Path
    decoder_tokenizer_path: Path
    base_dataset_config_path: Path

    @cached_property
    def base_dataset_config(self):
        return load_dataset_config(dataset_config_path=self.base_dataset_config_path)


def load_dataset_config(dataset_config_path, experiment_path=None, train_test_split=True) -> DatasetLoadConfig:
    configs = load_yaml(dataset_config_path)
    dataset_path = configs.get("dataset_path")
    save_dir = experiment_path if experiment_path is not None else dirname(dataset_path)
    dataset_type = configs.get("dataset_type")
    pre_processing_metadata: PreProcessingConfigs = get_pre_processing_configs(configs_id=dataset_type)
    episode_id_column = pre_processing_metadata.episode_id_column

    if train_test_split:

        if configs.get('train_split_path') is None or configs.get('test_split_path') is None:
            warnings.warn('train/test split paths not found, creating new splits now')
            dataset = pd.read_csv(dataset_path)
            test_split_size = configs.get("test_split_size")
            seed = configs.get("seed")
            normalization_data, test_data = create_stratified_splits(dataset=dataset,
                                                                     episode_id_column=episode_id_column,
                                                                     test_split_size=test_split_size,
                                                                     seed=seed)
            train_split_path = join(save_dir, 'train.csv')
            test_split_path = join(save_dir, 'test.csv')
            normalization_data.to_csv(train_split_path)
            test_data.to_csv(test_split_path)
            configs['train_split_path'] = train_split_path
            configs['test_split_path'] = test_split_path
    normalization_data_path = configs['train_split_path'] if train_test_split else dataset_path
    if configs.get("dataset_normalization_param_path") is None:
        warnings.warn('dataset normalization params path not found, creating normalization params now')
        dataset_normalization_param_path = join(save_dir, 'norm_dict.json')
        normalization_data = pd.read_csv(normalization_data_path)
        state_norm_dict = create_norm_dict(
            data=normalization_data,
            columns=pre_processing_metadata.state_vector_columns
        )
        action_norm_dict = create_ranged_norm_dict(
            data=normalization_data,
            columns=pre_processing_metadata.get_list_of_actions(),
            range_min=-1,
            range_max=1
        )

        norm_dict = {**state_norm_dict, **action_norm_dict}
        save_json(
            data=norm_dict,
            path=dataset_normalization_param_path
        )
        configs["dataset_normalization_param_path"] = dataset_normalization_param_path
    return DatasetLoadConfig(**configs)


def load_text_dataset(dataset_config_path) -> TextDatasetConfig:
    configs = load_yaml(dataset_config_path)
    return TextDatasetConfig(**configs)


def create_norm_dict(data, columns):
    norm_dict = {}
    for col in columns:
        norm_dict[col] = {"mean": data[col].mean(), "std": data[col].std()}
    return norm_dict


def create_ranged_norm_dict(data, columns, range_min, range_max):
    norm_dict = {}
    for col in columns:
        norm_dict[col] = {
            "min": float(data[col].min()),
            "max": float(data[col].max()),
            "range_min": range_min,
            "range_max": range_max
        }
    return norm_dict

from os import getenv

import pandas as pd
import pytest
from dotenv import load_dotenv

from config import PreProcessingChecksConfig
from utils import load_yaml

load_dotenv()


@pytest.fixture(scope='session')
def configs():
    configs_path = getenv('VALIDATION_CHECK_CONFIGS_FILE_PATH')
    configs_dict = load_yaml(configs_path)
    return PreProcessingChecksConfig(**configs_dict)


@pytest.fixture(scope='session')
def dataset():
    path = getenv('DATASET_PATH')
    return pd.read_csv(path)


@pytest.fixture
def required_variables(configs):
    return [*configs.state_space, *configs.action_space, *configs.misc_required_variables, configs.episode_id_column]


@pytest.fixture
def episode_id_column(configs):
    return configs.episode_id_column


@pytest.fixture
def variable_ranges(configs):
    return configs.variable_ranges


@pytest.fixture
def minimum_episode_size(configs):
    return configs.minimum_episode_size


@pytest.fixture
def state_action_space(configs):
    return [*configs.state_space, *configs.action_space]

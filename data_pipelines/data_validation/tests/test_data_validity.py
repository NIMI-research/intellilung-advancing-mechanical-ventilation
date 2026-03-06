import pandas as pd


def test_dataset_minimum_episode_size(dataset: pd.DataFrame, minimum_episode_size, episode_id_column):
    count_df = dataset.groupby(episode_id_column).size().reset_index(name='count').sort_values('count', ascending=True)
    episodes_atleast_min_size = count_df['count'] >= minimum_episode_size
    assert episodes_atleast_min_size.all(), f'Dataset contain episodes below minimum required size of {minimum_episode_size}.'


def test_all_required_columns_available(dataset: pd.DataFrame, required_variables):
    missing = set(required_variables) - set(dataset.columns)

    assert not missing, f'Missing columns: {missing}'


def test_required_columns_does_not_contain_nan_values(dataset: pd.DataFrame, required_variables):
    cols_with_nan = dataset[required_variables].isnull().any()
    assert not cols_with_nan.any(), f'These required columns have missing values: {cols_with_nan[cols_with_nan].index.tolist()}'

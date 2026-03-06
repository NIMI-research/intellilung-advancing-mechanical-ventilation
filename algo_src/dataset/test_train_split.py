from sklearn.model_selection import train_test_split


def _assign_quantile(size, quantiles):
    if size <= quantiles[0.25]:
        return 0
    elif size <= quantiles[0.5]:
        return 1
    elif size <= quantiles[0.75]:
        return 2
    else:
        return 3

def create_stratified_splits(dataset, episode_id_column, test_split_size, seed=None):
    """Stratified splitting based on episode length and daemo_discharge.
    Split stay_ids exclusively into train or test set"""
    episode_sizes = dataset.groupby(episode_id_column).size()
    quantiles = episode_sizes.quantile([0.25, 0.5, 0.75])
    dataset = dataset.copy()
    dataset['quantile'] = dataset[episode_id_column].map(lambda eid: _assign_quantile(episode_sizes[eid], quantiles))

    # Use median quantile for stay_id
    stay_stratification = dataset.groupby('stay_id').agg({'quantile': 'median', 'daemo_discharge': 'median'}).reset_index()
    stay_stratification['stratify_group'] = stay_stratification['quantile'].astype(str) + "_" + stay_stratification['daemo_discharge'].astype(str)

    # Perform stratified split on stay_id level
    train_stays, test_stays = train_test_split(stay_stratification['stay_id'], test_size=test_split_size, stratify=stay_stratification['stratify_group'], random_state=seed)
    dataset['split'] = dataset['stay_id'].apply(lambda x: 'train' if x in train_stays.values else 'test')

    # Split into final train and test sets
    train_df = dataset[dataset['split'] == 'train'].drop(columns=['split', 'quantile']).copy()
    test_df = dataset[dataset['split'] == 'test'].drop(columns=['split', 'quantile']).copy()

    return train_df, test_df


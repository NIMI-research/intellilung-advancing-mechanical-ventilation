from datetime import datetime
from os import getenv, makedirs
from os.path import join
from typing import Union, List

from dotenv import load_dotenv
from pydantic import BaseModel

from algorithms.offline_rl.cql import CQLConfig
from algorithms.offline_rl.hybrid_iql import TrainConfig
from reward.stacking import AddRewards
from reward.ventilator_free_days import VentilatorFreeReward
from utils.files import load_yaml, load_json, save_json

configs_classes = {
    'factored-CQL': CQLConfig,
    'Hybrid-IQL': TrainConfig
}


class RewardTuneConfigs(BaseModel):
    number_of_evaluations: int
    rewards: List[Union[AddRewards]]


def main(dataset_config_path, algorithm_configs, fqe_config_path, reward_tune_configs, device):
    algo = algorithm_configs['name']
    root_path = join(algorithm_configs['root_path'], f'reward-tune-{algo}-{datetime.now()}')
    algorithm_configs['root_path'] = join(root_path, 'tasks')

    makedirs(root_path, exist_ok=True)
    save_json(data=reward_tune_configs.model_dump(), path=join(root_path, 'reward_tune_configs.json'))

    for reward_index, reward_fn in enumerate(reward_tune_configs.rewards):
        experiment = {}
        for train_iter in range(reward_tune_configs.number_of_evaluations):
            config_class = configs_classes[algo]
            algo_config = config_class(**algorithm_configs)
            algo_config.experiment_folder_name = str(
                (reward_index * reward_tune_configs.number_of_evaluations) + train_iter)
            algo_config.save_only_last_checkpoint = True

            experiment['algo_config'] = algo_config.model_dump()
            experiment['fqe_config_path'] = fqe_config_path
            experiment['dataset_config_path'] = dataset_config_path
            experiment['exp_id'] = algo_config.experiment_path
            experiment['reward_fn'] = reward_fn.model_dump()
            experiment['device'] = device
            experiment['finished'] = False
            save_json(data=experiment, path=join(algo_config.experiment_path, 'experiment_config.json'))


if __name__ == "__main__":
    load_dotenv()
    main(
        dataset_config_path=getenv('DATASET_CONFIG_PATH'),
        algorithm_configs=load_yaml(getenv('ALGORITHM_CONFIGS_PATH')),
        fqe_config_path=getenv('DIST_FQE_CONFIG_PATH'),
        reward_tune_configs=RewardTuneConfigs(**load_json(getenv('REWARD_TUNE_CONFIGS'))),
        device=getenv('DEVICE', default='cpu')
    )

from os import getenv, makedirs
from dotenv import load_dotenv
from dataset.config import load_dataset_config
from dataset.load import load_behavior_policy_eval_dataset
from dataset.pre_processing_configs import get_pre_processing_configs
from estimators.physician import InitStateEstimator
from utils.files import load_yaml


def main(dataset_config_path, device, gamma: float):
    dataset_config_dict = load_yaml(dataset_config_path)
    dataset_type = dataset_config_dict['dataset_type']
    experiment_path = f'logs/behavior_policy_{dataset_type}_gamma={gamma}'
    makedirs(experiment_path, exist_ok=True)
    dataset_config = load_dataset_config(dataset_config_path=dataset_config_path,
                                         experiment_path=experiment_path, train_test_split=False)
    pre_processing_config = get_pre_processing_configs(configs_id=dataset_config.dataset_type)
    buffer = load_behavior_policy_eval_dataset(dataset=dataset_config.test_dataset_split,
                                               dataset_configs=dataset_config,
                                               pre_process_configs=pre_processing_config,
                                               device=device, flatten_hybrid_actions=True)
    physician_mdp_estimator = InitStateEstimator(rl_dataset=buffer.sample_all())
    avg_return = physician_mdp_estimator(gamma=gamma)
    print('Physician Returns (MDP Estimate): ', avg_return)


if __name__ == "__main__":
    load_dotenv()
    main(
        dataset_config_path=getenv('DATASET_CONFIG_PATH'),
        device=getenv('DEVICE', default='cpu'),
        gamma=float(getenv('GAMMA', default=0.996))
    )

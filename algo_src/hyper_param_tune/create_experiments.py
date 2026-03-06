from datetime import datetime
from os import getenv, makedirs
from os.path import join

from dotenv import load_dotenv

from utils.files import load_json, save_json

from itertools import product
from typing import List, Dict, Any
import copy

from utils.files import load_yaml


def set_nested_value(config: Dict[str, Any], key: str, value: Any) -> None:
    """
    Set a value in a potentially nested dict using dot notation.

    Args:
        config: Configuration dictionary
        key: Key path (e.g., 'param' or 'model.param' or 'optimizer.lr')
        value: Value to set
    """
    keys = key.split('.')
    current = config

    # Navigate to the parent of the target key
    for k in keys[:-1]:
        if k not in current:
            current[k] = {}
        current = current[k]

    # Set the final value
    current[keys[-1]] = value


def generate_hyperparameter_combinations(configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Generate all hyperparameter combinations for each algorithm configuration.

    Reads base algorithm config from file and overrides tune parameters.
    Parameters within the same tune item vary together.
    Different tune items create a cartesian product.

    Args:
        configs: List of algorithm configurations with tune parameters

    Returns:
        List of expanded configurations with algo_config containing overridden params
    """
    all_combinations = []

    for algo_config in configs:
        # Load base algorithm configuration
        base_algo_config = load_yaml(algo_config['base_algo_config_path'])

        # Extract non-tune fields
        base_fields = {k: v for k, v in algo_config.items() if k != 'tune'}

        # Get tune configurations
        tune_items = algo_config.get('tune', [])

        if not tune_items:
            # No tuning parameters, just use base config
            result_config = base_fields.copy()
            result_config['algo_config'] = base_algo_config
            all_combinations.append(result_config)
            continue

        # For each tune item, create parameter options
        tune_options = []
        for tune_item in tune_items:
            params = tune_item['params']
            values = tune_item['values']

            # Create options where all params get the same value
            options = []
            for value in values:
                option = {param: value for param in params}
                options.append(option)

            tune_options.append(options)

        # Generate cartesian product of all tune item options
        for combination in product(*tune_options):
            # Create a deep copy of base algo config
            algo_config_copy = copy.deepcopy(base_algo_config)

            # Override parameters from all tune items in this combination
            for param_set in combination:
                for param_name, param_value in param_set.items():
                    set_nested_value(algo_config_copy, param_name, param_value)

            # Create result config
            result_config = base_fields.copy()
            result_config['algo_config'] = algo_config_copy
            all_combinations.append(result_config)

    return all_combinations


def main(hyper_param_tune_config, device):
    root_path = join(hyper_param_tune_config['root_path'], f'hyper-param-tune-{datetime.now()}')
    tasks_root_path = join(root_path, 'tasks')

    makedirs(root_path, exist_ok=True)
    save_json(data=hyper_param_tune_config, path=join(root_path, 'hyper_param_tune_configs.json'))
    print("Root Path: ", root_path)

    configs = generate_hyperparameter_combinations(hyper_param_tune_config['tune_configs'])

    total_tasks = 0

    for i, config in enumerate(configs):

        for train_iter in range(config['number_of_evaluations']):
            experiment = {}
            experiment_folder_name = str(total_tasks)
            task_path = join(tasks_root_path, experiment_folder_name)
            makedirs(task_path, exist_ok=True)

            experiment['algo_config'] = config['algo_config']
            experiment['algo_config']['experiment_folder_name'] = experiment_folder_name
            experiment['algo_config']['root_path'] = join(root_path, 'tasks')

            experiment['algo_config']['save_only_last_checkpoint'] = True

            experiment['fqe_config_path'] = config['fqe_config_path']
            experiment['dataset_config_path'] = config['dataset_config_path']
            experiment['exp_id'] = task_path
            experiment['device'] = device
            experiment['finished'] = False
            save_json(data=experiment, path=join(task_path, 'experiment_config.json'))
            total_tasks += 1
    print("Total tasks:", total_tasks + 1)


if __name__ == "__main__":
    load_dotenv()
    main(
        hyper_param_tune_config=load_json(getenv('HYPER_PARAM_TUNE_CONFIGS')),
        device=getenv('DEVICE', default='cpu')
    )

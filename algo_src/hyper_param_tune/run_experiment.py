import argparse
from os.path import join

from dotenv import load_dotenv

from agents.configs import load_eval_configs
from algorithms.offline_rl.behavior_cloning import train as train_bc
from algorithms.offline_rl.cql import train as train_cql_factored
from algorithms.offline_rl.cql import train_cql, train_cql_no_addons
from algorithms.offline_rl.discrete_behavior_cloning import train as train_discrete_bc
from algorithms.offline_rl.edac import train as train_edac
from algorithms.offline_rl.hybrid_edac import train as train_hybrid_edac
from algorithms.offline_rl.iql import train as train_iql
from algorithms.offline_rl.hybrid_iql import train as train_hybrid_iql
from algorithms.offline_rl.hybrid_behavior_cloning import train as train_hybrid_behavior_cloning
from algorithms.offline_rl.discrete_iql import train as train_discrete_iql
from algorithms.eval.dist_fqe import train as train_dist_fqe
from utils.files import load_json, save_json
import os
import torch
from torch.nn.functional import cross_entropy
from actions.hybrid import get_continuous_action, get_discrete_action
from algorithms.eval.eval_ood_hybrid_ae import main as eval_ood_hybrid_ae, AEEvalConfigs

os.environ["WANDB_DISABLED"] = "true"

algorithm_trainers = {
    'CQL': train_cql_no_addons,
    'CF-CQL': train_cql_factored,
    'IQL': train_iql,
    'EDAC': train_edac,
    'BehaviorCloning': train_bc,
    'DiscreteBehaviorCloning': train_discrete_bc,
    'Hybrid-IQL': train_hybrid_iql,
    'Hybrid-EDAC': train_hybrid_edac,
    'C-CQL': train_cql,
    'Hybrid-BC': train_hybrid_behavior_cloning,
    'Discrete-IQL': train_discrete_iql
}


def main(task_path, ae_experiment_path):
    experiment_config = load_json(join(task_path, 'experiment_config.json'))
    if not experiment_config['finished']:
        algorithm_configs = experiment_config['algo_config']
        print(experiment_config)
        algo = algorithm_configs['name']

        device = experiment_config['device']

        trainer = algorithm_trainers[algo]

        print("\n============================================================")
        print(f"\n========== Step 1: Training Policy Using {algo} ==========")
        print("\n============================================================")

        experiment_path = trainer(
            dataset_config_path=experiment_config['dataset_config_path'],
            config=algorithm_configs,
            device=device
        )

        print("\n========================================================================")
        print("\n========== Step 2: Policy Evaluation using Distributional FQE ==========")
        print("\n========================================================================")

        train_dist_fqe(
            fqe_config_path=experiment_config['fqe_config_path'],
            experiment_path=experiment_path,
            device=device
        )

        print("\n========================================================================")
        print("\n========== Step 3: Evaluate Policy Distributional Mismatch ==========")
        print("\n========================================================================")
        algo_configs = load_json(join(experiment_path, 'config.json'))

        eval_ood_hybrid_ae(
            ae_configs=load_eval_configs(
                trainer_config=algo_configs,
                eval_config={'original_experiment_group_id': algo_configs['group_id']},
                config_class=AEEvalConfigs
            ),
            experiment_path=algo_configs['experiment_path'],
            ae_experiment_path=ae_experiment_path,
            device=device
        )

        experiment_config['finished'] = True
        save_json(data=experiment_config, path=join(task_path, 'experiment_config.json'))


if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_root_path', type=str)
    parser.add_argument('--task_id')
    args = parser.parse_args()
    main(
        task_path=join(args.experiment_root_path, 'tasks', str(args.task_id)),
        ae_experiment_path=os.getenv('AE_EXPERIMENT_PATH'),
    )

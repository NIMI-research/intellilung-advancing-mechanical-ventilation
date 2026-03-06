from os import getenv, environ
from os.path import join
from datetime import datetime

from dotenv import load_dotenv


from algorithms.offline_rl.cql import train as train_cql_factored, train_factored_only
from algorithms.offline_rl.cql import train_cql, train_cql_no_addons


from algorithms.offline_rl.hybrid_edac import train as train_hybrid_edac

from algorithms.offline_rl.hybrid_iql import train as train_hybrid_iql
from algorithms.offline_rl.discrete_iql import train as train_discrete_iql


from algorithms.eval.dist_fqe import train as train_dist_fqe
from eval_dataset_policy import train_fqe_on_dataset_policy
from generate_analysis_scripts_results import run_analysis_scripts
from utils.files import load_yaml

algorithm_trainers = {
    'CQL': train_cql_no_addons,
    'CF-CQL': train_cql_factored,
    'F-CQL': train_factored_only,
    'Hybrid-IQL': train_hybrid_iql,
    'Hybrid-EDAC': train_hybrid_edac,
    'C-CQL': train_cql,
    'Discrete-IQL': train_discrete_iql
}


def main(dataset_config_path, algorithm_configs, fqe_config_path, device):
    algo = algorithm_configs['name']
    algorithm_configs['root_path'] = join(algorithm_configs['root_path'], f'e2e-{algo}-{datetime.now()}')

    trainer = algorithm_trainers[algo]

    print("\n============================================================")
    print(f"\n========== Step 1: Training Policy Using {algo} ==========")
    print("\n============================================================")

    experiment_path = trainer(
        dataset_config_path=dataset_config_path,
        config=algorithm_configs,
        device=device
    )

    print("\n========================================================================")
    print("\n========== Step 2: Policy Evaluation using Distributional FQE ==========")
    print("\n========================================================================")

    train_dist_fqe(
        fqe_config_path=fqe_config_path,
        experiment_path=experiment_path,
        device=device
    )

    print("\n================================================================================")
    print("\n========== Step 3: Dataset Policy Evaluation using Distributional FQE ==========")
    print("\n================================================================================")
    dataset_policy_eval_path = train_fqe_on_dataset_policy(
        dataset_config_path=dataset_config_path,
        fqe_config_path=fqe_config_path,
        device=device,
        root_path=algorithm_configs['root_path']
    )

    print("\n=============================================================")
    print("\n========== Step 4: Running Policy Analysis Scripts ==========")
    print("\n=============================================================")

    run_analysis_scripts(
        experiment_path=experiment_path,
        device=device,
        behaviour_policy_eval_path=dataset_policy_eval_path
    )


if __name__ == "__main__":
    load_dotenv()
    main(
        dataset_config_path=getenv('DATASET_CONFIG_PATH'),
        algorithm_configs=load_yaml(getenv('ALGORITHM_CONFIGS_PATH')),
        fqe_config_path=getenv('DIST_FQE_CONFIG_PATH'),
        device=getenv('DEVICE', default='cpu')
    )

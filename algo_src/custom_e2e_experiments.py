from os import getenv, environ
from os.path import join
from datetime import datetime
from dotenv import load_dotenv
import torch
from torch.nn.functional import cross_entropy
from actions.hybrid import get_continuous_action, get_discrete_action
from agents.configs import load_eval_configs
from algorithms.offline_rl.cql import train as train_cql_factored, train_factored_only, train_cql_no_addons, train_cql
from algorithms.offline_rl.hybrid_edac import train as train_hybrid_edac
from algorithms.offline_rl.hybrid_iql import train as train_hybrid_iql
from algorithms.offline_rl.discrete_iql import train as train_discrete_iql
from algorithms.eval.eval_ood_hybrid_ae import main as eval_ood_hybrid_ae, AEEvalConfigs
from algorithms.eval.dist_fqe import train as train_dist_fqe
from eval_dataset_policy import train_fqe_on_dataset_policy
from generate_analysis_scripts_results import run_analysis_scripts
from utils.files import load_yaml, load_json

algorithm_trainers = {
    'CQL': train_cql_no_addons,
    'CF-CQL': train_cql_factored,
    'F-CQL': train_factored_only,
    'Hybrid-IQL': train_hybrid_iql,
    'Hybrid-EDAC': train_hybrid_edac,
    'C-CQL': train_cql,
    'Discrete-IQL': train_discrete_iql
}

experiments = [
    # {
    #     'config': load_yaml('configs/combined_mimic_eicu_hirid_hybrid/hybrid_iql_configs.yml'),
    #     'dataset_config_path': 'configs/combined_mimic_eicu_hirid_hybrid/dataset_config_mimic_test.yml',
    #     'repeat': 5,
    #     'extra_label': 'external_eval_mimic'
    # },

    # {
    #     'config': load_yaml('configs/combined_mimic_eicu_hirid/c_cql_config.yml'),
    #     'dataset_config_path': 'configs/combined_mimic_eicu_hirid/dataset_config_mimic_test.yml',
    #     'repeat': 5,
    #     'extra_label': 'external_eval_mimic'
    # },
    # {
    #     'config': load_yaml('configs/combined_mimic_eicu_hirid/cf_cql_config.yml'),
    #     'dataset_config_path': 'configs/combined_mimic_eicu_hirid/dataset_config_mimic_test.yml',
    #     'repeat': 5,
    #     'extra_label': 'external_eval_mimic'
    # },
    # {
    #     'config': load_yaml('configs/combined_mimic_eicu_hirid/vanilla_cql_config.yml'),
    #     'dataset_config_path': 'configs/combined_mimic_eicu_hirid/dataset_config_mimic_test.yml',
    #     'repeat': 5,
    #     'extra_label': 'external_eval_mimic'
    # },
    # {
    #     'config': load_yaml('configs/combined_mimic_eicu_hirid_hybrid/hybrid_edac_configs.yml'),
    #     'dataset_config_path': 'configs/combined_mimic_eicu_hirid_hybrid/dataset_config_mimic_test.yml',
    #     'repeat': 5,
    #     'extra_label': 'external_eval_mimic'
    # },
    # {
    #     'config': load_yaml('configs/combined_mimic_eicu_hirid/discrete_iql_configs.yml'),
    #     'dataset_config_path': 'configs/combined_mimic_eicu_hirid/dataset_config.yml',
    #     'repeat': 5,
    #     'extra_label': 'extra_eval_discrete_iql'
    # }

    # {
    #     'config': load_yaml('configs/combined_mimic_eicu_hirid/f_cql_config.yml'),
    #     'dataset_config_path': 'configs/combined_mimic_eicu_hirid/dataset_config.yml',
    #     'repeat': 5,
    #     'extra_label': 'extra_eval_f_cql',
    #     'ae_experiment_path': 'logs/Hybrid-BehaviorPolicyDensity-2025-07-28 13:53:03.424375'
    # },
    {
        'config': load_yaml('configs/combined_mimic_eicu_hirid/discrete_iql_configs.yml'),
        'dataset_config_path':  'configs/combined_mimic_eicu_hirid/dataset_config_mimic_test.yml',
        'repeat': 5,
        'extra_label': 'external_eval_mimic',
        'ae_experiment_path': 'logs/Hybrid-BehaviorPolicyDensity-2025-07-28 13:53:03.424375'
    },

]


def main(fqe_config_path, ae_experiment_path, device):
    for i, experiment in enumerate(experiments):
        print(f'--------------- Experiment: {i + 1}/{len(experiment)} ---------------')
        algorithm_configs = experiment['config']
        dataset_config_path = experiment['dataset_config_path']
        repeat = experiment['repeat']
        extra_label = experiment['extra_label']
        algo = algorithm_configs['name']
        algorithm_configs['root_path'] = join(algorithm_configs['root_path'], 'custom_experiments',
                                              f'e2e-{extra_label}-{algo}-{datetime.now()}')
        algorithm_configs['save_only_last_checkpoint'] = True
        for j in range(repeat):
            print(f'--------------- Repeat: {j + 1}/{repeat} ---------------')
            trainer = algorithm_trainers[algo]
            algorithm_configs['experiment_folder_name'] = str(j)

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

            print("\n========================================================================")
            print("\n========== Step 3: Evaluate Policy Distributional Mismatch ==========")
            print("\n========================================================================")
            algo_configs = load_json(join(experiment_path, 'config.json'))

            ae_experiment_path = experiment.get('ae_experiment_path', ae_experiment_path)

            eval_ood_hybrid_ae(
                ae_configs=load_eval_configs(
                    trainer_config=algo_configs,
                    eval_config={'original_experiment_group_id': algo_configs['group_id']},
                    config_class=AEEvalConfigs
                ),
                experiment_path=algo_configs['experiment_path'],
                ae_experiment_path=ae_experiment_path,
                device=getenv('DEVICE', default='cpu')
            )


if __name__ == "__main__":
    load_dotenv()
    main(
        fqe_config_path=getenv('DIST_FQE_CONFIG_PATH'),
        ae_experiment_path=getenv('AE_EXPERIMENT_PATH'),
        device=getenv('DEVICE', default='cpu')
    )

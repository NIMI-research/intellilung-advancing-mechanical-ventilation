from os import getenv
from analysis.log_actions import main as log_actions
from analysis.dist_compare import main as dist_compare
from analysis.action_compare import main as action_compare
from analysis.log_fqe_values import main as action_values

from dotenv import load_dotenv


def run_analysis_scripts(experiment_path, device, behaviour_policy_eval_path):
    print('################# Logging Actions #################')
    log_actions(
        experiment_path=experiment_path,
        device=device
    )
    print('################# Comparing AI vs Clinician Actions Distribution #################')
    dist_compare(
        experiment_path=experiment_path
    )
    print('################# Comparing AI vs Clinician Actions #################')
    action_compare(
        experiment_path=experiment_path
    )
    if behaviour_policy_eval_path:
        print('################# Logging AI and Clinician Action Values #################')
        action_values(
            experiment_path=experiment_path,
            device=device,
            behaviour_policy_eval_path=behaviour_policy_eval_path
        )


if __name__ == "__main__":
    load_dotenv()

    run_analysis_scripts(
        experiment_path=getenv('EXPERIMENT_PATH'),
        device=getenv('DEVICE', default='cpu'),
        behaviour_policy_eval_path=getenv('BEHAVIOUR_POLICY_EVAL_PATH', default=None)
    )

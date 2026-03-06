from os import makedirs
from os.path import join, exists

import wandb

from agents.base import Agent
from utils.files import save_json, load_json


class AgentEvaluator:
    def __init__(self, experiment_path: str):
        self.experiment_path = experiment_path
        makedirs(self.experiment_path, exist_ok=True)

    def eval(self, checkpoint_id: int, agent: Agent, panel_name='checkpoints', **kwargs):
        results = {}
        results_path = join(self.experiment_path, f'results.json')
        if exists(results_path):
            results = load_json(results_path)

        checkpoint_key = 'checkpoint'
        wandb.define_metric(f'checkpoints/*', step_metric=checkpoint_key)

        final_log = agent.eval(**kwargs)

        # log results locally
        results[str(checkpoint_id)] = final_log
        save_json(data=results, path=results_path)

        # log results wandb
        final_log = {f'{panel_name}/{key}': val for key, val in final_log.items()}
        wandb.log({**final_log, checkpoint_key: checkpoint_id})

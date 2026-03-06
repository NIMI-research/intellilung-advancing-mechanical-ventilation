from os import makedirs, getenv
from os.path import join

import wandb
from dotenv import load_dotenv
from tqdm import tqdm

from agents.base import Agent
from agents.configs import TrainerExperimentConfig
from utils.files import save_json

load_dotenv()
WANDB_DISABLE_CHECKPOINT = getenv('WANDB_DISABLE_CHECKPOINT', 'false').lower() == 'true'


class AgentTrainer:
    def __init__(self, agent: Agent, train_config: TrainerExperimentConfig):
        self.agent = agent
        self.train_config = train_config
        self.step_key = 'Step'

        makedirs(join(self.train_config.experiment_path, 'checkpoints'), exist_ok=True)
        save_json(data=self.train_config.model_dump(), path=join(self.train_config.experiment_path, 'config.json'))

        wandb.define_metric(self.step_key)

        wandb.define_metric('eval/*', step_metric=self.step_key)
        wandb.define_metric('train/*', step_metric=self.step_key)

    def save_checkpoint(self, checkpoints_folder_path, weights_folder_path, step):
        agent_save_path = join(checkpoints_folder_path, f'{step}.pkl')
        weights_save_path = join(weights_folder_path, f'{step}.pth')
        self.agent.save(agent_save_path)
        self.agent.save_weights(weights_save_path)
        if not WANDB_DISABLE_CHECKPOINT:
            wandb.save(join(checkpoints_folder_path, '*.pkl'))

    def eval_agent(self, eval_results, step, eval_results_save_path, infix, print_eval, **kwargs):
        eval_logs = self.agent.eval(**kwargs)
        if print_eval:
            print(eval_logs)
        eval_results[step] = eval_logs

        save_json(data=eval_results, path=eval_results_save_path)
        eval_logs = {f'eval/{infix}{key}': val for key, val in eval_logs.items()}
        wandb.log({**eval_logs, self.step_key: step})

    def train(self, panel_name_infix='', print_eval=False, **kwargs):
        total_steps = self.train_config.steps
        train_results = []
        eval_results = {}

        infix = f'{panel_name_infix}/' if not panel_name_infix == '' else panel_name_infix

        pbar = tqdm(
            range(total_steps),
            desc=f"Train Steps: ",
            position=0,
            leave=True,
        )

        save_path_root = join(self.train_config.experiment_path, str(panel_name_infix))
        eval_results_save_path = join(save_path_root, f'eval_results.json')
        train_results_save_path = join(save_path_root, f'train_results.json')
        checkpoints_folder_path = join(save_path_root, 'checkpoints')
        weights_folder_path = join(save_path_root, 'model_weights')
        makedirs(checkpoints_folder_path, exist_ok=True)
        makedirs(weights_folder_path, exist_ok=True)

        for step in pbar:
            eval_condition_partial = step == 0

            if step % self.train_config.eval_every == 0 or eval_condition_partial:
                self.eval_agent(
                    eval_results=eval_results,
                    step=step,
                    eval_results_save_path=eval_results_save_path,
                    infix=infix,
                    print_eval=print_eval,
                    **kwargs
                )

            learn_logs = self.agent.learn(**kwargs)
            pbar.set_postfix(learn_logs)

            if step % self.train_config.log_every == 0 or eval_condition_partial:
                train_results.append({step: learn_logs})

                save_json(data=train_results, path=train_results_save_path)

                learn_logs = {f'train/{infix}{key}': val for key, val in learn_logs.items()}
                wandb.log({**learn_logs, self.step_key: step})

            if step % self.train_config.checkpoint_every == 0 and step != 0 and not self.train_config.save_only_last_checkpoint:
                self.save_checkpoint(
                    checkpoints_folder_path=checkpoints_folder_path,
                    weights_folder_path=weights_folder_path,
                    step=step
                )

        self.eval_agent(
            eval_results=eval_results,
            step=total_steps,
            eval_results_save_path=eval_results_save_path,
            infix=infix,
            print_eval=print_eval,
            **kwargs
        )
        self.save_checkpoint(
            checkpoints_folder_path=checkpoints_folder_path,
            weights_folder_path=weights_folder_path,
            step=total_steps
        )

from datetime import datetime
from os import makedirs
from os.path import join
from typing import Type

from pydantic import BaseModel, computed_field


class ExperimentConfig(BaseModel):
    project: str
    name: str
    job_type: str
    group_id: str = str(datetime.now())
    root_path: str
    experiment_folder_name: str | None = None  # for manually setting the folder name

    @computed_field
    @property
    def experiment_path(self) -> str:
        if self.experiment_folder_name is not None:
            path = join(self.root_path, self.experiment_folder_name)
        else:
            path = join(self.root_path, f'{self.name}-{self.group_id}')
        makedirs(path, exist_ok=True)
        return path


class EvalExpConfig(BaseModel):
    project: str
    name: str
    job_type: str
    group_id: str = str(datetime.now())
    experiment_path: str = None


class TrainerConfig(BaseModel):
    steps: int = 1e6
    eval_every: int = 1e4
    log_every: int = 100
    checkpoint_every: int = 2e4
    save_only_last_checkpoint: bool = False
    state_encoder_path: str | None = None
    concat_raw_states_with_embeddings: bool = False


class TrainerExperimentConfig(TrainerConfig, ExperimentConfig):
    pass


def load_eval_configs(trainer_config: dict, eval_config: dict,
                      config_class: Type[EvalExpConfig]) -> EvalExpConfig:
    current = str(datetime.now())
    properties_to_copy = ['group_id', 'project']
    for prop in properties_to_copy:
        eval_config[prop] = trainer_config[prop]

    config = config_class(**eval_config)
    config.experiment_path = join(trainer_config['experiment_path'], 'eval', f'{config.name}',
                                  f'{current}')

    return config

from enum import Enum
from typing import List, Union, Type, TypeVar
from pydantic import BaseModel, ConfigDict

from actions.space import ActionSpace, get_list_of_actions
from reward.mortality import MortalityReward
from reward.range import RangeReward
from reward.stacking import AddRewards
from reward.ventilator_free_days import VentilatorFreeReward, VFDEachStep
from utils.files import load_json, load_yaml


class ActionType(str, Enum):
    discrete = 'discrete'
    continuous = 'continuous'


class PreProcessingConfigs(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, use_enum_values=True)
    episode_id_column: str
    timestep_column: str
    state_vector_columns: List[str]
    patient_alive_col: str
    reward_function: Union[AddRewards, RangeReward, VentilatorFreeReward, VFDEachStep, MortalityReward]
    action_space: ActionSpace
    historized_actions: bool = False
    history_len: int = 0
    vent_mode_action_masking: bool = False

    def get_list_of_actions(self):
        return get_list_of_actions(action_space=self.action_space)

    def get_list_of_states(self):
        return self.state_vector_columns


class TextDataPreProcessConfig(PreProcessingConfigs):
    use_text_states: bool = True
    use_text_actions: bool = True


PreProcessConfigType = TypeVar('PreProcessConfigType', PreProcessingConfigs, TextDataPreProcessConfig)


def load_pre_processing_configs(
        pre_process_configs_path: str,
) -> Union["PreProcessingConfigs", "TextDataPreProcessConfig"]:
    data = load_json(pre_process_configs_path)

    # Auto-pick if not provided (uses optional 'type' field, then tries both)

    for cls in (TextDataPreProcessConfig, PreProcessingConfigs):
        try:
            return cls(**data)
        except Exception:
            pass
    raise ValueError(
        "Could not infer config type; add 'type' = 'preprocessing' or 'text_data'."
    )


def get_pre_processing_configs(
        configs_id,
        config_type: Type[PreProcessConfigType] = PreProcessingConfigs
) -> PreProcessConfigType:
    pre_process_conf = load_yaml('configs/pre_processing_configs.yaml')[configs_id]
    return config_type(**pre_process_conf)

from typing import List, Dict

from pydantic import BaseModel

from utils import load_json


class PreProcessingChecksConfig(BaseModel):
    minimum_episode_size: int
    episode_id_column: str
    action_space: List[str]
    state_space: List[str]
    misc_required_variables: List[str]
    variable_ranges: Dict[str, Dict[str, float]] = load_json('configs/outlier_ranges.json')

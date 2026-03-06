from typing import List, Any

from pydantic import BaseModel, model_validator


class ActionSpace(BaseModel):
    continuous_actions: List[str] = []
    discrete_actions: List[str] = []

    @model_validator(mode="before")
    def validate_at_least_one_field(cls, values):
        # Check if both fields are empty
        if values.get('continuous_actions') == [] and values.get('discrete_actions') == []:
            raise ValueError("No Actions selected")
        return values


def get_list_of_actions(action_space: ActionSpace) -> List[str]:
    return action_space.continuous_actions + action_space.discrete_actions


def get_discrete_actions_list(action_space: ActionSpace) -> List[str]:
    return action_space.discrete_actions


def get_continuous_actions_list(action_space: ActionSpace) -> List[str]:
    return action_space.continuous_actions


def is_only_discrete(action_space: ActionSpace) -> bool:
    discrete_actions_list = get_discrete_actions_list(action_space=action_space)
    continuous_actions_list = get_continuous_actions_list(action_space=action_space)
    contains_discrete_actions = False if discrete_actions_list == [] else True
    contains_continuous_actions = False if continuous_actions_list == [] else True

    return contains_discrete_actions and not contains_continuous_actions


def is_only_continuous(action_space: ActionSpace) -> bool:
    discrete_actions_list = get_discrete_actions_list(action_space=action_space)
    continuous_actions_list = get_continuous_actions_list(action_space=action_space)
    contains_discrete_actions = False if discrete_actions_list == [] else True
    contains_continuous_actions = False if continuous_actions_list == [] else True

    return not contains_discrete_actions and contains_continuous_actions

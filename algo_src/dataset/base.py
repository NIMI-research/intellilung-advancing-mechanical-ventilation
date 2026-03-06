from dataclasses import dataclass
from typing import Optional

from tensordict import TensorDict
from torch import Tensor


@dataclass(kw_only=True)
class RLBatch:
    observations: Tensor
    actions: Tensor | TensorDict
    next_observations: Tensor
    rewards: Tensor
    terminals: Tensor
    ep_id: Optional[Tensor] = None
    time_step: Optional[Tensor] = None
    masks: Optional[Tensor] = None


@dataclass(kw_only=True)
class RLEvalBatch(RLBatch):
    next_actions: Tensor

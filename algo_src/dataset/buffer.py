import torch
from torch import Tensor, float32, randint, dtype
from dataset.base import RLBatch, RLEvalBatch


def find_first_indices(data, values_to_find) -> Tensor:

    data = data.flatten()
    indices = torch.zeros_like(values_to_find, dtype=torch.int32)
    for i, val in enumerate(values_to_find):
        index = torch.nonzero(torch.eq(data, val))
        indices[i] = index[0]
    return indices


def get_episode_initial_indices(ep_ids: Tensor):
    unique_ep_ids = ep_ids.unique()
    initial_indices = find_first_indices(ep_ids, unique_ep_ids).flatten()
    return initial_indices


class ReplayBuffer:
    def __init__(
            self,
            dataset: RLBatch,
            action_dtype: dtype,
            device
    ) -> None:
        self.action_dtype = action_dtype

        self.device = device

        self.observations = Tensor(dataset.observations).to(self.device).type(float32)
        self.next_observations = Tensor(dataset.next_observations).to(self.device).type(float32)
        self.actions = dataset.actions.to(self.device).to(self.device).type(action_dtype)
        self.rewards = Tensor(dataset.rewards).to(self.device).to(self.device).type(float32).reshape(-1, 1)
        self.terminals = Tensor(dataset.terminals).to(self.device).to(self.device).type(float32).reshape(-1, 1)
        self.epi_id = None if dataset.ep_id is None else Tensor(dataset.ep_id).to(self.device).type(float32).reshape(-1,
                                                                                                                     1)
        self.epi_id = self.epi_id.to(self.device)

        self.time_step = None if dataset.time_step is None else Tensor(dataset.time_step).to(self.device).type(
            float32).reshape(-1, 1)
        self.initial_indices = get_episode_initial_indices(self.epi_id).long().to(self.device)
        self.masks = None if dataset.masks is None else dataset.masks.to(self.device)

        self._ptr = len(self.observations)
        self._size = len(self.observations)

        self._indices = torch.randperm(self._size, device=self.device)
        self._cursor = 0

    def sample_epoch(self, batch_size: int) -> RLBatch:
        if self._cursor + batch_size > self._size:
            # reshuffle and restart epoch
            self._indices = torch.randperm(self._size, device=self.device)
            self._cursor = 0

        batch_indexes = self._indices[self._cursor:self._cursor + batch_size]
        self._cursor += batch_size

        return RLBatch(
            observations=self.observations[batch_indexes],
            actions=self.actions[batch_indexes],
            next_observations=self.next_observations[batch_indexes],
            terminals=self.terminals[batch_indexes],
            rewards=self.rewards[batch_indexes],
            ep_id=None if self.epi_id is None else self.epi_id[batch_indexes],
            time_step=None if self.time_step is None else self.time_step[batch_indexes],
            masks=None if self.masks is None else self.masks[batch_indexes]
        )

    def sample(self, batch_size: int) -> RLBatch:
        batch_indexes = randint(low=0, high=self._size, size=(batch_size,))

        return RLBatch(
            observations=self.observations[batch_indexes],
            actions=self.actions[batch_indexes],
            next_observations=self.next_observations[batch_indexes],
            terminals=self.terminals[batch_indexes],
            rewards=self.rewards[batch_indexes],
            ep_id=None if self.epi_id is None else self.epi_id[batch_indexes],
            time_step=None if self.time_step is None else self.time_step[batch_indexes],
            masks=None if self.masks is None else self.masks[batch_indexes]
        )

    def sample_initial(self) -> RLBatch:
        batch_indexes = self.initial_indices

        return RLBatch(
            observations=self.observations[batch_indexes],
            actions=self.actions[batch_indexes],
            next_observations=self.next_observations[batch_indexes],
            terminals=self.terminals[batch_indexes],
            rewards=self.rewards[batch_indexes],
            ep_id=None if self.epi_id is None else self.epi_id[batch_indexes],
            time_step=None if self.time_step is None else self.time_step[batch_indexes],
            masks=None if self.masks is None else self.masks[batch_indexes]
        )

    def sample_all(self) -> RLBatch:
        return RLBatch(
            observations=self.observations.clone(),
            actions=self.actions.clone(),
            next_observations=self.next_observations.clone(),
            terminals=self.terminals.clone(),
            rewards=self.rewards.clone(),
            ep_id=None if self.epi_id is None else self.epi_id.clone(),
            time_step=None if self.time_step is None else self.time_step.clone(),
            masks=self.masks
        )


class EvalBuffer(ReplayBuffer):

    def __init__(self, action_dtype: dtype, dataset: RLEvalBatch, device) -> None:
        super().__init__(dataset, action_dtype, device)
        self.next_actions = dataset.next_actions.to(self.device).to(self.device).type(action_dtype)

    def sample(self, batch_size: int) -> RLEvalBatch:
        batch_indexes = randint(low=0, high=self._size, size=(batch_size,))

        return RLEvalBatch(
            observations=self.observations[batch_indexes],
            actions=self.actions[batch_indexes],
            next_actions=self.next_actions[batch_indexes],
            next_observations=self.next_observations[batch_indexes],
            terminals=self.terminals[batch_indexes],
            rewards=self.rewards[batch_indexes],
            ep_id=None if self.epi_id is None else self.epi_id[batch_indexes],
            time_step=None if self.time_step is None else self.time_step[batch_indexes]
        )

    def sample_all(self) -> RLEvalBatch:
        return RLEvalBatch(
            observations=self.observations.clone(),
            actions=self.actions.clone(),
            next_actions=self.next_actions.clone(),
            next_observations=self.next_observations.clone(),
            terminals=self.terminals.clone(),
            rewards=self.rewards.clone(),
            ep_id=None if self.epi_id is None else self.epi_id.clone(),
            time_step=None if self.time_step is None else self.time_step.clone()
        )

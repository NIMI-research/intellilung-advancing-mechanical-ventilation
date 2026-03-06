from typing import Sequence

from torch import isin, Tensor

from dataset.base import RLBatch


class EpisodeGenerator:
    def __init__(
            self,
            rl_dataset: RLBatch
    ):
        self.batch = rl_dataset
        self.device = self.batch.observations.get_device()

    def slice(self, start, end):
        sliced_dataset = {}

        for key, value in self.batch.__dict__.items():
            if value is not None:
                sliced_dataset[key] = value[start:end]
        return type(self.batch)(**sliced_dataset)

    def slice_by_id(self, ep_id):
        sliced_dataset = {}
        ep_id_filter = isin(self.batch.ep_id.flatten(), Tensor([ep_id]).to(self.device))
        for key, value in self.batch.__dict__.items():
            if value is not None:
                sliced_dataset[key] = value[ep_id_filter]
        return type(self.batch)(**sliced_dataset)

    def __call__(self) -> Sequence[RLBatch]:
        start = 0
        episodes = []
        if self.batch.ep_id is None:
            terminals = self.batch.terminals
            for i in range(terminals.shape[0]):
                if terminals[i] or terminals[i]:
                    end = i + 1
                    episode = self.slice(start, end)
                    episodes.append(episode)
                    start = end
        else:
            episode_ids = self.batch.ep_id.unique().tolist()
            for ep_id in episode_ids:
                episode = self.slice_by_id(ep_id=ep_id)
                episodes.append(episode)

        return episodes

import torch

from dataset.base import RLBatch
from dataset.episode_generator import EpisodeGenerator


class InitialStateDatasetGenerator:

    def __init__(self, rl_dataset: RLBatch):
        episode_generator = EpisodeGenerator(rl_dataset=rl_dataset)
        self.episodes = episode_generator()

    def __call__(self) -> RLBatch:
        init_dataset_dict = {key: [] for key, value in self.episodes[0].__dict__.items() if value is not None}

        for episode in self.episodes:
            ep_dict = episode.__dict__
            ep_dict = {key: value for key, value in ep_dict.items() if value is not None}
            for key, value in ep_dict.items():
                init_dataset_dict[key].append(value[0])
        for key, value in init_dataset_dict.items():
            init_dataset_dict[key] = torch.stack(init_dataset_dict[key])

        return type(self.episodes[0])(**init_dataset_dict)

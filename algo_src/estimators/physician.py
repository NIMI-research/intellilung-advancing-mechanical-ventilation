from dataset.base import RLBatch
from dataset.episode_generator import EpisodeGenerator


class AvgEstimator:

    def __init__(self, rl_dataset: RLBatch):
        episode_generator = EpisodeGenerator(rl_dataset=rl_dataset)
        self.episodes = episode_generator()

    def __call__(self, gamma=0.99):

        total_return = 0
        total_steps = 0
        for episode in self.episodes:
            next_value = 0
            ep_len = len(episode.rewards)
            for t in reversed(range(ep_len)):
                next_value = episode.rewards[t] + gamma * next_value * (1 - episode.terminals[t])
                total_return += next_value
                total_steps += 1

        return (total_return / total_steps).item()


class InitStateEstimator:

    def __init__(self, rl_dataset: RLBatch):
        episode_generator = EpisodeGenerator(rl_dataset=rl_dataset)
        self.episodes = episode_generator()

    def __call__(self, gamma=0.99):

        total_return = 0
        total_steps = 0
        for episode in self.episodes:
            next_value = 0
            ep_len = len(episode.rewards)
            for t in reversed(range(ep_len)):
                next_value = episode.rewards[t] + gamma * next_value * (1 - episode.terminals[t])
            total_return += next_value
            total_steps += 1

        return (total_return / total_steps).item()

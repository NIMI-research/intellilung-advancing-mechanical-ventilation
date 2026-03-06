import pandas as pd
from torch import isin, Tensor

from dataset.base import RLBatch


class EpisodeRewardFilter:

    def __init__(self, batch: RLBatch):
        self.batch = batch

    def get_mask(self, reward_filter=lambda x: x > 0):
        device = self.batch.ep_id.get_device()
        episode_ids = self.batch.ep_id.unique().tolist()
        passed_episodes = []
        failed_episodes = []
        for ep_id in episode_ids:
            episode_reward = self.batch.rewards[self.batch.ep_id == ep_id].sum().item()
            if reward_filter(episode_reward):
                passed_episodes.append(ep_id)
            else:
                failed_episodes.append(ep_id)

        # batch_dict = self.batch.__dict__
        #
        # passed_episodes_filter = isin(self.batch.ep_id.flatten(), Tensor(passed_episodes).to(device))
        # passed_batch_dict = {key: value[passed_episodes_filter] for key, value in batch_dict.items()}
        #
        # failed_episode_filter = isin(self.batch.ep_id.flatten(), Tensor(failed_episodes).to(device))
        # failed_batch_dict = {key: value[failed_episode_filter] for key, value in batch_dict.items()}
        #
        # return RLBatch(**passed_batch_dict), RLBatch(**failed_batch_dict)
        return isin(self.batch.ep_id.flatten(), Tensor(passed_episodes).to(device))

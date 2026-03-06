import glob
from abc import abstractmethod
from os.path import join, getmtime

import dill
import torch

from dataset.buffer import ReplayBuffer


class Agent:
    @abstractmethod
    def learn(self, **kwargs):
        pass

    @abstractmethod
    def eval(self, **kwargs):
        pass

    def save(self, checkpoint_path):
        with open(checkpoint_path, 'wb') as f:
            dill.dump(self, f)

    def save_weights(self, weights_save_path):
        pass

    @classmethod
    def load(cls, save_path):
        with open(save_path, 'rb') as f:
            return dill.load(f)


class RLAgent(Agent):
    @abstractmethod
    def get_action(self, state, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def learn(self, buffer: ReplayBuffer, **kwargs):
        pass

    @abstractmethod
    def eval(self, buffer: ReplayBuffer, **kwargs):
        pass


class StateEncoder(Agent):

    @abstractmethod
    def state_emb_size(self):
        raise NotImplementedError

    @abstractmethod
    def encode(self, state, **kwargs):
        raise NotImplementedError


def load_state_encoder(state_encoder_path, concat_raw_states_with_embeddings) -> "StateEncoder":
    checkpoints = glob.glob(join(state_encoder_path, "checkpoints", "*"))
    checkpoints.sort(key=getmtime, reverse=True)

    state_encoder: "StateEncoder" = StateEncoder.load(save_path=checkpoints[0])

    if concat_raw_states_with_embeddings:
        original_encode = state_encoder.encode

        def encode_with_concat(x, *args, **kwargs):
            emb = original_encode(x, *args, **kwargs)
            # assumes x and emb are tensors with same batch dims
            return torch.cat([x, emb], dim=-1)

        state_encoder.encode = encode_with_concat

    return state_encoder

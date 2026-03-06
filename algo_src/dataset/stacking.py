from typing import Tuple

import numpy as np
import pandas as pd
import torch
from tensordict import TensorDict

from actions.hybrid import get_continuous_action, get_discrete_action, create_hybrid_action_tensor_dict
from dataset.base import RLBatch


def stack_history_torch(
        x: torch.Tensor,
        ep_id: torch.Tensor,
        forward_len: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Episode-aware forward sequence stacking for x with shape [B, D], torch-only.
    Returns:
      fwd: [B, H, D] gathered forward windows (padded by repeating the episode-end row)
      mask: [B, H]    1.0 = valid, 0.0 = padded (cross-episode or out-of-range)
    """
    B, D = x.shape
    H = forward_len
    device = x.device

    # Row and forward offset grids
    rows = torch.arange(B, device=device).view(B, 1)  # [B, 1]
    offsets = torch.arange(H, device=device).view(1, H)  # [1, H]
    raw_idx = rows + offsets  # [B, H]

    # Detect episode boundaries (where ep_id changes)
    is_episode_end = torch.zeros(B, dtype=torch.bool, device=device)
    ep_id = ep_id.squeeze()
    is_episode_end[:-1] = ep_id[:-1] != ep_id[1:]
    is_episode_end[-1] = True  # Last element is always an episode end

    # Get indices of all episode ends
    end_indices = torch.where(is_episode_end)[0]

    # For each position, find which episode it belongs to by finding
    # the first episode end that comes at or after it
    # Using searchsorted: for each row index, find the episode end
    ep_end_for_row = end_indices[torch.searchsorted(end_indices, torch.arange(B, device=device))]
    ep_end_for_row = ep_end_for_row.view(B, 1)  # [B, 1]

    # Clamp each row's forward sequence to its episode end
    safe_idx = torch.minimum(raw_idx, ep_end_for_row)  # [B, H]

    # Valid if we didn't exceed episode boundary
    mask = (raw_idx <= ep_end_for_row).to(x.dtype)  # [B, H]

    # Gather using advanced indexing
    fwd = x[safe_idx]  # [B, H, D]

    return fwd, mask.to(dtype=torch.bool)


def stack_history_rl_batch(batch: RLBatch,
                           history_len: int,
                           device) -> RLBatch:
    """
    Build histories WITHOUT concatenating actions into observations.

    observations[t]      -> [s_t,   s_{t-1}, ..., s_{t-H+1}]
    actions[t]           -> [a_t,   a_{t-1}, ..., a_{t-H+1}]
    next_observations[t] -> [s_{t+1}, s_t,   ..., s_{t-H+2}]

    masks: [B,H,1], 1 where the observation history slot is valid, else 0.
    """
    # Inputs
    s = batch.observations  # [B, Ds]
    sn = batch.next_observations  # [B, Ds]
    a = batch.actions  # [B, Da]

    ep_id = batch.ep_id.to(dtype=torch.long, device=s.device)

    # Histories (all torch)
    s_hist, s_mask = stack_history_torch(s, ep_id, history_len)  # [B,H,Ds], [B,H]

    if type(a) == TensorDict:
        a_hist_cont, _ = stack_history_torch(get_continuous_action(a), ep_id,
                                             history_len)
        a_hist_disc, _ = stack_history_torch(get_discrete_action(a), ep_id, history_len)
        a_hist = create_hybrid_action_tensor_dict(continuous_actions=a_hist_cont, discrete_actions=a_hist_disc)
    else:
        a_hist, _ = stack_history_torch(a, ep_id, history_len)  # [B,H,Da]
    sn_hist, _ = stack_history_torch(sn, ep_id, history_len)  # [B,H,Ds]

    masks = s_mask.to(device=device)  # [B,H,1]

    return RLBatch(
        observations=s_hist.to(device=device),  # [B,H,Ds]
        actions=a_hist.to(device=device),  # [B,H,Da]
        rewards=batch.rewards,
        terminals=batch.terminals,
        next_observations=sn_hist.to(device=device),  # [B,H,Ds]
        ep_id=batch.ep_id,
        time_step=batch.time_step,
        masks=masks,  # [B,H,1]
    )

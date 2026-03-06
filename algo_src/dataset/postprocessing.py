import pandas as pd


def drop_leading_trailing_zeros(group, min_rows=4):
    """
    Removes leading and trailing rows where `vent_mode = 0` and drops the episode
    if the resulting group has fewer than `min_rows` rows.
    """
    # Identify the indices where vent_mode is not 0
    non_zero_indices = group[group['vent_mode'] != 0].index
    
    # If no non-zero indices, return an empty DataFrame
    if non_zero_indices.empty:
        return pd.DataFrame()
    
    # Keep only the rows between the first and last non-zero index
    first_non_zero = non_zero_indices[0]
    last_non_zero = non_zero_indices[-1]
    trimmed_group = group.loc[first_non_zero:last_non_zero]
    
    # Drop the group if it has fewer than `min_rows` rows
    if len(trimmed_group) < min_rows:
        return pd.DataFrame()
    
    return trimmed_group


def adjust_post_extubation_interval(group):
    original_size = len(group)
    filtered_group = drop_leading_trailing_zeros(group)
    filtered_size = len(filtered_group)
    
    # Adjustment logic
    if not filtered_group.empty:
        adjustment = (original_size - filtered_size) * 60
        if group['post_extubation_interval'].iloc[0] != 90:
            filtered_group['post_extubation_interval'] += adjustment
    
    return filtered_group

def calculate_pause(group, timestep_column, default_pause=90):
    # Get the last time_interval of each mv_id group
    last_time_intervals = group.groupby('mv_id')[timestep_column].max()
    # Get the first time_interval of each mv_id group
    first_time_intervals = group.groupby('mv_id')[timestep_column].min()
    
    # Calculate time between last and first time_intervals of consecutive mv_id groups
    pauses = first_time_intervals.shift(-1) - last_time_intervals
    pauses.iloc[-1] = default_pause  # Set the last mv_id group's pause to default value
    
    # Map the pauses back to the original DataFrame
    return group['mv_id'].map(pauses)

def remove_episodes_with_long_zeros(df, threshold=6):
    """
    Removes entire episodes where there are streaks of `vent_mode = 0` of length `threshold` or longer.
    Returns the filtered DataFrame and the number of episodes dropped.
    """
    dropped_episode_count = 0  # To track how many episodes are dropped
    episodes_to_remove = []  # To store the episode IDs to be removed

    # Iterate through each episode group
    for episode_id, group in df.groupby('episode_id'):
        # Identify streaks of `vent_mode = 0`
        vent_zero_mask = (group['vent_mode'] == 0).astype(int)
        # use cumsum on vent_mode not equal 0, cumsum only increments if ne
        # as long as it doesnt increase, row is part of vent_mode 0 streak
        vent_zero_group_mask = group['vent_mode'].ne(0).cumsum()
        # group vent_zero_mask by groups
        vent_zero_counter = vent_zero_mask.groupby(vent_zero_group_mask)
        # get streak for each group
        consecutive_zero_streak = vent_zero_counter.cumsum()

        # Check if any streak meets or exceeds the threshold
        if (consecutive_zero_streak >= threshold).any():
            episodes_to_remove.append(episode_id)
            dropped_episode_count += 1

    # Filter out episodes to remove
    filtered_df = df[~df['episode_id'].isin(episodes_to_remove)].reset_index(drop=True)

    return filtered_df, dropped_episode_count

def remove_all_zero_vent_mode_groups(df):
    # Group by episode_id and check if any vent_mode value is non-zero
    non_zero_episodes = df.groupby('episode_id')['vent_mode'].any()

    # Filter the original DataFrame to keep only the rows with non-zero episodes
    df_filtered = df[df['episode_id'].isin(non_zero_episodes[non_zero_episodes].index)].reset_index(drop=True)

    return df_filtered

def fill_remaining_vent_mode(group):
    """
    Fills `vent_mode` column within an `episode_id` group:
    - Replace `0` with the value of the previous row.
    - If `0` is in the first row, replace it with the next available non-zero value.
    """
    # Forward fill within the group
    group['vent_mode'] = group['vent_mode'].replace(0, None)
    group['vent_mode'] = group['vent_mode'].ffill()

    # Backward fill to handle the case of the first row being 0
    group['vent_mode'] = group['vent_mode'].bfill()
    
    return group

def remove_vent_mode_zero(data, timestep_column):
    """Remove vent_mode zero from dataset based on specified conditions
    """
    data_wo_all_zero  = remove_all_zero_vent_mode_groups(data)
    data_wo_lead_trail = data_wo_all_zero.groupby('episode_id', group_keys=False).apply(drop_leading_trailing_zeros)
    data_wo_lead_trail = data_wo_lead_trail.groupby('episode_id', group_keys=False).apply(adjust_post_extubation_interval)
    filtered_df, _ = remove_episodes_with_long_zeros(data_wo_lead_trail, threshold=6)
    # Update mv_id
    # assign new mv_id within stay_id if new episode_id changes
    filtered_df['mv_id'] = (filtered_df.groupby('stay_id')['episode_id'].transform(lambda x: (x != x.shift()).cumsum() - 1))
    # Update pause until next
    filtered_df['pause_until_next'] = filtered_df.groupby('stay_id', group_keys=False).apply(lambda group: calculate_pause(group, timestep_column))
    df_wo_zero = filtered_df.groupby('episode_id', group_keys=False).apply(fill_remaining_vent_mode)
    return df_wo_zero


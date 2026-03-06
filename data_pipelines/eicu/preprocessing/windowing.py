import numpy as np
import pandas as pd

import preprocessing.demog as demog
import preprocessing.cleaning as cl
import common.utils as ut
import dicts.vars as var_dict


def clean_nanmedian(s):
    """nanmedian without warnings"""
    if s.isna().all():
        return np.nan
    return np.nanmedian(s)


def find_numeric_timepoint(col_data):
    col_uniques = col_data["value"].unique()
    if len(col_uniques) == 0:
        return np.nan
    elif len(col_uniques) == 1:
        return col_uniques[0]
    else:
        prioritized_data = col_data[col_data['priority']
                                    == col_data['priority'].min()]
        return clean_nanmedian(prioritized_data["value"])


def select_multicategorical(col_data, timepoint, time_window):
    """Select categorical value for the timepoint that had the longest duration in the time window.
    Return the value with the longest duration and the value of the last entry."""
    col_data_sorted = col_data.sort_values(by='offset')
    # Calculate differences between offsets
    differences = col_data_sorted['offset'].diff()
    # Handle first and last rows
    differences.iloc[0] = col_data_sorted.iloc[0]['offset'] - timepoint
    differences.iloc[-1] = timepoint + time_window - col_data_sorted.iloc[-1]['offset']
    # Ensure we are returning scalar values
    max_duration_value = col_data_sorted.loc[differences.idxmax(), 'value']
    last_value = col_data_sorted.iloc[-1]['value']
    return max_duration_value, last_value

def select_current_or_hist(current_row, timepoint, time_window, col, categorical_history):
    """Compare current entry and entry in history
    Return entry with highest duration and add latest to history if it was not used"""
    if categorical_history[col] is None:
        return current_row["value"], categorical_history
    else:
        cur_offset = ut.get_df_value(current_row["offset"])
        hist_offset = ut.get_df_value(categorical_history[col]["offset"])
        hist_val_duration = cur_offset - hist_offset
        cur_val_duration = timepoint + time_window - cur_offset

        if cur_val_duration >= hist_val_duration:
            categorical_timepoint_val = current_row["value"]
            categorical_history[col] = None
            return categorical_timepoint_val, categorical_history
        else:
            categorical_timepoint_val = categorical_history[col]["value"]
            categorical_history[col] = current_row
            return categorical_timepoint_val, categorical_history


def find_categorical_timepoint(col_data, timepoint, time_window, col, categorical_history):
    """Assigns categorical value to timepoints that had the longest duration during the time window
    """
    prioritized_data = col_data[col_data['priority'] == col_data['priority'].min()].drop_duplicates()
    col_uniques = prioritized_data["value"].unique()
    if len(col_uniques) == 0:
        # if no new value available, check for stored val from last timepoint
        if categorical_history[col] is None:
            return np.nan, categorical_history
        else:
            categorical_timepoint_val = categorical_history[col]["value"]
            categorical_history[col] = None
            return categorical_timepoint_val, categorical_history
    elif len(col_uniques) == 1:
        # if 1 new available
        # select which one to assign based on which category covers more of the timepoint
        # if latest one not used, store in history
        if categorical_history[col] is None:
            return col_uniques[0], categorical_history
        else:
            return select_current_or_hist(prioritized_data.iloc[0], timepoint, time_window, col, categorical_history)
    else:
        # if more than 1
        # select which one to assign based on which category covers more of the timepoint
        # assign latest to history
        timepoint_longest, timepoint_latest = select_multicategorical(
            prioritized_data, timepoint, time_window)
        categorical_timepoint_val, _ = select_current_or_hist(
            prioritized_data[prioritized_data['value'] == timepoint_longest].iloc[0], timepoint, time_window, col, categorical_history)
        categorical_history[col] = {'value': timepoint_latest, 'offset': timepoint}
        return categorical_timepoint_val, categorical_history
    
def assign_maxvaso4h(patient_episodes, n=4):
    """Get maximum vasopressor dosage within last n timesteps (default 4)"""
    patient_episodes['maxdrugs_vaso4h'] = patient_episodes['drugs_vaso4h'].rolling(window=n, min_periods=1).max()
    # Fill NaN values with 0
    patient_episodes['maxdrugs_vaso4h'] = patient_episodes['maxdrugs_vaso4h'].fillna(0)
    patient_episodes['drugs_vaso4h'] = patient_episodes['maxdrugs_vaso4h']
    patient_episodes.drop(columns=['maxdrugs_vaso4h'], inplace=True)
    patient_episodes = cl.replace_outliers_col("drugs_vaso4h", patient_episodes)
    return patient_episodes

def calc_urinout4h(patient_episodes, n=4):
    """Get cumulative urin out over n timesteps (default 4)"""
    patient_episodes['sum_urinout'] = patient_episodes['state_urin4h'].rolling(window=n, min_periods=1).sum()
    # Fill NaN values with 0
    patient_episodes['sum_urinout'] = patient_episodes['sum_urinout'].fillna(0)
    patient_episodes['state_urin4h'] = patient_episodes['sum_urinout']
    patient_episodes.drop(columns=['sum_urinout'], inplace=True)
    patient_episodes = cl.replace_outliers_col("state_urin4h", patient_episodes)
    return patient_episodes

def calc_fluidin_4h(patient_episodes, n=4):
    """Get cumulative fluid in over n timesteps (default 4)"""
    patient_episodes['sum_fluidin'] = patient_episodes['state_ivfluid4h'].rolling(window=n, min_periods=1).sum()
    # Fill NaN values with 0
    patient_episodes['sum_fluidin'] = patient_episodes['sum_fluidin'].fillna(0)
    patient_episodes['state_ivfluid4h'] = patient_episodes['sum_fluidin']
    patient_episodes.drop(columns=['sum_fluidin'], inplace=True)
    patient_episodes = cl.replace_outliers_col("state_ivfluid4h", patient_episodes)
    return patient_episodes

def meta4h_pipeline(patient_episodes, n=4):
    patient_episodes = assign_maxvaso4h(patient_episodes, n)
    patient_episodes = calc_urinout4h(patient_episodes, n)
    patient_episodes = calc_fluidin_4h(patient_episodes, n)
    return patient_episodes

def get_pause_in_days(i, relevant_mv_eps):
    if i == (len(relevant_mv_eps)-1):
        # Default to 90 days if no further episode
        return 90
    else:
        # convert offset minutes to days 
        return (relevant_mv_eps.loc[i+1, "vent_start"] - relevant_mv_eps.loc[i, "vent_end"])/60/24
    
def assign_eps_ids(state_vector):
    state_vector['combo'] = state_vector['stay_id'].astype(str) + '_' + state_vector['mv_id'].astype(str)
    id_mapping = {combo: i for i, combo in enumerate(state_vector['combo'].unique())}
    state_vector['episode_id'] = state_vector['combo'].map(id_mapping)
    state_vector.drop('combo', axis=1, inplace=True)
    return state_vector


def create_time_windows(stayid, time_window, patient_state_vector, variable_columns, patient_demo_df, patient_ventevents, required_variables):
    patient_episode_windows = []
    patient_missing_dict = {}
    categorical_history = {}
    for cat_col in var_dict.categorical_vars:
        categorical_history[cat_col] = None

    relevant_mv_eps = patient_ventevents[
                            patient_ventevents["vent_start"].notna() &
                            patient_ventevents["vent_end"].notna()
                        ].reset_index(drop=True)
    # for each mv episode
    for i, row in relevant_mv_eps.iterrows():
        start_offset = max(row["vent_start"], 0)
        end_offset = row["vent_end"]
        mv_duration = row["mv_duration"]

        unique_timepoints = [start_offset]
        while start_offset <= end_offset:
            start_offset += time_window
            unique_timepoints.append(start_offset)

        windowed_state_vector = {"stay_id": [], "mv_id":[], "offset": [], "mv_duration": [], "pause_until_next": []}

        for col in variable_columns:
            windowed_state_vector[col] = []

        for timepoint in unique_timepoints:
            if timepoint == unique_timepoints[-1]:
                break
            timepoint_data = patient_state_vector[((patient_state_vector["offset"] >= timepoint) & (
                patient_state_vector["offset"] < (timepoint+time_window)))]

            pause_until_next = get_pause_in_days(i, relevant_mv_eps)
            windowed_state_vector["offset"].append(timepoint)
            windowed_state_vector["stay_id"].append(stayid)
            windowed_state_vector["mv_id"].append(i)
            windowed_state_vector["mv_duration"].append(mv_duration)
            windowed_state_vector["pause_until_next"].append(pause_until_next)

            for col in variable_columns:
                # if categorical
                if col in var_dict.categorical_vars:
                    unique_timepoint_value, categorical_history = find_categorical_timepoint(
                        timepoint_data.loc[timepoint_data["variable"] == col, :].dropna(subset=["value"]), timepoint, time_window, col, categorical_history)
                # if numeric
                else:
                    unique_timepoint_value = find_numeric_timepoint(
                        timepoint_data.loc[timepoint_data["variable"] == col, :].dropna(subset=["value"]))
                windowed_state_vector[col].append(unique_timepoint_value)

        windowed_with_demog, missing_dict = demog.add_patient_demo(
            pd.DataFrame.from_dict(windowed_state_vector), patient_demo_df, end_offset, required_variables)
        # if dict still empty
        if not patient_missing_dict:
            patient_missing_dict = missing_dict.copy()
        else:
            for key in missing_dict:
                if key in missing_dict:
                    patient_missing_dict[key] += missing_dict[key]
                else:
                    patient_missing_dict[key] = missing_dict[key]

        if windowed_with_demog is not None:
            windowed_with_demog = demog.merge_TV(windowed_with_demog)
            patient_episode_windows.append(windowed_with_demog)
    if len(patient_episode_windows) > 0:
        patient_episode_windows_concat = pd.concat(patient_episode_windows, axis=0).reset_index(drop=True)
        patient_episode_windows_output = meta4h_pipeline(patient_episode_windows_concat)
    else:
        patient_episode_windows_output = None    

    return patient_episode_windows_output, patient_missing_dict




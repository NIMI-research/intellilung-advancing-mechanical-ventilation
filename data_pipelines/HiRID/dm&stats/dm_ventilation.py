import pandas as pd
import numpy as np
import polars as pl
from datetime import datetime
import os
import ast

PID = 'PatientID'

def correct_right_edge_vent(vent_status_arr, etco2_col):
    ''' Corrects the right edge of the ventilation status array, to pin-point the exact end with last etco2'''
    vent_status_arr_flipped = correct_left_edge_vent(np.flipud(vent_status_arr), np.flipud(etco2_col))
    return np.flipud(vent_status_arr_flipped)

def correct_left_edge_vent(vent_status_arr, etco2_col):
    ''' Corrects the left edge of the ventilation status array, to pin-point the exact  start with first etco2'''
    on_left_edge=False
    in_event=False

    for idx in range(len(vent_status_arr)):
        if vent_status_arr[idx]==1.0 and not in_event:
            in_event=True
            on_left_edge=True
        if in_event and vent_status_arr[idx]==0.0:
            in_event=False
            on_left_edge=False
        if on_left_edge and in_event:
            if vent_status_arr[idx]==0.0:
                in_event=False
                on_left_edge=False
            elif etco2_col[idx]>0.5:
                on_left_edge=False
            else:
                vent_status_arr[idx]=0.0

    return vent_status_arr

def delete_short_vent_events(vent_status_arr, short_event_min, time_intervall):
    ''' Delete short events in the ventilation status array'''
    in_event=False
    event_length=0
    for idx in range(len(vent_status_arr)):
        cur_state=vent_status_arr[idx]
        if in_event and cur_state==1.0:
            event_length+=time_intervall
        if not in_event and cur_state==1.0:
            in_event=True
            event_length=time_intervall
            event_start_idx=idx
        if in_event and (cur_state==0.0 or np.isnan(cur_state)):
            in_event=False
            if event_length<short_event_min:
                vent_status_arr[event_start_idx:idx]=0.0
    return vent_status_arr

def merge_short_vent_gaps(vent_status_arr, short_gap_min, time_intervall):
    ''' Merge short gaps in the ventilation status array'''
    in_gap=False
    gap_length=0
    before_gap_status=np.nan
    
    for idx in range(len(vent_status_arr)):
        cur_state=vent_status_arr[idx]
        if in_gap and (cur_state==0.0 or np.isnan(cur_state)):
            gap_length+=time_intervall
        elif not in_gap and (cur_state==0.0 or np.isnan(cur_state)):
            if idx>0:
                before_gap_status=vent_status_arr[idx-1]
            in_gap=True
            in_gap_idx=idx
            gap_length=time_intervall
        elif in_gap and cur_state==1.0:
            in_gap=False
            after_gap_status=cur_state
            if gap_length<=short_gap_min:
                vent_status_arr[in_gap_idx:idx]=1.0
    
    return vent_status_arr

def delete_short_vent_epidosed(vent_period_arr, hr_status_arr, short_event_min, time_intervall):
    ''' Delete short ventilation periods if no HR gap before '''
    in_event=False
    event_length=0
    for idx in range(len(vent_period_arr)):
        cur_state=vent_period_arr[idx]
        if in_event and cur_state==1.0:
            event_length+=time_intervall
        if not in_event and cur_state==1.0:
            in_event=True
            event_length=time_intervall
            event_start_idx=idx
        if in_event and cur_state==0.0:
            in_event=False

            # Short event at beginning of stay shall never be deleted...
            if event_start_idx==0:
                delete_event=False
            else:
                search_hr_idx=event_start_idx-1
                while search_hr_idx>=0:
                    if hr_status_arr[search_hr_idx]==1.0:
                        hr_gap_length=time_intervall*(event_start_idx-search_hr_idx)
                        delete_event=True
                        break
                    search_hr_idx-=1

                # Found no HR before event, do not delete event...
                if search_hr_idx==-1:
                    delete_event=False

            # Delete event in principle, then check if short enough...
            if delete_event:
                event_length+=hr_gap_length
                if event_length<=short_event_min:
                    vent_period_arr[event_start_idx:idx]=0.0
                    
    return vent_period_arr


def delete_low_density_hr_gap(status_arr, hr_status_arr, density_treshold):
    ''' Deletes gaps which are caused by likely leaving from icu'''
    in_event=False
    in_gap=False
    gap_idx=-1
    for idx in range(len(status_arr)):

        # Beginning of new event, not from inside gap
        if not in_event and not in_gap and status_arr[idx]==1.0:
            in_event=True

        # Beginning of potential gap that needs to be closed
        elif in_event and status_arr[idx]==0.0:
            in_gap=True
            gap_idx=idx
            in_event=False

        # The gap is over, re-assign the status of ventilation to merge the gap, enter new event
        if in_gap and status_arr[idx]==1.0:
            
            hr_sub_arr=hr_status_arr[gap_idx:idx]

            # Close the gap if the density of HR is too low in between
            if np.sum(hr_sub_arr)/hr_sub_arr.size<=density_treshold:
                status_arr[gap_idx:idx]=1.0
                
            in_gap=False
            in_event=True

    return status_arr

def check_variability_legacy(df, column_name, window_size, variability):
    # Calculate the rolling mean and standard deviation
    rolling_mean = df[column_name].rolling(window=window_size,min_periods=1).mean()
    rolling_std = df[column_name].rolling(window=window_size,min_periods=1).std()
    
    # Calculate the coefficient of variation which is a standardized measure of dispersion of a probability distribution or frequency distribution
    cv = (rolling_std / rolling_mean) * 100
    
    # Check if the CV is more than 15%
    variability_flag = cv > variability

    return variability_flag


def check_variability_10percent(df, column_name, window_size):
    # Calculate the rolling metrics
    rolling_median = df[column_name].rolling(window=window_size,min_periods=1).median()
    rolling_max = df[column_name].rolling(window=window_size,min_periods=1).max()
    rolling_min = df[column_name].rolling(window=window_size,min_periods=1).min()
    
    # Check if Max or Min is out of the range
    variability_flag = (rolling_min < 0.9 * rolling_median) & (rolling_max > 1.1 * rolling_median)

    return variability_flag

def check_variability_rr_low(df, column_name, window_size):
    # Calculate the rolling metrics
    rolling_median = df[column_name].rolling(window=window_size,min_periods=1).median()
    rolling_max = df[column_name].rolling(window=window_size,min_periods=1).max()
    rolling_min = df[column_name].rolling(window=window_size,min_periods=1).min()
    
    # Check if Max or Min is out of the range
    variability_flag = (rolling_min > 0.99 * rolling_median) & (rolling_max < 1.01 * rolling_median)

    return variability_flag
    
def process_patient(pid,df_resampled):
    hr_min_intervall=5; etCO2_min_intervall=10; tv_min_intervall=10; vent_group_intervall=60; airway_intervall=1440
    timestep_count = len(df_resampled)
    time_intervall = 5

    vent_votes_machanicalventilation=np.zeros(shape=(timestep_count)) #etco2_present
    vent_votes_invasiveventilation=np.zeros(shape=(timestep_count))
    vent_votes_niv=np.zeros(shape=(timestep_count))
    vent_votes_score=np.zeros(shape=(timestep_count))
    hr_state=np.zeros(shape=(timestep_count))

    #map controlled modes
    d = ast.literal_eval("{1:1, 2:2, 3:2, 4:3, 5:3, 6:4, 7:4, 8:5, 9:6, 10:6, 11:7, 12:8, 13:9, 129:8}")
    df_resampled.loc[:, 'dm_vent_mode_subgroup'] = df_resampled["vm3017"].map(d)
    d = ast.literal_eval("{1:1, 2:2, 3:2, 4:2, 5:2, 6:3, 7:4, 8:5, 9:6}")
    df_resampled.loc[:, 'dm_vent_mode_group'] = df_resampled["dm_vent_mode_subgroup"].map(d)

    for jdx in range(1,timestep_count):
        ##checked from here
        vote_score=0

        # HR as marker of presence
        win_hr=df_resampled[max(0,jdx-hr_min_intervall):jdx].vm2001
        if (win_hr>0.5).any():
            hr_state[jdx]=1

        # EtCO2 requirement
        win_etco2=df_resampled[max(0,jdx-etCO2_min_intervall):jdx].vm3003
        if (win_etco2>0.5).any():
            vote_score+=2
            vent_votes_machanicalventilation[jdx]=1

        # Ventilation group requirement
        win_vent_group_index=df_resampled[max(0,jdx-vent_group_intervall):jdx].dm_vent_mode_group.last_valid_index()
        if win_vent_group_index is not None:
            if df_resampled.loc[win_vent_group_index].dm_vent_mode_group in [2.0,3.0]: #controled, spontanious
                vote_score+=1
            elif df_resampled.loc[win_vent_group_index].dm_vent_mode_group in [1.0]: #standby
                vote_score-=1
            elif df_resampled.loc[win_vent_group_index].dm_vent_mode_group in [4.0,5.0,6.0]: #niv, high flow
                vote_score-=2

        # TV presence requirement
        win_tv_group_index=df_resampled[max(0,jdx-tv_min_intervall):jdx].vm3006.last_valid_index()
        if win_tv_group_index is not None and df_resampled.loc[win_tv_group_index].vm3006>50:
            vote_score+=1

        # Airway requirement
        win_airway_index=df_resampled[max(0,jdx-airway_intervall):jdx].vm3019.last_valid_index()
        if win_airway_index is not None:
            if df_resampled.loc[win_airway_index].vm3019 in [1,2]:
                vote_score+=2
            if df_resampled.loc[win_airway_index].vm3019 in [3,4,5,6]:
                vote_score-=1

        #overall score
        vent_votes_score[jdx]=vote_score
        if vote_score>=4:
            vent_votes_invasiveventilation[jdx]=1

    #correct vent episodes over the stay
    vent_votes_machanicalventilation = correct_left_edge_vent(vent_votes_machanicalventilation, df_resampled.vm3003)
    vent_votes_machanicalventilation = correct_right_edge_vent(vent_votes_machanicalventilation, df_resampled.vm3003)
    vent_votes_invasiveventilation = correct_left_edge_vent(vent_votes_invasiveventilation, df_resampled.vm3003)
    vent_votes_invasiveventilation = correct_right_edge_vent(vent_votes_invasiveventilation, df_resampled.vm3003)

    vent_votes_niv = (vent_votes_invasiveventilation==0) & (vent_votes_machanicalventilation==1)
    vent_votes_invasiveventilation = merge_short_vent_gaps(vent_votes_invasiveventilation, 30,time_intervall)
    vent_votes_niv = merge_short_vent_gaps(vent_votes_niv, 30,time_intervall)
    vent_votes_invasiveventilation = delete_low_density_hr_gap(vent_votes_invasiveventilation, hr_state, 0.5)
    vent_votes_niv = delete_low_density_hr_gap(vent_votes_niv, hr_state, 0.5)
    vent_votes_invasiveventilation = delete_short_vent_epidosed(vent_votes_invasiveventilation, hr_state, 30, time_intervall)
    vent_votes_niv = delete_short_vent_epidosed(vent_votes_niv, hr_state, 15, time_intervall)
    
    #deduct general ventilation mode
    df_resampled['is_inv_vent'] = vent_votes_invasiveventilation == 1
    df_resampled['dm_vent_inv_state'] = vent_votes_invasiveventilation == 1
    df_resampled['dm_vent_niv_state'] = vent_votes_niv
    df_resampled['variability_rr_high'] = check_variability_10percent(df_resampled, "vm3002", 12) # rr max and min are more than 10% above/below average over the last 1h
    df_resampled['variability_rr_low'] = check_variability_rr_low(df_resampled, "vm3002", 12)# rr max and min are within of 1% of the median over the last 1h
    df_resampled['variability_tv_high'] = check_variability_10percent(df_resampled, "vm3006", 12) # tv max and min are more than 10% above/below average over the last 1h
    df_resampled['variability_pi_high'] = check_variability_10percent(df_resampled, "vm3007", 12)# Pinsp max and min are more than 10% above/below average over the last 1h

    df_resampled['dm_vent_contr_mode'] = df_resampled.dm_vent_mode_group.ffill() == 2.0 #temporät nicht genau
    df_resampled['is_controlled'] = False
    df_resampled['is_controlled'] = (df_resampled.dm_vent_inv_state & ~df_resampled.variability_rr_high & df_resampled.variability_rr_low) #very low rr variability -> is controlled
    df_resampled['is_controlled'] = df_resampled.is_controlled | (df_resampled.dm_vent_inv_state & df_resampled.dm_vent_mode_subgroup.isin([2.0,4.0,5.0]) & ~df_resampled.variability_rr_high & ~df_resampled.variability_tv_high)  #volumen controllierte beatmung
    df_resampled['is_controlled'] = df_resampled.is_controlled | (df_resampled.dm_vent_inv_state & df_resampled.dm_vent_inv_state.isin([3.0, 5.0]) & ~df_resampled.variability_rr_high & ~df_resampled.variability_pi_high)  #volumen controllierte beatmung
    df_resampled['dm_vent_controlled_ventilation_merged'] = merge_short_vent_gaps(df_resampled['is_controlled'], 15,time_intervall)
    
    df_selected = df_resampled[["PatientID", "AbsDatetime", "dm_vent_inv_state", "dm_vent_niv_state",'is_controlled', 'dm_vent_controlled_ventilation_merged', 'dm_vent_contr_mode', 'dm_vent_mode_group', 'dm_vent_mode_subgroup']]
    df_selected.PatientID = pid

    return df_selected
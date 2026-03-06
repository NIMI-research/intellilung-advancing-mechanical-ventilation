import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Tuple

from common.config import settings

import common.kbase as kb 

# Function for binning demographics variables
def bin_variables(df: pd.DataFrame) -> pd.DataFrame:    

    # daemo_age 
    ## Bin Width = 8
    ## Bin centers = 22, 30, 38...94
    age_bins = np.arange(18, 99, 8) 
    df['age_binned'] = pd.cut(df['daemo_age'], bins=age_bins, labels=age_bins[:-1] + 4, right=False)

    # daemo_weight
    ## Bin Width = 4
    ## Bin centers = 42, 46, 50...138
    weight_bins = np.arange(39, 144, 4)
    df['daemo_weight_binned'] = pd.cut(df['daemo_weight'], bins=weight_bins, labels=weight_bins[:-1] + 3, right=False)
    df['daemo_weight_binned'] = df['daemo_weight_binned'].astype(float)
    df['daemo_weight_binned'] = df['daemo_weight_binned'].astype('int16')
    ## Cap at 138 as max threshold to avoid outliers
    df.loc[df['daemo_weight_binned'] > 135, 'daemo_weight_binned'] = 138

    # daemo_height
    ## Bin Width = 4
    ## Bin centers = 154, 158, 162, 166...198
    height_bins = np.arange(152, 202, 4)
    df['daemo_height_binned'] = pd.cut(df['daemo_height'], bins=height_bins, labels=height_bins[:-1] + 2, right=True)
    df['daemo_height_binned'] = df['daemo_height_binned'].astype(float)
    df['daemo_height_binned'] = df['daemo_height_binned'].astype('int16')
    ## Cap at 158 as min threshold to avoid outliers
    df.loc[df['daemo_height_binned'] < 157, 'daemo_height_binned'] = 158

    # daemo_ideal_weight
    ## Bin Width = 4
    ## Bin centers = 22, 26, 30...90
    # ideal_weight_bins = np.arange(20, 99, 4)
    # df['daemo_ideal_weight_binned'] = pd.cut(df['daemo_ideal_weight'], bins=ideal_weight_bins, labels=ideal_weight_bins[:-1] + 2, right=False)
    
    return df

# Function for removing urine4h and ivfluid4h values before 4h
def remove_4h(df: pd.DataFrame) -> pd.DataFrame:
    # Define the number of rows corresponding to the first 4 hours
    min_data_points = 4 * 12  # 48 rows, 4h of time
    # Select only the necessary columns to optimize memory usage
    relevant_columns = ['PatientID', 'state_ivfluid4h', 'state_urin4h']
    subset_df = df[relevant_columns].copy()
    # Calculate a cumulative count of rows within each PatientID
    subset_df['row_num'] = subset_df.groupby('PatientID').cumcount()
    # Set the first 4 hours (rows 0 to 47) of each PatientID to 0
    subset_df.loc[subset_df['row_num'] < min_data_points, ['state_ivfluid4h', 'state_urin4h']] = 0
    # Drop the temporary column to clean up
    subset_df.drop(columns='row_num', inplace=True)
    # Update the original DataFrame with the modified values
    df[['state_ivfluid4h', 'state_urin4h']] = subset_df[['state_ivfluid4h', 'state_urin4h']]

    return df


# Function for converting variable units
def convert_units(df: pd.DataFrame) -> pd.DataFrame:
    # Convert unit for each column present in the var_outlier_ranges dictionary 
    for col in tqdm(df.columns):
        if col in kb.unit_conversion.keys():
            scaling = kb.unit_conversion[col]["scale"]
            df.loc[:, col] = df.eval(f"{col} * {scaling}")
    df['state_urin4h'] = df['state_urin4h'] / 12  # convert to real sum of mL/h over the last 4h
    df['state_ivfluid4h'] = df['state_ivfluid4h'] / 12  # convert to real sum of mL/h over the last 4h
            
    return df

# Function for removing outliers 
def remove_outliers(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:

    total_patients = df['PatientID'].nunique()
    print("Unique patients are: ", total_patients)

    # List to store DataFrame fragments for each column
    outliers_summary_fragments = []

    # Remove outliers for each column present in the var_outlier_ranges dictionary
    for col in tqdm(df.columns):
        if col in kb.var_outlier_ranges.keys():
            min_outlier, max_outlier = kb.var_outlier_ranges[col]
            original_dtype = df[col].dtype  # Store original data type

            lower_outlier_mask = df[col] < min_outlier
            higher_outlier_mask = df[col] > max_outlier
            unique_lower_outliers = df.loc[lower_outlier_mask, 'PatientID'].nunique()
            unique_higher_outliers = df.loc[higher_outlier_mask, 'PatientID'].nunique()

            # Create a DataFrame fragment for the current column
            fragment = pd.DataFrame(
                {'Variable': [col],
                 'Patients_Lower_Than_Min_Outlier': [unique_lower_outliers],
                 'Percentage_Lower_Than_Min_Outlier': [round(unique_lower_outliers / total_patients * 100, 2)],
                 'Patients_Higher_Than_Max_Outlier': [unique_higher_outliers],
                 'Percentage_Higher_Than_Max_Outlier': [round(unique_higher_outliers / total_patients * 100, 2)]}
            )

            outliers_summary_fragments.append(fragment)

            df[col] = np.where((df[col] >= min_outlier) & (df[col] <= max_outlier), df[col], np.nan)
            df[col] = df[col].astype(original_dtype)  # Convert back to original data type

    # Concatenate all DataFrame fragments into a single DataFrame
    outliers_summary = pd.concat(outliers_summary_fragments, ignore_index=True)

    return df, outliers_summary

# Function for transforming, creating and re-encoding variables
def transform_var(df: pd.DataFrame) -> pd.DataFrame:
    # Create unique PatientID-mv_id identifier episode_id
    df['episode_id'] = df.groupby(['PatientID', 'mv_id']).ngroup() + 1
    # daemo_discharge: 1 = alive, 0 = dead
    df['daemo_discharge'] = np.where(df['daemo_discharge'] == "alive", 1, 0)
    # state_airtype: converting decimal values into NaN (all values should be integer from 1 to 6)
    df['state_airtype'] = df['state_airtype'].apply(lambda x: np.nan if not float(x).is_integer() else x) 
    # state_airtype: IntelliLung's encoding for state_airtype
    df['state_airtype'] = df['state_airtype'].replace({1: 2, 2: 1, 3: 0, 4: 0, 5: 0, 6: 0}) 
    # vent_mode: IntelliLung's encoding for vent_mode
    df['vent_mode'] = df['vent_mode'].replace({1: 0, 2: 2, 3: 2, 4: 3, 5: 3, 6: 2, 7: 2, 8: 0, 
                                               9: 3, 10: 3, 11: 0, 12: 0, 13:0})
    # Creating vent_pinsp-peep variable for availability purposes (recalculated later in time_windowing.py)
    df['vent_pinsp-peep'] = np.where(df['vent_pinsp'].isna() | df['vent_peep'].isna(), np.nan, df['vent_pinsp'] - df['vent_peep'])
    # Creating vent_vtnorm variable, the ratio of tidal volume (vent_vt) and patient ideal weight
    df['vent_vtnorm'] = df['vent_vt'] / df['daemo_ideal_weight']

    # Filling missing vital_map values ussing formula: MAP = DBP + 1/3(SBP−DBP)
    missing_map_mask = (df['vital_map'].isnull())
    dbp_mask = (~df['vital_DBP'].isnull())
    sbp_mask = (~df['vital_SBP'].isnull())
    calc_mask = missing_map_mask & dbp_mask & sbp_mask
    df.loc[calc_mask, 'vital_map'] = df.loc[calc_mask, 'vital_DBP'] + \
        (1/3) * (df.loc[calc_mask, 'vital_SBP'] -
                 df.loc[calc_mask, 'vital_DBP'])

    # Reindex columns and convert data types
    col_reindex = [col for col in kb.var_order if col in df.columns]
    df = df.reindex(columns = col_reindex)
    df = df.astype(kb.transform_column_dtypes(df))
    return df

# Function for forward propagating valid (non-null) values for each PatientID and mv_id
def fill_na(df: pd.DataFrame) -> pd.DataFrame:
    # Taking all columns after episode_id
    column_start_index = df.columns.get_loc("episode_id")
    columns_to_fill = df.columns[column_start_index + 1:] 
    var_na = [col for col in columns_to_fill if df[col].isna().sum() != 0] #lists only columns with null values
    print(var_na)
    # Forward fill by patient and ventilation episode
    df[var_na] = df.groupby(['PatientID', 'mv_id'])[var_na].ffill() 
    return df

# Function for creating new vent variables according to vent_mode settings
def process_for_different_vent_modes(group: pd.DataFrame):
    """
    When vent_mode=2, vent_vt is an action, not an observation:
    - vent_vt_obs should be vent_vt from previous timepoint
    - vent_vt_action should be current value
    - vent_pinsp should be current value

    When vent_mode=3, vent_pinsp-peep is an action, current vent_pinsp is not an observation:
    - vent_pinsp should be vent_pinsp from previous timepoint
    - vent_pinsp-peep should be based on current vent_pinsp
    - vent_vt should be current value
    """
    # Add previous timepoint values for reference
    group["vent_vt_prev"] = group["vent_vtnorm"].shift(1)
    group["vent_pinsp_prev"] = group["vent_pinsp"].shift(1)

    # Start by setting default values
    group["vent_vt_obs"] = group["vent_vtnorm"]
    group["vent_vt_action"] = group["vent_vtnorm"]

    # Rule for vent_mode = 2
    group.loc[group["vent_mode"] == 2, "vent_vt_obs"] = group["vent_vt_prev"].bfill()
    ## bfill(): imputes first value in group which is NaN
    ## vent_pinsp remains the same

    # Rule for vent_mode = 3
    group.loc[group["vent_mode"] == 3, "vent_pinsp"] = group["vent_pinsp_prev"].bfill()

    # Drop auxiliary columns if not needed
    group.drop(columns=["vent_vt_prev", "vent_pinsp_prev"], inplace=True)

    return group
    

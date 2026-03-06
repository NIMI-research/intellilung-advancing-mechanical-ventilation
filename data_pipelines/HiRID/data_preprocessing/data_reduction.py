import pandas as pd
import numpy as np
from tqdm import tqdm

import pandas as pd

def filter_patients_with_nan_episodes(df: pd.DataFrame, variables: list, set: list) -> pd.DataFrame:
    """
    Filters out patients if any of their ventilation episodes have at least one variable 
    in 'variables' completely missing (NaN) across all time points.

    Parameters:
        df (pd.DataFrame): Input dataframe containing 'PatientID', 'mv_id', and measurement variables.
        variables (list): List of variables to check for missing values.

    Returns:
        pd.DataFrame: Filtered dataframe without patients having invalid episodes.
    """

    # Identify ventilation episodes where ANY variable in 'variables' is entirely NaN
    invalid_episodes = df.groupby(["PatientID", "mv_id"])[variables].apply(lambda x: x.isna().all().any())

    # Filter out patients whose episodes contain at least one fully NaN variable
    filtered_df = df[~df[["PatientID", "mv_id"]].apply(tuple, axis=1).isin(invalid_episodes[invalid_episodes].index)]

    print("Number of patients in time-windowed state vector, for", set, "variables, is:", 
        len(df['PatientID'].unique()), "with", df.index.size, "rows")
    print("Number of patients after removing episodes with all nan for at least one variable, for", set, "variables, is:", 
        len(filtered_df['PatientID'].unique()), "with", filtered_df.index.size, "rows\n")
    
    return filtered_df

# Function for implementing cutting points, by removing rows from the dataframe where any value in the specified columns is NaN
def cutting_point(df: pd.DataFrame, col_nan_list: np.array, set: str) -> pd.DataFrame:
    # Filter columns with nan
    columns = [col for col in col_nan_list] 
    
    # Iterate over each PatientID
    print("\nImplementing cutting points:")
    df_output = pd.DataFrame()
    for patient_id in tqdm(df['PatientID'].unique()):
        patient_df = df[df['PatientID'] == patient_id]
        
        # Iterate over each mv_id within the PatientID
        for mv_id in patient_df['mv_id'].unique():
            mv_df = patient_df[patient_df['mv_id'] == mv_id].copy()
            
            # Iterate over the rows in mv_df
            for idx, row in mv_df.iterrows():
                # Check if any value in the specified columns is NaN
                if row[columns].isnull().any():
                    # Drop the row if any value in the specified columns is NaN
                    mv_df.drop(idx, inplace=True)
                else:
                    # All values are valid, break the loop for the current mv_id
                    break
            
            # Append the cut mv_df to the final DataFrame 
            df_output = pd.concat([df_output, mv_df], ignore_index=True)

    print("Number of patients in the final time window state vector after cutting point implementation, for", set, "variables, is:", 
          len(df_output['PatientID'].unique()), "with", df_output.index.size, "rows")
    
    return df_output.sort_values(["PatientID", "mv_id", "time_interval"])

# Function for resetting time intervals by groups beginning with 60 min
def reset_time_intervals(df: pd.DataFrame) -> pd.DataFrame:
    # Group by 'PatientID' and 'mv_id' and reset time intervals
    df = df.sort_values(['PatientID', 'mv_id', 'time_interval']) 
    df['time_interval'] = (df.groupby(['PatientID', 'mv_id']).cumcount() + 1) * 60 
    return df

# Function to filter out episodes with <4h length, as they may have this duration after implementing cutting points
def filter_episode4h(df: pd.DataFrame, set: str) -> pd.DataFrame:

    ## Calculate the maximum time_interval for each unique combination of PatientID and mv_id
    max_time_interval = df.groupby(['PatientID', 'mv_id'])['time_interval'].max().reset_index()
    max_time_interval['mv_duration_new'] = max_time_interval['time_interval'] / 1440 # from minutes to days
    ## Merge to assign mv_duration_new back to the original dataframe
    df_output = pd.merge(df, max_time_interval[['PatientID', 'mv_id', 'mv_duration_new']], 
                on=['PatientID', 'mv_id'], how='left')
    ## Filter out episodes with <4h length (1/6 days)
    df_output = df_output.loc[df_output['mv_duration_new'] >= 1/6, :]
    # Eliminate the column 'mv_duration_new' from the dataset
    df_output = df_output.drop(columns=['mv_duration_new'])

    print("Number of patients in the final time window state vector after cutting point implementation \nand filtering by >=4h episode length, for", set, "variables, is:", 
          len(df_output['PatientID'].unique()), "with", df_output.index.size, "rows\n")

    return df_output.sort_values(["PatientID", "mv_id", "time_interval"])


import os
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import multiprocessing as mp

from tqdm import tqdm
from typing import Tuple

from common.config import settings
import common.kbase as kb 

# Set path to source tables
source_tables_path = os.path.join(settings.source_path, settings.path_to_source_tables)
source_tables_ventilation = "dm_ventilation"

# Function for filtering patient cohort and extracting ventilated data
def vent_patients(demog_table: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:

    # Subfolder with ventilation parquet files
    subfolder = os.path.join(source_tables_path, source_tables_ventilation)
    parquet_files = [os.path.join(subfolder, file) for file in os.listdir(subfolder) if file.endswith(".parquet")]
    # Limit the list of Parquet files in settings from common.config for demo testing
    if settings.PARQUET_TEST_NUM == 0:
        parquet_files_to_read = parquet_files
    else:
        parquet_files_to_read = parquet_files[:settings.PARQUET_TEST_NUM] 

    # Read Parquet files from this subfolder, load them into a DataFrame extracting only the required columns
    vent_df = pd.DataFrame()
    print("Iterating through parquet files from", source_tables_ventilation, "folder:")
    for parquet_file in tqdm(parquet_files_to_read):
        # Read Parquet file and select only the required columns
        df = pq.read_table(parquet_file, columns=['PatientID','AbsDatetime', 'dm_vent_inv_state']).to_pandas()
        vent_df = pd.concat([vent_df, df], ignore_index=True)
    vent_df = vent_df.rename(columns=kb.column_name_conversions)
    vent_df_full = vent_df.copy()
    vent_df_full = vent_df_full.astype(kb.transform_column_dtypes(vent_df_full))
    
    # Unique patients from dm_ventilation files before applying ventilation filtering
    unique_patients_no_filter = vent_df['PatientID'].unique()
    # Identify the intervals where vent_invas=1 and calculate the accumulated duration of vent_invas=1 in 5-minute intervals
    vent_df['cum_duration'] = vent_df.groupby(['PatientID', (vent_df['vent_invas'] != 1).cumsum()]).cumcount() * 5
    # Group consecutive invasivaly ventilated or non-ventilated rows for each PatientID
    vent_df['mv_id'] = vent_df.groupby('PatientID')['vent_invas'].transform(lambda x: (x != x.shift()).cumsum())

    # Add mv_duration to each 5-min row and reset mv_id to build vent_df 
    vent_df = vent_df[vent_df['vent_invas']==1]
    vent_df['mv_duration'] = vent_df.groupby(['PatientID', 'mv_id'])['AbsDatetime'].transform(lambda x: (x.max() - x.min()) / pd.Timedelta(days=1))
    vent_df = vent_df.sort_values(['PatientID', 'mv_id'])
    vent_df['mv_id'] = (vent_df.groupby('PatientID')['mv_id'].transform(lambda x: pd.factorize(x)[0] + 1))
    vent_df = vent_df.astype(kb.transform_column_dtypes(vent_df))
    # print("\nvent_df: ", vent_df.info())
    # Aggregate episodes to build vent_episodes
    vent_episodes = vent_df.groupby(['PatientID', 'mv_id']).agg(
        start_vent_episode=('AbsDatetime', 'min'),
        end_vent_episode=('AbsDatetime', 'max'),
        mv_duration=('mv_duration', 'max')
    ).reset_index()
    # print("\nvent_episodes: ", vent_episodes.info())
    
    # Flowchart of patient cohort
    """
    We defined 'patient cohort' as the patients fulfilling the following criteria:
    - Patients with invasive ventilation.
    - >=4h episodes.
    - Patients with valid and in-range values for height and weight.
    """
    ## Total patients
    print("\nUnique patients from dm_ventilation files before applying any ventilation filtering: ", 
          len(unique_patients_no_filter))
    ## Patients with invasive ventilation
    print("Unique patients with invasive ventilation: ", len(vent_episodes['PatientID'].unique()))
    # Patients with episodes >=4h (1/6 days)
    vent_episodes = vent_episodes[vent_episodes['mv_duration']>=1/6]
    print("Unique patients with >=4h episodes: ", len(vent_episodes['PatientID'].unique()))
    ## Patients with empty or outlier values for weight and/or height
    columns_demog_table = ['PatientID', 'dod', 'inicu_death', 'inhospital_death', 
                           'intime', 'outtime', 'los', 'daemo_discharge']
    vent_episodes = pd.merge(vent_episodes, demog_table[columns_demog_table], on='PatientID', how='inner')
    vent_episodes = vent_episodes[['PatientID', 
                                  'mv_id', 'mv_duration', 'intime', 'start_vent_episode', 'end_vent_episode', 'outtime', 'los', 
                                  'daemo_discharge', 'dod', 'inicu_death', 'inhospital_death']]
    unique_patients_cohort = vent_episodes['PatientID'].unique()
    print("Unique patients after removing patients without valid weight and height: ", len(unique_patients_cohort))

    # Export parquet file
    kb.export_parquet(vent_episodes, "hirid_vent_episodes.parquet", "hirid_vent_episodes")

    # vent_episodes: table containing information on >=4h invasivaly ventilated episodes for cohort patients 
    # unique_patients_cohort: ndarray, a list of cohort patients with PatientID
    return vent_episodes, unique_patients_cohort, vent_df_full

# Function for merging consecutive invasive mechanical ventilation (IMV) episodes with time gap less than 6 hours
def combine_episodes(df: pd.DataFrame) -> pd.DataFrame:

    # Create nº of mv episodes
    df['number_episodes'] = df.groupby('PatientID')['mv_id'].transform('count')

    # Create variable 'pause_until_next', the time in days until next >=4h IMV episode
    ## If there is only one episode for a patient or for the last episodee in multiple episodes, assign a fixed value of 90 days
    df['pause_until_next'] = 90
    df['pause_until_next'] = df['pause_until_next'].astype(float)
    ## Iterate over each PatientID with more than one episode
    for patient_id in df[df['number_episodes'] > 1]['PatientID'].unique():
        patient_df = df[df['PatientID'] == patient_id]
        mv_ids = patient_df['mv_id'].unique()
        for i in range(len(mv_ids) - 1):
            current_mv_id = mv_ids[i]
            next_mv_id = mv_ids[i + 1]
            # Get the end of the current_mv_id and the start of the next_mv_id
            end_current_episode = patient_df[patient_df['mv_id'] == current_mv_id]['end_vent_episode'].iloc[0]
            start_next_episode = patient_df[patient_df['mv_id'] == next_mv_id]['start_vent_episode'].iloc[0]
            pause_days = (start_next_episode - end_current_episode).total_seconds() / 86400
            # Assign the pause_days to the rows of current_mv_id
            df.loc[(df['PatientID'] == patient_id) & (df['mv_id'] == current_mv_id), 'pause_until_next'] = pause_days
    
    # Create combination variable
    df['combination'] = np.where(df['pause_until_next'] < 0.25, "YES", "NO")

    # Group consecutive episodes with time gap less than 6 hours
    ## Create a helper column 'group_id' to identify groups of consecutive episodes to combine
    group_id = []
    current_group = 0

    for idx, row in df.iterrows():
        if idx == 0 or df.loc[idx - 1, 'combination'] == "NO":
            current_group += 1
        group_id.append(current_group)

    df['group_id'] = group_id

    ## Aggregate the episodes based on the group_id
    aggregated = df.groupby(['group_id']).agg(
        {
            'PatientID': 'first',
            'mv_id': 'first',  
            'mv_duration': 'sum',  
            'intime': 'first', 
            'start_vent_episode': 'first', 
            'end_vent_episode': 'last',  
            'outtime': 'last', 
            'los': 'last', 
            'daemo_discharge': 'last', 
            'dod': 'last', 
            'inicu_death': 'last', 
            'inhospital_death': 'last', 
            'number_episodes': 'last', 
            'pause_until_next': 'last', 
            'combination': 'first'  
        }
    ).reset_index(drop=True)

    ## Recalculate mv_duration to take into account aggregated episodes
    aggregated['mv_duration'] = round((aggregated['end_vent_episode'] - aggregated['start_vent_episode']).dt.total_seconds() / 86400, 4)

    return aggregated

# Function for adding ventilator-related variables to vent_episodes dataframe
def add_ventilator_days(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(['PatientID', 'start_vent_episode']).reset_index(drop=True)
    # Create time at day30 from ventilation start per patient
    df['Day30'] = df.groupby('PatientID')['start_vent_episode'].transform('min') + pd.Timedelta(days=30)
    # Create sum of all ventilation episodes in days per patient
    df['total_mv_duration'] = df.groupby('PatientID')['mv_duration'].transform('sum')
    # Create max end_vent_episode time (in case there are multiple episodes)
    df['max_vent_time'] = df.groupby('PatientID')['end_vent_episode'].transform('max')
    # Calculate the number of days since the last ventilation episode until death. 
    ## If the patient is discharged alive, assign a fixed value of 90 days
    df['post_extubation_interval'] = np.where(df['daemo_discharge']=="dead", 
                                            ((df['dod'] - df['max_vent_time']).dt.total_seconds() / 86400).round(2), 90)

    # Create ventilator-free days from los and total_mv_duration
    df['VFD'] = df['los'] - df['total_mv_duration']

    # Calculate ventilator-free days at day 30 based on different discharge, lenght of stay and ventilation conditions:
    conditions = [
        (df['outtime'] >= df['Day30']) & (df['max_vent_time'] < df['Day30']),
        (df['outtime'] >= df['Day30']) & (df['max_vent_time'] >= df['Day30']),
        (df['dod'] < df['Day30']),
        (df['outtime'] < df['Day30']) & (df['daemo_discharge'] == "alive")
    ]
    values = [
        30 - df['total_mv_duration'],                              
        np.maximum(30 - df['total_mv_duration'], 0), #equal to 0 if there is one single mv episode for >30 days                                                         
        ((df['dod'] - df['max_vent_time']).dt.total_seconds() / 86400).round(2), # for multiple mv episodes, in-VFD are not considered                
        30 - df['total_mv_duration']                               
    ]
    labels = [
    'Ventilation ended before Day30 and patient discharged after Day30',  
    'Ventilation continued after Day30',                               
    'Discharged dead before Day30',                                
    'Discharged alive before Day30'                               
]
    df['VFD30'] = np.select(conditions, values, default=np.nan)
    # Assign labels for different types of condition
    df['condition'] = np.select(conditions, labels, default='Unclassified')

    # columns_to_round = ['mv_duration', 'los', 'total_mv_duration', 'post_extubation_interval', 'VFD', 'VFD30']
    # df[columns_to_round] = round(df[columns_to_round], 2)
    return df.sort_values(['PatientID', 'mv_id'])

# Function for assigning mv_id to ventilation raw data
def assign_mvid(vent_df_full: pd.DataFrame, vent_episodes: pd.DataFrame, multiproc: bool = False) -> pd.DataFrame:
    vent_df_assigned = pd.DataFrame()
    print("\nAssigning mv_id to each 5-min interval row:")
    for patient in tqdm(vent_df_full['PatientID'].unique()):
        patient_df = vent_df_full[vent_df_full['PatientID'] == patient][['PatientID', 'AbsDatetime', 'vent_invas']].copy()
        episode_df = vent_episodes[vent_episodes['PatientID'] == patient]
        for _, row in episode_df.iterrows():
            mask = (patient_df['AbsDatetime'] >= row['start_vent_episode']) & (patient_df['AbsDatetime'] <= row['end_vent_episode'])
            patient_df.loc[mask, 'mv_id'] = row['mv_id']
        vent_df_assigned = pd.concat([vent_df_assigned, patient_df], ignore_index=True)
    ## Remove NaN values in mv_id
    vent_df_assigned = vent_df_assigned.dropna(subset=['mv_id'])
    return vent_df_assigned

# Function for creating time windows using multiprocessing to accelerate computation time
def create_assign_mvid_fast(vent_df_full: pd.DataFrame, vent_episodes: pd.DataFrame):
    """
    Assign mv_id to a large dataset using multiprocessing for efficiency.
    """
    results = []
    df = []

    unique_patient_ids = vent_episodes["PatientID"].unique()
    tasks = min(100, len(unique_patient_ids))
    split_patient_ids = np.array_split(unique_patient_ids, tasks)

    with mp.Pool(settings.num_of_cores) as pool:
        pbar = tqdm(total=tasks, desc="Assigning mv_id [%]", position=0)

        def update(*a):
            pbar.update()

        for i in range(tasks):
            # Subset data for each chunk
            chunk_vent_df = vent_df_full[vent_df_full["PatientID"].isin(split_patient_ids[i])]
            chunk_vent_episodes = vent_episodes[vent_episodes["PatientID"].isin(split_patient_ids[i])]

            results.append(
                pool.apply_async(
                    assign_mvid,
                    args=(chunk_vent_df, chunk_vent_episodes, True),
                    callback=update,
                )
            )

        # Collect results
        for result in results:
            try:
                df.append(result.get())
            except Exception as e:
                print(f"Error in multiprocessing: {e}")
                raise

        pbar.close()

    # Combine all results and sort
    combined_df = pd.concat(df, ignore_index=True).sort_values(by=["PatientID", "AbsDatetime"]).reset_index(drop=True)
    # Apply column type transformations
    combined_df = combined_df.astype(kb.transform_column_dtypes(combined_df))

    return combined_df
    


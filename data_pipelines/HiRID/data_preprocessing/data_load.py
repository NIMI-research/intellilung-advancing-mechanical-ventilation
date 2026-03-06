
import os
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import subprocess

from tqdm import tqdm
from typing import Tuple

import data_preprocessing.data_cleaning as dc
from common.config import settings
import common.kbase as kb 

# Set path to source tables
source_tables_path = os.path.join(settings.source_path, settings.path_to_source_tables)

# List of column names from source files to drop as they are not within IntelliLung's variables of interest
# vm1 and vm2 for weight and height are removed from dm source files and are taken from demographic/static table
list_var_to_drop = ['vm1', 'vm2', 'vm2201', 'vm2202', 'vm2203', 
                        'vm2204', 'vm2205', 'vm2206', 'vm2207', 'vm5001', 'vm5096', 'vm5097',
                            'vm5098', 'vm5099', 'vm2201_bolus', 'vm2202_bolus', 'vm2203_bolus',
                                'vm2204_bolus', 'vm2205_bolus', 'vm2206_bolus', 'vm2207_bolus',
                                    'vm2218_bolus', 'vm2219_bolus']

# Function for creating static/demographic table from source tables
def create_static_table() -> pd.DataFrame:

    path_general_table = os.path.join(source_tables_path, "static_table/general_table.csv")
    # Check if the file exists
    if not os.path.exists(path_general_table):
        raise FileNotFoundError(
            f"Error: The file 'general_table.csv' was not found. Please ensure the 'general_table.csv' "
            "file is present in the specified directory."
        )

    # Paths to dm_merged parquet files
    mdp = os.path.join(settings.source_path, "source_tables/dm_merged/")
    parquet_files = [os.path.join(mdp, f) for f in os.listdir(mdp) if f.endswith('.parquet')]

    # Load data
    dall = []
    dtimes = []

    for file in parquet_files:
        data = pq.read_table(file).to_pandas()
        selected_columns = data.iloc[:, :4]  # Selecting first 4 columns
        dtimes.append(selected_columns[['PatientID', 'AbsDatetime']])
        data = data.dropna(subset=['vm1', 'vm2'])
        
        # Aggregating multiple measurements of weight (vm1) and height (vm2) by the mean 
        grouped = data.groupby('PatientID', as_index=False)
        data = grouped.agg({
            'vm1': lambda x: x.mean(skipna=True),
            'vm2': lambda x: x.mean(skipna=True)
        })
        
        dall.append(data)

    dall = pd.concat(dall).sort_values(by='PatientID')
    upids = pd.DataFrame({'origin': ['dm_merged', 'dm_merged with valid height&weight'],
                        'N': [len(pd.concat(dtimes)['PatientID'].unique()), dall['PatientID'].nunique()]})

    # Renaming columns
    statDat = dall.rename(columns={'vm1': 'weight', 'vm2': 'height'})

    # Getting in/out times
    dtimes = pd.concat(dtimes).dropna().drop_duplicates()
    intime = dtimes.groupby('PatientID')['AbsDatetime'].first()
    outtime = dtimes.groupby('PatientID')['AbsDatetime'].last()
    los = ((outtime - intime).dt.total_seconds() / 86400).astype('float32') # Length of stay in days

    Ts = pd.DataFrame({'pid': intime.index, 'intime': intime.values, 'outtime': outtime.values, 'los': los})
    statDat = statDat.merge(Ts, left_on='PatientID', right_on='pid', how='left').drop(columns=['pid'])

    # Loading additional data from general_table.csv
    gt = pd.read_csv(path_general_table)
    print("Unique patients in general_table.csv are: ", gt['patientid'].nunique())
    print(f"weight/height information missing for {upids.iloc[0,1] - upids.iloc[1,1]} patients.")
    gt = gt.sort_values(by='patientid')
    gt = gt[gt['patientid'].isin(statDat['PatientID'])]
    gt['admissiontime'] = pd.to_datetime(gt['admissiontime'], format='%Y-%m-%d %H:%M:%S', utc=True)
    statDat = statDat.merge(gt[['patientid', 'sex', 'age', 'discharge_status']], left_on='PatientID', right_on='patientid', how='left')
    statDat.drop(columns=['patientid'], inplace=True)

    # Calculating ideal body weight
    statDat['ideal_body_weight'] = 50 + 0.91 * (statDat['height'] - 152.4)
    statDat.loc[statDat['sex'] == 'F', 'ideal_body_weight'] = 45.5 + 0.91 * (statDat['height'] - 152.4)

    # Handling discharge status and dod (date of dead)
    statDat = statDat.rename(columns={'discharge_status': 'discharge_location'})
    discharge_map = {'dead': 1, 'alive': 0}
    statDat['inhospital_death'] = statDat['discharge_location'].map(discharge_map).fillna(0).astype(int)
    statDat['inicu_death'] = statDat['inhospital_death']
    statDat['dod'] = pd.NaT
    statDat['dod'] = statDat['dod'].astype('datetime64[ns]')
    statDat.loc[statDat['discharge_location'] == 'dead', 'dod'] = statDat['outtime']

    # Applying exclusion criteria
    initial_patient_count = statDat['PatientID'].nunique()
    statDat = statDat[(statDat['weight'] >= 40) & (statDat['weight'] <= 140)]
    statDat = statDat[(statDat['height'] >= 155) & (statDat['height'] <= 200)]
    excluded_patient_count = initial_patient_count - statDat['PatientID'].nunique()
    print(f"Number of patients excluded by weight and height criteria: {excluded_patient_count}")

    # Handling missing discharge locations
    statDat['discharge_location'] = statDat['discharge_location'].fillna('alive')
    statDat['sex'] = statDat['sex'].map({'M': 0, 'F': 1}).astype('category')

    # Rename column names, convert data types and export demographic/static table
    demog_table = statDat.rename(columns=kb.column_name_conversions)
    demog_table = demog_table.astype(kb.transform_column_dtypes(demog_table))
    print(demog_table.info())
    print("Unique patients in demog_table after excluding nans and outliers from weight and height are: ", demog_table['PatientID'].nunique())
    # demog_table.to_csv(os.path.join(settings.source_path, "source_tables/static_table/HiRID_static_table.csv"), 
    #                    sep = ',', index=False)

    # demog_table: pd.DataFrame, containing demographic information for each patient
    ## daemo_sex: binary variable, 1 for Female, 0 for Male
    ## daemo_age: age in years
    ## daemo_weight: weight in Kg
    ## daemo_ideal_weight: ideal weight in Kg based on height following an equation.
    ## daemo_height: heigh in cm
    return demog_table

# Function for extracting parquet files from source directory and joining them. Folder names must begin with dm, for instance:
# dm_fluid, dm_merged, dm_vasoactive, dm_ventilation
def combine_parquets(state_vector_raw: pd.DataFrame, cohort_patients_list: np.array) -> pd.DataFrame:

    # Get a list of all subfolders
    subfolders = [os.path.join(source_tables_path, folder) for folder in os.listdir(source_tables_path) 
                if os.path.isdir(os.path.join(source_tables_path)) and folder.startswith("dm")]
    
    # Iterate over each subfolder
    print("Iterating through subfolders:")
    for subfolder in tqdm(subfolders):
        # Get a list of all Parquet files in the current subfolder
        parquet_files = [os.path.join(subfolder, file) for file in os.listdir(subfolder) if file.endswith(".parquet")]

        # Limit the list of Parquet files in settings from common.config for demo testing
        if settings.PARQUET_TEST_NUM == 0:
            parquet_files_to_read = parquet_files
        else:
            parquet_files_to_read = parquet_files[:settings.PARQUET_TEST_NUM]

        # Read Parquet files from each subfolder and load them into a DataFrame
        subfolder_df = pd.DataFrame()
        print("Iterating through parquet files from", os.path.basename(subfolder), "folder:")
        for parquet_file in tqdm(parquet_files_to_read):
            df = pq.read_table(parquet_file).to_pandas()
            df = df[df['PatientID'].isin(cohort_patients_list)] #select included patient cohort
            float_columns = df.select_dtypes(include=['float64']).columns
            df[float_columns] = df[float_columns].astype(np.float32) #reduce mempory by converting float64 columns to float32
            subfolder_df = pd.concat([subfolder_df, df], ignore_index=True)

        # Drop not needed variables
        subfolder_df = subfolder_df.drop(list_var_to_drop, axis=1, errors='ignore')
        # print(state_vector_raw.columns)

        # Perform the successive joins with source files from each subfolder
        if state_vector_raw.empty:
            state_vector_raw = subfolder_df
        else:
            state_vector_raw = state_vector_raw.join(subfolder_df.set_index(["PatientID", "AbsDatetime"]), how="inner", on=["PatientID", "AbsDatetime"])

    state_vector_raw = state_vector_raw.sort_values(['PatientID', 'AbsDatetime'])
    print("\nUnique patients in combined dataframe are: ", state_vector_raw['PatientID'].nunique())
    
    # Rename column names to IntelliLung's variable schema using the equivalence dictionary and convert data types to reduce memory
    print("#### Renaming column names to IntelliLung's variable schema and converting data types")
    state_vector_raw = state_vector_raw.rename(columns=kb.column_name_conversions)
    
    # Ensure safe type conversions
    for col, dtype in kb.transform_column_dtypes(state_vector_raw).items():
        try:
            state_vector_raw[col] = state_vector_raw[col].astype(dtype)
        except ValueError:
            print(f"Skipping conversion of column {col} to {dtype} due to unsafe conversion.")

    # Remove values (convert to 0) before 4h for cumulative 4h variables 
    # This step must be previous to later filterings to have proper 4h variables, if not we could be removing valid values
    print("#### Removing urine4h and ivfluid4h values before 4h")
    state_vector_raw = dc.remove_4h(state_vector_raw)

    # # Filter for invasive ventilation rows
    # state_vector_raw = state_vector_raw[state_vector_raw['vent_invas'] == 1]

    # # Show information about the combined data
    # print("\n######### Information of combined DataFrame: #########")
    # print(state_vector_raw.info())

    return state_vector_raw

# Function for merging tables (demog_table, state_vector_raw, vent_episodes)
def merge_tables(demog_table: pd.DataFrame, df: pd.DataFrame, vent_episodes: pd.DataFrame, 
                    vent_df_assigned: pd.DataFrame, cohort_patients_list: np.array) -> pd.DataFrame:
  
    # Bin variables in demog_table
    demog_table = dc.bin_variables(demog_table)
    # # Quality control for bin_variables
    # bin_variables = demog_table.copy()
    # bin_variables = demog_table[['PatientID', 'daemo_age', 'age_binned', 'daemo_weight', 'daemo_weight_binned', 
    #                          'daemo_height', 'daemo_height_binned', 'daemo_ideal_weight', 'daemo_ideal_weight_binned']]
    
    # Eliminate original demographic variables and replace them with binned versions
    demog_table = demog_table.drop(['daemo_age', 
                                    'daemo_weight', 
                                    'daemo_height'],
                                    axis=1)
    demog_table = demog_table.rename(columns={
        'age_binned': 'daemo_age',
        'daemo_weight_binned': 'daemo_weight',
        'daemo_height_binned': 'daemo_height'
    })
    demog_table = demog_table.astype(kb.transform_column_dtypes(demog_table))
    # print(demog_table.info())

    # Join source tables with static table (with demographic data), vent_episodes (ventilation information) and vent_df_assigned (mv_duration and mv_id)  
    ## Merge by 5-min intervals
    df = df.merge(vent_df_assigned[['PatientID', 'AbsDatetime', 'mv_id']], 
            how="inner", on=["PatientID", "AbsDatetime"]).reset_index(drop=True)
    ## Merge by episodes
    df = df.merge(vent_episodes[['PatientID', 'mv_id', 'mv_duration', 'post_extubation_interval', 'pause_until_next', 'VFD', 'VFD30']], 
                  how="inner", on=['PatientID', 'mv_id'])
    ## Merge by patients
    df = df.merge(demog_table[['PatientID', 'daemo_sex', 'daemo_age', 'daemo_weight', 'daemo_height', 'daemo_discharge', 
                     'daemo_ideal_weight']], how='inner', on='PatientID')

    # Include additional demographic varaibles in the static/demog table
    demog_table = demog_table.merge(vent_episodes[['PatientID', 'total_mv_duration', 'VFD30']], how='inner', on='PatientID')
    demog_table = demog_table.drop_duplicates(subset='PatientID').reset_index(drop=True)
    demog_table = demog_table[['PatientID', 'los', 'daemo_age', 'daemo_sex', 
                                        'daemo_ideal_weight', 'daemo_weight', 'daemo_height', 
                                        'total_mv_duration', 'VFD30', 
                                        'daemo_discharge', 'inhospital_death', 'inicu_death']].copy()
    kb.export_parquet(demog_table, "hirid_demog_table.parquet", "hirid_demog_table")
    # Apply standardized column names and units
    demog_table['database'] = 2
    demog_table['daemo_discharge'] = np.where(demog_table['daemo_discharge'] == "alive", 1, 0)
    demog_table['total_mv_duration'] = demog_table['total_mv_duration']*24 #convert to hours
    demog_table = demog_table.rename(columns={'PatientID': 'stay_id'})
    cols = ['database'] + [col for col in demog_table.columns if col != 'database']
    demog_table = demog_table[cols]
    demog_table.to_csv((os.path.join(settings.source_path, settings.output_save_path, "hirid_demog_table.csv")), index=False)

    print("\nUnique patients after merging all source tables are: ", df['PatientID'].nunique())
        
    col_reindex = [col for col in kb.var_order if col in df.columns]
    df = df.reindex(columns = col_reindex)

    print("\nUnique patients in state_vector_raw are: ", df['PatientID'].nunique())
    # print(df.info())

    # df: pd.DataFrame, state_vector_raw on cohort patients in 5-min intervals (without any further filtering), as it was captured from HiRID.
        ## episode_id is a primary key associated with unique combinations of PatientID and mv_id
    # demog_table: pd.DataFrame, containing specific demographic information for each patient

    return df, demog_table




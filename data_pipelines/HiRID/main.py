# -------------------- ENVIRONMENT ----------------------
import pandas as pd
import numpy as np
import time
import pyarrow.parquet as pq
import os

from common.config import settings
import common.kbase as kb 

import data_preprocessing.data_filtering as df
import data_preprocessing.data_load as dl 
import data_preprocessing.data_cleaning as dc
import data_preprocessing.time_windowing as tw 
import data_preprocessing.data_reduction as dr

def main():
    # -------------------- DATA FILTERING & LOADING ----------------------
    start_time = time.perf_counter()
    print("######## Starting Data Load ########")

    # Load and filter static/demographic table 
    print("\n#### Load static/demographic table:")
    demog_table = dl.create_static_table()

    # Extract and filter patients and episodes that fulfil the inclusion and ventilation criteria. Build episode dataframe
    print("\n#### Extracting IMV episodes and ventilation data:\n")
    vent_episodes, cohort_patients_list, vent_df_full = df.vent_patients(demog_table) 
    # Merge consecutive invasive mechanical ventilation (IMV) episodes with time gap less than 6 hours
    vent_episodes = df.combine_episodes(vent_episodes)
    # Add vent variables to episodes table
    vent_episodes = df.add_ventilator_days(vent_episodes)
    # Assign episode mv_id to raw ventilation data
    vent_df_reduced = df.create_assign_mvid_fast(vent_df_full, vent_episodes)

    # Extract files and join tables 
    print("\n#### Extracting and merging source tables:\n")
    state_vector_raw = pd.DataFrame()
    state_vector_raw = dl.combine_parquets(state_vector_raw, cohort_patients_list)

    # Merge across tables
    print("\n#### Merging source tables with demographic and ventilation data:\n")
    state_vector_raw, demog_table = dl.merge_tables(demog_table, state_vector_raw, vent_episodes, vent_df_reduced, cohort_patients_list)
    var_NaN_raw, var_noNaN_raw = kb.save_nan_var(state_vector_raw) # Save variables with NaN and no NaN

    # Export raw state vector in parquet format
    # state_vector_raw contains raw data on patient cohort in 5-min intervals, selecting >=4h episodes of invasive mechanical ventilation (IMV) 
    kb.export_parquet(state_vector_raw, settings.state_vector_raw_name, "state_vector_raw")

    print(f"\nData Load is finished and took: {round((time.perf_counter() - start_time)/60, 2)} minutes")

    # -------------------- DATA CLEANING & STANDARDIZATION ----------------------
    start_time = time.perf_counter()
    print("######## Starting Data Cleaning ########\n")

    print("\nUnique patients in state_vector_raw are:", state_vector_raw['PatientID'].nunique())
    print("Rows in state_vector_raw are:", state_vector_raw.index.size, "with cohort patients and >=4h IMV episodes")
    state_vector_not_imputed = state_vector_raw.copy()
    state_vector_not_imputed = state_vector_not_imputed.sort_values(['PatientID', 'AbsDatetime']).reset_index(drop=True)

    # Converting units
    print('\n#### Converting units')
    state_vector_not_imputed = dc.convert_units(state_vector_not_imputed)

    # Removing outliers - converting them into NaNs
    print('\n#### Removing outliers')
    state_vector_not_imputed, outlier_summary = dc.remove_outliers(state_vector_not_imputed)
    outlier_summary.to_csv(os.path.join(settings.source_path, settings.output_save_path, "outlier_summary.csv"), 
                        sep = ',', index=False)
    var_NaN_not_imputed, var_noNaN_not_imputed = kb.save_nan_var(state_vector_not_imputed) # Save variables with NaN and no NaN

    # Transform, create and re-encode variables
    print('\n#### Transforming, creating and re-encoding variables')
    state_vector_not_imputed = dc.transform_var(state_vector_not_imputed)

    # Export not-imputed state vector df in parquet format
    ## state_vector_not_imputed dataframe contains selected ventilation data on patient cohort in 5-min intervals 
    ## after applying unit conversion and outlier removal
    kb.export_parquet(state_vector_not_imputed, settings.state_vector_not_imputed_name, "state_vector_not_imputed")

    # Imputation: Filling NA values
    print('\n#### Propagating non-null values forward for each PatientID and mv_id')
    state_vector_imputed = state_vector_not_imputed.copy()
    state_vector_imputed = dc.fill_na(state_vector_imputed)
    # print(state_vector_imputed.info())
    var_NaN_imputed, var_noNaN_imputed = kb.save_nan_var(state_vector_imputed) # Save variables with NaN and no NaN

    # Export imputed state vector in parquet format
    ## state_vector_imputed dataframe contains propagated data in 5-min intervals, after forward propagation of non-null values.
    kb.export_parquet(state_vector_imputed, settings.state_vector_imputed_name, "state_vector_imputed")

    print(f"\nData Cleaning is finished and took: {round((time.perf_counter() - start_time)/60, 2)} minutes")

    # -------------------- TIME WINDOWS CREATION ----------------------
    start_time = time.perf_counter()
    print("######## Starting Data Vectorization ########\n")

    # Create time windows
    print('\n#### Creating time windows for non-imputed state vector')
    resolution_subname = "_" + str(settings.resolution) + "min"
    if settings.create_nonimputed == 1:
        state_vector_time_windows_notimputed = tw.create_time_windows_fast(state_vector_not_imputed) #not imputed time windowed 
        # state vector for statistical purposes and comparison across databases
        kb.export_parquet(state_vector_time_windows_notimputed, "hirid_state_vector_time_windows_notimputed" + resolution_subname + ".parquet", 
                    "state_vector_time_windows_notimputed" + resolution_subname)
        
    print('\n#### Creating time windows for imputed state vector')
    state_vector_time_windows_imputed = tw.create_time_windows_fast(state_vector_imputed) #imputed df for further processing in the present code
    # print(state_vector_time_windows_imputed.info())
    var_NaN_time_windows, var_noNaN_time_windows = kb.save_nan_var(state_vector_time_windows_imputed) # Save variables with NaN and no NaN
    kb.export_parquet(state_vector_time_windows_imputed, "hirid_state_vector_time_windows_imputed" + resolution_subname + ".parquet", 
                    "state_vector_time_windows_imputed" + resolution_subname)

    # dfSummary(state_vector_time_windows)

    print(f"\nData Vectorization is finished and took: {round((time.perf_counter() - start_time)/60, 2)} minutes")

    # -------------------- CUTTING POINT & SELECTION OF SET VARIABLES ----------------------

    # Filter the time windowed state vector by selected set variables. Change combinations for distinct set of variables of interest
    set_variables = kb.set0_variables + kb.set1_variables + kb.set2a_variables + kb.set2b_variables
    set_filtered = [var for var in set_variables if var not in kb.set_low_availability_variables]
    set_variables_name = "setfull" # change for distinct output name
    state_vector_time_windows_set = state_vector_time_windows_imputed[set_filtered]
    var_NaN_time_windows_set, var_noNaN_time_windows_set = kb.save_nan_var(state_vector_time_windows_set) # Save variables with NaN and no NaN

    # Implement cutting point, reset time_intervals to resolution start time and filter resulting episodes by >4h duration
    print('\n#### Implementing cutting points')
    state_vector_time_windows_set_cut = dr.filter_patients_with_nan_episodes(state_vector_time_windows_set, var_NaN_time_windows_set, set_variables_name)
    state_vector_time_windows_set_cut = dr.cutting_point(state_vector_time_windows_set_cut, var_NaN_time_windows_set, set_variables_name)
    state_vector_time_windows_set_cut = dr.reset_time_intervals(state_vector_time_windows_set_cut)
    state_vector_time_windows_set_cut_filtered = dr.filter_episode4h(state_vector_time_windows_set_cut, set_variables_name).reset_index(drop=True)
    ## Save variables with NaN and no NaN (all columns should have non-NaN values)
    var_NaN_time_windows_set_cut, var_noNaN_time_windows_set_cut = kb.save_nan_var(state_vector_time_windows_set_cut_filtered)
    print("The columns with nan in the final reduced state_vector are:", var_NaN_time_windows_set_cut)

    # Standardization of columns to combine with other database state vectors
    state_vector_time_windows_set_cut_filtered.rename(columns={'PatientID': 'stay_id'}, inplace=True)
    print(state_vector_time_windows_set_cut_filtered.info())

    # Export state vector time windows
    kb.export_parquet(state_vector_time_windows_set, "hirid_state_vector_time_windows" + resolution_subname + "_" + set_variables_name + ".parquet", 
                    "state_vector_time_windows" + resolution_subname + "_" + set_variables_name)
    kb.export_parquet(state_vector_time_windows_set_cut_filtered, "hirid_state_vector" + resolution_subname + "_" + 
                    set_variables_name + "ai_ready" + ".parquet", "state_vector_time_windows" + resolution_subname + "_" + set_variables_name + "_cut")

    # -------------------- QUALITY CONTROL ----------------------
    # Final checking in final ai-ready state vector
    resolution_subname = "_" + str(settings.resolution) + "min"
    set_variables_name = "setfull" # change for distinct output name
    hirid_ai_ready = pd.read_parquet(os.path.join(settings.source_path, settings.output_save_path, "hirid_state_vector" + resolution_subname + "_" + set_variables_name + "ai_ready" + ".parquet")) 
    print("Unique patients in final ai_ready state vector:", hirid_ai_ready['stay_id'].nunique())
    print("Rows in final ai_ready state vector:", hirid_ai_ready.index.size)

    if hirid_ai_ready['stay_id'].nunique() == state_vector_time_windows_set_cut_filtered['stay_id'].nunique() and hirid_ai_ready.index.size == state_vector_time_windows_set_cut_filtered.index.size:
        print("Quality Control passed successfully\n")
    else:
        print("Check final dataframes\n")

    print(hirid_ai_ready.info())

    # Export to csv
    hirid_ai_ready.to_csv((os.path.join(settings.source_path, settings.output_save_path, 
                                                "hirid_state_vector_60min_setfullai_ready.csv")), 
                        index=False)

    # # Data Validation Instructions:
    # # Create a .env file in the data_pipelines/data_validation folder.
    # # Ensure the final state vector is placed in the data_validation folder with the same name as specified in .env.example.
    # # Follow the instructions in the data_validation README.
    # VALIDATION_CHECK_CONFIGS_FILE_PATH='configs/validation_checks_hirid.yml'
    # DATASET_PATH='hirid_state_vector_60min_setfullai_ready.csv'

if __name__ == "__main__":
    main()


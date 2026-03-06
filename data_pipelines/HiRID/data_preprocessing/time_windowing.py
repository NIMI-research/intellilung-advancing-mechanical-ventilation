import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm
import pyarrow.parquet as pq
import warnings
import multiprocessing as mp

from common.config import settings
import common.kbase as kb 
import data_preprocessing.data_cleaning as dc

def create_time_windows(state_vector: pd.DataFrame, resolution: int, multiproc: bool = False) -> pd.DataFrame:
    full_df = []
    #'AbsdDatetime' is modified in this function and vent_vt_obs and vent_vt_action are newly created afterwards
    col_reindex = [col for col in kb.var_order if col not in ['AbsDatetime', 'vent_vt_obs', 'vent_vt_action']]

    # Create a new AbsDatetime column with incremental minutes starting from 0, in steps of 5, for each PatientID
    state_vector['AbsDatetime'] = state_vector.groupby('PatientID').cumcount() * 5
    state_vector = state_vector.rename(columns={"AbsDatetime": "time_interval"})
    # print(state_vector.head())
    # print(state_vector.info())

    # Iterating through each patient
    for patient in tqdm(state_vector["PatientID"].unique()):
        # Extracting data for the specific patient
        patient_data = state_vector.loc[state_vector["PatientID"] == patient, :]

        # Iterating through each patient's mv_id
        mv_id_true = 0
        stay_start_time = patient_data["time_interval"].values[0]
        for mv_id in patient_data["mv_id"].unique():
            # Extracting data for the specific MV
            mv_data = patient_data.loc[patient_data["mv_id"] == mv_id, :]
            start_time = mv_data["time_interval"].values[0]
            end_time = mv_data["time_interval"].values[-1]
            if start_time == end_time:
                continue  # Skip if the end time is the same as the start time for the ventilation episode mv_id
            
            # Generation of time windows
            ## Calculate the time windows in minutes
            time_diff = end_time - start_time
            windows = (1 + np.arange(np.ceil((time_diff + 1) / resolution))) * resolution 

            ## Create a DataFrame with shape according to time windows (windows) and state_vector columns
            mat = np.zeros(shape=(len(windows), len(col_reindex)))
            mat.fill(np.nan)
            df = pd.DataFrame(data=mat, columns=col_reindex)
            ## Fill with PatientID and time intervals, along with static variables
            df.loc[:, "PatientID"] = patient
            df.loc[:, "time_interval"] = windows
            static_var = kb.static_var
            df.loc[:, static_var] = mv_data.loc[mv_data.index[0], static_var].values
            # print(df.info())

            # Iterating through time windows
            N = len(windows)
            for i, window in enumerate(windows):
                ## Extracting datapoints for the resolution interval
                window_data = mv_data.loc[
                    mv_data["time_interval"].between(
                        left = start_time + window - resolution, 
                        right = start_time + window,
                        inclusive = "left"
                    ), list(col_reindex)].reset_index(drop=True)
                # print(window_data.info())

                window_data.drop("time_interval", axis=1, inplace=True)

                # Fill the time window with the median for continous variables and the mode for categorical ones
                # Specific cases are set for fluid intake and output variables, filling with last value
                with warnings.catch_warnings(): #prevent output warnings when encountering NaNs
                    warnings.filterwarnings(
                        "ignore", 
                        message="All-NaN slice encountered", 
                        category=RuntimeWarning
                    ) 
                   
                    df.loc[i, kb.num_var] = np.nanmedian(window_data[kb.num_var], axis=0)

                    for cat in kb.cat_var:
                        df.loc[i, cat] = stats.mode(window_data[cat], keepdims=True)[0][0]

                    df.loc[i, 'dm_total_in'] = window_data['dm_total_in'].values[-1]
                    df.loc[i, 'dm_total_out'] = window_data['dm_total_out'].values[-1]
                    df.loc[i, 'state_cumfluids'] = window_data['state_cumfluids'].values[-1]
                    df.loc[i, 'state_ivfluid4h'] = window_data['state_ivfluid4h'].values[-1]
                    df.loc[i, 'state_urin4h'] = window_data['state_urin4h'].values[-1]
                    df.loc[i, 'drugs_vaso4h'] = window_data['drugs_vaso4h'].values[-1] #max vasopressor in the last 4h

            if len(df) != 0:
                # Makes time_interval absolute by adding the time difference between the beginning of the ICU stay and the beginning of the
                # current MV treatment
                if mv_id_true != 0:
                    df["time_interval"] += start_time - stay_start_time

                df.loc[:, "mv_id"] = mv_id_true + 1 #beginning the mv_id at 1
                mv_id_true += 1

                # TODO the propagation here does carry-forward but that might not be optimal for fluids and drugs
                # TODO introduce SAH periods (values should only be propagated for a select few time intervals)
                # propagate_measurements(data=df, stayid_weight_data=weight_data, start_time=start_time)
                full_df.append(df)

    if len(full_df) != 0:
        full_df = (pd.concat(full_df).sort_values(by=["PatientID", "mv_id", "time_interval"], ignore_index=True))
        col_reindex = [col for col in col_reindex if col in full_df.columns]
        full_df = full_df.reindex(columns = col_reindex)
        # full_df = full_df.astype(kb.transform_column_dtypes(full_df))
        return full_df

    else:
        return pd.DataFrame()

# Function for creating time windows using multiprocessing to accelerate computation time
def create_time_windows_fast(data: pd.DataFrame):
    """"""
    results = []
    state_vectors = []

    unique_stay_ids = data["PatientID"].unique()
    tasks = min(100, len(unique_stay_ids))
    split_stay_ids = np.array_split(unique_stay_ids, tasks)

    with mp.Pool(settings.num_of_cores) as pool:
        pbar = tqdm(total=tasks, desc="Windowing [%]", position=0)

        def update(*a):
            pbar.update()

        for i in range(tasks):
            results.append(
                pool.apply_async(
                    create_time_windows,
                    args=(
                        data.loc[data["PatientID"].isin(split_stay_ids[i]), :],
                        settings.resolution,
                        True,
                    ),
                    callback=update,
                )
            )
        for result in results:
            result = result.get()
            state_vectors.append(result)

        pbar.close()

    state_vectors = pd.concat(state_vectors, ignore_index=True).sort_values(by=["PatientID", "mv_id", "time_interval"]).reset_index(drop=True)

    # Calculate new composite variables or recalculate existing ones as these may have changed after applying time windowing
    ## Recalculate vent_pinsp-peep  
    state_vectors['vent_pinsp-peep'] = np.where(state_vectors['vent_pinsp'].isna() | state_vectors['vent_peep'].isna(),
                                                 np.nan, state_vectors['vent_pinsp'] - state_vectors['vent_peep'])
    ## Recalculate state_cumfluids. It was previously calculated from HiRID source tables (dm_total_in-dm_total_out), but these variables
    ## take into account all types of fluids. IntelliLung's state_cumfluids is exclusively iv fluid intake minus urine fluid output
    state_vectors = state_vectors.drop(["state_cumfluids"], axis=1)
    state_vectors["state_cumfluids"] = np.where(state_vectors['state_ivfluid4h'].isna() | state_vectors['state_urin4h'].isna(),
                                                 np.nan, state_vectors["state_ivfluid4h"] - state_vectors["state_urin4h"])
    ## Create new vent variables
    state_vectors = state_vectors.groupby("PatientID").apply(dc.process_for_different_vent_modes).reset_index(drop=True)
    
    # Reindex columns and convert data types
    col_reindex = [col for col in kb.var_order if col in state_vectors.columns]
    state_vectors['daemo_sex'] = state_vectors['daemo_sex'].astype(int)
    state_vectors['daemo_discharge'] = state_vectors['daemo_discharge'].astype(int)
    state_vectors = state_vectors.reindex(columns = col_reindex)
    state_vectors = state_vectors.astype(kb.transform_column_dtypes(state_vectors))

    return state_vectors.sort_values(["PatientID", "mv_id", "time_interval"])


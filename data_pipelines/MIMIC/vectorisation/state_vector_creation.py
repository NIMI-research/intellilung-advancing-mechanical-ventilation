import sys
import copy 

import numpy as np
import pandas as pd
import multiprocessing as mp

from tqdm import tqdm
from typing import Optional

from common.config import settings
import common.utility.utility as util
import common.knowledge_base.kbase as kb


def binary_search(arr, x):
    low = 0
    high = len(arr) - 1
    mid = 0

    while low <= high:
        mid = (high + low) // 2

        # If x is greater, ignore left half
        if arr[mid] < x:
            low = mid + 1

        # If x is smaller, ignore right half
        elif arr[mid] > x:
            high = mid - 1

        # means x is present at mid
        else:
            return mid

    # If we reach here, then the element was not present
    return -1


def create_state_vectors(data: pd.DataFrame, patients_data: pd.DataFrame, multiproc=True):
    state_vectors = []
    state_priorities = []

    if "mv_id" not in data.columns:
        data["mv_id"] = 0

    for stay_id in data["stay_id"].unique() if multiproc else tqdm(data["stay_id"].unique(), desc="Creating state vectors"):
        stay_id_data = data.loc[data["stay_id"] == stay_id, :]

        height = stay_id_data.loc[stay_id_data["label"] == kb.unif_vars.height, "valuenum"].round().mean()

        for mv_id in stay_id_data["mv_id"].unique():
            mv_stay_data = stay_id_data.loc[stay_id_data["mv_id"] == mv_id, :]

            mv_stay_timepoints = mv_stay_data["charttime"].unique()

            # Creating empty state vectors to hold data from the patient's MV
            mat = np.empty(
                shape=((len(mv_stay_timepoints), len(kb.vectors.variables["var_names"]))),
                dtype=kb.reduce_memory_based_on_column("valuenum", dtype_only=True),
            )
            mat.fill(np.nan)
            df = pd.DataFrame(data=mat, columns=kb.vectors.variables["var_names"])

            # Filling state vectors
            start_time = 0
            df["mv_id"] = mv_id
            df["timepoints"] = mv_stay_timepoints

            patient = patients_data.loc[patients_data["stay_id"] == stay_id, :]
            cols_to_update = ["stay_id", "daemo_age", "daemo_sex"]
            patient_cols = ["stay_id", "age", "gender"]
            df[cols_to_update] = patient.iloc[0][patient_cols].values

            # Making a new priority datafrake to store measurement priorities
            df_prio = copy.deepcopy(df)

            var_data_grouped = mv_stay_data.groupby("label")
            for var_key, var_data in var_data_grouped:
                # data_to_add = mv_stay_data.loc[mv_stay_data["label"] == var_key, ["charttime", "valuenum", "priority"]]
                # if not len(data_to_add.index):
                if var_key not in kb.vectors.variables["var_names"] or var_data.empty:
                    # if var_key == kb.unif_vars.gender:
                    #     df[var_key] = patient["gender"].values[0]
                    continue

                data_to_add = util.get_unique_timepoint_data(data=var_data, timepoints=mv_stay_timepoints, time_index="charttime")
                if data_to_add["charttime"].values[0] > start_time:
                    start_time = data_to_add["charttime"].values[0]

                # Inserting data into the new dataframes
                df.loc[df["timepoints"].isin(data_to_add["charttime"].values), var_key] = data_to_add["valuenum"].values
                df_prio.loc[df_prio["timepoints"].isin(data_to_add["charttime"].values), var_key] = data_to_add["priority"].values
            else:
                # Filling missing data for the starting time (time when all vars appeared at least once)
                # ind = util.binary_search(df["timepoints"], start_time)
                ind = np.searchsorted(df["timepoints"], start_time, side="left")
                vars_to_fill = [var for var in set(kb.mv_reqs.unified_vars.keys()).difference([kb.unif_vars.weight]) if np.isnan(df.at[ind, var])]
                
                # Fillings the first valid index with latest value for each variable
                df.loc[ind, vars_to_fill] = [
                    next((val for i in reversed(df.index[:ind]) if not np.isnan(val := df.at[i, var_key])), np.nan) for var_key in vars_to_fill
                ]
                df_prio.loc[ind, vars_to_fill] = [
                    next((val for i in reversed(df_prio.index[:ind]) if not np.isnan(val := df_prio.at[i, var_key])), np.nan) for var_key in vars_to_fill
                ]

                df = util.impute_weight_in_vector(
                    mv_data=df,
                    stayid_weight_data=stay_id_data.loc[stay_id_data["label"] == kb.unif_vars.weight, :],
                    time_index_long="charttime",
                    value_index_long="valuenum",
                    time_index_vec="timepoints",
                    value_index_vec=kb.unif_vars.weight,
                )

                # Cutting off everything before the first valid index for the state vectors
                df = df.loc[ind:, :]
                df_prio = df_prio.loc[ind:, :]

                if not df.empty:
                    df.loc[:, kb.unif_vars.height] = height
                    state_vectors.append(df)
                    state_priorities.append(df_prio)

    if len(state_vectors):
        state_vectors = pd.concat(state_vectors, ignore_index=True)
        state_priorities = pd.concat(state_priorities, ignore_index=True).fillna(100)
    else:
        state_vectors = pd.DataFrame()
        state_priorities = pd.DataFrame()

    return state_vectors, state_priorities


def create_state_vectors_fast(data: pd.DataFrame, patients_data: pd.DataFrame):
    """"""
    results = []
    state_vectors = []
    state_priorities = []

    unique_stay_ids = data["stay_id"].unique()
    tasks = min(100, len(unique_stay_ids))
    split_stay_ids = np.array_split(unique_stay_ids, tasks)

    with mp.Pool(settings.num_of_cores) as pool:
        pbar = tqdm(total=tasks, desc="Creating state vectors [%]", position=0)

        def update(*a):
            pbar.update()

        for i in range(tasks):
            results.append(
                pool.apply_async(
                    create_state_vectors,
                    args=(
                        data.loc[data["stay_id"].isin(split_stay_ids[i]), :],
                        patients_data.loc[patients_data["stay_id"].isin(split_stay_ids[i]), :],
                    ),
                    callback=update,
                )
            )
        for result in results:
            result = result.get()
            state_vectors.append(result[0])
            state_priorities.append(result[1])

        pbar.close()

    state_vectors = pd.concat(state_vectors, ignore_index=True).sort_values(by=["stay_id", "timepoints"]).reset_index(drop=True)
    state_priorities = pd.concat(state_priorities, ignore_index=True).sort_values(by=["stay_id", "timepoints"]).reset_index(drop=True)

    if state_vectors.empty:
        print("State vectors are empty. Exiting...")
        sys.exit()
    
    return state_vectors, state_priorities

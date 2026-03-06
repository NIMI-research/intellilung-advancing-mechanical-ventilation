import os
import re

import numpy as np
import pandas as pd

from tqdm import tqdm

from common.config import settings

import common.knowledge_base.kbase as kb


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def pdone():
    print(f'{bcolors.OKGREEN}Done{bcolors.ENDC}')

# def drop_subset_time_intervals(data: pd.DataFrame):
#     rows_to_drop = np.zeros(len(data.index), dtype=bool)
#     for row_num in data.index:
#         left_side = data["starttime"].between(data.at[row_num,"starttime"],data.at[row_num,"endtime"],inclusive='neither')
#         right_side = data["endtime"].between(data.at[row_num,"starttime"],data.at[row_num,"endtime"],inclusive='neither')
#         rows_to_drop |= left_side & right_side
#     fixed_data = data.drop(rows_to_drop.loc[rows_to_drop.values].index)
#     return fixed_data


def handle_overlapping_time_intervals(data: pd.DataFrame):
    # Initialize a list to store merged intervals
    merged_intervals = []
    
    # Sort the dataframe by start time to simplify the merging process
    data.sort_values(by="starttime", ignore_index=True, inplace=True)
    
    # Initialize variables to track the current merged interval
    current_start = None
    current_end = None
    cnt = 0
    
    # Iterate over each row in the sorted dataframe
    for _, row in data.iterrows():
        if current_start is None:  # If this is the first interval
            current_start = row["starttime"]
            current_end = row["endtime"]
        elif row["starttime"] <= current_end:  # If the current interval overlaps with the current merged interval
            current_end = max(current_end, row["endtime"])
        else:  # If the current interval does not overlap with the current merged interval
            # Append the current merged interval to the list
            merged_intervals.append((data.at[0, "stay_id"], cnt, current_start, current_end))
            # Start a new merged interval with the current row
            current_start = row["starttime"]
            current_end = row["endtime"]
            cnt += 1
    
    # Append the last merged interval to the list
    if current_start is not None:
        merged_intervals.append((data.at[0, "stay_id"], cnt, current_start, current_end))
    
    # Convert the list of merged intervals to a DataFrame
    merged_data = pd.DataFrame(merged_intervals, columns=["stay_id", "mv_id", "starttime", "endtime"])
    
    return merged_data


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


def process_input_to_float(x):
    if pd.isna(x):
        return None
    try:
        x = float(x)
        return x
    except:
        pattern = r"(\d+)(?![^\(]*\))"
        match = re.search(pattern, x)
        return float(match.group(1)) if match else None


def get_unique_timepoint_data(data: pd.DataFrame, timepoints: np.array, time_index: str):
    indices_to_keep = []
    if len(data.index) == len(data[time_index].unique()):
        return data

    for timepoint in data[time_index].unique():
        data_to_check = data.loc[data[time_index] == timepoint, :].copy()

        if len(data_to_check.index) == 1:
            indices_to_keep.append(data_to_check.index.values[0])
        else:
            indices_to_keep.append(data_to_check["priority"].idxmin())

    fixed_data = data.loc[indices_to_keep, :]
    return fixed_data
        
def read_sql_query(query_file: str, connection):
    with open(os.path.join(settings.path_to_sql_queries,query_file)) as f:
        query = f.read()
        with connection.cursor() as cur:
            cur.execute(query)
            data = cur.fetchall()
            cols = []
            for col_desc in cur.description:
                cols.append(col_desc[0])
        df = pd.DataFrame(data=data,columns=cols)
        df = df.astype(kb.reduce_memory_based_on_column(df.columns))
    return df

def get_mv_times(procedures: pd.DataFrame):
    all_mv_times = []
    for stay_id in tqdm(procedures["stay_id"].unique()):
        mv_times = procedures.loc[
            (procedures["stay_id"]==stay_id) & (procedures["itemid"]==kb.mv_reqs.mv_reqs["Invasive Ventilation"]), 
            ["stay_id","starttime","endtime"]
        ].reset_index(drop=True)
        if mv_times.empty: continue
        mv_times = handle_overlapping_time_intervals(data=mv_times)
        mv_times = mv_times.sort_values(by="starttime",ignore_index=True)
        all_mv_times.append(mv_times)
    all_mv_times = pd.concat(all_mv_times,ignore_index=True)
    return all_mv_times

def merge(list1, list2):
    merged_list = [(list1[i], list2[i]) for i in range(0, len(list1))]
    return merged_list

def convert_to_h_m_s(seconds: int):
    import datetime
    return str(datetime.timedelta(seconds=seconds))



def impute_weight_in_vector(
    mv_data: pd.DataFrame,
    stayid_weight_data: pd.DataFrame,
    time_index_long: str = "charttime",
    value_index_long: str = "valuenum",
    time_index_vec: str = "timepoints",
    value_index_vec: str = "daemo_weight",
    start_time: float = 0,
):
    if stayid_weight_data.empty:
        return mv_data

    if np.isnan(mv_data[value_index_vec]).any():
        # Adding datapoints from within MV timeframes
        interp_indices = mv_data.loc[
            (mv_data[time_index_vec] >= stayid_weight_data[time_index_long].values[0])
            & (mv_data[time_index_vec] <= stayid_weight_data[time_index_long].values[-1])
            & (np.isnan(mv_data[value_index_vec]))
        ].index

        if interp_indices.any():
            mv_data.loc[interp_indices, value_index_vec] = np.interp(
                mv_data.loc[interp_indices, time_index_vec].values + start_time,
                stayid_weight_data[time_index_long].values,
                stayid_weight_data[value_index_long].values,
            ).astype(kb.reduce_memory_based_on_column(value_index_long, dtype_only=True))

            # Copying values of the last available datapoint for MV timepoints that came afterwards
            if np.isnan(mv_data.loc[interp_indices[-1] + 1 :, value_index_vec]).any():
                mv_data.loc[interp_indices[-1] + 1 :, value_index_vec] = mv_data.at[interp_indices[-1], value_index_vec]
        # Copying values of the last available datapoint for MV timepoints that came afterwards
        elif mv_data.at[0, time_index_vec] > stayid_weight_data.at[stayid_weight_data.index[-1], value_index_long]:
            mv_data.loc[:, value_index_vec] = stayid_weight_data.at[stayid_weight_data.index[-1], value_index_long]

    return mv_data

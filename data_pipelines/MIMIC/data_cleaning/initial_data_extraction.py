import os
import numpy as np
import pandas as pd
import multiprocessing as mp

from tqdm import tqdm
from collections import Counter

import common.utility as util
import common.knowledge_base.kbase as kb


def encode_string_data(data: pd.DataFrame) -> pd.DataFrame:
    """Converts string data found in the dataframe `value` column to categorical values.
    This will also output .json files with conversion dicts into a `categ` directory
    found in the knowledge_base/dicts directory.
    The .json files will follow the naming convention: variable name + '_categorical.json'"""

    # Getting all indices of values that are strings and their itemids
    data_tbe_index = data.loc[pd.to_numeric(data["value"], errors="coerce").isna()].index
    itemids = data.loc[data_tbe_index, "itemid"].unique()

    # Making a dictionary with unified variable names and the itemids with string values
    category_dict = {}
    for itemid in itemids:
        key = kb.get_uni_var_from_rel_var(kb.get_data_from_itemid(itemid))
        if key not in category_dict:
            category_dict[key] = []
        category_dict[key].append(int(itemid))

    kb.categorical_data_vars = category_dict

    # Creating the `categ` directory and saving the main categorical dict
    if not os.path.isdir(os.path.join(kb.path_to_kb, f"dicts/categ")):
        os.mkdir(os.path.join(kb.path_to_kb, f"dicts/categ"))
    util.save_to_knowledge_base_dicts({"categories": category_dict}, "categ/categorical_vars")

    # Creating additional files to help understand the new values for each unified variable
    for key in category_dict:
        # Checks if mapping already exists for a variable and then applies it
        if key in list(kb.categorical_data_dicts.keys()):
            mapped_data = data.loc[data["itemid"].isin(category_dict[key]), "value"].map(kb.categorical_data_dicts[key])
            data.loc[data["itemid"].isin(category_dict[key]), "value"] = mapped_data

        # If mapping doesn't exist, the function creates its own mapping and saves it as a .json file
        else:
            data.loc[data["itemid"].isin(category_dict[key]), "value"], index = pd.factorize(data.loc[data["itemid"].isin(category_dict[key]), "value"])
            util.save_to_knowledge_base_dicts({key: {i: val for i, val in enumerate(index)}}, f"categ/{key}_categorical")

    data = data.dropna(subset=["value"]).reset_index(drop=True)

    data["value"] = pd.to_numeric(data["value"], errors="raise")
    data.rename(columns={"value": "valuenum"}, inplace=True)

    return data


def process_ecmo_sweep(data: pd.DataFrame) -> pd.DataFrame:
    def extract_float_from_volume(volume_str):
        try:
            # Split the string and extract the numeric part
            numeric_part = volume_str.split(" ")[0]

            # Convert the numeric part to a float
            return float(numeric_part)
        except ValueError:
            # Handle the case where conversion to float fails
            return np.nan

    for itemid in kb.get_unified_itemids("ecmo_sweep"):
        data.loc[data["itemid"] == itemid, "value"] = data.loc[data["itemid"] == itemid, "value"].apply(extract_float_from_volume)

    return data


def process_stay_id_data_fast(data: pd.DataFrame, ventilation: pd.DataFrame):
    all_mv_times = []
    extracted_data = []

    data_vars = set(
        [
            key
            for key in kb.mv_reqs.unified_vars.keys()
            if (len(kb.mv_reqs.unified_vars[key]) != 0)
            & ("ecmo" not in key)
            & ("urin" not in key)
            & ("cumfluids" not in key)
            & ("ivfluid" not in key)
            & ("drugs" not in key)
            & ("daemo" not in key)
            & ("vent_inv" not in key)
        ]
    )

    req_vars = set(
        [
            var
            for var in kb.mv_reqs.unified_vars.keys()
            if (len(kb.get_unified_itemids(var)) != 0) and (kb.get_data_from_itemid(kb.get_unified_itemids(var)[0], "set") == 1)
        ]
    ).difference(set(["daemo_discharge", "vent_invas"]))

    missing_vars = {key: 0 for key in data_vars}

    # for stay_id in tqdm(data["stay_id"].unique(),desc=f"Process {pos}",position=pos,leave=False):
    for stay_id in data["stay_id"].unique():
        mv_times = ventilation.loc[
            (ventilation["stay_id"] == stay_id) & (ventilation["status"].isin(["InvasiveVent", "Tracheostomy"])), 
            ["stay_id", "starttime", "endtime"]
        ].reset_index(drop=True)

        if mv_times.empty:
            continue

        stay_id_data = data.loc[data["stay_id"] == stay_id, :].sort_values(by="charttime")
        mv_times = util.handle_overlapping_time_intervals(data=mv_times)
        mv_times = mv_times.sort_values(by="starttime", ignore_index=True)

        # Compares all the present variables in the patient's data to all the variables
        # Creates a missingness dictionary that is used to shown the extent of variable missingness
        uniq_vars = set(stay_id_data["label"].unique())
        if data_vars.difference(uniq_vars) != set():
            for key in data_vars.difference(uniq_vars):
                missing_vars[key] += 1

        # Skips patients who do not meet the minimum requirement of having all variables within SET 1
        # if req_vars.difference(uniq_vars) != set():
        #     continue

        height_data = stay_id_data.loc[stay_id_data["label"] == kb.unif_vars.height].copy()
        if height_data.empty:
            continue

        weight_data = stay_id_data.loc[stay_id_data["label"] == kb.unif_vars.weight].copy()
        if weight_data.empty:
        #     stay_id_data.loc[stay_id_data["label"] == kb.unif_vars.weight, "valuenum"] = np.median(
        #         stay_id_data.loc[stay_id_data["label"] == kb.unif_vars.weight, "valuenum"].values
        #     )
        # else:
            continue

        # Extracting times of mechanical ventilation from patient stays in the ICU
        extracted_stay_id_data = stay_id_data.copy()
        extracted_stay_id_data["mv_id"] = -1

        for i, ind in enumerate(mv_times.index):
            if (mv_times.at[ind, "endtime"] - mv_times.at[ind, "starttime"]) < 4 * 3600:
                continue

            in_range_times = extracted_stay_id_data["charttime"].between(mv_times.at[ind, "starttime"], mv_times.at[ind, "endtime"])
            if not in_range_times.any():
                continue

            extracted_stay_id_data.loc[in_range_times, "mv_id"] = i

            # Copying height & weight data to the beginning of each MV period
            height_data.loc[:, "charttime"] = extracted_stay_id_data.loc[in_range_times, "charttime"].unique().min()
            height_data["mv_id"] = i

            weight_data.loc[:, "charttime"] = extracted_stay_id_data.loc[in_range_times, "charttime"].unique().min()
            weight_data["mv_id"] = i

            extracted_stay_id_data = pd.concat([extracted_stay_id_data, height_data, weight_data], ignore_index=True)

        extracted_stay_id_data = extracted_stay_id_data.loc[extracted_stay_id_data["mv_id"].ge(0), :]
        if extracted_stay_id_data.empty:
            continue

        # if not weight_data.empty:
        #     weight_data_to_append = wi.impute_weight(weight_data=weight_data, mv_times=mv_times)

        #     if not weight_data_to_append.empty:
        #         weight_data_to_append["mv_id"] = -1
        #         for i, ind in enumerate(mv_times.index):
        #             weight_data_to_append.loc[
        #                 weight_data_to_append["charttime"].between(mv_times.at[ind, "starttime"], mv_times.at[ind, "endtime"]), "mv_id"
        #             ] = i

        #         extracted_stay_id_data = pd.concat([extracted_stay_id_data, weight_data_to_append], ignore_index=True)

        all_mv_times.append(mv_times)

        extracted_data.append(extracted_stay_id_data)

    if len(extracted_data) != 0:
        all_mv_times = pd.concat(all_mv_times, ignore_index=True)
        extracted_data = pd.concat(extracted_data, ignore_index=True)
        return all_mv_times, extracted_data, missing_vars
    else:
        return pd.DataFrame(), pd.DataFrame(), missing_vars


def extract_mv_data_fast(data: pd.DataFrame, ventilation: pd.DataFrame, num_cores: int):
    """"""
    extracted_data = []
    all_mv_times = []
    missing = []
    results = []

    unique_stay_ids = data["stay_id"].unique()
    tasks = 100
    split_stay_ids = np.array_split(unique_stay_ids, tasks)

    with mp.Pool(num_cores) as pool:
        pbar = tqdm(total=tasks, desc="Extracting MV data", position=0)

        def update(*a):
            pbar.update()

        for i in range(tasks):
            results.append(
                pool.apply_async(
                    process_stay_id_data_fast,
                    args=(
                        data.loc[data["stay_id"].isin(split_stay_ids[i]), :],
                        ventilation.loc[ventilation["stay_id"].isin(split_stay_ids[i]), :],
                    ),
                    callback=update,
                )
            )
        for result in results:
            result_tuple = result.get()
            all_mv_times.append(result_tuple[0])
            extracted_data.append(result_tuple[1])
            missing.append(result_tuple[2])

        pbar.close()

    extracted_data = pd.concat(extracted_data).sort_values(by=["stay_id", "charttime"]).reset_index(drop=True)
    all_mv_times = pd.concat(all_mv_times, ignore_index=True)
    missing = dict(sum((Counter(d) for d in missing), Counter()))

    return extracted_data, all_mv_times


def drop_non_cohort_patients(data: pd.DataFrame, demog: pd.DataFrame, all_mv_times: pd.DataFrame):
    icu_stays_non_cohort_ages = demog.loc[demog["age"] < 18, "stay_id"]
    icu_stays_invalid_dod = demog.loc[demog["dod"] < demog["intime"], "stay_id"]

    # Filtering out patients without height or weight data
    icu_stays_with_height = data.loc[data["label"] == kb.unif_vars.height, "stay_id"].unique()
    icu_stays_with_weight = data.loc[data["label"] == kb.unif_vars.weight, "stay_id"].unique()

    # Filtering out patients without common actions required by the AI
    # common_ids = set(data["stay_id"].unique())
    # for var in [kb.unif_vars.rr, kb.unif_vars.fio2, kb.unif_vars.peep]:
    #     common_ids = common_ids & set(data.loc[data["label"] == var, "stay_id"].unique())
    # icu_stays_with_actions = list(common_ids)

    data_drop_indeces = np.array([])

    ###### remove younger than 18 ######
    data_drop_indeces = np.append(data_drop_indeces, data.loc[data["stay_id"].isin(icu_stays_non_cohort_ages)].index)

    ###### remove patients with negative date of death (before ICU intime) ######
    data_drop_indeces = np.append(data_drop_indeces, data.loc[data["stay_id"].isin(icu_stays_invalid_dod)].index)

    ###### remove patients without height ######
    data_drop_indeces = np.append(data_drop_indeces, data.loc[~data["stay_id"].isin(icu_stays_with_height)].index)

    ###### remove patients without weight ######
    data_drop_indeces = np.append(data_drop_indeces, data.loc[~data["stay_id"].isin(icu_stays_with_weight)].index)

    ###### remove patients without ACTION data
    # data_drop_indeces = np.append(data_drop_indeces, data.loc[~data["stay_id"].isin(icu_stays_with_actions)].index)

    if len(data_drop_indeces):
        data.drop(np.unique(data_drop_indeces), inplace=True)
        data.reset_index(inplace=True, drop=True)

    all_mv_times = all_mv_times.loc[all_mv_times["stay_id"].isin(data["stay_id"].unique()), :].reset_index(drop=True)

    return data, all_mv_times

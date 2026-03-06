import numpy as np
import pandas as pd

from tqdm import tqdm
from typing import Optional

import common.knowledge_base.kbase as kb


def standardize_data(data: pd.DataFrame, unit_label: Optional[str] = "valueuom"):
    unit_flag = unit_label in data.columns
    ###### convert height from inch to cm ######
    mask = data["itemid"].isin(kb.get_var_itemids_by_unit(var=kb.unif_vars.height, unit="Inch"))
    data.loc[mask, "valuenum"] *= 2.54
    if unit_flag:
        data.loc[mask, unit_label] = "cm"

    ###### convert Temp from F to C ######
    mask = data["itemid"].isin(kb.get_var_itemids_by_unit(var=kb.unif_vars.temp, unit="°F"))
    data.loc[mask, "valuenum"] = (data.loc[mask, "valuenum"] - 32) * 5 / 9
    if unit_flag:
        data.loc[mask, unit_label] = "°C"

    ###### convert Hb ######
    data.loc[data["itemid"].isin(kb.get_unified_itemids(kb.unif_vars.hb)), "valuenum"] *= 0.6206

    ###### convert weight from lbs to kg ######
    mask = data["itemid"].isin(kb.get_var_itemids_by_unit(var=kb.unif_vars.weight, unit=None))
    data.loc[mask, "valuenum"] = data.loc[mask, "valuenum"] * 0.453592
    if unit_flag:
        data.loc[mask, unit_label] = "kg"

    ##### convert blood_gluco from mg/dL to mmol/L
    mask = data["itemid"].isin(kb.get_var_itemids_by_unit(var=kb.unif_vars.gluco, unit=None))
    data.loc[mask, "valuenum"] = data.loc[mask, "valuenum"] * 0.0555
    if unit_flag:
        data.loc[mask, unit_label] = "mmol/L"

    ##### convert blood_magnes from mg/dL to mmol/L
    mask = data["itemid"].isin(kb.get_var_itemids_by_unit(var=kb.unif_vars.magnes, unit=None))
    data.loc[mask, "valuenum"] = data.loc[mask, "valuenum"] * 0.4114
    if unit_flag:
        data.loc[mask, unit_label] = "mmol/L"

    ##### convert blood_calcium from mg/dL to mmol/L
    mask = data["itemid"].isin(kb.get_var_itemids_by_unit(var=kb.unif_vars.ca, unit=None))
    data.loc[mask, "valuenum"] = data.loc[mask, "valuenum"] * 0.2495
    if unit_flag:
        data.loc[mask, unit_label] = "mmol/L"

    ##### convert state_bun from mg/dL to mmol/L
    mask = data["itemid"].isin(kb.get_var_itemids_by_unit(var=kb.unif_vars.bun, unit=None))
    data.loc[mask, "valuenum"] = data.loc[mask, "valuenum"] * 0.357
    if unit_flag:
        data.loc[mask, unit_label] = "mmol/L"

    ##### convert blood_ast from mg/dL to mmol/L
    mask = data["itemid"].isin(kb.get_var_itemids_by_unit(var=kb.unif_vars.ast, unit=None))
    data.loc[mask, "valuenum"] = data.loc[mask, "valuenum"] * 0.0167
    if unit_flag:
        data.loc[mask, unit_label] = "umol/s/L"

    ##### convert blood_alt from mg/dL to mmol/L
    mask = data["itemid"].isin(kb.get_var_itemids_by_unit(var=kb.unif_vars.alt, unit=None))
    data.loc[mask, "valuenum"] = data.loc[mask, "valuenum"] * 0.0167
    if unit_flag:
        data.loc[mask, unit_label] = "umol/s/L"

    ##### convert blood_crea from mg/dL to umol/L
    mask = data["itemid"].isin(kb.get_var_itemids_by_unit(var=kb.unif_vars.crea, unit=None))
    data.loc[mask, "valuenum"] = data.loc[mask, "valuenum"] * 88.4017
    if unit_flag:
        data.loc[mask, unit_label] = "umol/L"

    ##### convert blood_bili from mg/dL to umol/L
    mask = data["itemid"].isin(kb.get_var_itemids_by_unit(var=kb.unif_vars.bili, unit=None))
    data.loc[mask, "valuenum"] = data.loc[mask, "valuenum"] * 17.1037
    if unit_flag:
        data.loc[mask, unit_label] = "umol/L"

    ##### convert vital_co from L/min to ml/min
    mask = data["itemid"].isin(kb.get_var_itemids_by_unit(var="vital_co", unit="L/min"))
    data.loc[mask, "valuenum"] = data.loc[mask, "valuenum"] * 1000
    if unit_flag:
        data.loc[mask, unit_label] = "ml/min"

    return data


def remove_outliers(data: pd.DataFrame, ranges: dict) -> pd.DataFrame:
    data_drop_indeces = np.array([])
    # Iterate through each key in the `ranges` dict that holds outlier ranges
    for key, (min_normal, max_normal) in ranges.items():
        # Extract a subset of the data for the current label
        temp_data = data.loc[data["label"] == key, :]
        if not temp_data.empty:
            mask = (temp_data["valuenum"] < min_normal) | (temp_data["valuenum"] > max_normal)

            # Check for a special case for the "height" label
            # Selects the entire data for a patient whose height is an outlier
            if key in [kb.unif_vars.height, kb.unif_vars.weight]:
                indices = data.loc[data["stay_id"].isin(temp_data.loc[mask, "stay_id"].unique())].index
            else:
                indices = data.loc[temp_data[mask].index].index

            data_drop_indeces = np.append(data_drop_indeces, indices)

    # Remove data that falls outside of the normal value range
    if len(data_drop_indeces):
        data.drop(np.unique(data_drop_indeces), inplace=True)
        data.reset_index(inplace=True, drop=True)

    return data


def clean_and_standardise_data(data: pd.DataFrame, unit_label: Optional[str] = "valueuom") -> pd.DataFrame:
    ###### Remove zero values ###### PEEP, HILFSDRUCK, BEa
    df = data.loc[
        (data["valuenum"] <= 0) & (~data["label"].isin([kb.unif_vars.peep, kb.unif_vars.press_sup, kb.unif_vars.be] + list(kb.categorical_data_dicts.keys())))
    ]
    data_drop_indeces = df.index

    data.drop(data_drop_indeces, inplace=True)
    data.reset_index(drop=True, inplace=True)

    data = standardize_data(data=data, unit_label=unit_label)
    data = remove_outliers(data=data, ranges=kb.mv_reqs.outlier_ranges)

    valid_stay_ids = data.groupby("stay_id")["label"].apply(lambda x: {"daemo_height", "daemo_height"}.issubset(x.values))
    data = data.loc[data["stay_id"].isin(valid_stay_ids[valid_stay_ids].index)]

    # req_vars = set(
    #     [
    #         var
    #         for var in kb.mv_reqs.unified_vars.keys()
    #         if (len(kb.get_unified_itemids(var)) != 0) and (kb.get_data_from_itemid(kb.get_unified_itemids(var)[0], "set") == 1)
    #     ]
    # ).difference(set(["daemo_discharge", "vent_invas"]))

    # data_drop_indeces = np.array([])
    # for stay_id in tqdm(data["stay_id"].unique(), desc="Cleaning data"):
    #     stay_id_data = data.loc[data["stay_id"] == stay_id, :]
    #     uniq_vars = set(stay_id_data["label"].unique())

        # Skips patients who do not meet the minimum requirement of having all variables within SET 1
        # if req_vars.difference(uniq_vars) != set():
        #     data_drop_indeces = np.append(data_drop_indeces, stay_id_data.index)

    # data.drop(data_drop_indeces, inplace=True)
    data.reset_index(drop=True, inplace=True)

    return data

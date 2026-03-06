import re
import json
import math
from time import time
import numpy as np
import pandas as pd

import dicts.vars as var_dict

with open("eicu/dicts/target_units.json", "r") as f:
    TARGET_UNITS = json.load(f)

with open("eicu/dicts/vaso_target_units.json", "r") as f:
    VASO_TARGET_UNITS = json.load(f)

with open("eicu/dicts/outlier_ranges.json") as f:
    OUTLIER_RANGES = json.load(f)


def cast_clean_int(x):
    """Age column includes value >89 and empty values"""
    try:
        x_int = int(x)
    except ValueError as e:
        x = re.sub(r'[^0-9]', '', x)
        x_int = int(x) if x != '' else 0
    return x_int


def cast_clean_float(x):
    """Remove elements interferring with float conversions
    like %"""
    try:
        x_float = float(x)
    except ValueError as e:
        # if not just chars
        if re.match(r'.*[0-9].*', x):
            x = re.sub(r'[^0-9\.]', '', x)
            x_float = float(x) if x != '' else 0.0
        # if empty
        elif x.strip() == '':
            x_float = np.nan
        else:
            x_float = x
    return x_float


def remove_invalid_values(data):
    """Remove known invalid values from variable"""
    for var in var_dict.invalid_vals:
        mask = (data["variable"] == var) & (
            data["value"].isin(var_dict.invalid_vals[var]))
        data.loc[mask, "value"] = np.nan
    return data


def match_units(varname, unit):
    if varname in TARGET_UNITS:
        return TARGET_UNITS[varname] == unit
    elif not isinstance(unit, str) and math.isnan(unit):
        return True
    else:
        raise KeyError(varname, "has no target unit for", unit)


def match_units_vectorized(varnames, units):
    result = np.zeros_like(varnames, dtype=bool)

    for varname, unit in TARGET_UNITS.items():
        mask = varnames == varname
        result[mask] = result[mask] | (units[mask] != unit)
        # what about rows not in target units but where unit is given

    return result


def convert_units(data):
    """Convert measured units to target units as specified
    Note/TODO: after conversion loinc and unit of converted rows are not uptodate anymore"""
    # nonmatch_mask = data.apply(lambda x: match_units(x.variable, x.units), axis=1)
    nonmatch_mask = match_units_vectorized(
        data['variable'].values, data['units'].values)
    if any(nonmatch_mask):
        data.loc[nonmatch_mask, "value"] = apply_conversion_dict_vectorized(
            data[nonmatch_mask]['loinc'].values, data[nonmatch_mask]['value'].values)
        # data.loc[nonmatch_mask, "value"] = data[nonmatch_mask].apply(
        #         lambda x: apply_conversion_dict(x.loinc, x.value), axis=1)

    # for loinc in unit_conversions:
    #     loinc_mask = data["loinc"] == loinc
    #     b = data[loinc_mask]
    #     if any(loinc_mask):
    #         data.loc[loinc_mask, "value"] = data[loinc_mask].apply(
    #             lambda x: apply_conversion_dict(x.loinc, x.value), axis=1)
    return data


def fahrenheit2celsius(value, args=None):
    return (value - 32) * 5/9


def mg_dl2g_l(value, args=None):
    return value/100


def g_dl2g_l(value, args=None):
    return value*10


def mmol2umol(value, args=None):
    return value*1000


def mg_dl2mmol_l(value, ratio):
    return value * ratio


unit_conversions = {
    # "dyn.s/cm5/m^2 - dyn.s/cm5/m^3"
    # "8834-4": "* 1",

    # "g/dl - g/l"
    "1751-7": (g_dl2g_l, None),

    # "mmol/l - g/l"
    # "54347-0": "* 1",

    # "l/min - ml"
    # "8736-1": "* 1",
    # "8737-9": "* 1",
    # "76519-8": "* 1",

    # "ml - ml/4h"
    # "8975-5": "/ 4",
    # "8985-4": "/ 4",
    # "9187-6": "/ 4",

    # "ml/h - ml/4h"
    # "8976-3": "* 4",
    # "9188-4": "* 4",

    # "g/dl - mmol/l"
    "718-7": (mg_dl2mmol_l, 0.6206),
    # "30350-3": "* 1",
    # "30352-9": "* 1",

    # "mg/dl - mmol/l"
    # "49765-1": "* 1",
    "17861-6": (mg_dl2mmol_l, 0.2495),
    # "59470-5": "* 1",
    # "59471-3": "* 1",
    # "17863-2": "* 1",
    # "38230-9": "* 1",
    "21377-7": (mg_dl2mmol_l, 0.4114),
    # "19123-9": "* 1",
    # "32698-3": "* 1",
    # "75940-7": "* 1",
    # "30313-1": "* 1",
    "6299-2": (mg_dl2mmol_l, 0.357),
    # "3094-0": "* 1",
    # "59032-3": "* 1",
    # "30242-2": "* 1",
    # "30241-4": "* 1",
    # "51829-0": "* 1",
    # "14118-4": "* 1",
    # "41652-9": "* 1",
    "2339-0": (mg_dl2mmol_l, 0.0555),
    # "41651-1": "* 1",
    # "32016-8": "* 1",
    # "41653-7": "* 1",

    # "1/min/ml/kg - ml/min"
    # "90566-1": "* 1",

    # "U/l - umol/l"
    "1920-8": (mg_dl2mmol_l, 0.0167),
    # "48136-6": "* 1",
    "76625-3": (mg_dl2mmol_l, 0.0167),
    # "1742-6": "* 1",
    # "77144-4": "* 1",

    # "mg/dl - umol/l"
    "38483-4": (mg_dl2mmol_l, 88.4017),
    "42719-5": (mg_dl2mmol_l, 17.1037),
    # "59827-6": "* 1",
    # "59828-4": "* 1",
    # "1975-2": "* 1",

    # "mmol/l - umol/l"
    "77140-2": (mmol2umol, None),
    "59826-8": (mmol2umol, None),
    "54363-7": (mmol2umol, None),
    "89872-6": (mmol2umol, None),
    "89871-8": (mmol2umol, None),
    "14631-6": (mmol2umol, None),
    "97770-2": (mmol2umol, None),
    "77137-8": (mmol2umol, None),

    # F - C
    "8310-5": (fahrenheit2celsius, None)
}


def apply_conversion_dict(loinc, value):
    try:
        f, args = unit_conversions[loinc]
        return f(value, args)
    except TypeError:
        return value


def apply_conversion_dict_vectorized(loinc, value):
    result = np.full_like(value, value)

    for loinc_u, (f, args) in unit_conversions.items():
        mask = loinc == loinc_u
        result[mask] = f(value[mask], args)

    return result


def replace_outliers_col(col, data):
    """Replace outliers of specified column with the value of the previous row, 
    or cap to threshold for the first row."""
    lower_threshold = OUTLIER_RANGES[col]["Threshold low"][0]
    upper_threshold = OUTLIER_RANGES[col]["Threshold high"][0]
    
    # Identify outliers
    outlier_mask = (data[col] < lower_threshold) | (data[col] > upper_threshold)
    
    data = data.copy()  # avoid function modifying references outside of scope
    for idx in data[outlier_mask].index:
        if idx == 0:  # First row
            if data.loc[idx, col] < lower_threshold:
                data.loc[idx, col] = lower_threshold
            elif data.loc[idx, col] > upper_threshold:
                data.loc[idx, col] = upper_threshold
        else:  # Use the value from the previous row
            data.loc[idx, col] = data.loc[idx - 1, col]
    
    return data

def remove_outliers_col(values, col):
    """Remove outliers for the specified column.
    thres_col can be set to use thresholds from another key"""
    lower_threshold = OUTLIER_RANGES[col]["Threshold low"][0]
    upper_threshold = OUTLIER_RANGES[col]["Threshold high"][0]
    # Identify outliers
    outlier_mask = (values < lower_threshold) | (values > upper_threshold)
    # Replace outlier values in the column with nan
    # Make an explicit copy to avoid warnings
    values = values.copy()
    values[outlier_mask] = np.nan

    return values

def remove_outliers(data):
    """Drop outlier values based on thresholds."""
    outlier_mask = pd.Series(False, index=data.index)  # Initialize a boolean mask
    data = data.copy()  # avoid function modifying references outside of scope
    for var, thresholds in OUTLIER_RANGES.items():
        variable_mask = data["variable"] == var
        # Apply thresholds only to the relevant rows
        low, high = thresholds["Threshold low"][0], thresholds["Threshold high"][0]
        relevant_values = data.loc[variable_mask, "value"]
        variable_outliers = (relevant_values < low) | (relevant_values > high)
        outlier_mask.loc[variable_mask] |= variable_outliers
    # Filter the DataFrame to remove rows flagged as outliers
    return data.loc[~outlier_mask].reset_index(drop=True)


def apply_dict_encodings(data, subset_var, dictionariy, is_numeric=False):
    subset_mask = data['variable'] == subset_var
    if subset_mask.any():
        if not is_numeric:
            subset_values = data.loc[subset_mask,
                                     'value'].astype(str).str.strip()
        else:
            subset_values = data.loc[subset_mask, 'value']

        replaced_values = subset_values.replace(dictionariy)
        data.loc[subset_mask, "value"] = replaced_values.astype(float)

    return data


def encode_strings(data):
    """Convert string columns to numeric according to dictionary encodings"""
    cat_dictionaries = {"vent_mode": var_dict.vent_mode_groups,
                        "vent_invas": var_dict.vent_invas_groups,
                        "state_airtype": var_dict.state_airtype_groups}
    for categ in cat_dictionaries:
        data = apply_dict_encodings(
            data, categ, cat_dictionaries[categ])
    # vaso should be numeric, but has some strings which indicate 0
    data = apply_dict_encodings(
        data, "drugs_vaso4h", var_dict.vaso_encodings, is_numeric=True)
    return data


def drop_problematic_dev(data, variables):
    """This function is only for dev, to temporarily problematic variables"""
    variable_mask = data["variable"].isin(variables)
    data.drop(data[variable_mask][[isinstance(x, str)
              for x in data[variable_mask].value]].index, inplace=True)
    data = data.reset_index(drop=True)

    return data


def dose2ned(data):
    # Vasopressors target unit 2 ned
    ned_ratios = {
        # dopamine
        "4370-3": 0.01,
        "4363-8": 0.01,
        # vasopressin
        "4369-5": 2.5,
        "4393-5": 2.5,
        # Angiotensin II
        "4373-7": 2.5,
        "4372-9 ": 2.5,
        # Metaraminol
        "4376-0": 8,
        "4375-2": 8,
        # Phenylephrine
        "4379-4": 0.06,
        "4378-6": 0.06,
        # Hydroxocobalamin
        "4380-2": 0.02,
        # TODO 4382-8, 4381-0
        "4385-1": 0.04,
        "4384-4": 0.04,
        "4387-7": 10,
        "4388-5": 10
    }
    data["value"] *= data["loinc"].map(ned_ratios).fillna(1)
    return data


def convert2min(value, cur_units):
    if "min" not in cur_units:
        if "h" in cur_units:
            value /= 60
    return value


def convert2mcg(value, cur_units):
    units2mcg_dict = {
        "mg": 1000,
        "nanograms": 0.001
    }

    if "mcg" not in cur_units:
        if "mg" in cur_units:
            value *= units2mcg_dict.get("mg", 1)
        elif "nanograms" in cur_units:
            value *= units2mcg_dict.get("nanograms", 1)
    return value


def convert2kgnorm(value, cur_units, weight):
    if "kg" not in cur_units:
        value /= weight
    return value


def convert2kgdenorm(value, cur_units, weight):
    if "kg" in cur_units:
        value *= weight
    return value


def convert2umin(row):
    # unique units to be handled
    # 'units/h', 'units/kg/min', 'units/kg/h', 'units/min'
    value = row['value']
    units = row['units']
    weight = row['daemo_weight']
    cur_units = units.split("/")

    value = convert2min(value, cur_units)
    if not pd.isnull(weight):
        value = convert2kgdenorm(value, cur_units, weight)
    return value


def convert2mcgkgmin(row):
    # unique units to be handled
    # 'mcg/min', 'mcg/kg/min', 'mcg/kg/h',
    # 'mg/min', 'mg/h', 'mcg/h', 'mg/kg/min', 'ml', 'nanograms/kg/min',
    value = row['value']
    units = row['units']
    weight = row['daemo_weight']
    cur_units = units.split("/")

    value = convert2min(value, cur_units)
    value = convert2mcg(value, cur_units)
    if not pd.isnull(weight):
        value = convert2kgnorm(value, cur_units, weight)

    return value


def standardize_vaso(data, demog):
    """Convert vasopressor values to target values for NED
    Note/TODO: after conversion loinc and unit of converted rows are not uptodate anymore"""
    result_df = demog[demog["variable"] == "daemo_weight"].groupby(
        ["patientunitstayid"])["value"].min().reset_index()
    result_df.rename(columns={"value": "daemo_weight"}, inplace=True)
    data = pd.merge(data, result_df, on="patientunitstayid", how="left")

    for key, loincs_to_norm in VASO_TARGET_UNITS.items():
        norm_mask = (data["loinc"].isin(loincs_to_norm)) & (
            data["units"] != key)
        if norm_mask.any():
            if key == "mcg/kg/min":
                data.loc[norm_mask, "value"] = data[norm_mask].apply(
                    convert2mcgkgmin, axis=1)
            elif key == "U/min":
                data.loc[norm_mask, "value"] = data[norm_mask].apply(
                    convert2umin, axis=1)
    data.drop(columns=["daemo_weight"], inplace=True)
    data = dose2ned(data)
    return data


def clean_data(df, startTime):
    # convert str columns to numeric
    print("Encoding strings")
    df = encode_strings(df)
    print("at ", str(time() - startTime), "seconds")
    print("Cleaning data...")
    df = remove_invalid_values(df)
    print("at ", str(time() - startTime), "seconds")
    print("...")
    df["value"] = df["value"].apply(cast_clean_float)
    print("at ", str(time() - startTime), "seconds")
    print("Converting units...")
    df = convert_units(df)  # Converting the data in desired units
    print("at ", str(time() - startTime), "seconds")
    print("Removing outliers...")
    df = remove_outliers(df)
    print("at ", str(time() - startTime), "seconds")

    return df

def filter_below_duration(ventevents, min_duration):
    """Filter rows below minimum duration"""
    duration_in_days = min_duration/24
    return ventevents[ventevents["mv_duration"] >= duration_in_days].reset_index(drop=True)

def minutes2days(value):
    return value/60/24

def merge_overlapping_intervals(df):
    """
    Merges overlapping intervals of vent_start and vent_end for each stay_id.

    Parameters:
        df (pd.DataFrame): DataFrame with columns 'stay_id', 'vent_start', 'vent_end'.

    Returns:
        pd.DataFrame: DataFrame with merged intervals for each stay_id.
    """
    merged_results = []

    # Group by stay_id
    for stay_id, group in df.groupby("stay_id"):
        # Sort by vent_start
        group = group.sort_values(by="vent_start").reset_index(drop=True)
        
        # Initialize the first interval
        current_start = group.loc[0, "vent_start"]
        current_end = group.loc[0, "vent_end"]
        mv_duration = group.loc[0, "hours_of_vent"]/24  # duration in days
        
        # Iterate through intervals
        for i in range(1, len(group)):
            next_start = group.loc[i, "vent_start"]
            next_end = group.loc[i, "vent_end"]
            
            # Check for overlap
            if next_start <= current_end:  # Overlap exists
                current_end = max(current_end, next_end)  # Extend the interval
            else:  # No overlap, save the current interval and start a new one
                mv_duration = minutes2days(current_end - current_start)
                merged_results.append({"stay_id": stay_id, "mv_duration": mv_duration,
                                       "vent_start": current_start, "vent_end": current_end})
                current_start = next_start
                current_end = next_end
        
        # Add the last interval for the group
        mv_duration = minutes2days(current_end - current_start)
        merged_results.append({"stay_id": stay_id, "mv_duration": mv_duration,
                               "vent_start": current_start, "vent_end": current_end})
    
    # Create a DataFrame from the results
    merged_df = pd.DataFrame(merged_results)
    return merged_df

def preprocess_ventevents(ventevents, min_duration):
    """Merge overlapping intervals
    Filter episodes for minimum duration
    """
    ventevents = merge_overlapping_intervals(ventevents)
    ventevents = filter_below_duration(ventevents, min_duration)
    return ventevents

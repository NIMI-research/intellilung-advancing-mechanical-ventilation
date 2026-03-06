import warnings

import numpy as np
import pandas as pd
import multiprocessing as mp
from tqdm import tqdm

import common.utility.utility as util
import common.knowledge_base.kbase as kb
from common.config import settings


warnings.filterwarnings(action="ignore", message="All-NaN slice encountered")


def find_missing_timepoints(timepoints: pd.Series, resolution: int):
    missingness_dict = {}
    for i in timepoints.index[:-1]:
        if timepoints[i + 1] - timepoints[i] > resolution:
            missingness_dict.update({i: np.floor((timepoints[i + 1] - timepoints[i]) / resolution)})
    return missingness_dict


def propagate_measurements(data: pd.DataFrame):
    unified_vars = list(set(kb.mv_reqs.unified_vars.keys()))
    data.loc[:, unified_vars] = data.loc[:, unified_vars].fillna(method="ffill")

    return data


def calculate_ned_scale_and_conversion(row):
    unit_value = kb.conversion_dict[row["valueuom"]]

    real_key = next(key for key in kb.target_conversions.keys() if key in kb.get_data_from_itemid(row["itemid"]))
    drug_dict = kb.target_conversions[real_key]
    target_unit_value = kb.conversion_dict[drug_dict["unit"]]
    scale = drug_dict["scale"]

    return unit_value / target_unit_value * scale

def is_outside_range(number, range_start, range_end):
    return not (range_start <= number <= range_end)

def set_first_three_values_to_zero(df, patient_column, columns_to_zero):
    """
    Sets the first 3 values of specified columns to 0 for each unique patient.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        patient_column (str): The column name that identifies individual patients.
        columns_to_zero (list): List of columns where the first 3 values should be set to 0 for each patient.

    Returns:
        pd.DataFrame: DataFrame with updated values.
    """
    # Apply the operation to each group (patient)
    df = df.copy()
    for _, group in df.groupby(patient_column):
        indices_to_zero = group.index[:3]  # First 3 indices for each patient
        df.loc[indices_to_zero, columns_to_zero] = 0
    return df


# Calculates the DRUGS_VASO4H variable
def calculate_drug_inputs(df_m, patient_drug_inputs, tmpts, i):
    for j, tmpt in enumerate(tmpts):
        # extracts all injected drugs over the period of 4h before the currently selected timepoint
        tmp = patient_drug_inputs.loc[
            patient_drug_inputs["charttime"].lt(tmpt)
            & patient_drug_inputs["endtime"].gt(tmpt - 4 * 3600),
            :,
        ]
        # if no drugs were injected set to 0 and continue
        if tmp.empty:
            df_m.at[j, "drugs_vaso4h"] = 0
            continue

        # Rates have not yet been standardised here. A special type of conversion to NED is done below
        rates = tmp["value"] * 60 / df_m.at[j, "daemo_weight"]  # Mass / kg / Time [min]

        neds = rates * tmp.apply(calculate_ned_scale_and_conversion, axis=1)

        # Checks for any drugs that were injected since the start of the 4h period
        calc_times = tmp.loc[tmp["charttime"] >= tmpt - 4 * 3600, "charttime"].values

        # Calculates MAXIMUM dose of Vasopressors for given timepoints (no new drugs were injected)
        if len(calc_times) == 0:
            # If there are no starting timepoints present within the start-to-end interval, then the max vasopressor dose is
            # administered at the start (and stays the same throughout if there are no end times)
            t = np.sum(neds)
        else:
            # Calculates the sum of vasopressor NE doses and picks the maximum
            ned_sums = np.array([
                np.sum(neds[(tmp["charttime"] <= calc_time) & (tmp["endtime"] >= calc_time)]) 
                for calc_time in np.append(calc_times, tmpt)
            ])
            t = np.max(ned_sums)

        # Checking if the max vasopressors are outside the normal range
        if is_outside_range(t, *kb.mv_reqs.outlier_ranges["drugs_vaso4h"]):
            # If this is the first timestep, then a value must be imputed, otherwise NaN values can be placed here
            # If the NaN value does remain after windowing, then the carry forward propagation will eliminate it
            if i == 0:
                df_m.at[j, "drugs_vaso4h"] = np.clip(t, *kb.mv_reqs.outlier_ranges["drugs_vaso4h"])
            else:
                df_m.at[j, "drugs_vaso4h"] = np.nan
        else:
            df_m.at[j, "drugs_vaso4h"] = t

    return df_m

# Calculates the STATE_IVFLUID4H variable
def calculate_fluid_inputs(df_m, patient_fluid_inputs, tmpts, i):
    for j, tmpt in enumerate(tmpts):
        tmp = patient_fluid_inputs.loc[
            patient_fluid_inputs["charttime"].lt(tmpt)
            & patient_fluid_inputs["endtime"].gt(tmpt - 4 * 3600),
            :,
        ]
        if tmp.empty:
            df_m.at[j, "state_ivfluid4h"] = 0
            continue

        volumes = tmp["value"] * (np.minimum(tmpt, tmp["endtime"]) - np.maximum(tmpt - 4 * 3600, tmp["charttime"]))

        # Temporarily storing the total volume of iv fluids over 4h
        t = np.sum(volumes)

        # Checking if the total volume is outside the normal range
        if is_outside_range(t, *kb.mv_reqs.outlier_ranges["state_ivfluid4h"]):
            # If this is the first timestep, then a value must be imputed, otherwise NaN values can be placed here
            # If the NaN value does remain after windowing, then the carry forward propagation will eliminate it
            if i == 0:
                df_m.at[j, "state_ivfluid4h"] = np.clip(t, *kb.mv_reqs.outlier_ranges["state_ivfluid4h"])
            else:
                df_m.at[j, "state_ivfluid4h"] = np.nan
        else:
            df_m.at[j, "state_ivfluid4h"] = t

    return df_m

# Calculates the STATE_URIN4H variable
def calculate_urin_4h(df_m, patient_outputs, tmpts, i):
    for j, tmpt in enumerate(tmpts):
        tmp = patient_outputs.loc[
            patient_outputs["charttime"].between(tmpt - 4 * 3600, tmpt, inclusive="both"),
            :,
        ]
        if tmp.empty:
            df_m.at[j, "state_urin4h"] = 0
            continue

        # Temporarily storing the total volume of urin over 4h
        t = np.sum(tmp["value"].values)

        # Checking if the total volume is outside the normal range
        if is_outside_range(t, *kb.mv_reqs.outlier_ranges["state_urin4h"]):
            # If this is the first timestep, then a value must be imputed, otherwise NaN values can be placed here
            # If the NaN value does remain after windowing, then the carry forward propagation will eliminate it
            if i == 0:
                df_m.at[j, "state_urin4h"] = np.clip(t, *kb.mv_reqs.outlier_ranges["state_urin4h"])
            else:
                df_m.at[j, "state_urin4h"] = np.nan
        else:
            df_m.at[j, "state_urin4h"] = t

    return df_m


def create_time_windows(data: pd.DataFrame, priors: pd.DataFrame, inputevents: pd.DataFrame, outputevents: pd.DataFrame, resolution: int, multiproc: bool = False):
    full_data = []

    # Preparidng urin data
    urin_output = outputevents.loc[
        outputevents["itemid"].isin(kb.get_unified_itemids("state_urin4h")), 
        ["stay_id", "charttime", "value", "valueuom"]
    ]
    if not urin_output["valueuom"].eq("ml").all():
        # Standardising urin output
        urin_output["value"] *= urin_output["valueuom"].apply(lambda x: kb.conversion_dict[x] / kb.conversion_dict["ml"])
        urin_output["valueuom"] = "ml"

    # Preparidng drug data
    drug_inputs = inputevents.loc[
        inputevents["itemid"].isin(kb.get_unified_itemids("drugs_vaso4h")), 
        ["stay_id", "itemid", "charttime", "endtime", "value", "valueuom"]
    ]
    drug_inputs["value"] /= (drug_inputs["endtime"] - drug_inputs["charttime"])

    # Preparidng injected fluids data
    fluid_inputs = inputevents.loc[
        inputevents["itemid"].isin(kb.get_unified_itemids("state_ivfluid4h")), 
        ["stay_id", "charttime", "endtime", "value", "valueuom"]
    ]
    fluid_inputs["value"] /= (fluid_inputs["endtime"] - fluid_inputs["charttime"])          # Liquid volume / Time [s]
    if not fluid_inputs["valueuom"].eq("ml").all():
        # Dropping any rows from fluid_inputs where "valueuom" isn't in kb.conversion_dict
        fluid_inputs = fluid_inputs[fluid_inputs["valueuom"].isin(kb.conversion_dict)]

        # Standardising rates
        fluid_inputs["value"] *= fluid_inputs["valueuom"].apply(lambda x: kb.conversion_dict[x] / kb.conversion_dict["ml"])
        fluid_inputs["valueuom"] = "ml"

    # Group datasets by 'stay_id' upfront
    grouped_data = data.groupby("stay_id")
    grouped_priors = priors.groupby("stay_id")
    grouped_drug_inputs = drug_inputs.groupby("stay_id")
    grouped_fluid_inputs = fluid_inputs.groupby("stay_id")
    grouped_urin_output = urin_output.groupby("stay_id")

    # Cycling through patients
    for stay_id, stay_id_data in tqdm(grouped_data, desc="Creating time windows") if not multiproc else grouped_data:
        # Extracting grouped data for current STAY ID
        stay_id_data_prio = grouped_priors.get_group(stay_id)
        patient_drug_inputs = (
            grouped_drug_inputs.get_group(stay_id)
            if stay_id in grouped_drug_inputs.groups
            else pd.DataFrame(columns=["stay_id", "itemid", "charttime", "endtime", "value", "valueuom"])
        )
        patient_fluid_inputs = (
            grouped_fluid_inputs.get_group(stay_id)
            if stay_id in grouped_fluid_inputs.groups
            else pd.DataFrame(columns=["stay_id", "charttime", "endtime", "value", "valueuom"])
        )
        patient_outputs = (
            grouped_urin_output.get_group(stay_id)
            if stay_id in grouped_urin_output.groups
            else pd.DataFrame(columns=["stay_id", "charttime", "value", "valueuom"])
        )

        # Nested grouping by 'mv_id'
        mv_groups = stay_id_data.groupby("mv_id")
        prio_mv_groups = stay_id_data_prio.groupby("mv_id")

        # Cycling through a patient's multiple MV stays if they exist
        mv_id_true = 0
        stay_start_time = stay_id_data["timepoints"].values[0]
        for mv_id, mv_data in mv_groups:
            # Extracting specific MV data
            mv_prio = prio_mv_groups.get_group(mv_id)

            start_time = mv_data["timepoints"].iloc[0]
            end_time = mv_data["timepoints"].iloc[-1]
            if start_time == end_time:
                continue

            times = (1 + np.arange(np.ceil((end_time - start_time + 1) / resolution))) * resolution

            mat = np.zeros(shape=(len(times), len(kb.mv_reqs.unified_vars.keys())))
            mat.fill(np.nan)
            df = pd.DataFrame(data=mat, columns=kb.mv_reqs.unified_vars.keys())

            df.loc[:, "stay_id"] = stay_id
            df.loc[:, "timepoints"] = times
            df.loc[:, "daemo_age"] = mv_data.at[mv_data.index[0], "daemo_age"]

            N = len(times)
            for i, time in enumerate(times):
                # Extracting datapoints for the 1 hour interval
                df_m = mv_data.loc[
                    mv_data["timepoints"].between(
                        left=start_time + time - resolution, 
                        right=start_time + time,
                        inclusive="left"
                        ), 
                        ["timepoints"] + list(kb.mv_reqs.unified_vars.keys()
                    )
                ].reset_index(drop=True)

                df_m_prio = mv_prio.loc[
                    mv_prio["timepoints"].between(
                        left=start_time + time - resolution, 
                        right=start_time + time,
                        inclusive="left"
                        ), 
                        list(kb.mv_reqs.unified_vars.keys()
                    )
                ].reset_index(drop=True)

                # If df_m is empty, that timepoint in the final time windows dataframe will be NaN and the previous value
                # will be carried over.
                if df_m.empty:
                    # Treats the current value for categoricals as the initial value for the following time window if there is
                    # a following time window
                    if i < N - 1:
                        df.loc[i + 1, list(kb.categorical_data_vars.keys())] = df.loc[i, list(kb.categorical_data_vars.keys())]

                    # Prevents gaps in final dataframe due to no data in the current time window
                    df.loc[i, kb.mv_reqs.unified_vars.keys()] = df.loc[i - 1, kb.mv_reqs.unified_vars.keys()]
                    continue

                # The value chosen is the one that lasted the longest during the 1h resolution time
                for cat in kb.categorical_data_vars.keys():
                    if (i > 0) and np.isnan(df_m.at[0, cat]):
                        df_m.at[0, cat] = df.at[i, cat]

                    df_m.loc[:, cat].ffill(inplace=True)

                    # Memorising the change for the following iteration
                    if i < N - 1:
                        df.at[i + 1, cat] = df_m[cat].iloc[-1]

                    # Selecting the category which lasted the longest during the time window
                    if len(df_m.loc[:, cat].unique()) != 1:
                        tmp = df_m.index[df_m[cat].diff() != 0].tolist()
                        time_tmp = df_m.loc[tmp, "timepoints"].values
                        time_tmp[0] = start_time + time - resolution
                        time_list = np.append(time_tmp, start_time + time)

                        # Changing all values to the same category because a median will be calculated
                        df_m.loc[:, cat] = df_m.at[tmp[np.argmax(time_list[1:] - time_list[:-1])], cat]

                tmpts = df_m["timepoints"].values                
                calculate_drug_inputs(df_m, patient_drug_inputs, tmpts, i)
                calculate_fluid_inputs(df_m, patient_fluid_inputs, tmpts, i)
                calculate_urin_4h(df_m, patient_outputs, tmpts, i)

                df_m.drop("timepoints", axis=1, inplace=True)

                # Excluding measurements that are not of highest priority within the 1 hour interval
                min_values = df_m_prio.min()
                mask = df_m_prio.where(df_m_prio.eq(min_values), np.nan)
                mask = mask.where(mask.notna(), np.nan)
                mask[mask.notna()] = 1

                assert list(df_m.columns) == list(mask.columns), "Columns of data and priors are not the same or not in the same order"
                assert len(df_m.index) == len(mask.index), "Interval data and the masking don't have the same amount of datapoints"

                df.loc[i, kb.mv_reqs.unified_vars.keys()] = np.nanmedian(df_m * mask, axis=0)
            if len(df) != 0:
                # Makes timepoints absolute by adding the time difference between the beginning of the ICU stay and the beginning of the
                # current MV treament
                if mv_id_true != 0:
                    df["timepoints"] += start_time - stay_start_time

                df.loc[:, "mv_id"] = mv_id_true
                mv_id_true += 1

                propagate_measurements(data=df)
                full_data.append(df)

    if len(full_data) != 0:
        full_data = (
            pd.concat(full_data).sort_values(by=["stay_id", "mv_id", "timepoints"], ignore_index=True).reindex(kb.vectors.variables["var_names"], axis=1)
        )
        full_data = set_first_three_values_to_zero(full_data, patient_column="stay_id", columns_to_zero=["state_ivfluid4h", "state_urin4h"])

        return full_data.astype(kb.reduce_memory_based_on_column(full_data.columns))
    else:
        return pd.DataFrame()


def create_time_windows_fast(data: pd.DataFrame, priors: pd.DataFrame, inputevents: pd.DataFrame, outputevents: pd.DataFrame):
    """"""
    results = []
    state_vectors = []

    unique_stay_ids = data["stay_id"].unique()
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
                        data.loc[data["stay_id"].isin(split_stay_ids[i]), :],
                        priors.loc[priors["stay_id"].isin(split_stay_ids[i]), :],
                        inputevents.loc[inputevents["stay_id"].isin(split_stay_ids[i]), :],
                        outputevents.loc[outputevents["stay_id"].isin(split_stay_ids[i]), :],
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

    state_vectors = pd.concat(state_vectors, ignore_index=True).sort_values(by=["stay_id", "mv_id", "timepoints"]).reset_index(drop=True)
    return state_vectors

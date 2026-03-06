import numpy as np
import pandas as pd

from tqdm import tqdm

import common.utility.utility as util
import common.knowledge_base.kbase as kb


def compute_sys_and_dia_to_mean(interval_data: pd.DataFrame, sys_label: str, dia_label: str, mean_label: str, label_categories):
    sys_lc_label = sys_label.lower().replace(" ", "_") + "_value"
    dia_lc_label = dia_label.lower().replace(" ", "_") + "_value"
    mean_lc_label = mean_label.lower().replace(" ", "_") + "_value"

    merge_df = pd.merge(
        left=interval_data.loc[interval_data["label"] == sys_label, ["charttime", "valuenum"]].rename(columns={"valuenum": sys_lc_label}),
        right=interval_data.loc[interval_data["label"] == dia_label, ["charttime", "valuenum"]].rename(columns={"valuenum": dia_lc_label}),
        on="charttime",
        how="inner",
    )

    mean_timepoints = interval_data.loc[interval_data["label"] == mean_label, "charttime"].unique()
    merge_df.set_index("charttime", drop=True, inplace=True)
    indices_to_drop = set(merge_df.index).intersection(set(mean_timepoints))
    merge_df.drop(indices_to_drop, inplace=True)
    if merge_df.empty:
        return merge_df

    merge_df[mean_lc_label] = 1 / 3 * merge_df[sys_lc_label] + 2 / 3 * merge_df[dia_lc_label]

    # Dropping newly calculated values which fall out of the normal range
    min_outlier, max_outlier = kb.mv_reqs.outlier_ranges[mean_label]
    mask = (merge_df[mean_lc_label] <= min_outlier) | (merge_df[mean_lc_label] >= max_outlier)
    merge_df.drop(np.unique(merge_df[mask].index), inplace=True)

    df = pd.DataFrame(np.zeros((len(merge_df.index), len(interval_data.columns))), columns=interval_data.columns)

    df["stay_id"] = interval_data.iat[0, interval_data.columns.get_loc("stay_id")]
    df["itemid"] = kb.get_var_itemid(kb.mv_reqs.unified_vars[mean_label][0])
    df["charttime"] = merge_df.index.values
    df["valuenum"] = merge_df[mean_lc_label].values
    df["label"] = pd.Categorical([mean_label] * len(df.index), categories=label_categories)

    return df


def compute_missing_values(data: pd.DataFrame):
    """Computes the missing ART and PAP mean values based on their sys and dia values."""

    _map = kb.unif_vars.art_press_mean
    _sap = kb.unif_vars.art_press_sys
    _dap = kb.unif_vars.art_press_dia

    for stay_id in tqdm(data["stay_id"].unique(), desc="Computing missing values"):
        stay_id_data = data.loc[data["stay_id"] == stay_id, :]

        for i in stay_id_data["mv_id"].unique():
            interval_data = stay_id_data.loc[stay_id_data["mv_id"] == i, :]

            # Calculating missing mean-arterial-pressure values
            bools = np.logical_and(
                interval_data[_map].isna(),
                interval_data[_sap].notna() & interval_data[_dap].notna(),
            )

            if bools.any():
                interval_data.loc[bools, _map] = 1 / 3 * interval_data.loc[bools, _sap].copy() + 2 / 3 * interval_data.loc[bools, _dap].copy()

                # Checking for created outliers and dropping them
                min_outlier, max_outlier = kb.mv_reqs.outlier_ranges[_map]
                mask = (interval_data[_map] <= min_outlier) | (interval_data[_map] >= max_outlier)

                # TODO decide how to handle outlier data:
                #   1) delete entire row
                #   2) set only the values in question ['sys_press_art','dia_press_art','mean_press_art'] to NaN
                # data.drop(np.unique(interval_data[mask].index), inplace=True)
                data.loc[interval_data[mask].index,['sys_press_art','dia_press_art','mean_press_art']] = np.nan

            # interval_data = compute_sys_and_dia_to_mean(interval_data=interval_data,sys_label='sys_press_art',dia_label='dia_press_art',mean_label='mean_press_art',label_categories=data["label"].cat.categories)

            # df = compute_sys_and_dia_to_mean(interval_data=interval_data,sys_label="PAP sys",dia_label="PAP dia",mean_label='mean_press_pul_art',label_categories=data["label"].cat.categories)
            # if not df.empty: data_to_append.append(df)

    return data.reset_index(drop=True)


# Group by "stay_id" and apply the rules
def process_for_different_vent_modes(group: pd.DataFrame):
    # Add previous timepoint values for reference
    group["vent_vt_prev"] = group[kb.unif_vars.tv_norm].shift(1)
    group["vent_pinsp_prev"] = group["vent_pinsp"].shift(1)

    # Rule for vent_mode = 2
    group.loc[group["vent_mode"] == 2, "vent_vt_obs"] = group["vent_vt_prev"].bfill()       # bfill(): imputes first value in group which is NaN
    group.loc[group["vent_mode"] == 2, "vent_vt_action"] = group[kb.unif_vars.tv_norm]
    # vent_pinsp remains the same

    # Rule for vent_mode = 3
    group.loc[group["vent_mode"] == 3, "vent_pinsp"] = group["vent_pinsp_prev"].bfill()
    group.loc[group["vent_mode"] == 3, "vent_vt_obs"] = group[kb.unif_vars.tv_norm]
    # The value of vent_vt_action won't be used
    group.loc[group["vent_mode"] == 3, "vent_vt_action"] = group[kb.unif_vars.tv_norm]

    # Drop auxiliary columns if not needed
    group.drop(columns=["vent_vt_prev", "vent_pinsp_prev"], inplace=True)

    return group


def TV_normalization(data: pd.DataFrame):
    for stay_id in data["stay_id"].unique():
        stay_id_data = data.loc[data["stay_id"] == stay_id, :]

        gender = stay_id_data[kb.unif_vars.gender].values[0]
        height = stay_id_data[kb.unif_vars.height].values[0]
        # Clipping ranges were added to Ideal Body Weight (IBW) calculation
        if gender == 0:  # male
            IBW = np.clip(50.0 + 0.91 * (height - 152.4), 20, 80)
        else:  # female = 1
            IBW = np.clip(45.5 + 0.91 * (height - 152.4), 20, 80)
        data.loc[data["stay_id"] == stay_id, kb.unif_vars.tv_norm] = np.clip(stay_id_data[kb.unif_vars.tv] / IBW, 0.8573020703844999, 42.618977980194714)
    return data

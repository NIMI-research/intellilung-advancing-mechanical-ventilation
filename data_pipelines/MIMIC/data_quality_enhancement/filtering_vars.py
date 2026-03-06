import pandas as pd
import numpy as np

from tqdm import tqdm
from typing import Union

mostly_available_columns = [
    "blood_sao2",
    "blood_ast",
    "blood_alt",
    "vital_SBP",
    "vital_DBP",
    "blood_caion",
    "blood_lac",
    "blood_be",
    "blood_paco2",
    "blood_pao2",
    "vent_suppress",
    "blood_PTT",
    "blood_INR",
    "daemo_weight",
    "blood_ph",
    "state_temp",
    "blood_magnes",
    "blood_wbc",
    "vital_spo2",
    "blood_hco3",
    "blood_plat",
    "blood_hb",
    "blood_chlorid",
    "blood_hct",
    "blood_sodium",
    "vital_map",
    "vent_pinsp",
    "blood_potas",
    "vent_mairpress",
    "vent_rrspont",
    "vent_rrtot",
    "daemo_height",
    "vital_hr",
    "state_urin4h",
    "daemo_sex",
    "state_cumfluids",
    "drugs_vaso4h",
    "state_ivfluid4h",
    "daemo_age",
    "vent_vt_obs",
    "vent_peep",
    "vent_fio2",
    "vent_inspexp",
    "vent_vt_action",
    "vent_mode",
    "vent_pinsp-peep",
    "timepoints",
    "daemo_discharge"
]


def filter_variables(df_mimic_iv: pd.DataFrame, ai_ready: bool = False):
    # Check if MV IDs are in ascending order without skips for each patient
    # assert df_mimic_iv.groupby('stay_id')['mv_id'].apply(
    #     lambda x: list(range(0, len(np.unique(x)))) == sorted(np.unique(x))
    # ).all(), "Some patients have missing MV periods!"

    # Filtering out columns that are not in the list
    if ai_ready:
        df = df_mimic_iv[df_mimic_iv[mostly_available_columns].isna().sum(axis=1) == 0]
    else:
        df = df_mimic_iv[["stay_id", "mv_id", "episode_id"] + mostly_available_columns]

    # Adjust MV IDs to correct any skips created by deletion
    episode_ids = df["episode_id"].unique()
    df = df_mimic_iv[df_mimic_iv["episode_id"].isin(episode_ids)]
    df = df.groupby("episode_id").filter(lambda x: len(x) > 1)

    for _, group in tqdm(df.groupby('stay_id'), desc="Adjusting MV IDs"):
        expected_ids = list(range(0, len(group["mv_id"].unique())))
        if not group['mv_id'].unique().tolist() == expected_ids:
            new_ids = {old_id: new_id for old_id, new_id in zip(group['mv_id'].unique(), expected_ids)}
            df.loc[df['stay_id'] == group['stay_id'].iloc[0], 'mv_id'] = df.loc[df['stay_id'] == group['stay_id'].iloc[0], 'mv_id'].map(new_ids)


    return df


def add_duration_and_pause_info(data: pd.DataFrame, demo: pd.DataFrame, mv_times: pd.DataFrame) -> Union[pd.DataFrame, list]:
    days_to_seconds_scale = 24 * 60 * 60

    # Creating new dataframe to hold important info
    # stay_id is made into the index so it needs to be dropped from regular columns

    # Incorporate time of death into pause_until_next
    mv_times_new = mv_times.merge(
        demo[['stay_id', 'dod']], on='stay_id', how='left'
    )

    # ***********************************************************
    # TODO: ADD CHECK AND DELETE ANY EPISODES STARTING POST DEATH
    # ***********************************************************
    # Make sure end of MV is tied to time of death (there were cases where dod < endtime)
    mv_times_new["endtime"] = mv_times_new.apply(
        lambda row: min(row["endtime"], row["dod"]) if pd.notna(row["dod"]) else row["endtime"],
        axis=1
    )
    # Calculate duration of MV and pause until the following episode
    mv_times_new = mv_times_new.groupby("stay_id").apply(
        lambda gdf: gdf.assign(
            mv_duration=lambda x: (x["endtime"] - x["starttime"]) / days_to_seconds_scale,
            pause_until_next=lambda x: (x["starttime"].shift(-1) - x["endtime"]) / days_to_seconds_scale
        )
    ).reset_index(drop=True)

    # # Adjust pause_until_next for last episode based on death_time
    # mv_times_new['pause_until_next'] = mv_times_new.apply(
    #     lambda row: row['pause_until_next']
    #     if pd.notna(row['pause_until_next'])                                    # keep calculated pause if it exists
    #     else min((row['dod'] - row['endtime']) / days_to_seconds_scale, 90)     # calculate pause in case of death but cap at 90 days
    #     if pd.notna(row['dod']) and pd.isna(row['pause_until_next'])                # BUT ONLY if the patient died and this is the last episode
    #     else row['pause_until_next'],
    #     axis=1
    # ).fillna(90)  # Default pause if no death or next episode

    # Adjust for not have pause for last episode
    mv_times_new['pause_until_next'] = mv_times_new.apply(
        lambda row: row['pause_until_next']
        if pd.notna(row['pause_until_next'])                                    # keep calculated pause if it exists
        else 90,
        axis=1
    )

    # Create post_extubation_interval column
    mv_times_new['post_extubation_interval'] = mv_times_new.apply(
        lambda row: min((row['dod'] - row['endtime']) / days_to_seconds_scale, 90)
        if pd.notna(row['dod'])
        else 90,
        axis=1
    ).fillna(90)

    # Drop unnecessary columns
    mv_times_new = mv_times_new.drop(["starttime", "endtime", "dod"], axis=1)

    # run check for validity of MV_ID
    invalid_ids = mv_times.groupby('stay_id')['mv_id'].apply(
        lambda group: (group != range(0, len(group))).any()
    )

    if invalid_ids.any():
        print("Error: Invalid mv_id detected for the following stay_ids:")
        print(invalid_ids[invalid_ids].index.tolist())
    else:
        print("All mv_id assignments are valid.")


    result = pd.merge(data, mv_times_new, how='left', left_on=['stay_id', 'mv_id'], right_on=['stay_id', 'mv_id'])

    return result, ["mv_duration", "pause_until_next", "post_extubation_interval"]

import datetime
import os
import time

import numpy as np
from common.config import settings

import common.utility as util
import common.knowledge_base.kbase as kb

import data_cleaning.data_loading as dl
import data_cleaning.initial_data_extraction as ide

import vectorisation.state_vector_creation as vec
import data_quality_enhancement.time_window_creation as twc

import data_quality_enhancement as dqe

from data_cleaning.cleaning import clean_and_standardise_data

"""
STEPS IN THE DATA PREPROCESSING PIPELINE:
    1) State vector creation
    2) Data quality enhancement (computing missing values, VT normalisation) & time window creation  
"""


def main():
    # If data is loaded from a postgres database
    # --------------------------------------------------------
    # -------------------- LOADING DATA ----------------------
    # --------------------------------------------------------
    kb.regenerate_knowledge_base()
    kb.unif_vars = kb.Unif_Vars()

    util.generate_categ_dicts()

    # Loading measurements, demographics and procedures
    data, demo, ventilation, inputevents, outputevents = dl.load_data_from_csvs()

    # --------------------------------------------------------
    # ----------- DATA CLEANING & STANDARDISATION ------------
    # --------------------------------------------------------
    start_time = time.perf_counter()
    print("[######## Starting the data preprocessing pipeline ########]")

    print("Converting `ecmo_sweep` to float...", end=" ")
    data = ide.process_ecmo_sweep(data)
    util.pdone()

    print("Encoding string data...", end=" ")
    data = ide.encode_string_data(data=data)
    util.pdone()

    print("Collecting patient data related to: ", list(kb.mv_reqs.mv_reqs.keys()))
    # data, all_mv_times = ide.process_stay_id_data_fast(data=data, ventilation=ventilation)
    data, all_mv_times = ide.extract_mv_data_fast(data=data, ventilation=ventilation, num_cores=settings.num_of_cores)

    print("Dropping non-cohort patients...", end=" ")
    data, all_mv_times = ide.drop_non_cohort_patients(data=data, demog=demo, all_mv_times=all_mv_times)
    util.pdone()

    print("Standardizing the data and dropping outliers...", end=" ")
    data = clean_and_standardise_data(data=data)
    util.pdone()

    # --------------------------------------------------------
    # ------------- STATE VECTOR CREATION --------------------
    # --------------------------------------------------------

    # Taking only the first N patients to test the output
    if settings.stay_id_test_num > 0:
        print(f"Extracting the first {settings.stay_id_test_num} stay ids for testing...", end=" ")
        stay_ids = data["stay_id"].unique()
        data = data.loc[data["stay_id"].isin(stay_ids[: settings.stay_id_test_num])]
        all_mv_times = all_mv_times.loc[all_mv_times["stay_id"].isin(stay_ids[: settings.stay_id_test_num])]
        util.pdone()

    # Creating state vectors
    # data, priors = vec.create_state_vectors(data=data, patients_data=demo, multiproc=False)
    data, priors = vec.create_state_vectors_fast(data=data, patients_data=demo)

    # Value of 1 is taken for patients who are alive after the treatment
    # Value of 0 is taken for patients who died either in hospital or after 90 days post treatment
    data["daemo_discharge"] = 1
    data.loc[data["stay_id"].isin(demo.loc[demo["dod"].notna(), "stay_id"]), "daemo_discharge"] = 0

    # --------------------------------------------------------
    # --------------- DATA QUALITY ENHANCEMENT ---------------
    # --------------------------------------------------------

    # Defining moments when ECMO was active
    data.loc[:, "ecmo_active"] = data.loc[:, ["ecmo_sweep", "ecmo_bloodflow", "ecmo_rpm"]].notna().any(axis=1).astype(int)

    # Computing missing values
    data = dqe.compute_missing_values(data=data)

    # Creating time windows
    # data = twc.create_time_windows(data, priors=priors, inputevents=inputevents, outputevents=outputevents, resolution=settings.resolution, multiproc=False)
    data = twc.create_time_windows_fast(data=data, priors=priors, inputevents=inputevents, outputevents=outputevents)

    print("Excluding patients with under 4h of MV state vectors...", end=" ")
    assert data.groupby('stay_id')['mv_id'].apply(
        lambda x: list(range(0, len(np.unique(x)))) == sorted(np.unique(x))
    ).all(), "Some patients have missing MV periods!"
    valid_data_ids = data.groupby(["stay_id", "mv_id"]).apply(lambda x: len(x) >= 4)
    data = data[data.set_index(["stay_id", "mv_id"]).index.isin(valid_data_ids[valid_data_ids].index)].reset_index(drop=True)
    del valid_data_ids
    util.pdone()

    # Calculate cum_fluid_balance
    data["state_cumfluids"] = np.clip(data["state_ivfluid4h"] - data["state_urin4h"], *kb.mv_reqs.outlier_ranges["state_cumfluids"])

    print("TV normalization...", end=" ")
    data = dqe.TV_normalization(data=data)
    util.pdone()

    # Adding vent_pinsp-peep column
    data["vent_pinsp-peep"] = data["vent_pinsp"] - data["vent_peep"]

    # Apply processing to each group
    data["vent_vt_obs"] = np.nan
    data["vent_vt_action"] = np.nan
    data = data.groupby("stay_id").apply(dqe.process_for_different_vent_modes).reset_index(drop=True)
    data = data[data["vent_mode"] != 0].reset_index(drop=True)

    # Cleaning up the state vector of unneeded or currently unused states
    columns_to_cleanup = ["ext_ready", "is_spont", "state_icd", "iv_fluid_in_4h", "vent_invas", "ethnic"]
    data.drop(columns_to_cleanup, axis=1, inplace=True)

    print("Creating Episode IDs...", end=" ")
    data["episode_id"] = data.groupby(["stay_id", "mv_id"], sort=False).ngroup()
    util.pdone()

    print("Dropping episodes with negative vent_pinsp-peep...", end=" ")
    episode_ids_to_drop = data[data["vent_pinsp-peep"] < 0].groupby("episode_id")["episode_id"].unique().index
    data = data[~data["episode_id"].isin(episode_ids_to_drop)].reset_index(drop=True)
    util.pdone()

    print("Dropping episodes with less than 4 rows of data...", end=" ")
    episode_counts = data.groupby("episode_id").size()
    valid_episodes = episode_counts[episode_counts >= 4].index
    data = data[data["episode_id"].isin(valid_episodes)].reset_index(drop=True)
    util.pdone()

    print("Adjusting MV IDs and Episode IDs...", end=" ")
    for stay_id in data["stay_id"].unique():
        stay_id_data = data.loc[data["stay_id"] == stay_id, :]
        new_ids = {old_id: new_id for old_id, new_id in zip(stay_id_data["mv_id"].unique(), list(range(0, len(stay_id_data["mv_id"].unique()))))}
        data.loc[data["stay_id"] == stay_id, "mv_id"] = data.loc[data["stay_id"] == stay_id, "mv_id"].map(new_ids)
    data["episode_id"] = data.groupby(["stay_id", "mv_id"], sort=False).ngroup()
    util.pdone()
    

    # Adding MV duration, pause and reward
    print("Adding MV duration, pause, and reward info to AI-ready dataset...", end=" ")
    data_ai = data.copy(deep=True)
    data_ai, additional_ai_columns = dqe.add_duration_and_pause_info(data=data_ai, demo=demo, mv_times=all_mv_times)
    util.pdone()

    print("Filtering out less frequent variables")
    data = dqe.filter_variables(data)
    data_ai = dqe.filter_variables(data_ai, ai_ready=True)

    print("Creating mortality column", end=" ")
    def add_mortality_column(data):
        data["daemo_morta"] = data["daemo_discharge"]
        data["daemo_morta"] = data["daemo_morta"].replace({0: 1, 1: 0})
        
        return
    
    add_mortality_column(data=data)
    add_mortality_column(data=data_ai)
    util.pdone()
    
    data["timepoints"] /= 60
    data_ai["timepoints"] /= 60

    def get_current_date():
        current_time = datetime.datetime.now()
        formatted_date = current_time.strftime("%Y_%B_%d")
        return formatted_date

    current_date = get_current_date()
    print(f"The current date is", current_date)
    if not os.path.exists(os.path.join(settings.output_save_path, current_date)):
        os.mkdir(os.path.join(settings.output_save_path, current_date))

    output_save_path = os.path.join(settings.output_save_path, current_date)

    print("Saving the result to csv file...", end=" ")
    final_columns = dqe.mostly_available_columns
    final_columns.append("daemo_morta")

    front_items = ["stay_id", "mv_id", "episode_id"]
    final_columns = front_items + [item for item in np.sort(final_columns) if item not in front_items]
  
    data = data[final_columns]
    data_ai = data_ai[final_columns + additional_ai_columns]

    data.to_csv(os.path.join(output_save_path, settings.state_vectors_output_name + ".csv"), columns=final_columns, index=False)
    
    data_ai.to_csv(
        os.path.join(output_save_path, settings.state_vectors_output_name + "_ai_ready.csv"), 
        columns=final_columns + additional_ai_columns, 
        index=False
    )
    util.pdone()

    print(f"Pipeline is finished and took: {util.convert_to_h_m_s(time.perf_counter() - start_time)}.")


if __name__ == "__main__":
    main()

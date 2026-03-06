import os
import json
import tqdm

import pandas as pd
import multiprocessing as mp
from time import time
from datetime import datetime
from dotenv import load_dotenv

import extract.extract as ex
import preprocessing.filtering as filt
import preprocessing.windowing as wind
import preprocessing.imputation as imp
import preprocessing.cleaning as cl
import dicts.vars as var_dict


if __name__ == "__main__":
    startTime = time()
    # load environment variables
    load_dotenv()
    # set directories
    sourcepath = os.getenv("SOURCE_PATH")
    vitals_dir = os.path.join(sourcepath, "vitalPeriodic")
    out_dir = os.getenv("OUT_DIR")
    if (out_dir == "") or (out_dir is None):
        out_dir = os.path.join(
            "eicu", f"{datetime.fromtimestamp(startTime).strftime('%y%m%d%H%M%S%f')}_state_vectors")
    interm_vector_dir = f"{out_dir}/interm_state_vectors"
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    if not os.path.isdir(interm_vector_dir):
        os.mkdir(interm_vector_dir)
    if (os.getenv("QUERY_DB") == "True"):
        from extract.querydb import query_db
        query_db(startTime, sourcepath)
    num_cores = int(os.getenv("NUM_CORES"))

    # get required variables
    required_variables = os.getenv("REQUIRED_VARIABLES")
    required_variables_all, required_filter = filt.get_reqvars(required_variables)
    with open(f"{out_dir}/preprocessing_configs.json", "w") as f:
        json.dump({"required_variables":required_variables_all}, f)    

    # Set project variables according to project requirements
    MIN_HOURS_DURATION = 4
    TIME_WINDOW_DURATION = 60

    print("Loading data...")
    data_df = ex.dynamic_from_csv(sourcepath)
    demo_df = ex.static_from_csv(sourcepath)
    ventevents = pd.read_csv(os.path.join(
        sourcepath, "ventevents.csv"))
    print("at ", str(time() - startTime), "seconds")

    print("Preprocessing vent episodes...")
    ventevents = cl.preprocess_ventevents(ventevents, MIN_HOURS_DURATION)
    print("at ", str(time() - startTime), "seconds")

    print("Selecting vent patients...")
    vent_patientids = ventevents["stay_id"].unique()
    data_df = data_df[data_df["patientunitstayid"].isin(
        vent_patientids)].reset_index(drop=True)
    demo_df = demo_df[demo_df["patientunitstayid"].isin(
        vent_patientids)].reset_index(drop=True)
    print("at ", str(time() - startTime), "seconds")

    # NOTE used to drop currently unused variable
    data_df = cl.drop_problematic_dev(data_df, ["vent_invas"])
    print("Cleaning demo...")
    cleaned_demo = cl.clean_data(demo_df, startTime)
    print("Cleaning data...")
    cleaned_data = cl.clean_data(data_df, startTime)

    print("Filtering patients without required demographics or vent mode...")
    filtered_demo, filtered_data, missing_dict = filt.filter_pats_static(
        cleaned_demo, cleaned_data)
    filtered_demo.to_pickle(f"{out_dir}/filtered_demo.pkl")
    print("at ", str(time() - startTime), "seconds")

    print("Standardize vasopressors...")
    filtered_data = cl.standardize_vaso(filtered_data, filtered_demo)
    filtered_data.to_pickle(f"{out_dir}/filtered_data.pkl")
    print("at ", str(time() - startTime), "seconds")

    # filtered_demo = pd.read_pickle(f"{out_dir}/filtered_demo.pkl")
    # filtered_demo["value"] = filtered_demo["value"].apply(cl.cast_clean_float)
    # filtered_data = pd.read_pickle(f"{out_dir}/filtered_data.pkl")
    # filtered_data["value"] = filtered_data["value"].apply(cl.cast_clean_float)

    for vitals_file in os.listdir(vitals_dir):
        if vitals_file.startswith("vitalPeriodic"):
            print(vitals_file)
            vitals_file_basename = os.path.splitext(vitals_file)[0]
            vitals_df = pd.read_csv(os.path.join(vitals_dir, vitals_file))
            # Filtering data
            slice_ids = set(vitals_df["patientunitstayid"].unique()).intersection(
                filtered_data["patientunitstayid"].unique())
            filtered_data_slice = filtered_data[filtered_data["patientunitstayid"].isin(
                slice_ids)]
            filtered_vitals_slice = vitals_df[vitals_df["patientunitstayid"].isin(
                slice_ids)]

            print("Cleaning vitals...")
            cleaned_vitals = cl.clean_data(vitals_df, startTime)
            chunk_data = pd.concat(
                [cleaned_vitals, filtered_data_slice], axis=0).reset_index(drop=True)
            print("at ", str(time() - startTime), "seconds")

            print("Filtering patients...")
            all_state_vectors, missing_dict = filt.filter_by_reqvar(chunk_data, required_filter)
            print("at ", str(time() - startTime), "seconds")

            all_state_vectors.to_pickle(
                f"{interm_vector_dir}/{vitals_file_basename}_initial_long_vector.pkl")            
            with open(f"{out_dir}/missing_dict.json", "w") as f:
                json.dump(missing_dict, f)

            # all_state_vectors = pd.read_pickle(f"{interm_vector_dir}/{vitals_file_basename}_initial_long_vector.pkl")
            # Create time windows
            print("Create time windows...")
            unique_patientunitstayids = all_state_vectors["patientunitstayid"].unique()
            all_windowed_list = []
            variable_columns = var_dict.all_variables

            # # single thread
            # for stayid in tqdm.tqdm(unique_patientunitstayids):
            #     patient_state_vector = all_state_vectors[all_state_vectors["patientunitstayid"] == stayid].reset_index(
            #         drop=True)
            #     patient_ventevents = ventevents[ventevents["stay_id"]== stayid].reset_index(drop=True)
            #     windowed_vector, missing_dict = wind.create_time_windows(
            #         stayid, TIME_WINDOW_DURATION, patient_state_vector, variable_columns, filtered_demo[filtered_demo["patientunitstayid"] == stayid], 
            #         patient_ventevents, required_variables_all)
            #     all_windowed_list.append(windowed_vector)

            results = []
            with mp.Pool(num_cores) as pool:
                for stayid in tqdm.tqdm(unique_patientunitstayids):
                    stay_df = all_state_vectors[all_state_vectors["patientunitstayid"] == stayid]
                    stay_ventevents = ventevents[ventevents["stay_id"] == stayid].reset_index(
                        drop=True)
                    results.append(pool.apply_async(
                        wind.create_time_windows, args=(stayid,
                                                        TIME_WINDOW_DURATION,
                                                        stay_df, variable_columns,
                                                        filtered_demo[filtered_demo["patientunitstayid"] == stayid],
                                                        stay_ventevents,
                                                        required_variables_all)))

                print("Appending Results: ")
                for result in tqdm.tqdm(results, total=len(results)):
                    result_tuple = result.get()
                    all_windowed_list.append(result_tuple[0])

                    stay_missing_dict = result_tuple[1]
                    for key in stay_missing_dict:
                        if key in missing_dict:
                            missing_dict[key] += stay_missing_dict[key]
                        else:
                            missing_dict[key] = stay_missing_dict[key]

            all_windowed_state_vectors = pd.concat(
                all_windowed_list, axis=0).reset_index(drop=True)
            print("at", str(time() - startTime), "seconds")

            with open(f"{out_dir}/missing_dict.json", "w") as f:
                json.dump(missing_dict, f)
            all_windowed_state_vectors.to_pickle(
                f"{interm_vector_dir}/{vitals_file_basename}_windowed_state_vector.pkl")

            # all_windowed_state_vectors = pd.read_pickle(f"{interm_vector_dir}/{vitals_file_basename}_windowed_state_vector.pkl")
            print("Compute missing variables...")
            all_windowed_state_vectors = imp.compute_calculable(
                all_windowed_state_vectors)
            print("at", str(time() - startTime), "seconds")

            # Handling missing data
            print("Impute missing variables...")
            all_filled_list = []
            min_trajectory_len = MIN_HOURS_DURATION*60 / \
                TIME_WINDOW_DURATION    # based on proposal

            # # single thread
            # for stayid in tqdm.tqdm(all_windowed_state_vectors["stay_id"].unique()):
            #     patient_state_vector = all_windowed_state_vectors[all_windowed_state_vectors["stay_id"] == stayid].reset_index(
            #         drop=True)
            #     all_filled_list.append(imp.forward_imputation(patient_state_vector, min_trajectory_len))

            results = []
            with mp.Pool(num_cores) as pool:
                for stayid in tqdm.tqdm(all_windowed_state_vectors["stay_id"].unique()):
                    patient_state_vector = all_windowed_state_vectors[all_windowed_state_vectors["stay_id"] == stayid].reset_index(
                        drop=True)
                    results.append(pool.apply_async(
                        imp.forward_imputation, args=(patient_state_vector, min_trajectory_len, required_variables_all)))
                for result in tqdm.tqdm(results, total=len(results)):
                    all_filled_list.append(result.get())

            all_filled_state_vectors = pd.concat(
                all_filled_list, axis=0).reset_index(drop=True)
            print("at", str(time() - startTime), "seconds")

            print("Handle mode dependent variables...")
            all_filled_state_vectors = imp.calc_mode_dependent_variables(
                all_filled_state_vectors)
            print("at", str(time() - startTime), "seconds")
            all_filled_state_vectors.to_pickle(
                f"{interm_vector_dir}/{vitals_file_basename}_state_vectors_eicu.pkl")

    print("Creating final state vector...")
    final_state_vector = ex.merge_state_vectors(interm_vector_dir)
    fsv_w_epsid = wind.assign_eps_ids(final_state_vector)
    fsv_w_epsid.to_csv(f"{out_dir}/state_vectors_eicu.csv", index=False)
    fsv_w_epsid[fsv_w_epsid.columns[~fsv_w_epsid.isnull().any()].tolist()].to_csv(
        f"{out_dir}/state_vectors_eicu_ai_ready.csv", index=False)
    print("at", str(time() - startTime), "seconds")

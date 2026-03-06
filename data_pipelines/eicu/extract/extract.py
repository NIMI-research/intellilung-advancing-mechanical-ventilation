import os
import pandas as pd
import dicts.vars as var_dict

no_offset_tables = [
                    "patient", 
                    # "apacheApsVar",
                    # "apachePatientResult", 
                    # "apachePredVar"
                    ]

def dynamic_from_csv(sourcepath):
    data_csv_concat = None
    for filename in os.listdir(sourcepath):
        if filename.endswith(".csv"):
            if os.path.splitext(filename)[0] not in no_offset_tables:
                print(filename)
                data_csv = pd.read_csv(os.path.join(sourcepath, filename))
                if data_csv_concat is None:
                    data_csv_concat = data_csv.copy()
                else:
                    data_csv_concat = pd.concat(
                        [data_csv_concat, data_csv], axis=0)
    return data_csv_concat


def static_from_csv(sourcepath):
    data_csv_concat = None
    for filename in os.listdir(sourcepath):
        if filename.endswith(".csv"):
            if os.path.splitext(filename)[0] in no_offset_tables:
                print(filename)
                data_csv = pd.read_csv(os.path.join(sourcepath, filename))
                if data_csv_concat is None:
                    data_csv_concat = data_csv.copy()
                else:
                    data_csv_concat = pd.concat(
                        [data_csv_concat, data_csv], axis=0)
    return data_csv_concat

def get_patients_with_vent(data):
    return data[data["variable"].isin(var_dict.mapped_vent_vars)]["patientunitstayid"].unique()

def merge_state_vectors(sourcepath):
    data_concat = None
    for filename in os.listdir(sourcepath):
        if filename.endswith("state_vectors_eicu.pkl"):
            print(filename)
            data_pkl = pd.read_pickle(os.path.join(sourcepath, filename))
            if data_concat is None:
                data_concat = data_pkl.copy()
            else:
                data_concat = pd.concat(
                    [data_concat, data_pkl], axis=0)
    return data_concat

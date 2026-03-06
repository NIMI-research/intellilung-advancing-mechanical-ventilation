import math
import preprocessing.cleaning as cl

def is_float(string):
    try:
        float_value = float(string)
        return True
    except ValueError:
        return False
    
def filter_patients_wo_reqdemog(demo_df, req_demo):
    missing_dict = {}
    patients_with_req_demog = set(demo_df['patientunitstayid'])
    patient_num = len(patients_with_req_demog)
    for var in req_demo:
        patients_with_var = set(demo_df.loc[(demo_df["variable"] == var) & (demo_df["value"].notnull()), "patientunitstayid"].unique())
        missing_dict[var] = patient_num - len(patients_with_var)
        patients_with_req_demog = patients_with_req_demog.intersection(patients_with_var)
    
    demo_df = demo_df[demo_df["patientunitstayid"].isin(patients_with_req_demog)].reset_index(drop=True)
    return demo_df, missing_dict


def add_patient_demo(patient_state_vector, patient_demo_df, ventend_offset, required_variables):
    """Add demographic data to each time step"""
    missing_dict = {}

    # add weight
    admit_weight = patient_demo_df.loc[patient_demo_df["variable"] == "daemo_weight", "value"]
    if len(admit_weight.values) > 0:
        admit_weight = cl.cast_clean_float(admit_weight.mean())
        patient_state_vector["daemo_weight"] = admit_weight
    else:
        missing_dict["daemo_weight"] = 1
        return None, missing_dict

    # add gender
    gender_dict = {"Female": 1, "Male": 0}
    sex = patient_demo_df.loc[patient_demo_df["variable"] == "daemo_sex", "value"]
    no_sex = False
    if len(sex.values) > 0:
        sex = sex.values[0]
        if is_float(sex):
            sex = float(sex)
            if not math.isnan(sex):
                patient_state_vector["daemo_sex"] = int(sex)
            else:
                missing_dict["daemo_sex"] = 1
                no_sex = True
        elif sex in gender_dict:
            patient_state_vector["daemo_sex"] = gender_dict[sex]
        else:
            missing_dict["daemo_sex"] = 1
            no_sex = True
    else:
        missing_dict["daemo_sex"] = 1
        no_sex = True
    if no_sex:
        return None, missing_dict

    # add survival status
    status_dict = {"expired": 0, "alive": 1}
    status_df = patient_demo_df[
        (patient_demo_df["variable"] == "daemo_discharge") & 
        (patient_demo_df["value"].notna())
    ]

    if not status_df.empty:
        # Get latest discharge offset
        max_offset_index = status_df["offset"].idxmax()
        max_offset = status_df.at[max_offset_index, "offset"]
        # Get latest status
        latest_status = status_df.at[max_offset_index, "value"].lower()
        patient_state_vector["daemo_discharge"] = status_dict[latest_status]

        # Calculate post_extubation_interval
        if status_dict[latest_status] == 0: # if patient died
            # interval: time of ventend until time of death
            # convert offset minutes to days
            # NOTE: ventend_offset can be slightly larger than dischargetime due to query specific issue, fix with max
            patient_state_vector["post_extubation_interval"] = max(0, (max_offset - ventend_offset)/60/24)
        else:
            # Default to 90 days if the patient is alive
            patient_state_vector["post_extubation_interval"] = 90
    else:
        missing_dict["daemo_discharge"] = 1
        return None, missing_dict

    # add height
    admit_height = patient_demo_df.loc[patient_demo_df["variable"]
                                    == "daemo_height", "value"]

    if len(admit_height.values) > 0:
        admit_height = cl.cast_clean_float(admit_height.mean())
        patient_state_vector["daemo_height"] = admit_height
        # if none available, discard patient
    else:
        missing_dict["daemo_height"] = 1
        return None, missing_dict
    
    # add age
    age = patient_demo_df.loc[patient_demo_df["variable"] == "daemo_age", "value"]

    if len(age.values) > 0:
        age = cl.cast_clean_int(age.iloc[0])
        patient_state_vector["daemo_age"] = age
    else:
        missing_dict["daemo_age"] = 1
        if "daemo_age" in required_variables:
            return None, missing_dict

    return patient_state_vector, missing_dict

def merge_TV(patient_state_vector):
    """Merge normalized and unnormalized Tidal Volume into 1 column"""
    # calculate IBW based on gender
    if patient_state_vector["daemo_sex"].loc[patient_state_vector.index[0]]:
        ibw = (patient_state_vector["daemo_height"] - 152.4) * 0.91 + 45.5
    else:
        ibw = (patient_state_vector["daemo_height"] - 152.4) * 0.91 + 50.0

    # unnormalize vt_norm
    patient_state_vector['vent_vtnorm'] = patient_state_vector["vent_vtnorm"] * ibw
    # clean outlier based on vent_vt: previously not done because ot defined explicitly for vt_norm
    patient_state_vector['vent_vtnorm'] = cl.remove_outliers_col(patient_state_vector["vent_vtnorm"], "vent_vt")
    # re-normalize
    patient_state_vector['vent_vtnorm'] = patient_state_vector["vent_vtnorm"] / ibw
    
    # calculate missing vt based on vtnorm
    missing_vt_mask = patient_state_vector['vent_vt'].isnull()
    if len(patient_state_vector[missing_vt_mask]["vent_vtnorm"]) > 0:
        patient_state_vector.loc[missing_vt_mask, 'vent_vt'] = patient_state_vector[missing_vt_mask]["vent_vtnorm"] * ibw
    # calculate missing vtnorm based on vt
    missing_vtnorm_mask = patient_state_vector['vent_vtnorm'].isnull()
    patient_state_vector.loc[missing_vtnorm_mask, 'vent_vtnorm'] = patient_state_vector[missing_vtnorm_mask]["vent_vt"] / ibw

    return patient_state_vector

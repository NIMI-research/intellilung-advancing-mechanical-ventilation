
import ast
import dicts.vars as var_dict
import preprocessing.demog as demog

def get_reqvars(required_variables):
    """Get list of required variables and required variables ex demographic variables"""
    if required_variables is None:
        required_filter = list(
                var_dict.sets_dict["1_exdaemo"]) + list(var_dict.required_tidal)
        required_all = required_filter+list(var_dict.sets_dict["1_daemo"])
    else:
        required_all = ast.literal_eval(required_variables)
        # remove demog and 4h variables because they are only added during windowing
        required_filter = [x for x in required_all if (not x.startswith("daemo_") and ("4h" not in x))]
    return required_all, required_filter
        

def filter_pats_static(cleaned_demo, cleaned_data):
    """Filter patients who do not have required static data:
    - demographics
    - vent_mode
    """
    # Filter by demog
    filtered_demo, missing_dict = demog.filter_patients_wo_reqdemog(
        cleaned_demo, var_dict.sets_dict["1_daemo"])
    # Filter by vent_mode
    # Get patientids with vent_mode
    pats_w_vent_mode = cleaned_data[cleaned_data["variable"]
                                    == "vent_mode"]["patientunitstayid"].unique()
    # Filter patients
    filtered_demo = cleaned_demo[cleaned_demo["patientunitstayid"].isin(
        pats_w_vent_mode)].reset_index(drop=True)
    filtered_data = cleaned_data[cleaned_data["patientunitstayid"].isin(
        pats_w_vent_mode)].reset_index(drop=True)
    return filtered_demo, filtered_data, missing_dict

def filter_by_reqvar(df, reqvars):
    # Filter the data to include only required variables
    filtered_data = df[df["variable"].isin(
        reqvars)]

    # Check if all required variables have at least one valid value for each patient
    valid_values = (
        filtered_data
        .groupby(["patientunitstayid", "variable"])["value"]
        .apply(lambda x: ~x.isnull().all())  # True if at least one non-NaN value exists
        .unstack(fill_value=False)  # Create DataFrame with variables as columns; fill missing with False
    )
    # If required variable was not present in valid_values, set it to False
    valid_values = valid_values.reindex(columns=reqvars, fill_value=False)

    # Filter patients who have at least one valid value for ALL required variables
    patients_with_all_vars = valid_values[valid_values.all(axis=1)].index
    all_state_vectors = df[df["patientunitstayid"].isin(
        patients_with_all_vars)]

    missing_dict = valid_values.apply(lambda col: (~col).sum()).to_dict()

    return all_state_vectors, missing_dict


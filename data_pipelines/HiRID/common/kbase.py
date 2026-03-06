import os
import json
import sys
import pandas as pd
from tqdm import tqdm
from common.config import settings

# Load dictionaries
with open(os.path.join(os.path.dirname(__file__), "dicts/HiRID_var_map.json")) as f:
    column_name_conversions = json.load(f)

with open(os.path.join(os.path.dirname(__file__), "dicts/variable_outlier_ranges.json")) as f:
    var_outlier_ranges = json.load(f)

with open(os.path.join(os.path.dirname(__file__), "dicts/unit_conversion.json")) as f:
    unit_conversion = json.load(f)

# Variable order for state vectors
var_order = [
    'PatientID', 'AbsDatetime', 'time_interval', 'mv_id', 'episode_id', 
    'mv_duration', 'pause_until_next', 'post_extubation_interval', 'VFD', 'VFD30',
    'daemo_sex', 'daemo_age', 'daemo_weight', 'daemo_ideal_weight', 
    'daemo_height', 'daemo_discharge',
    'vital_hr', 'vital_map', 'vital_SBP', 'vital_DBP', 'vital_mpap', 'vital_co', 'vital_spo2',  'vital_cvp',
    'blood_svo2', 'blood_lac', 'blood_prbc', 'blood_ffp', 'blood_paco2', 'blood_pao2', 'blood_sao2', 'blood_ast', 
    'blood_alt', 'blood_billi', 'blood_album', 'blood_gluco', 'blood_be', 'blood_hco3', 'blood_ph', 'blood_potas', 'blood_sodium', 
    'blood_chlorid', 'blood_caion', 'blood_calcium', 'blood_magnes', 'blood_crea', 'blood_wbc', 'blood_hb', 'blood_plat', 
    'blood_PTT', 'blood_INR',
    'state_temp', 'state_bun', 
    'dm_total_in', 'dm_total_out', 'state_cumfluids',
    'state_ivfluid4h', 'state_urin4h', 
    'drugs_vaso4h',
    'vent_rrtot', 'vent_etco2', 'vent_fio2', 'vent_mairpress', 'vent_vt', 'vent_vtnorm', 'vent_vt_obs', 'vent_vt_action',
    'vent_pinsp', 'vent_peep', 'vent_pinsp-peep',
    'vent_invas', 'state_airtype', 'vent_mode', 
    'dm_vent_mode_group', 'dm_vent_mode_subgroup', 'dm_vent_contr_mode', 
    'is_controlled', 'dm_vent_controlled_ventilation_merged',
    'dm_vent_niv_state'
    ]

# List of variables for time windowing
## Categorical variables
cat_var = ['vent_invas', 'state_airtype', 'vent_mode', 'dm_vent_mode_group', 'dm_vent_mode_subgroup', 'dm_vent_contr_mode', 
    'is_controlled', 'dm_vent_controlled_ventilation_merged', 'dm_vent_niv_state']
## Numeric variables (except for 4h variables and fluid/vasopressor-related variables)
num_var = ['vital_hr', 'vital_map', 'vital_SBP', 'vital_DBP', 'vital_mpap', 'vital_co', 'vital_spo2',  'vital_cvp',
    'blood_svo2', 'blood_lac', 'blood_prbc', 'blood_ffp', 'blood_paco2', 'blood_pao2', 'blood_sao2', 'blood_ast', 
    'blood_alt', 'blood_billi', 'blood_album', 'blood_gluco', 'blood_be', 'blood_hco3', 'blood_ph', 'blood_potas', 'blood_sodium', 
    'blood_chlorid', 'blood_caion', 'blood_calcium', 'blood_magnes', 'blood_crea', 'blood_wbc', 'blood_hb', 'blood_plat', 
    'blood_PTT', 'blood_INR',
    'state_temp', 'state_bun', 
    'vent_rrtot', 'vent_etco2', 'vent_fio2', 'vent_mairpress', 'vent_vt', 'vent_vtnorm', 
    'vent_pinsp', 'vent_peep', 'vent_pinsp-peep']
# Static/demographic and ventilation values for each specific patient and episode
static_var = ['daemo_sex', 'daemo_age', 'daemo_weight', 'daemo_ideal_weight', 'daemo_height', 'daemo_discharge', 
              'episode_id', 'mv_duration', 'pause_until_next', 'post_extubation_interval', 'VFD', 'VFD30']

# List of static variables (set0)
set0_variables = [
    'PatientID', 'time_interval', 'mv_id', 'episode_id', 'mv_duration', 'pause_until_next', 'post_extubation_interval', 'VFD', 'VFD30'
]
# List of set1 variables
set1_variables = [
    'daemo_sex', 'daemo_weight', 'daemo_height', 'daemo_discharge',
    'vital_map', 'vital_hr', 'vital_spo2', 
    'vent_fio2', 'vent_pinsp', 'vent_peep', 'vent_pinsp-peep', 'vent_rrtot', 
    'vent_invas', 'vent_vt', 'vent_vtnorm', 'vent_mode', 'vent_vt_obs', 'vent_vt_action'
]
# List of set2a variables
set2a_variables = [
    'blood_be', 'blood_hco3', 'blood_ph', 'vital_mpap', 'vital_DBP', 'vital_SBP', 'blood_INR', 'blood_PTT', 
    'daemo_age', 'blood_calcium', 'blood_chlorid', 'blood_caion', 'blood_magnes', 'blood_potas', 'blood_sodium', 
    'vent_etco2', 'blood_paco2', 'blood_pao2', 'blood_sao2', 'blood_svo2', 'blood_plat', 'blood_hb', 'blood_wbc', 
    'state_bun', 'blood_crea', 'blood_album', 'blood_alt', 'blood_ast', 'blood_billi', 'blood_lac', 'blood_gluco', 
    'state_temp', 'vent_mairpress', 'state_airtype', 'blood_ffp', 'blood_prbc'
]
# List of set2b variables
set2b_variables = [
    'drugs_vaso4h', 'vital_cvp', 'state_cumfluids', 'state_ivfluid4h', 'vital_co', 'state_urin4h'
]
# List of variables with low availability to be removed
set_low_availability_variables = [
    'blood_album', 'blood_alt', 'blood_ast', 'blood_billi', 'blood_calcium', 'blood_crea', 'blood_magnes', 
    'blood_PTT', 'blood_svo2', 'state_bun', 'state_temp', 'vital_mpap', 'vital_co'
]

# List with IntelliLung-named variables undergoing unit conversion
unit_converted_intellilung_variables = list(unit_conversion.keys())

# List with Hirid's original-named variables undergoing unit conversion
unit_conversion_hirid_originals = []
for intellilung_variable in unit_converted_intellilung_variables:
    hirid_original_variable = column_name_conversions.get(intellilung_variable)
    if hirid_original_variable:
        unit_conversion_hirid_originals.append(hirid_original_variable)

# Function for exporting dataframe in parquet format
def export_parquet(df: pd.DataFrame, output_name, file_name_for_printing):
    if not os.path.exists(os.path.join(settings.source_path, settings.output_save_path)):
        os.mkdir(os.path.join(settings.source_path, settings.output_save_path))
    print("\nConverting to parquet:", file_name_for_printing)
    with tqdm(total=1) as pbar:
        df.to_parquet(os.path.join(settings.source_path, settings.output_save_path, output_name))
        pbar.update(1)

# Function for printing variables with NaN values
def save_nan_var(df: pd.DataFrame):
    var_not_nan = [col for col in df.columns if df[col].isna().sum() == 0]  # Columns without NaN values
    var_nan = [col for col in df.columns if col not in var_not_nan]  # Columns with NaN values
    return var_nan, var_not_nan

# Function for converting data types to reduce memory for computation
def transform_column_dtypes(df: pd.DataFrame):
    type_dict = {
        "PatientID": "Int32",
        "AbsDatetime": "datetime64[ns]",
        "time_interval": "Int32",
        "mv_id": "Int8",
        "episode_id": "Int16",
        "mv_duration": "float32",
        "pause_until_next": "float32",
        "VFD": "float32",
        "VFD30": "float32",
        "post_extubation_interval": "float32",
        "daemo_sex": "category",
        "daemo_age": "Int8",
        "daemo_weight": "int16",
        "daemo_ideal_weight": "float32",
        "daemo_height": "int16",
        "daemo_discharge": "category",
        "vital_hr": "float32",
        "vital_map": "float32",
        "vital_SBP": "float32",
        "vital_DBP": "float32",
        "vital_mpap": "float32",
        "vital_co": "float32",
        "vital_spo2": "float32",
        "vital_cvp": "float32",
        "blood_svo2": "float32",
        "blood_lac": "float32",
        "blood_prbc": "float32",
        "blood_ffp": "float32",
        "blood_paco2": "float32",
        "blood_pao2": "float32",
        "blood_sao2": "float32",
        "blood_ast": "float32",
        "blood_alt": "float32",
        "blood_billi": "float32",
        "blood_album": "float32",
        "blood_gluco": "float32",
        "blood_be": "float32",
        "blood_hco3": "float32",
        "blood_ph": "float32",
        "blood_potas": "float32",
        "blood_sodium": "float32",
        "blood_chlorid": "float32",
        "blood_caion": "float32",
        "blood_calcium": "float32",
        "blood_magnes": "float32",
        "blood_crea": "float32",
        "blood_wbc": "float32",
        "blood_hb": "float32",
        "blood_plat": "float32",
        "blood_PTT": "float32",
        "blood_INR": "float32",
        "dm_total_in": "float32",
        "dm_total_out": "float32",
        "state_temp": "float32",
        "state_bun": "float32",
        "state_cumfluids": "float32",
        "state_ivfluid4h": "float32",
        "state_urin4h": "float32",
        "drugs_vaso4h": "float32",
        # "dm_vasoactive_current": "float32",
        "vent_rrtot": "float32",
        "vent_etco2": "float32",
        "vent_fio2": "float32",
        "vent_mairpress": "float32",
        "vent_vt": "float32",
        "vent_vt_obs": "float32", 
        "vent_vt_action": "float32",
        "vent_vtnorm": "float32",
        "vent_pinsp": "float32",
        "vent_peep": "float32",
        #"vent_suppress": "float32",
        "vent_pinsp-peep": "float32",
        "vent_invas": "category",
        "state_airtype": "category",
        "vent_mode": "category",
        "dm_vent_mode_group": "category",
        "dm_vent_mode_subgroup": "category",
        "dm_vent_contr_mode": "category",
        "is_controlled": "category",
        "dm_vent_controlled_ventilation_merged": "category",
        "dm_vent_niv_state": "category"
    }
    return {key: val for key, val in type_dict.items() if key in df.columns}
    

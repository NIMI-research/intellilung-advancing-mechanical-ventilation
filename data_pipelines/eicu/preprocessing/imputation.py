import numpy as np
import pandas as pd

import common.utils as ut
import dicts.vars as var_dict


def compute_calculable(data_df):
    """Fill missing variables that can be calculated"""

    # filling vital_map
    missing_mask = (data_df['vital_map'].isnull())
    dbp_mask = (~data_df['vital_DBP'].isnull())
    sbp_mask = (~data_df['vital_SBP'].isnull())

    calc_mask = missing_mask & dbp_mask & sbp_mask
    data_df.loc[calc_mask, 'vital_map'] = data_df.loc[calc_mask, 'vital_DBP'] + \
        (1/3) * (data_df.loc[calc_mask, 'vital_SBP'] -
                 data_df.loc[calc_mask, 'vital_DBP'])

    return data_df


def find_first_valindex_all(windowed_state_vector, var_list):
    """"Return index at which listed variables occured at least once
    If first occurence cannot be found, return None
    """
    val_indices = []
    for col in var_list:
        try:
            indice = windowed_state_vector[col].first_valid_index()
            if indice is not None:
                val_indices.append(indice)
            else:
                return None
        except KeyError:
            return None
    return max(val_indices)


def fill_first_index(first_index, patient_state_vector):
    """Filled based on last available"""
    # Reset index to ensure positional indexing is correct
    patient_state_vector_filled = patient_state_vector.copy().reset_index(drop=True)
    for colnum in range(len(patient_state_vector_filled.columns)):
        if np.isnan(ut.get_df_value(patient_state_vector_filled.iloc[first_index, colnum])):
            col_last_valid_i = patient_state_vector_filled.iloc[:first_index, colnum].last_valid_index(
            )
            if col_last_valid_i is not None:
                patient_state_vector_filled.iloc[first_index,
                                                 colnum] = patient_state_vector_filled.iloc[col_last_valid_i, colnum]
    return patient_state_vector_filled.iloc[first_index:, :].reset_index(drop=True)


def carry_forward(patient_state_vector):
    """Carry forward last available data point until next is available"""
    patient_state_vector_filled = patient_state_vector.copy().reset_index(drop=True)
    for colnum in range(len(patient_state_vector_filled.columns)):
        for rownum in range(1, len(patient_state_vector_filled)):
            current_val = ut.get_df_value(
                patient_state_vector_filled.iloc[rownum, colnum])
            if np.isnan(current_val):
                col_last_valid_i = patient_state_vector_filled.iloc[:rownum, colnum].last_valid_index(
                )
                if col_last_valid_i is not None:
                    patient_state_vector_filled.iloc[rownum,
                                                     colnum] = ut.get_df_value(patient_state_vector_filled.iloc[col_last_valid_i, colnum])
    return patient_state_vector_filled


def forward_imputation(patient_state_vector, min_trajectory_len, var_list=var_dict.sets_dict["1_exdaemo"]):
    results = []
    for mv_id, mv_eps in patient_state_vector.groupby("mv_id"):
        first_index = find_first_valindex_all(
            mv_eps, var_list)
        if first_index is not None and ((len(mv_eps) - first_index) >= min_trajectory_len):
            sub_patient_state_vector = fill_first_index(
                first_index, mv_eps)
            results.append(carry_forward(sub_patient_state_vector))

    if results:  # Ensure results is not empty
        return pd.concat(results, axis=0, ignore_index=True)
    return None


def handle_conditional_actions(windowed_vector):
    """
    When vent_mode=2, vent_vt is an action, not an observation:
    - vent_vt_obs should be vent_vt from previous timepoint
    - vent_vt_action should be current value
    - vent_pinsp should be current value

    When vent_mode=3, vent_pinsp-peep is an action, current vent_pinsp is not an observation:
    - vent_pinsp should be vent_pinsp from previous timepoint
    - vent_pinsp-vent_peep should be based on current vent_pinsp
    - vent_vt should be current value
    """
    # vent_vt
    windowed_vector['vent_vt_action'] = windowed_vector['vent_vtnorm']
    windowed_vector['vent_vt_obs'] = windowed_vector['vent_vtnorm']
    # Use previous row when vent_mode is 2
    windowed_vector.loc[windowed_vector['vent_mode'] == 2,
                        'vent_vt_obs'] = windowed_vector['vent_vtnorm'].shift(1)
    # Handle the first row: replace NaN in 'vent_vt_obs' with the current row value
    windowed_vector['vent_vt_obs'] = windowed_vector['vent_vt_obs'].fillna(
        windowed_vector['vent_vtnorm'])

    # vent_pinsp
    windowed_vector['vent_pinsp_obs'] = windowed_vector['vent_pinsp']
    # Use previous row when vent_mode is 3
    windowed_vector.loc[windowed_vector['vent_mode'] == 3,
                        'vent_pinsp'] = windowed_vector['vent_pinsp_obs'].shift(1)
    # Handle the first row: replace NaN in 'vent_pinsp' with the current row value
    windowed_vector['vent_pinsp'] = windowed_vector['vent_pinsp'].fillna(
        windowed_vector['vent_pinsp_obs'])
    windowed_vector = windowed_vector.drop(columns=['vent_pinsp_obs'])

    return windowed_vector


def calc_mode_dependent_variables(all_filled_state_vectors):
    all_filled_state_vectors["vent_pinsp-peep"] = all_filled_state_vectors["vent_pinsp"] - \
        all_filled_state_vectors["vent_peep"]
    # drop vent_pinsp-peep below 0
    all_filled_state_vectors = all_filled_state_vectors.groupby(['stay_id', 'mv_id']).filter(
        lambda x: (x['vent_pinsp-peep'] >= 0).all()).reset_index(drop=True)
    all_filled_state_vectors = all_filled_state_vectors.groupby(
        ['stay_id', 'mv_id'], group_keys=False).apply(handle_conditional_actions)

    return all_filled_state_vectors

import pandas as pd
import numpy as np
import polars as pl
import sys
import os
import matplotlib.pyplot as plt
from datetime import datetime
    
def process_patient(patient_id, ds_merged):
    time_intervall= 5

    va_columns = ['vm2201','vm2202', 'vm2207']
    va_columns_factors = { 'vm2201': 1, 'vm2202': 1, 'vm2207': 2.5}

    ds_merged = ds_merged.filter((pl.col('PatientID')==patient_id))
    ds_merged = ds_merged.select(["PatientID", "AbsDatetime", "vm1"] + va_columns + [f'{element}_bolus' for element in va_columns]).collect().to_pandas()

    ds_merged["dm_vasoactive_current"] = 0
    #adopt to er weight dosage
    ds_merged["dm_vasoactive_current"] = 0

    #adopt to er weight dosage
    ds_merged['vm1'] = ds_merged['vm1'].ffill().fillna(80)
    ds_merged["vm2201"] = ds_merged["vm2201"] / ds_merged['vm1']
    ds_merged["vm2202"] = ds_merged["vm2202"] / ds_merged['vm1']
    ds_merged["vm2207"] = ds_merged["vm2207"] / ds_merged['vm1']

    for col in va_columns:
        ds_merged[col] = ds_merged[col].ffill().fillna(0)
        ds_merged[col+"_bolus"] = ds_merged[col+"_bolus"].ffill().fillna(0)
        ds_merged[col + "_equipotent"] = ((ds_merged[col]+ds_merged[col+"_bolus"]) * va_columns_factors[col]).fillna(0)
        ds_merged["dm_vasoactive_current"] = ds_merged["dm_vasoactive_current"] + ds_merged[col + "_equipotent"]
        
    ds_merged["vasopress4h"]  = ds_merged['dm_vasoactive_current'].rolling(window=int(60/time_intervall*4), min_periods=1).max()
    
    return ds_merged.loc[:,['PatientID', 'AbsDatetime', 'dm_vasoactive_current', 'vasopress4h']]
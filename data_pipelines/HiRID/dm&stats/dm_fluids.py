import pandas as pd
import numpy as np
import polars as pl
import sys
import os
import matplotlib.pyplot as plt
from datetime import datetime

#Function to convert cummulative variables to discrete increases (only positive values)
def convert_cumulative_to_discrete(ds_merged, col_name):
    df_cont = ds_merged.select([pl.col('AbsDatetime'), pl.col(col_name)]).filter(pl.col(col_name).is_not_null()).with_columns(pl.col(col_name).diff().alias(col_name + '_diff')).filter(pl.col(col_name) >= 0).collect().to_pandas()
    if not df_cont.empty:
        df_cont.loc[0,col_name+'_diff'] = df_cont.loc[0,col_name]
        df_cont = df_cont[df_cont[col_name+'_diff'] >= 0]
        return df_cont.loc[:,['AbsDatetime',col_name+'_diff']]
    else:
        return pd.DataFrame(columns={'AbsDatetime': pd.Series(dtype='datetime64[ns]') ,col_name+'_diff': pd.Series(dtype=np.float64)})

#Function to convert flow rate to volume
def convert_flowrate_to_volume(ds_merged, col_name):
    df_cont = ds_merged.select([pl.col('AbsDatetime'), pl.col(col_name)]).filter(pl.col(col_name).is_not_null()).collect().to_pandas()
    if not df_cont.empty:
        df_cont = df_cont.resample('5min', on='AbsDatetime').mean().reset_index()
        df_cont.ffill(inplace=True)
        df_cont.loc[:,col_name+'_volume_ml_5min'] = df_cont.loc[:,col_name] / 12
        return df_cont.loc[:,['AbsDatetime',col_name+'_volume_ml_5min']]    
    else:
        return pd.DataFrame(columns={'AbsDatetime': pd.Series(dtype='datetime64[ns]') ,col_name+'_volume_ml_5min': pd.Series(dtype=np.float64)})
    
def process_patient(patient_id, ds_merged):
    #loop over all cummulative variables of interest and add them to one dataframe
    df_cont = pd.DataFrame()
    cum_columns = ['vm5096','vm5097', 'vm5098', 'vm5099']
    for col_name in cum_columns:
        if df_cont.columns.empty:
            df_cont = convert_cumulative_to_discrete(ds_merged, col_name)
        else:
            df_cont = df_cont.merge(convert_cumulative_to_discrete(ds_merged, col_name), on='AbsDatetime', how='outer')

    #loop over all flow variables of interest        
    rate_columns = ['vm5001']
    for col_name in rate_columns:
        if df_cont.columns.empty:
            df_cont = convert_flowrate_to_volume(ds_merged, col_name)
        else:
            df_cont = df_cont.merge(convert_flowrate_to_volume(ds_merged, col_name), on='AbsDatetime', how='outer')
    df_cont.fillna(0,inplace=True)

    if df_cont.empty:
        return pd.DataFrame()
    
    df_cont = df_cont.resample('5min', on='AbsDatetime').sum() #resample to 5 minutes for calculations over time
    df_cont = df_cont.assign(dm_total_in=df_cont.vm5098_diff.cumsum()) #total fluid intake so far over stay
    df_cont = df_cont.assign(dm_total_out=df_cont.vm5099_diff.cumsum()) #total fluid output so far over stay
    df_cont = df_cont.assign(cum_fluid_balance=df_cont.dm_total_in - df_cont.dm_total_out) #cumulative balance so  far over stay (difference between intake and output)
    df_cont = df_cont.assign(Iv_fluid_in_4h=df_cont.vm5098_diff.rolling(4*12, min_periods=1).sum()) #total fluid intake over last 4 hours
    df_cont = df_cont.assign(urine_out_4h=df_cont.vm5001_volume_ml_5min.rolling(4*12, min_periods=1).sum()) #total urine output over last 4 hours (volume in ml) - cave this is ffilled which is logicaly not correct but does not leak informaiton from the future
    df_cont.reset_index(inplace=True)
    df_cont['PatientID'] = patient_id
    
    return df_cont.loc[:,['PatientID', 'AbsDatetime', 'dm_total_in', 'dm_total_out', 'cum_fluid_balance', 'Iv_fluid_in_4h', 'urine_out_4h']]
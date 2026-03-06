import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime
import argparse

from pyspark.sql import functions as sf
sys.path.append('/cluster/home/faltysm/git/2021_ICUpipe/DataFrame/Common')
from spark_common import get_spark_session
from pyspark.sql.window import Window

from dm_merged_preprocessing import process_data_patient

import polars as pl

spark = get_spark_session(8, 1024, 64)

def read_reference_table(varref_path):
    """
    Read variableid-metavariableid mapping table for the merge step
    """
    varref = pd.read_csv(varref_path)
    varref.drop(varref.index[varref.variableid.isnull()], inplace=True)
    varref.loc[:, "variableid"] = varref.variableid.astype(int)
    varref.set_index("variableid", inplace=True)
    
    return varref

'''wraper loops through a set of patients'''
def process_batch(startid, stopid):
    #this goes to the patient looper
    d_monvals = spark.read.parquet(os.path.join(SOURCE_PATH, 'observation_tables'))
    d_pharma = spark.read.parquet(os.path.join(SOURCE_PATH, 'pharma_records'))
    
    varref = read_reference_table('intellilung_varref_hirid.csv')
    varref.metavariableid = varref.metavariableid.astype(int)
    df_admissions = []
    output_cols = ["PatientID", "AbsDatetime"] + [f"vm{vid}" for vid in sorted(varref['metavariableid'].unique())] + [f"vm{vid}_bolus" for vid in sorted(varref.loc[varref.type == 'pharma', 'metavariableid'].unique())]

    for pid in range(startid, stopid + 1):
        print (pid)
        observed = d_monvals.filter(sf.col(PID) == pid).where(sf.col('variableid').isin(varref[varref.type == "observed"].index.values.tolist())).withColumn("invalidated", sf.substring(sf.bin(sf.col("status")),-2,1).cast('int')).filter("invalidated==0").toPandas()
        pharma = d_pharma.filter(sf.col(PID) == pid).where(sf.col('pharmaid').isin(varref[varref.type == "pharma"].index.values.tolist())).withColumn("started", sf.substring(sf.bin(sf.col("recordstatus")),-3,1).cast('int')).withColumn("stoped", sf.substring(sf.bin(sf.col("recordstatus")),-9,1).cast('int')).toPandas()
        
        df_pid = process_data_patient(pid, observed, pharma, varref)
        df_pid.loc[:, list(set(output_cols).difference(set(df_pid.columns)))] = np.nan
        df_admissions.append(df_pid)

    df = pd.concat(df_admissions).sort_values(["PatientID", "AbsDatetime"])  
    df = df[output_cols]

    # explicitly setting type to be consistent across chunks (important to process using pyspark)
    df["PatientID"] = df["PatientID"].astype('int32')
    print (df.columns)
    df.iloc[:, 2:]= df.iloc[:, 2:].astype('float64')

    output_dir_results = os.path.join(OUTPUT_DIR, datetime.today().strftime('%Y-%m-%d')) 

    if not os.path.exists(output_dir_results): 
        os.makedirs(output_dir_results)

    if not df.empty:
        p_df = pl.from_pandas(df)
        p_df.write_parquet(os.path.join(output_dir_results, "merged_{}_{}.parquet".format(startid,stopid)))

if __name__=="__main__": 
    parser=argparse.ArgumentParser()
    # CONSTANTS 
    OUTPUT_DIR = '/cluster/work/grlab/clinical/hirid2/research/faltysm/Intellilung/dm_merged'
    LOG_DIR="/cluster/work/grlab/clinical/hirid2/research/faltysm/ICU_pipe/logs"
    SOURCE_PATH="/cluster/work/grlab/clinical/hirid_public/v1.1.1"
    VARID = 'variableid'
    PID = 'patientid'
    VALUE = 'value'
    DATETIME = 'datetime'

    output_dir_results = os.path.join(OUTPUT_DIR, datetime.today().strftime('%Y-%m-%d'))  
    
    # Input paths
    parser.add_argument("--pid_start", help="First batch id to be processed", type=int)
    parser.add_argument("--pid_stop", help="Last batch id to be processed", type=int)
    parser.add_argument("--run_mode", default="INTERACTIVE", help="Should job be run in batch or interactive mode")
    parser.add_argument("--output_path", default=OUTPUT_DIR, help="Path to store results")

    args=parser.parse_args()
    assert(args.run_mode in ["CLUSTER", "INTERACTIVE"]) 

    if args.run_mode=="CLUSTER":
        sys.stdout=open(os.path.join(LOG_DIR,"{}_Volume_{}_{}.stdout".format(datetime.today().strftime('%Y-%m-%d'), args.pid_start,args.pid_stop)),'w')
        sys.stderr=open(os.path.join(LOG_DIR,"{}_Volume_{}_{}.stderr".format(datetime.today().strftime('%Y-%m-%d'), args.pid_start,args.pid_stop)),'w')

    process_batch(args.pid_start,args.pid_stop)

    print ("success")
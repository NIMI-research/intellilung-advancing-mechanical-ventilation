import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime
import argparse
import polars as pl

from dm_fluids import process_patient

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
    df_collection = []
    for pid in range(startid,stopid+1):
        print(f"Processing patient {pid}")
        ds_merged = pl.scan_parquet(os.path.join(SOURCE_PATH, 'dm_merged', '2023-10-08', '*.parquet')).filter((pl.col('PatientID')==pid))
        df_collection.append(process_patient(pid, ds_merged))

    df = pd.concat(df_collection).sort_values(["PatientID", "AbsDatetime"])

    # explicitly setting type to be consistent across chunks (important to process using pyspark)
    df["PatientID"] = df["PatientID"].astype('int32')
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
    OUTPUT_DIR = '/cluster/work/grlab/clinical/hirid2/research/faltysm/Intellilung/dm_fluid'
    LOG_DIR="/cluster/work/grlab/clinical/hirid2/research/faltysm/ICU_pipe/logs"
    SOURCE_PATH="/cluster/work/grlab/clinical/hirid2/research/faltysm/Intellilung" 
    
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
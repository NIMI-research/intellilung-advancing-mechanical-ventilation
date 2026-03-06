import pandas as pd
import numpy as np
import polars as pl
import sys
import os
from datetime import datetime
import argparse

 
def process_column(columns, block_id):
    ds_merged = pl.scan_parquet(os.path.join(SOURCE_PATH, 'dm_merged', 'latest', '*.parquet'))
    ds_fluid = pl.scan_parquet(os.path.join(SOURCE_PATH, 'dm_fluid', 'latest', '*.parquet'))
    ds_vasoactive = pl.scan_parquet(os.path.join(SOURCE_PATH, 'dm_vasoactive', 'latest', '*.parquet'))
    ds_ventilation = pl.scan_parquet(os.path.join(SOURCE_PATH, 'dm_ventilation', 'latest', '*.parquet'))

    ds_joined = ds_merged.join(ds_fluid, on=["PatientID", "AbsDatetime"], how="outer")
    ds_joined = ds_joined.join(ds_vasoactive, on=["PatientID", "AbsDatetime"], how="outer")
    ds_joined = ds_joined.join(ds_ventilation, on=["PatientID", "AbsDatetime"], how="outer")

    Npata = {}
    Npatap = {}
    Tavar1 = {}
    Tavar2 = {}
    Tavar3 = {}
    Trvar1 = {}
    Trvar2 = {}
    Trvar3 = {}
    Trvarend1 = {}
    Trvarend2 = {}
    Trvarend3 = {}
    Nppat1 = {}
    Nppat2 = {}
    Nppat3 = {}

    #patient count in DS
    Npat = ds_joined.select(['PatientID']).collect().n_unique()

    #patient lenght of stay and admission and discharge time based on first and last HR measurement
    ds_first_hr = ds_joined.filter(pl.col("vm2001").is_not_null()).select(pl.col(['PatientID','AbsDatetime'])).rename({"AbsDatetime": "time_of_admission"}).groupby(pl.col('PatientID')).first()
    ds_last_hr = ds_joined.filter(pl.col("vm2001").is_not_null()).select(pl.col(['PatientID','AbsDatetime'])).rename({"AbsDatetime": "time_of_discharge"}).groupby(pl.col('PatientID')).last()
    ds_los = ds_first_hr.join(ds_last_hr, on="PatientID").with_columns((pl.col("time_of_discharge") - pl.col("time_of_admission")).cast(pl.Float32, strict=False).alias("los")).select(pl.col(["PatientID", "los", "time_of_admission", "time_of_discharge"]))

    for col in columns:
        print(col)
        #calculate simple value quantiles
        summary_statistics = pl.concat(
            [
            ds_joined.select(col).quantile(0.1), #values 0.1 quantile
            ds_joined.select(col).quantile(0.5), #values 0.5 quantile
            ds_joined.select(col).quantile(0.9) #values 0.9 quantile
            ]
        ).collect().transpose(include_header=True).rename({"column_0": "Val1", "column_1": "Val2", "column_2": "Val3"})
        
        #first and last measurements, with no measurements considert before and after the admission/discharge
        ds_first = ds_joined.filter(pl.col(col).is_not_null()).select(pl.col(['PatientID','AbsDatetime'])).rename({"AbsDatetime": "firstValue_tmp"}).groupby(pl.col('PatientID')).first()
        ds_last = ds_joined.filter(pl.col(col).is_not_null()).select(pl.col(['PatientID','AbsDatetime'])).rename({"AbsDatetime": "lastValue_tmp"}).groupby(pl.col('PatientID')).last()
        ds_count = ds_joined.filter(pl.col(col).is_not_null()).select(pl.col(['PatientID','AbsDatetime'])).groupby(pl.col('PatientID')).count()
        ds_timediff = ds_first.join(ds_last, on="PatientID").join(ds_los, on="PatientID").join(ds_count, on="PatientID")
        ds_timediff = ds_timediff.with_columns(pl.min(["time_of_discharge", pl.col("lastValue_tmp")]).alias("lastValue"))
        ds_timediff = ds_timediff.with_columns(pl.max(["time_of_admission", pl.col("firstValue_tmp")]).alias("firstValue"))

        #time intervall between first and last
        ds_timediff = ds_timediff.with_columns((pl.col("lastValue") - pl.col("firstValue")).cast(pl.Float32, strict=False).alias("time_diff"))

        #time intervall between last measurement and discharge
        ds_timediff = ds_timediff.join(ds_los, on="PatientID").with_columns((pl.col("time_of_discharge") - pl.col("lastValue")).cast(pl.Float32, strict=False).alias("time_before_discharge"))

        #above time intervalls in percent
        ds_timediff = ds_timediff.join(ds_los, on="PatientID").with_columns((pl.col("time_diff") / pl.col("los")).cast(pl.Float32, strict=False).alias("time_diff_rel"))
        ds_timediff = ds_timediff.join(ds_los, on="PatientID").with_columns((pl.col("time_before_discharge") / pl.col("los")).cast(pl.Float32, strict=False).alias("time_before_discharge_rel"))

        #select columns and collect
        ds_timediff = ds_timediff.select("time_diff", "time_diff_rel", "time_before_discharge_rel", "count").collect()

        Tavar1[col]=ds_timediff["time_diff"].quantile(0.25)
        Tavar2[col]=ds_timediff["time_diff"].quantile(0.50)
        Tavar3[col]=ds_timediff["time_diff"].quantile(0.75)

        Trvar1[col]=ds_timediff["time_diff_rel"].quantile(0.25)
        Trvar2[col]=ds_timediff["time_diff_rel"].quantile(0.50)
        Trvar3[col]=ds_timediff["time_diff_rel"].quantile(0.75)

        Trvarend1[col]=ds_timediff["time_before_discharge_rel"].quantile(0.25)
        Trvarend2[col]=ds_timediff["time_before_discharge_rel"].quantile(0.50)
        Trvarend3[col]=ds_timediff["time_before_discharge_rel"].quantile(0.75)

        Nppat1[col]=ds_timediff["count"].quantile(0.25)
        Nppat2[col]=ds_timediff["count"].quantile(0.50)
        Nppat3[col]=ds_timediff["count"].quantile(0.75)

        for cl in [Tavar1,Tavar2,Tavar3]:
            if cl[col] != None:
                cl[col]=cl[col]/3600/1000000000
        
    col_names = ["Tavar1", "Tavar2" ,"Tavar3" , "Trvar1", "Trvar2", "Trvar3", "Trvarend1", "Trvarend2", "Trvarend3", "Nppat1", "Nppat2", "Nppat3"]
    i=0
    for cl in [Tavar1, Tavar2 ,Tavar3 , Trvar1, Trvar2, Trvar3, Trvarend1, Trvarend2, Trvarend3, Nppat1, Nppat2, Nppat3]:
        df = pl.from_dict(cl).transpose().rename({"column_0": col_names[i]})
        summary_statistics = pl.concat([summary_statistics, df], how="horizontal")
        i = i + 1

    output_dir_results = os.path.join(OUTPUT_DIR, datetime.today().strftime('%Y-%m-%d')) 

    if not os.path.exists(output_dir_results): 
        os.makedirs(output_dir_results)

    summary_statistics = summary_statistics.select([
        summary_statistics.columns[0],  # Keep the first column as it is
        *[summary_statistics[col].cast(pl.Float64) for col in summary_statistics.columns[2:]]  # Cast the remaining columns to float64
    ])
    summary_statistics.write_parquet(os.path.join(output_dir_results, "statistics_{}.parquet".format(block_id)))


if __name__=="__main__": 
    parser=argparse.ArgumentParser()
    # CONSTANTS 
    OUTPUT_DIR = '/cluster/work/grlab/clinical/hirid2/research/faltysm/Intellilung/statistics'
    LOG_DIR="/cluster/work/grlab/clinical/hirid2/research/faltysm/ICU_pipe/logs"
    SOURCE_PATH="/cluster/work/grlab/clinical/hirid2/research/faltysm/Intellilung" 
    
    # Input paths
    parser.add_argument("--column", help="First batch id to be processed", type=str)
    parser.add_argument("--blockid", help="id number unique", type=int)
    parser.add_argument("--run_mode", default="INTERACTIVE", help="Should job be run in batch or interactive mode")
    parser.add_argument("--output_path", default=OUTPUT_DIR, help="Path to store results")

    args=parser.parse_args()
    assert(args.run_mode in ["CLUSTER", "INTERACTIVE"]) 

    if args.run_mode=="CLUSTER":
        sys.stdout=open(os.path.join(LOG_DIR,"{}_statistics_{}.stdout".format(datetime.today().strftime('%Y-%m-%d'), args.blockid)),'w')
        sys.stderr=open(os.path.join(LOG_DIR,"{}_statistics_{}.stderr".format(datetime.today().strftime('%Y-%m-%d'), args.blockid)),'w')

    process_column([args.column],args.blockid)

    print ("success")




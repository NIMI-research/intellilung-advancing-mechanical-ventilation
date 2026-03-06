import subprocess
import os
import os.path
import numpy as np
import polars as pl

SOURCE_PATH="/cluster/work/grlab/clinical/hirid2/research/faltysm/Intellilung"
ds_merged = pl.scan_parquet(os.path.join(SOURCE_PATH, 'dm_merged', 'latest', '*.parquet'))
ds_fluid = pl.scan_parquet(os.path.join(SOURCE_PATH, 'dm_fluid', 'latest', '*.parquet'))
ds_vasoactive = pl.scan_parquet(os.path.join(SOURCE_PATH, 'dm_vasoactive', 'latest', '*.parquet'))
ds_ventilation = pl.scan_parquet(os.path.join(SOURCE_PATH, 'dm_ventilation', 'latest', '*.parquet'))

ds_joined = ds_merged.join(ds_fluid, on=["PatientID", "AbsDatetime"], how="outer")
ds_joined = ds_joined.join(ds_vasoactive, on=["PatientID", "AbsDatetime"], how="outer")
ds_joined = ds_joined.join(ds_ventilation, on=["PatientID", "AbsDatetime"], how="outer")

i = 0
col_selection = [col for col in ds_joined.columns if col not in ['PatientID', 'AbsDatetime']]
for col in col_selection:
    i = i + 1
    LOG_DIR="/cluster/work/grlab/clinical/hirid2/research/faltysm/ICU_pipe/logs"
    job_name="statistics_{}".format(col)
    mem_in_mbytes = 8000
    n_cpu_cores = 2
    n_compute_hours = 4

    compute_script_path="/cluster/home/faltysm/git/2021_ICUpipe/DataFrame/HiRID/IntelliLung/statistics.py"

    log_result_file=os.path.join(LOG_DIR, "{}_RESULT.txt".format(job_name))
    log_error_file=os.path.join(LOG_DIR, "{}_ERROR.txt".format(job_name))

    subprocess.call(["source activate ds_p38_base"],shell=True)



    cmd_line=" ".join(["sbatch", "--mem-per-cpu {}".format(mem_in_mbytes), 
                                   "-n", "{}".format(n_cpu_cores),
                                   "--time", "{}:00:00".format(n_compute_hours),
                                   "--mail-type FAIL",
                                   "--exclude compute-biomed-15",
                                   "--job-name","{}".format(job_name), "-o", log_result_file, "-e", log_error_file, "--wrap",
                                   '\"python3', compute_script_path, "--run_mode CLUSTER",
                                   "--blockid {}".format(i), 
                                   "--column {}".format(col), '\"'])
     
    print (cmd_line)
    subprocess.call([cmd_line], shell=True)

print ("number of jobs: " + str(i))
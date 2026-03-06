import os
import time

import pandas as pd

from tqdm import tqdm
from common.config import settings

import common.utility.utility as util
import common.knowledge_base.kbase as kb


def load_data_from_csvs():
    start_time = time.perf_counter()

    pb = tqdm(desc="Loading files", total=5)
    data_path = settings.input_data_path

    demo = pd.read_csv(os.path.join(data_path, "mimiciv", f"mimiciv_demo.csv"), sep="|", header=0)
    pb.update()

    ventilation = pd.read_csv(os.path.join(data_path, "mimiciv", f"mimiciv_ventilation.csv"), sep="|", header=0)
    pb.update()

    inputevents = pd.read_csv(os.path.join(data_path, "mimiciv", f"mimiciv_inputs.csv"), sep="|", header=0)
    pb.update()

    outputevents = pd.read_csv(os.path.join(data_path, "mimiciv", f"mimiciv_outputs.csv"), sep="|", header=0)
    pb.update()

    chartevents = pd.read_csv(
        os.path.join(data_path, "mimiciv", f"mimiciv_chartevents.csv"),
        sep="|",
        header=0,
        dtype={"stay_id": "int32", "charttime": "float64", "value": "object", "valueuom": "object", "itemid": "int32"},
    )
    pb.update()

    pb.close()
    print(f"Loading took: {time.perf_counter()-start_time} seconds.")

    print("Reducing memory requirement for tables...", end=" ")
    chartevents = chartevents.astype(kb.reduce_memory_based_on_column(chartevents.columns))
    inputevents = inputevents.astype(kb.reduce_memory_based_on_column(inputevents.columns))
    outputevents = outputevents.astype(kb.reduce_memory_based_on_column(outputevents.columns))
    util.pdone()

    return chartevents, demo, ventilation, inputevents, outputevents


def load_data(conn):
    start_time = time.perf_counter()

    pb = tqdm(desc="Loading files", total=5)
    data_path = settings.input_data_path

    # Loading demographics
    demo = util.read_sql_query(query_file="base/mimiciv_demographics.sql", connection=conn)
    pb.update()

    # Loading ventilation data which contains times when Mechanical Ventilation was started
    ventilation = util.read_sql_query(query_file="base/mimiciv_ventilation.sql", connection=conn)
    pb.update()

    inputevents = util.read_sql_query(query_file="generated/mimiciv_inputevents_query.sql", connection=conn)
    pb.update()

    outputevents = util.read_sql_query(query_file="generated/mimiciv_outputevents_query.sql", connection=conn)
    pb.update()

    ingredientevents = util.read_sql_query(query_file="generated/mimiciv_ingredientevents_query.sql", connection=conn)
    # Loading chartevents from postgres
    chartevents = util.read_sql_query(query_file="generated/mimiciv_chartevents_query.sql", connection=conn)  # loads around 1 GB of data
    pb.update()

    pb.close()
    print(f"Loading took: {time.perf_counter()-start_time} seconds.")

    if not os.path.exists(os.path.join(settings.input_data_path, "mimiciv")):
        print("Creating a folder for csv files...", end=" ")
        os.makedirs(os.path.join(settings.input_data_path, "mimiciv"))
        util.pdone()
    
    print("Saving to csv...", end=" ")
    demo.to_csv(os.path.join(data_path, "mimiciv", f"mimiciv_demo.csv"), sep="|", header=demo.columns, index=False)

    ventilation.to_csv(os.path.join(data_path, "mimiciv", f"mimiciv_ventilation.csv"), sep="|", header=ventilation.columns, index=False)

    inputevents.to_csv(os.path.join(data_path, "mimiciv", f"mimiciv_inputs.csv"), sep="|", header=inputevents.columns, index=False)

    outputevents.to_csv(os.path.join(data_path, "mimiciv", f"mimiciv_outputs.csv"), sep="|", header=outputevents.columns, index=False)

    chartevents = pd.concat([chartevents, ingredientevents], ignore_index=True)
    chartevents.to_csv(os.path.join(data_path, "mimiciv", f"mimiciv_chartevents.csv"), sep="|", header=chartevents.columns, index=False)
    util.pdone()

    print("Reducing memory requirement for tables...", end=" ")
    chartevents = chartevents.astype(kb.reduce_memory_based_on_column(chartevents.columns))
    inputevents = inputevents.astype(kb.reduce_memory_based_on_column(inputevents.columns))
    outputevents = outputevents.astype(kb.reduce_memory_based_on_column(outputevents.columns))
    util.pdone()

    return chartevents, demo, ventilation, inputevents, outputevents

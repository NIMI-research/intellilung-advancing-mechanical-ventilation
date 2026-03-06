import os
import psycopg2
from time import time

import pandas as pd

def query_db(startTime, sourcepath):
    # full queries take multiple hours to run
    conn = psycopg2.connect(dbname=os.getenv("DBNAME"), user=os.getenv("DBUSER"),
                    options=os.getenv('DBOPTIONS'))
    conn.autocommit = True
    print("Querying database")
    run_queries(conn, "eicu/sql", sourcepath)
    print("Querying db took", str(time() - startTime), "seconds")

def run_queries(conn, sql_path, out_dir):
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    for filename in os.listdir(sql_path):
        if filename.endswith(".sql"):
            print(filename)
            with open(os.path.join(sql_path, filename), "r") as f:
                result = pd.read_sql_query(f.read(), conn)
                result.to_csv(os.path.join(out_dir, os.path.splitext(
                    filename)[0]+".csv"), index=False)
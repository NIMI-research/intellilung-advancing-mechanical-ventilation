import os
import psycopg2

from common.config import settings

import common.utility as util
import common.knowledge_base.kbase as kb

import data_cleaning.data_loading as dl


"""
STEPS IN THE DATA EXTRACTION PIPELINE:
    1) Data cleaning & Standardization
        - Extract data about patients who have been to the ICU (~40k patients)
        - Extract patients who received mechanical ventilation
        - Remove Null entries
"""


def main():

    with psycopg2.connect(dbname=settings.mimic_dbname, user=settings.postgres_user,
                          password=settings.postgres_pass) as conn:
        # --------------------------------------------------------
        # ------------ SETTING UP KNOWLEDGE BASE DATA ------------
        # --------------------------------------------------------
        # Loading d_items table which contains all chartevent labels and item_ids
        d_items = util.read_sql_query(query_file="base/mimiciv_ditems.sql", connection=conn)

        # Regenerating important dicts in case of a change. This part of the code loads the variables excel file and based on that
        # produces dictionaries for the knowledge base.
        if settings.setup:
            print("Generating dictionaries for knowledge base...", end=" ")

            util.generate_mv_unified_vars(
                d_items=d_items,
                path_to_excel_file=os.path.join(settings.input_data_path, "mimiciv",
                                                "actions-rewards-variable-availability.xlsx"),
                path_to_sql_file=settings.path_to_sql_queries,
            )
            util.generate_dicts(
                d_items=d_items, path_to_excel_file=os.path.join(settings.input_data_path, "mimiciv",
                                                                 "actions-rewards-variable-availability.xlsx"),
                conn=conn
            )
            kb.regenerate_knowledge_base()
            util.pdone()

        # --------------------------------------------------------
        # -------------------- SAVING DATA ----------------------
        # --------------------------------------------------------
        # Loading measurements, demographics and procedures
        dl.load_data(conn=conn)


if __name__ == "__main__":
    main()

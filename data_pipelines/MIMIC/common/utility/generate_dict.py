import os
import json

import numpy as np
import pandas as pd

import common.utility as util
import common.knowledge_base.kbase as kb

from common.config import settings


def save_to_knowledge_base_dicts(data: dict, name: str) -> None:
    with open(os.path.join(kb.path_to_kb, f"dicts/{name}.json"), "w") as f:
        for key in data:
            if isinstance(data[key], np.ndarray):
                data[key] = data[key].tolist()
        json.dump(data, f, indent=4)


def load_basis_table(path_to_excel_file):
    df = pd.read_excel(
        path_to_excel_file, 
        header=0, 
        usecols=["variables names", "MIMIC\n(itemid)", "table/category", "LOINC"], 
        sheet_name="Variables MIMIC IV"
    )
    prior_df = pd.read_excel(
        path_to_excel_file,
        header=0,
        usecols=[
            "full naming schema",
            "Variables names",
            "Threshold low",
            "Threshold high",
            "sah period",
            "Set",
            "Priority",
            "LOINC",
            "target unit",
        ],
        sheet_name="Variables index",
    )

    # Rename columns for ease of use
    df = df.rename(columns={"variables names": "var_name", "MIMIC\n(itemid)": "itemid"})
    prior_df = prior_df.rename(
        columns={
            "full naming schema": "full_label",
            "Variables names": "var_name",
            "Threshold low": "th_low",
            "Threshold high": "th_high",
            "sah period": "sah",
            "Set": "set",
            "Priority": "prior",
            "target unit": "target_unit",
        }
    )
    # Create new dataframe to hold threshold and sah info
    additional_info = prior_df.loc[prior_df["full_label"].notna(), ["full_label", "th_low", "th_high", "sah", "set", "target_unit"]].copy()

    # Fill missing values in 'new_label' and 'var_name' columns with the previous non-null value
    df["var_name"].fillna(method="ffill", inplace=True)
    # Get rid of blank spaces at the beginning and end of LOINC code string
    df["LOINC"] = df["LOINC"].str.strip()

    # Fill missing values in 'new_label' and 'var_name' columns with the previous non-null value
    prior_df.loc[:, ["full_label", "var_name"]] = prior_df.loc[:, ["full_label", "var_name"]].fillna(method="ffill")
    # Replace NaN values in 'prior' column with 0
    prior_df["prior"].replace(np.nan, 0, inplace=True)

    # Drop rows where 'LOINC' is NaN
    prior_df.drop(prior_df.loc[prior_df["LOINC"].isna()].index, inplace=True)
    prior_df.reset_index(drop=True, inplace=True)
    # Get rid of blank spaces at the beginning and end of LOINC code string
    prior_df["LOINC"] = prior_df["LOINC"].str.strip()

    # Convert 'itemid' column to numeric, replace NaN with 0, and cast to int32
    df["itemid"] = pd.to_numeric(df["itemid"], errors="coerce", downcast="integer").replace(np.nan, 0).astype("int32")

    # Drop rows where 'table/category' starts with "Scores"
    df.drop(df.loc[df["table/category"].str.startswith("Scores").fillna(False)].index, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Merge the two dataframes based on the 'LOINC' column
    df = df.merge(prior_df[["full_label", "prior", "LOINC"]], on=["LOINC"], how="left")

    # Drop rows where 'prior' is NaN and 'itemid' is not 0
    df.drop(df.loc[df["prior"].isna() & (df["itemid"] != 0)].index, inplace=True)
    # Fill remaining NaN values in 'prior' with 0
    df.loc[df["prior"].isna(), "prior"] = 0
    # Convert 'prior' column to numeric and downcast to integer
    df["prior"] = pd.to_numeric(df["prior"], downcast="integer", errors="coerce")

    # Update 'var_name' for rows where 'new_label' is not NaN
    ind = df.loc[df["full_label"].notna()].index
    df.loc[ind, "var_name"] = df.loc[ind, "full_label"].values

    return df, additional_info


# Adjust the code so that it works for each of the tables that the itemids in the excel table connect to
def get_linked_table_names(d_items: pd.DataFrame, df: pd.DataFrame):
    tables = {
        "chartevents": [],
        "inputevents": [],
        "outputevents": [],
        "ingredientevents": [],
        "datetimeevents": []
    }
    for itemid in df.loc[df["itemid"] > 0, "itemid"]:
        table = d_items.loc[d_items["itemid"] == itemid, "linksto"].values[0]

        if table not in tables.keys():
            continue

        tables[table].append(itemid)

    return tables


def generate_mv_unified_vars(d_items: pd.DataFrame, path_to_excel_file, path_to_sql_file):
    # Base table with variable names, itemids, priorities, category, LIONC code
    df, _ = load_basis_table(path_to_excel_file=path_to_excel_file)

    # Dict with tablenames and lists of itemids belonging to them
    tables = get_linked_table_names(d_items=d_items, df=df)

    # The two row formats that will be used to edit the query templates
    row_format = "when A.itemid in ({}) then '{}'"
    prior_row_format = "when A.itemid = {} then {}"

    # Creating a dict with tablenames linking to their respective SQL queries
    query_dict = {}
    for table in tables:
        with open(os.path.join(path_to_sql_file, "templates", f"mimiciv_{table}_base_template.sql"), "r") as f:
            sql_query = f.read()
            query_dict.update({table: sql_query})

    var_dict = {}
    item_id_list = {table: [] for table in tables.keys()}

    for var in df["var_name"].unique():
        item_ids = df.loc[df["var_name"] == var, "itemid"].values
        item_ids = item_ids[item_ids > 0]
        # Listing all measurement names that are included within a variable
        var_dict.update({var: [d_items.at[d_items.loc[d_items["itemid"] == item_id, :].index[0], "label"] for item_id in item_ids]})

        if np.size(item_ids[item_ids > 0]) == 0:
            continue

        # Adding to SQL quries for each concerned table from the listed itemids
        for table in query_dict:
            temp = set(tables[table]) & set(item_ids)
            if not temp:
                continue

            # Adding priorities
            for itemid in temp:
                insert_pos = query_dict[table].find("-- insert itemids and priorities here")
                query_dict[table] = (
                    query_dict[table][:insert_pos]
                    + prior_row_format.format(itemid, df.loc[df["itemid"] == itemid, "prior"].values[0])
                    + "\n"
                    + "   " * 8
                    + query_dict[table][insert_pos:]
                )

            # Inserting the variable names conresponding to a set of itemids
            temp = ",".join(map(str, temp))

            insert_pos = query_dict[table].find("-- insert itemids and names here")
            query_dict[table] = query_dict[table][:insert_pos] + row_format.format(temp, var) + "\n" + "   " * 8 + query_dict[table][insert_pos:]

            # The item_id_list is appended in such a way so that the final list of IDs takes a new line for each different variable
            item_id_list[table].append(temp)

    join_str = ",\n" + "   " * 8

    # Check if the folder for generated queries exists and if not create it
    generated_folder = os.path.join(settings.path_to_sql_queries, "generated")
    if not os.path.exists(generated_folder):
        os.makedirs(generated_folder)
    
    # Adding a set of itemids to look through
    for table in query_dict:
        table_item_id_list = join_str.join(item_id_list[table])
        insert_pos = query_dict[table].find("-- insert just itemids here")
        query_dict[table] = query_dict[table][:insert_pos] + table_item_id_list + "\n" + "   " * 8 + query_dict[table][insert_pos:]

        with open(os.path.join(settings.path_to_sql_queries, "generated", f"mimiciv_{table}_query.sql"), "w") as f:
            f.write(query_dict[table])

    save_to_knowledge_base_dicts(var_dict, "mv_unified_vars")
    kb.mv_reqs = kb.MechanicalVent()

    return


def generate_dicts(d_items: pd.DataFrame, path_to_excel_file, conn):
    """Generates two dictionaries based on the unified_vars dictionary and the d_items table.
    1) mv_relevant_varables -- a dictionary containing measurement labels and the parameters of that measurement (itemid, priority, unit and table).
    2) vector_variables -- a dictionary containing all the variables that should be included in the state vectors.
    """

    var_dict = {}
    uni_var_info = {}
    prior_df, info = load_basis_table(path_to_excel_file=path_to_excel_file)
    tables = get_linked_table_names(d_items, prior_df)

    # The unified_vars dictionary must already exists for the code to work.
    # The mv_relevant_variables and vector_variables are made here
    duplicates = {}

    for uni_var in kb.mv_reqs.unified_vars:
        if (uni_var in info["full_label"].values) and pd.notna(info.loc[info["full_label"] == uni_var, "th_low"].values[0]):
            uni_var_info.update(
                {
                    uni_var: [
                        util.process_input_to_float(info.loc[info["full_label"] == uni_var, "th_low"].values[0]),
                        util.process_input_to_float(info.loc[info["full_label"] == uni_var, "th_high"].values[0]),
                    ]
                }
            )
        for key in kb.mv_reqs.unified_vars[uni_var]:
            row = d_items.loc[d_items["label"] == key, :]

            value = row["itemid"].values
            if len(value) > 1:
                if key not in duplicates:
                    num = len(value)
                    duplicates.update({key: num})
                else:
                    duplicates[key] -= 1
                    num = duplicates[key]

                print(f"There are multiple id codes for label {key}. It will be renamed to {key + f'-{num}'}.")
                key = key + f"-{num}"
            else:
                num = 1

            var_dict.update(
                {
                    key: {
                        "itemid": int(value[num - 1]),
                        "var": uni_var,
                        "unit": row["unitname"].values[num - 1] if row["unitname"].values[num - 1] != "None" else None,
                        "th_low": util.process_input_to_float(info.loc[info["full_label"] == uni_var, "th_low"].values[0]),
                        "th_high": util.process_input_to_float(info.loc[info["full_label"] == uni_var, "th_high"].values[0]),
                        "prior": int(prior_df.loc[prior_df["itemid"] == int(value[num - 1]), "prior"].values[0]),
                        "set": info.loc[info["full_label"] == uni_var, "set"].values[0],
                        "table": next((table for table in tables if int(value[num - 1]) in tables[table]), None),
                    }
                }
            )

    save_to_knowledge_base_dicts(uni_var_info, "variable_outlier_ranges")

    # Extracting existing variables
    df_list = []
    with conn.cursor() as cur:
        for table in tables:
            query = f"""
            select distinct A.itemid
                from mimiciv_icu.{table} A 
                    where A.itemid in {tuple(tables[table])} ;"""

            cur.execute(query)
            data = cur.fetchall()
            cols = []
            for col_desc in cur.description:
                cols.append(col_desc[0])

            df_list.append(pd.DataFrame(data=data, columns=cols))

    df = pd.concat(df_list, ignore_index=True)

    # Dropping keys that don't appear in chartevents
    keys_to_drop = []
    for key in var_dict:
        if var_dict[key]["itemid"] not in df["itemid"].values:
            keys_to_drop.append(key)

    for key in keys_to_drop:
        var_dict.pop(key)

    vector_vars = ["stay_id", "mv_id", "timepoints", "daemo_age"]
    vector_vars.extend(kb.mv_reqs.unified_vars.keys())
    vec_dict = {"var_names": vector_vars}

    save_to_knowledge_base_dicts(vec_dict, "vector_variables")
    save_to_knowledge_base_dicts(var_dict, "mv_relevant_vars")

    unit_dict = {}
    for key in kb.mv_reqs.unified_vars:
        if len(kb.mv_reqs.unified_vars[key]) == 0:
            continue
        unit_dict.update(
            {
                key: {
                    "initial": list(set([var_dict[var]["unit"] for var in kb.mv_reqs.unified_vars[key] if var in var_dict])),
                    "target": info.loc[info["full_label"] == key, "target_unit"].replace(np.nan, None).values[0],
                }
            }
        )
        if key in unit_dict:
            if len(unit_dict[key]) == 0:
                unit_dict[key] = [None]

    save_to_knowledge_base_dicts(unit_dict, "mv_unified_vars_units")

    generate_categ_dicts()


def generate_categ_dicts():
    # Making dictionaries that map categorical values (N different categories) to a minimal set of values (X, where X << N)
    cat_dict_def = pd.read_excel(os.path.join(settings.input_data_path, "mimiciv", "Dict_cath_variables.xlsx"), sheet_name="MIMIC_IV")

    cat_dict_def["category"] = pd.to_numeric(cat_dict_def["category"], errors="coerce")

    cat_dict_def["full_name"].ffill(inplace=True)
    for categ in cat_dict_def["full_name"].unique():
        categ_dict = {key: int(cat) for key, cat in cat_dict_def.loc[cat_dict_def["full_name"] == categ, ["value_str", "category"]].values if pd.notna(cat)}
        kb.categorical_data_dicts.update({categ: categ_dict})
        util.save_to_knowledge_base_dicts({categ: categ_dict}, f"categ/{categ}_categorical")

import psycopg2
import pandas as pd
import numpy as np
import os
import re


def gen_varrow_sql(cohort_list):
    varrow_tables = {
        "customLab": ["labothername", "labotheroffset", "labothervaluetext"],
        "lab": ["labname", "labResultOffset", "labResultText"],
        # largest table
        "nurseCharting": ["nursingchartcelltypevallabel", "nursingChartEntryOffset", "nursingChartValue"],
        "respiratoryCharting": ["respchartvaluelabel", "respChartEntryOffset", "respChartValue"],
        "physicalExam": ["physicalexamvalue", "physicalexamoffset", "physicalexamtext"],
        "infusionDrug": ["drugname", "infusionoffset", "drugrate"],
        "intakeOutput": ["cellpath", "intakeoutputentryoffset", "cellvaluenumeric"],
        "pastHistory": ["pasthistoryvaluetext", "pasthistoryenteredoffset", "pasthistoryvalue"],
        "nurseCare": ["cellattribute", "nursecareentryoffset", "cellattributevalue"]
    }

    for table, table_cols in varrow_tables.items():
        print(table)
        table_vars = df[df["table"] == table]
        if len(table_vars) > 0:
            if table == "nurseCharting":
                var_val_dict = table_vars[["eICU", "variable_column", "value_column"]].set_index(
                    'eICU').apply(tuple, axis=1).to_dict()
            thegroup = table_vars.groupby('eICU')['LOINC']
            # check if more than 1 loinc for var
            loincs_per_drugname = thegroup.nunique()
            a = loincs_per_drugname[loincs_per_drugname > 1]
            if len(a) > 0:
                print("more than 1 loinc for var")
                print(a)
            id_table_mapping = thegroup.apply(list).to_dict()
            cols = table_cols
            var_loinc_list = []
            loinc_prio_case_list = []
            var_name_list = []
            var_unit_list = []
            for var in id_table_mapping.keys():
                var_col = var_val_dict[var][0] if table == "nurseCharting" else cols[0]
                var_loinc_list.append(
                    f"WHEN {var_col} = '{var}' THEN '{id_table_mapping[var][0]}'")
                var_name_list.append(
                    f"WHEN {var_col} = '{var}' THEN '{var_name_mapping[var][0]}'")
                # unit assignment from excel
                unit = unit_mapping[var][0]
                if unit == "nan":
                    unit = ""
                var_unit_list.append(
                    f"WHEN {var_col} = '{var}' THEN '{unit}'")
                if id_table_mapping[var][0] in loinc_prio_mapping:
                    try:
                        prio = int(
                            loinc_prio_mapping[id_table_mapping[var][0]][0])
                        loinc_prio_case_list.append(
                            f"WHEN {var_col} = '{var}' THEN {prio}")
                    except ValueError:
                        print("No prio for", id_table_mapping[var][0])
                        pass

            if len(var_name_list) > 0:
                var_name_cases = " ".join(var_name_list)
                var_name_case_query = f", CASE {var_name_cases} END AS variable"
            else:
                print("no loinc mapping", table)
            if len(var_loinc_list) > 0:
                var_loinc_cases = " ".join(var_loinc_list)
                var_loinc_case_query = f", CASE {var_loinc_cases} END AS LOINC"
            else:
                print("no loinc mapping", table)
            if len(loinc_prio_case_list) > 0:
                loinc_prio_cases = " ".join(loinc_prio_case_list)
                loinc_prio_case_query = f", CASE {loinc_prio_cases} END AS Priority"
            else:
                print("no prio mapping", table)
                loinc_prio_case_query = ""

            # only tabel that has units
            # if table == "lab":
            #     var_unit_cases_query = f", labmeasurenameinterface as units"
            # el
            if len(var_unit_list) > 0:
                var_unit_cases = " ".join(var_unit_list)
                var_unit_cases_query = f", CASE {var_unit_cases} END AS units"
            else:
                print("no unit mapping")
                var_unit_cases_query = ""

            cohort_query = " AND patientunitstayid IN (" + \
                ", ".join(cohort_list) + ")"

            if table == "nurseCharting":
                or_conds = []
                for table_var_col in table_vars["variable_column"].unique():
                    varlist = set([str(x).strip()
                                   for x in table_vars[table_vars["variable_column"] == table_var_col]["eICU"].unique()])
                    where_cond = "', '".join(varlist)
                    or_conds.append(f"{table_var_col} IN ('{where_cond}')")
                where_cond = "WHERE (" + " OR ".join(or_conds) + ")"
            else:
                varlist = set([str(x).strip()
                               for x in table_vars["eICU"].unique()])
                where_cond = "', '".join(varlist)
                where_cond = f"WHERE {var_col} IN ('{where_cond}')"

            query = f"SELECT patientunitstayid{var_name_case_query} ,{cols[1]} AS offset, {cols[2]} AS value{loinc_prio_case_query}{var_loinc_case_query}{var_unit_cases_query} FROM {table} {where_cond}{cohort_query}{limitn};"
            # print(query)
            # Test query
            if test_query:
                cursor.execute(query)
                results = cursor.fetchall()
                print(results)
                conn.commit()
            with open(f"eicu/sql/{table}.sql", "w") as f:
                f.write(query)


def gen_varcol_sql(cohort_list):
    varcol_tables = {
        "vitalPeriodic": "observationOffset",
        "vitalAPeriodic": "observationOffset",
        "respiratoryCare": "respCareStatusOffset",
        # "diagnosis":"diagnosisOffset"
    }
    for table in varcol_tables:
        print(table)
        table_vars = df[df["table"] == table]

        thegroup = table_vars.groupby('eICU')['LOINC']
        # check if more than 1 loinc for var
        loincs_per_drugname = thegroup.nunique()
        a = loincs_per_drugname[loincs_per_drugname > 1]
        if len(a) > 0:
            print("more than 1 loinc for var")
            print(a)
        id_table_mapping = thegroup.apply(list).to_dict()
        if table == "vitalPeriodic":
            chunk_size = 5000
            for i in range(0, len(cohort_list)+1, chunk_size):
                querylist = []
                patient_chunk = cohort_list[i:i+chunk_size]
                cohort_query = " AND patientunitstayid IN (" + ", ".join(
                    patient_chunk) + ")"
                for col in table_vars["eICU"].unique():
                    loinc = id_table_mapping[col][0]
                    priority = loinc_prio_mapping[loinc][0]
                    # unit assignment from excel
                    unit = unit_mapping[col][0]
                    if unit == "nan":
                        unit = ""
                    querylist.append(
                        f"SELECT patientunitstayid, '{var_name_mapping[col][0]}' AS variable, {varcol_tables[table]} AS offset, {col} AS value, {priority} AS priority, '{loinc}' AS loinc, '{unit}' AS units FROM {table} WHERE {col} IS NOT NULL{cohort_query}")
                query = " UNION ALL ".join(querylist) + f"{limitn};"
                # print(query)
                if test_query:
                    cursor.execute(query)
                    results = cursor.fetchall()
                    print(results)
                    conn.commit()
                with open(f"eicu/sql/{table}_{i+chunk_size}.sql", "w") as f:
                    f.write(query)
        else:
            querylist = []
            cohort_query = " AND patientunitstayid IN (" + \
                ", ".join(cohort_list) + ")"
            for col in table_vars["eICU"].unique():
                loinc = id_table_mapping[col][0]
                priority = loinc_prio_mapping[loinc][0]
                # unit assignment from excel
                unit = unit_mapping[col][0]
                if unit == "nan":
                    unit = ""
                querylist.append(
                    f"SELECT patientunitstayid, '{var_name_mapping[col][0]}' AS variable, {varcol_tables[table]} AS offset, {col} AS value, {priority} AS priority, '{loinc}' AS loinc, '{unit}' AS units FROM {table} WHERE {col} IS NOT NULL{cohort_query}")
            query = " UNION ALL ".join(querylist) + f"{limitn};"
            # print(query)
            if test_query:
                cursor.execute(query)
                results = cursor.fetchall()
                print(results)
                conn.commit()
            with open(f"eicu/sql/{table}.sql", "w") as f:
                f.write(query)

    # # chunking for tables which are too large
    # chunksize = 200000000
    # large_tables = {"vitalPeriodic": [
    #     "observationOffset", "vitalperiodicid", "2188637668"]}
    # for table in large_tables:
    #     table_vars = df[df["table"] == table]
    #     thegroup = table_vars.groupby('eICU')['LOINC']
    #     # check if more than 1 loinc for var
    #     a = table_vars[thegroup.transform('count') > 1]
    #     if len(a) > 0:
    #         print(a)
    #     id_table_mapping = thegroup.apply(list).to_dict()
    #     for i in range(0, int(large_tables[table][2])+1, chunksize):
    #         id_from = i
    #         id_to = i+chunksize
    #         querylist = []
    #         for col in table_vars["eICU"].unique():
    #             loinc = id_table_mapping[col][0]
    #             priority = loinc_prio_mapping[loinc][0]
    #             querylist.append(
    #                 f"SELECT patientunitstayid, '{var_name_mapping[col][0]}' AS variable, {large_tables[table][0]} AS offset, {col} AS value, {priority} AS priority, '{loinc}' AS loinc FROM {table} WHERE {col} IS NOT NULL AND {large_tables[table][1]}>={id_from} AND {large_tables[table][1]}<{id_to}")
    #         query = " UNION ALL ".join(querylist) + f"{limitn};"
    #         # print(query)
    #         if test_query:
    #             cursor.execute(query)
    #             results = cursor.fetchall()
    #             print(results)
    #             conn.commit()
    #         with open(f"eicu/sql/get_{table}_vars_{id_to}.sql", "w") as f:
    #             f.write(query)


def gen_intake_sql(cohort_list):
    # Read the Excel sheet into a pandas DataFrame
    intakexls = pd.read_excel('eICU/intake_4h_eICU_wLOINC.xlsx')
    intakexls["drugname"] = intakexls["drugname"].apply(
        lambda x: str(x).strip())
    intakexls["LOINC"] = intakexls["LOINC"].apply(lambda x: str(x).strip())
    intakexls["unit"] = intakexls["unit"].apply(lambda x: str(x).strip())
    intake_unit_mapping = intakexls.groupby(
        'drugname')['unit'].apply(list).to_dict()

    varrow_tables = {
        "infusionDrug": ["drugname", "infusionoffset", "drugrate"]
    }
    for table, table_cols in varrow_tables.items():
        print(table)
        thegroup = intakexls.groupby('drugname')['LOINC']
        # check if more than 1 loinc for var
        loincs_per_drugname = thegroup.nunique()
        a = loincs_per_drugname[loincs_per_drugname > 1]
        if len(a) > 0:
            print("more than 1 loinc for var")
            print(a)
        id_table_mapping = thegroup.apply(list).to_dict()
        cols = table_cols
        varlist = set([str(x).strip() for x in intakexls["drugname"].unique()])
        where_cond = "', '".join(varlist)
        var_loinc_list = []
        loinc_prio_case_list = []
        var_unit_list = []
        for var in id_table_mapping.keys():
            var_loinc_list.append(
                f"WHEN {cols[0]} = '{var}' THEN '{id_table_mapping[var][0]}'")
            unit = intake_unit_mapping[var][0]
            if unit == "nan":
                unit = ""
            var_unit_list.append(
                f"WHEN {cols[0]} = '{var}' THEN '{unit}'")
            if id_table_mapping[var][0] in loinc_prio_mapping:
                try:
                    prio = int(loinc_prio_mapping[id_table_mapping[var][0]][0])
                    loinc_prio_case_list.append(
                        f"WHEN {cols[0]} = '{var}' THEN {prio}")
                except ValueError:
                    # print("No prio for", id_table_mapping[var][0])
                    pass

        if len(var_loinc_list) > 0:
            var_loinc_cases = " ".join(var_loinc_list)
            var_loinc_case_query = f", CASE {var_loinc_cases} END AS LOINC"
        else:
            print("no loinc mapping", table)
        if len(loinc_prio_case_list) > 0:
            loinc_prio_cases = " ".join(loinc_prio_case_list)
            loinc_prio_case_query = f", CASE {loinc_prio_cases} END AS Priority"
        else:
            print("no prio mapping", table)
            loinc_prio_case_query = ""
        if len(var_unit_list) > 0:
            var_unit_cases = " ".join(var_unit_list)
            var_unit_cases_query = f", CASE {var_unit_cases} END AS units"
        else:
            print("no unit mapping")
            var_unit_cases_query = ""

        cohort_query = " AND patientunitstayid IN (" + \
            ", ".join(cohort_list) + ")"
        query = f"SELECT patientunitstayid, 'state_ivfluid4h' as variable,{cols[1]} AS offset, {cols[2]} AS value{loinc_prio_case_query}{var_loinc_case_query}{var_unit_cases_query} FROM {table} WHERE {cols[0]} IN ('{where_cond}'){cohort_query}{limitn};"
        # print(query)
        # Test query
        if test_query:
            cursor.execute(query)
            results = cursor.fetchall()
            print(results)
            conn.commit()
        with open(f"eicu/sql/intake4h.sql", "w") as f:
            f.write(query)


def gen_nonoffset_sql(cohort_list):
    no_offset_tables = [
        # "apacheApsVar",
        # "apachePatientResult",
        # "apachePredVar"
    ]
    for table in no_offset_tables:
        print(table)
        table_vars = df[df["table"] == table]

        thegroup = table_vars.groupby('eICU')['LOINC']
        # check if more than 1 loinc for var
        loincs_per_drugname = thegroup.nunique()
        a = loincs_per_drugname[loincs_per_drugname > 1]
        if len(a) > 0:
            print("more than 1 loinc for var")
            print(a)
        id_table_mapping = thegroup.apply(list).to_dict()
        querylist = []
        cohort_query = " AND patientunitstayid IN (" + \
            ", ".join(cohort_list) + ")"
        for col in table_vars["eICU"].unique():
            loinc = id_table_mapping[col][0]
            priority = loinc_prio_mapping[loinc][0]
            unit = unit_mapping[col][0]
            if unit == "nan":
                unit = ""
            querylist.append(
                f"SELECT patientunitstayid, '{var_name_mapping[col][0]}' AS variable, '' AS offset, {col} AS value, {priority} AS priority, '{loinc}' AS loinc, '{unit}' AS units FROM {table} WHERE {col} IS NOT NULL{cohort_query}")
        query = " UNION ALL ".join(querylist) + f"{limitn};"
        # print(query)
        if test_query:
            cursor.execute(query)
            results = cursor.fetchall()
            print(results)
            conn.commit()
        with open(f"eicu/sql/{table}.sql", "w") as f:
            f.write(query)


def gen_patient_sql(cohort_list):
    table = "patient"
    print(table)
    table_vars = df[df["table"] == table]

    thegroup = table_vars.groupby('eICU')['LOINC']
    id_table_mapping = thegroup.apply(list).to_dict()
    querylist = []
    cohort_query = " AND patientunitstayid IN (" + ", ".join(cohort_list) + ")"
    offset_cols = {
        "unitdischargestatus": "unitdischargeoffset",
        "hospitaldischargestatus": "hospitaldischargeoffset"
    }
    for col in table_vars["eICU"].unique():
        loinc = id_table_mapping[col][0]
        priority = loinc_prio_mapping[loinc][0]
        unit = unit_mapping[col][0]
        if unit == "nan":
            unit = ""
        var_offset = offset_cols.get(col, "''")
        # age column is not numeric due to >89, therefore for UNION ALL, all other cols have to be converted,
        # else UNION types character varying and numeric cannot be matched
        var_query = f"{col}::VARCHAR"

        querylist.append(
            f"SELECT patientunitstayid, '{var_name_mapping[col][0]}' AS variable, {var_offset}::VARCHAR AS offset, {var_query} AS value, {priority} AS priority, '{loinc}' AS loinc, '{unit}' AS units FROM {table} WHERE {col} IS NOT NULL{cohort_query}")
    query = " UNION ALL ".join(querylist) + f"{limitn};"
    # print(query)
    if test_query:
        cursor.execute(query)
        results = cursor.fetchall()
        print(results)
        conn.commit()
    with open(f"eicu/sql/{table}.sql", "w") as f:
        f.write(query)

def get_ventmode_sql(cohort_list):
    vent_mode_cols = ['Tidal Volume (set)', 'Set Vt (Servo,LTV)', 'Set Vt (Drager)',
                  'Adult Con Setting Set Vt', 'Inspiratory Pressure, Set', 'Pressure to Trigger PS']
    cohort_query = " AND patientunitstayid IN (" + ", ".join(cohort_list) + ")"
    vent_mode_values = "', '".join(vent_mode_cols)
    query = f"""
    SELECT 
        patientunitstayid, 
        'vent_mode' AS variable, 
        respChartEntryOffset AS offset, 
        respchartvaluelabel AS value, 
        1 AS priority, 
        '' AS units 
    FROM respiratoryCharting 
    WHERE respchartvaluelabel IN ('{vent_mode_values}')
    {cohort_query}
    {limitn};
    """
    if test_query:
        cursor.execute(query)
        results = cursor.fetchall()
        print(results)
        conn.commit()
    with open(f"eicu/sql/vent_mode_supp.sql", "w") as f:
        f.write(query)

def cast_clean_int(x):
    """Age column includes value >89 and empty values"""
    try:
        x_int = int(x)
    except ValueError as e:
        x = re.sub(r'[^0-9]', '', x)
        x_int = int(x) if x != '' else 0
    return x_int


def get_cohort_ids():
    query = "SELECT * FROM patient;"
    patients = pd.read_sql_query(query, conn)
    patients['age'] = patients['age'].apply(cast_clean_int)
    patient_agefilter = patients.loc[(patients.age >= 18)]
    patient_icutimefilter = patient_agefilter.loc[(patient_agefilter.hospitaladmitoffset <= 0) & (
        (patient_agefilter.unitdischargeoffset + patient_agefilter.hospitaladmitoffset) >= 240)]
    cohort_ids = patient_icutimefilter['patientunitstayid'].unique()
    np.savetxt("cohort_ids.txt", cohort_ids)
    return cohort_ids


def get_vent_cohort_ids():
    ventevents = pd.read_parquet("eicu/csvs_new/ventevents_issue82.parquet")
    return ventevents["stay_id"].unique()


def run_queries():
    for filename in os.listdir("eicu/sql/"):
        if filename.endswith(".sql"):
            print(filename)
            with open("eicu/sql/"+filename, "r") as f:
                result = pd.read_sql_query(f.read(), conn)
                result.to_csv("eicu/csvs_new/" +
                              os.path.splitext(filename)[0]+".csv", index=False)


conn = psycopg2.connect(dbname="eicu", user="jason")
conn.autocommit = True
cursor = conn.cursor()

# Read the Excel sheet into a pandas DataFrame
df = pd.read_excel(
    'eicu/actions-rewards-variable-availability.xlsx', sheet_name="Variables eICU")
df["eICU"] = df["eICU"].apply(lambda x: re.sub(r"\'", "\''", str(x).strip()))
df["LOINC"] = df["LOINC"].apply(lambda x: str(x).strip())
df["unit"] = df["unit"].apply(lambda x: str(x).strip())

df = df[df['Variables names'].notna() & (df['Variables names'] != '') & df['table'].notna(
) & (df['table'] != '') & df['eICU'].notna() & (df['eICU'] != '')]

print("Relevant tables", df['table'].unique())

var_index = pd.read_excel(
    'eicu/actions-rewards-variable-availability.xlsx', sheet_name="Variables index")
var_index["LOINC"] = var_index["LOINC"].apply(lambda x: str(x).strip())
# var_index["Priority"] = var_index["Priority"].apply(lambda x: str(x).strip())
var_index["full naming schema"] = var_index["full naming schema"].apply(
    lambda x: str(x).strip())

loinc_prio_mapping = var_index.groupby(
    'LOINC')['Priority'].apply(list).to_dict()
loinc_var_mapping = var_index.groupby(
    'LOINC')['full naming schema'].apply(list).to_dict()
var_name_mapping = df.groupby(
    'eICU')['full naming schema'].apply(list).to_dict()
unit_mapping = df.groupby('eICU')['unit'].apply(list).to_dict()

test_query = False
limitn = " LIMIT 10" if test_query else ""
# large_table = True

cohort_ids = get_vent_cohort_ids()
# cohort_ids = np.loadtxt("cohort_ids.txt")
cohort_list = [str(int(x)) for x in cohort_ids]
gen_varcol_sql(cohort_list)
gen_varrow_sql(cohort_list)
gen_intake_sql(cohort_list)
gen_patient_sql(cohort_list)
get_ventmode_sql(cohort_list)
run_queries()

conn.close()

import json
import pandas as pd


def gen_outlier_ranges():
    var_index = pd.read_excel(
        'eicu/actions-rewards-variable-availability.xlsx', sheet_name="Variables index")
    var_index["full naming schema"] = var_index["full naming schema"].apply(
        lambda x: str(x).strip())

    var_index = var_index[var_index['Variables names'].notna() & (var_index['Variables names'] != '') & var_index['Threshold low'].notna(
    ) & (var_index['Threshold low'] != '') & var_index['Threshold high'].notna() & (var_index['Threshold high'] != '')]

    var_threshold_mapping = var_index.groupby('full naming schema').agg({
        'Threshold low': list,
        'Threshold high': list
    }).to_dict(orient='index')

    with open("eicu/outlier_ranges.json", "w") as f:
        json.dump(var_threshold_mapping, f)


def gen_vent_encodings():
    var_index = pd.read_excel(
        'eicu/Dict_cath_variables.xlsx', sheet_name="eICU")
    var_index["value_str"] = var_index["value_str"].apply(
        lambda x: str(x).strip())
    grouped = var_index.groupby('full_name').apply(
        lambda x: dict(zip(x['value_str'], x['category'])))
    for full_name, group_dict in grouped.items():
        with open(f"eicu/{full_name}.json", 'w') as f:
            json.dump(group_dict, f)


def gen_conversion_dict():
    var_index = pd.read_excel(
        'eicu/actions-rewards-variable-availability.xlsx', sheet_name="Variables index")
    var_index["full naming schema"] = var_index["full naming schema"].apply(
        lambda x: str(x).strip())
    var_index["LOINC"] = var_index["LOINC"].apply(lambda x: str(x).strip())
    # != returns rows with both column NaN. Use additional XOR (^) to get rows where only 1 is NaN
    conversion_vars = var_index[var_index['target unit'].notna() & var_index['unit'].notna() & (
        var_index['target unit'] != var_index['unit']) | (var_index['target unit'].isna() ^ var_index['unit'].isna())]
    conversion_vars[["full naming schema", "target unit", "unit",
                     "LOINC"]].drop_duplicates().to_csv("eicu/conversion_vars.csv")
    print()


def gen_target_dict():
    var_index = pd.read_excel(
        'eicu/actions-rewards-variable-availability.xlsx', sheet_name="Variables index")
    var_index["full naming schema"] = var_index["full naming schema"].apply(
        lambda x: str(x).strip())
    var_index["LOINC"] = var_index["LOINC"].apply(lambda x: str(x).strip())
    # != returns rows with both column NaN. Use additional XOR (^) to get rows where only 1 is NaN
    # TODO drop 4h target units caused handled separately
    target_dict = var_index[var_index['target unit'].notna()][["full naming schema", "target unit"]].drop_duplicates(
    ).groupby('full naming schema')["target unit"].apply(list).to_dict()
    with open("eicu/target_units.json", "w") as f:
        json.dump(target_dict, f)


def gen_set_dict():
    var_index = pd.read_excel(
        'eicu/actions-rewards-variable-availability.xlsx', sheet_name="Variables index")
    var_index["full naming schema"] = var_index["full naming schema"].apply(
        lambda x: str(x).strip())
    var_index["Set"] = var_index["Set"].apply(lambda x: str(x).strip())
    # != returns rows with both column NaN. Use additional XOR (^) to get rows where only 1 is NaN
    sets_dict = var_index[var_index['Set'].notna()][["full naming schema", "Set"]].drop_duplicates(
    ).groupby('Set')['full naming schema'].apply(list).to_dict()
    with open("eicu/sets_dict.json", "w") as f:
        json.dump(sets_dict, f)


def get_invas_dict():
    var_index = pd.read_excel(
        'eicu/map_o2admindevice_eicu.xlsx', sheet_name="Tabelle1")
    var_index["Value"] = var_index["Value"].apply(
        lambda x: str(x).strip())
    var_index = var_index.drop_duplicates()
    invas_dict = dict(zip(var_index["Value"], var_index["Encoding"]))
    with open("eicu/invas_dict.json", "w") as f:
        json.dump(invas_dict, f)


def get_vaso_target_units():
    var_index = pd.read_excel(
        'eicu/actions-rewards-variable-availability.xlsx', sheet_name="Variables index")
    var_index["full naming schema"] = var_index["full naming schema"].apply(
        lambda x: str(x).strip())
    var_index["LOINC"] = var_index["LOINC"].apply(lambda x: str(x).strip())
    vaso_vars = var_index[var_index["full naming schema"] == "drugs_vaso4h"]

    target_dict = vaso_vars[vaso_vars['target unit'].notna()][["target unit", "LOINC"]].drop_duplicates(
    ).groupby('target unit')["LOINC"].apply(list).to_dict()
    with open("eicu/vaso_target_units.json", "w") as f:
        json.dump(target_dict, f)

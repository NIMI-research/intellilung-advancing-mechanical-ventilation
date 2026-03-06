import os
import json
import sys

from typing import Union
import common.utility.utility as util

path_to_kb = os.path.dirname(__file__)


class Patients:
    def __init__(self) -> None:
        abs_path = os.path.dirname(__file__)
        with open(os.path.join(abs_path, "dicts/req_careunits.json")) as f:
            self.req_careunits = json.load(f)["careunits"]


class MechanicalVent:
    def __init__(self):
        abs_path = os.path.dirname(__file__)
        with open(os.path.join(abs_path, "dicts/base_mv_reqs.json")) as f:
            self.mv_reqs = json.load(f)

        with open(os.path.join(abs_path, "dicts/mv_relevant_vars.json")) as f:
            try:
                self.relevant_variables = json.load(f)
            except:
                self.relevant_variables = {}

        if os.path.exists(os.path.join(abs_path, "dicts/mv_unified_vars.json")):
            with open(os.path.join(abs_path, "dicts/mv_unified_vars.json")) as f:
                try:
                    self.unified_vars = json.load(f)
                except:
                    self.unified_vars = {}

        with open(os.path.join(abs_path, "dicts/variable_outlier_ranges.json")) as f:
            try:
                self.outlier_ranges = json.load(f)
            except:
                self.outlier_ranges = {}


conversion_dict = {"mg": 1e-3, "mcg": 1e-6, "ng": 1e-9, "L": 1, "ml": 1e-3, "units": 1}
with open(os.path.join(os.path.dirname(__file__), "dicts/conversion_dict.json")) as f:
    target_conversions = json.load(f)


class Vector:
    def __init__(self) -> None:
        abs_path = os.path.dirname(__file__)
        with open(os.path.join(abs_path, "dicts/vector_variables.json")) as f:
            self.variables = json.load(f)


class Unif_Vars:
    def __init__(self):
        try:
            self.weight = next((key for key in mv_reqs.unified_vars if any("Weight" in meas for meas in mv_reqs.unified_vars[key])), None)
            self.height = next((key for key in mv_reqs.unified_vars if any("Height" in meas for meas in mv_reqs.unified_vars[key])), None)

            self.peep = next((key for key in mv_reqs.unified_vars if any("PEEP" in meas for meas in mv_reqs.unified_vars[key])), None)
            self.fio2 = next((key for key in mv_reqs.unified_vars if any("FiO2" in meas for meas in mv_reqs.unified_vars[key])), None)
            self.rr = next((key for key in mv_reqs.unified_vars if any("RR" in meas for meas in mv_reqs.unified_vars[key])), None)

            self.be = next((key for key in mv_reqs.unified_vars if any("Base Excess" in meas for meas in mv_reqs.unified_vars[key])), None)

            self.press_sup = next((key for key in mv_reqs.unified_vars if any("PSV Level" in meas for meas in mv_reqs.unified_vars[key])), None)

            self.gender = next((key for key in mv_reqs.unified_vars if any("Gender" in meas for meas in mv_reqs.unified_vars[key])), None)

            self.hb = next((key for key in mv_reqs.unified_vars if any("Hemoglobin" in meas for meas in mv_reqs.unified_vars[key])), None)
            self.temp = next((key for key in mv_reqs.unified_vars if any("Temperature" in meas for meas in mv_reqs.unified_vars[key])), None)

            self.art_press_sys = next((key for key in mv_reqs.unified_vars if any("Pressure systolic" in meas for meas in mv_reqs.unified_vars[key])), None)
            self.art_press_dia = next((key for key in mv_reqs.unified_vars if any("Pressure diastolic" in meas for meas in mv_reqs.unified_vars[key])), None)
            self.art_press_mean = next((key for key in mv_reqs.unified_vars if any("Pressure mean" in meas for meas in mv_reqs.unified_vars[key])), None)

            self.tv_norm = "vent_vtnorm"
            self.tv = next((key for key in mv_reqs.unified_vars if any("Tidal Volume" in meas for meas in mv_reqs.unified_vars[key])), None)

            self.gluco = next((key for key in mv_reqs.unified_vars if any("Glucose" in meas for meas in mv_reqs.unified_vars[key])), None)

            self.magnes = next((key for key in mv_reqs.unified_vars if any("Magnesium" in meas for meas in mv_reqs.unified_vars[key])), None)

            self.ca = next((key for key in mv_reqs.unified_vars if any("Calcium non-ionized" in meas for meas in mv_reqs.unified_vars[key])), None)

            self.bun = next((key for key in mv_reqs.unified_vars if any("BUN" in meas for meas in mv_reqs.unified_vars[key])), None)

            self.ast = next((key for key in mv_reqs.unified_vars if any("AST" in meas for meas in mv_reqs.unified_vars[key])), None)
            self.alt = next((key for key in mv_reqs.unified_vars if any("ALT" in meas for meas in mv_reqs.unified_vars[key])), None)

            self.crea = next((key for key in mv_reqs.unified_vars if any("Creatinine" in meas for meas in mv_reqs.unified_vars[key])), None)

            self.bili = next((key for key in mv_reqs.unified_vars if any("Bilirubin" in meas for meas in mv_reqs.unified_vars[key])), None)

            if not self.weight:
                raise Exception("The weight variable is not present in the unified variables")
            if not self.height:
                raise Exception("The height variable is not present in the unified variables")

            if not self.peep:
                raise Exception("The PEEP variable is not present in the unified variables")
            if not self.fio2:
                raise Exception("The FiO2 variable is not present in the unified variables")
            if not self.rr:
                raise Exception("The RR variable is not present in the unified variables")

            if not self.be:
                raise Exception("The BE variable is not present in the unified variables")

            if not self.press_sup:
                raise Exception("The pressure support variable is not present in the unified variables")
        except Exception as e:
            print(f"{util.bcolors.FAIL}{e}{util.bcolors.ENDC}")
            sys.exit()


patient_reqs = Patients()
mv_reqs = MechanicalVent()
vectors = Vector()
unif_vars = None
unif_vars_info = None
categorical_data_vars = None
categorical_data_dicts = {}


def regenerate_knowledge_base():
    global patient_reqs
    global mv_reqs
    global vectors

    patient_reqs = Patients()
    mv_reqs = MechanicalVent()
    vectors = Vector()


def reduce_memory_based_on_column(columns: list, dtype_only: bool = False):
    type_dict = {
        "stay_id": "int32",
        "charttime": "float64",
        "starttime": "float64",
        "endtime": "float64",
        "itemid": "int32",
        "valuenum": "float32",
        "label": "category",
        "mv_id": "uint8",
        "age": "uint8",
        "gender": "uint8",
    }
    return {key: val for key, val in type_dict.items() if key in columns} if not dtype_only else type_dict[columns]


def update_vector_variables():
    vectors.variables["var_names"] = ["stay_id", "mv_id", "timepoints", "age"] + sorted(list(mv_reqs.unified_vars.keys()))


def get_var_itemid(var: str):
    if var not in mv_reqs.relevant_variables:
        return None
    return mv_reqs.relevant_variables[var]["itemid"]


def get_var_unit(var: str):
    if var not in mv_reqs.relevant_variables:
        return None
    return mv_reqs.relevant_variables[var]["unit"]


def get_unified_itemids(var: str):
    if var not in mv_reqs.unified_vars:
        return []
    return [get_var_itemid(var=key) for key in mv_reqs.unified_vars[var] if key in mv_reqs.relevant_variables]


def get_var_itemids_by_unit(var: str, unit: Union[str, None]):
    if not var in mv_reqs.unified_vars:
        return None
    return [get_var_itemid(var=v) for v in mv_reqs.unified_vars[var] if ((get_var_unit(var=v) == unit) if unit else not get_var_unit(var=v))]


def get_var_prior_by_itemid(itemid: int):
    return next((mv_reqs.relevant_variables[var]["prior"] for var in mv_reqs.relevant_variables if mv_reqs.relevant_variables[var]["itemid"] == itemid), None)


# def get_var_unit_by_itemid(itemid: int):


def get_data_from_itemid(itemid: int, res: str = None):
    if not res:
        return next((var for var in mv_reqs.relevant_variables if mv_reqs.relevant_variables[var]["itemid"] == itemid), None)
    else:
        return next((mv_reqs.relevant_variables[var][res] for var in mv_reqs.relevant_variables if mv_reqs.relevant_variables[var]["itemid"] == itemid), None)


def get_uni_var_from_rel_var(var: str):
    return next((uni_var for uni_var in mv_reqs.unified_vars if var in mv_reqs.unified_vars[uni_var]), None)

import json

with open("eicu/dicts/sets_dict.json", "r") as f:
    sets_dict = json.load(f)

with open("eicu/dicts/vent_mode.json", "r") as f:
    vent_mode_groups = json.load(f)

with open("eicu/dicts/vent_invas.json", "r") as f:
    vent_invas_groups = json.load(f)

with open("eicu/dicts/state_airtype.json", "r") as f:
    state_airtype_groups = json.load(f)

all_variables = set([item for setlist in sets_dict.values() for item in setlist])

categorical_vars = ["vent_invas", "vent_mode", "state_airtype"]

invalid_vals = {
    "vent_rsbi": ["M"],
    "state_ivfluid4h": ["ERROR", "UD"]
}

vaso_encodings = {
    "OFF": 0,
    "OFF\\.br\\": 0,
    "Documentation undone": 0
}

required_tidal = [
    "vent_vtnorm",
    "vent_vt"
]

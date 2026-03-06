import pandas as pd


def test_variable_ranges_dict_contain_required_variables(variable_ranges, state_action_space):
    variable_ranges_keys = set(variable_ranges.keys())
    state_action_space = set(state_action_space)
    variables_not_in_ranges_dict = list(state_action_space - variable_ranges_keys)
    assert state_action_space.issubset(
        variable_ranges_keys), f'Following variables missing from variable ranges file: {variables_not_in_ranges_dict}'


def test_required_variables_in_range(dataset: pd.DataFrame, variable_ranges, state_action_space):
    lows = {col: variable_ranges[col]["low"][0] for col in state_action_space}
    highs = {col: variable_ranges[col]["high"][0] for col in state_action_space}

    out_of_range_columns = []
    for var in state_action_space:
        var_in_range = dataset[var].between(lows[var], highs[var], inclusive='both').all()
        if not var_in_range:
            out_of_range_columns.append(var)

    assert not out_of_range_columns, f'Following variables with out of range values: {out_of_range_columns}'


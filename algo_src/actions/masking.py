def mask_actions_with_empty_bin(actions, list_of_actions, action_ranges):
    vent_mode_index = list_of_actions.index('vent_mode')
    vent_vt_index = list_of_actions.index('vent_vt_action')
    vent_pisn_peep_index = list_of_actions.index('vent_pinsp-peep')

    vent_mode = actions[:, vent_mode_index]

    vent_vt_null_action_bin = len(action_ranges['vent_vt_action']) - 1
    vent_pinsp_peep_null_action_bin = len(action_ranges['vent_pinsp-peep']) - 1

    is_volume_controlled_mode = vent_mode == 0
    is_pressure_controlled_mode = vent_mode == 1

    actions[is_volume_controlled_mode, vent_pisn_peep_index] = vent_pinsp_peep_null_action_bin
    actions[is_pressure_controlled_mode, vent_vt_index] = vent_vt_null_action_bin
    return actions

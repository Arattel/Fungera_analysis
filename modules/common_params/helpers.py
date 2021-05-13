def init_instruction_set(name):
    if name == 'base':
        from modules.instruction_sets.instruction_sets import base as instructions
    elif name in ['jump_directed', 'jump_direction_independent']:
        from modules.instruction_sets.instruction_sets import jump_call as instructions
    elif name == 'error_correction':
        from modules.instruction_sets.instruction_sets import error_correction as instructions
    return instructions


def init_deltas(name):
    if name == 'error_correction':
        from modules.deltas import error_correction_deltas as deltas
    else:
        from modules.deltas import base_deltas as deltas
    return deltas

from dataclasses import dataclass, field
from typing import List, Tuple, Dict

default_ancestors: Dict = {
    'base': 'initial.gen',
    'jump_directed': 'jump_to_coord_improved.gen',
    'jump_direction_independent': 'direction_agnostic_call.gen',
    'error_correction': 'initial_interpreted.gen'
}


@dataclass
class Config:
    """
    instruction_set: base, jump_directed, jump_direction_independent, error_correction
    """

    instruction_set: str = 'error_correction'
    memory_size: Tuple[int] = (500, 500)
    info_display_size: Tuple[int] = (50, 50)
    scroll_step: int = 50
    kill_organisms_ratio: float = 0.5
    memory_full_ratio: float = 0.90
    cycle_gap: int = 10
    random_rate: int = 2000
    stack_length: int = 8
    organism_death_rate: int = 100
    kill_if_no_child: int = 25000
    autosave_rate: Tuple = (40, 1)
    penalize_parasitism: int = 100
    random_seed: int = 38
    use_mutations: bool = False
    dump_full_snapshots: bool = True
    simulation_name: str = 'fungera_direction_independent_fixed_seed_38'

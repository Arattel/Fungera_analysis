from dataclasses import dataclass, field
from typing import List, Tuple, Dict

default_ancestors: Dict = {
    'base': 'initial.gen',
    'jump_directed': 'jump_to_coord_improved.gen',
    'jump_direction_independent': 'direction_agnostic_call.gen',
    'error_correction': 'initial_interpreted.gen'
}

default_cache_files: Dict[str, str] = {
    'base': 'data/base_fungera_cache.pkl',
    'jump_directed': 'data/jump_directed_cache.pkl',
    'jump_direction_independent': 'data/jump_direction_independent_cache.pkl',
    'error_correction': 'data/error_correction_cache.pkl'
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
    random_rate: int = 400
    stack_length: int = 8
    organism_death_rate: int = 10
    kill_if_no_child: int = 25000
    autosave_rate: Tuple = (75, 1)
    penalize_parasitism: int = 100
    random_seed: int = 38
    use_mutations: bool = True
    dump_full_snapshots: bool = True
    simulation_name: str = 'perkele!'
    max_derivative_points: int = 15
    smoothing_window: int = 40

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Config:
    memory_size: Tuple[int] = (900, 900)
    info_display_size: Tuple[int] = (50, 50)
    scroll_step: int = 50
    kill_organisms_ratio: float = 0.5
    memory_full_ratio: float = 0.90
    cycle_gap: int = 10
    random_rate: int = 20
    stack_length: int = 8
    organism_death_rate: int = 100
    kill_if_no_child: int = 25000
    autosave_rate: Tuple = (1, 1)
    penalize_parasitism: int = 100
    random_seed: int = 38
    simulation_name: str = 'Entropy_experiment_normal_save_rate_seed_38_kill_rate_20'

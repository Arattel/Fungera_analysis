import numpy as np

base_deltas = {
    'left': np.array([0, -1]),
    'right': np.array([0, 1]),
    'up': np.array([-1, 0]),
    'down': np.array([1, 0]),
}

error_correction_deltas = {
    'left': np.array([0, -3]),
    'right': np.array([0, 3]),
    'up': np.array([-3, 0]),
    'down': np.array([3, 0]),
}

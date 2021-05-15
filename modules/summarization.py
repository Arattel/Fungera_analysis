import os
import pickle
from typing import List

import pandas as pd
import numpy as np
from scipy.signal import argrelextrema


def get_prefixes(filenames: List[str]) -> List[str]:
    prefixes = list(set(map(lambda x: x.split('_cycle_')[0], filenames)))
    prefixes.sort()
    return prefixes


def preprocess_records(records) -> pd.DataFrame:
    cycles = list(map(lambda x: x['cycle'], records))
    entropies = list(map(lambda x: x['entropy'], records))
    df = pd.DataFrame({
        'cycles': cycles,
        'entropies': entropies
    })
    df = df.sort_values(by='cycles')
    return df


def get_metrics_for_prefix(prefix, files, directory):
    filenames = list(filter(lambda x: x.startswith(prefix), files))
    filenames = list(filter(lambda x: x.endswith('.snapshot2'), filenames))
    filepaths = list(map(lambda x: os.path.join(directory, x), filenames))
    records = []
    for filepath in filepaths:
        try:
            records.append(pickle.load(open(filepath, 'rb')))
        except Exception:
            print(filepath)
    return records


def find_indices_with_biggest_derivative(df, max_points=10):
    ts_diff = np.diff(df.entropies).argsort()[::-1]
    return ts_diff[:max_points]


def get_summary(df, max_derivative_points=15):
    local_max = argrelextrema(df.entropies.values, np.greater)
    local_min = argrelextrema(df.entropies.values, np.less)
    derivatives = find_indices_with_biggest_derivative(df, max_points=max_derivative_points)
    return np.concatenate([df.cycles.values[local_max], df.cycles.values[local_min], df.cycles.values[derivatives]])


def get_summarization_cycles(metrics_df: pd.DataFrame, smoothing_window: int = None,
                             max_derivative_points: int = 10):
    metrics_df.entropies = metrics_df.entropies.rolling(smoothing_window).mean()
    metrics_df = metrics_df.iloc[smoothing_window - 1:, :]
    summary_cycles = get_summary(df=metrics_df, max_derivative_points=max_derivative_points)
    return np.sort(summary_cycles)

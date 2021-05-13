import os
import argparse
from threading import Thread, Event
import toml
import numpy as np
import modules.window as w
from conf.config import Config
from dataclasses import asdict
import shutil
from modules.common_params.helpers import init_instruction_set, init_deltas


class RepeatedTimer(Thread):
    def __init__(self, interval, function, args=None, kwargs=None):
        Thread.__init__(self)
        self.interval = interval
        self.function = function
        self.args = args if args is not None else []
        self.kwargs = kwargs if kwargs is not None else {}
        self.finished = Event()
        self.start()

    def cancel(self):
        self.finished.set()

    def run(self):
        while not self.finished.wait(self.interval[0]):
            if not self.finished.is_set():
                self.function(*self.args, **self.kwargs)
                self.interval[0] *= self.interval[1]
        self.finished.set()


instructions = init_instruction_set(Config.instruction_set)
deltas = init_deltas(Config.instruction_set)
instructions_set_name = Config.instruction_set


def load_config():
    _config = asdict(Config())

    for key in _config:
        value = _config[key]
        _config[key] = np.array(value) if isinstance(value, tuple) else value

    return _config


is_running = False

config = load_config()

parser = argparse.ArgumentParser(
    description='Fungera - two-dimentional artificial life simulator'
)
parser.add_argument('--name', default=config['simulation_name'], help='Simulation name')
parser.add_argument(
    '--seed', type=int, help='Random seed',
    default=config['random_seed']
)

parser.add_argument(
    '--random_rate', type=int, help='Random rate',
    default=config['random_rate']
)
parser.add_argument(
    '--out', type=str, help='Output file',
)
parser.add_argument(
    '--input', type=str, help='Output file',
    default='../fungera/basic_fungera_experiments/'
)
parser.add_argument(
    '--state', default='new', help='State file to load (new/last/filename)',
)

line_args = parser.parse_args()

config['random_seed'] = line_args.seed
config['simulation_name'] = line_args.name
config['random_rate'] = line_args.random_rate
config['snapshot_to_load'] = line_args.state

screen = None

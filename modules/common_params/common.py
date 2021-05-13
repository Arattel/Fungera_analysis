import curses
import os
import argparse
from threading import Thread, Event
import toml
import numpy as np
import modules.window as w
from conf.config import Config, default_ancestors
from modules.common_params.helpers import init_instruction_set, init_deltas
from dataclasses import asdict
import shutil
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s|%(filename)s|%(lineno)s| %(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    filename='example.log',
)
logger = logging.getLogger(__name__)


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


colors = {
    'parent_bold': 1,
    'child_bold': 2,
    'ip_bold': 3,
    'parent': 4,
    'child': 5,
    'ip': 6,
}


def init_curses():
    terminal_size = shutil.get_terminal_size(fallback=(120, 50))
    if terminal_size.columns < 100 or terminal_size.lines < 25:
        print(
            'Terminal size is too small. The terminal size must be at least (100, 25)'
        )
        print(
            'Your terminal size: ({}, {})'.format(
                terminal_size.columns, terminal_size.lines
            )
        )
        exit(0)
    _screen = w.Window(curses.initscr())
    _screen.setup()

    curses.noecho()
    curses.cbreak()
    curses.curs_set(0)

    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(colors['parent_bold'], curses.COLOR_WHITE, 126)
    curses.init_pair(colors['ip_bold'], curses.COLOR_WHITE, 160)
    curses.init_pair(colors['child_bold'], curses.COLOR_WHITE, 128)
    curses.init_pair(colors['parent'], curses.COLOR_WHITE, 27)
    curses.init_pair(colors['ip'], curses.COLOR_WHITE, 117)
    curses.init_pair(colors['child'], curses.COLOR_WHITE, 33)
    return _screen


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
    '--state', default='new', help='State file to load (new/last/filename)',
)
parser.add_argument(
    '--seed', type=int, help='Random seed',
    default=config['random_seed']
)

parser.add_argument(
    '--random_rate', type=int, help='Random rate',
    default=config['random_rate']
)

parser.add_argument(
    '--instruction_set',
    type=str,
    default=Config.instruction_set,
    choices=list(default_ancestors.keys())
)

parser.add_argument('--no_snapshots', action='store_true')

line_args = parser.parse_args()

config['snapshot_to_load'] = line_args.state
config['random_seed'] = line_args.seed
config['simulation_name'] = line_args.name
config['random_rate'] = line_args.random_rate
config['dump_full_snapshots'] = not line_args.no_snapshots

print(config)
instructions_set_name = line_args.instruction_set

instructions = init_instruction_set(instructions_set_name)
deltas = init_deltas(instructions_set_name)

screen = init_curses()

import pickle
import traceback
import glob
import os
import numpy as np
import modules.common_params.common_headless as c
import modules.memory_classes.memory_headless as m
import modules.queues.queue_headless as q
import modules.organisms.organism_headless as o
import math
import json
from fungera_headless import FungeraHeadless
import sys

sys.modules['modules.memory'] = m
sys.modules['modules.queue'] = q
sys.modules['modules.organism'] = o
sys.modules['modules.common'] = c


def get_organism_commands(start, size, memory):
    return memory.memory_map[
           start[0]: start[0] + size[0],
           start[1]: start[1] + size[1],
           ]


dir = 'test'

if __name__ == '__main__':
    with open(c.config['snapshot_to_load'], 'rb') as f:
        state = pickle.load(f)
        queue = state['queue']
        memory = state['memory']
        for i, organism in enumerate(queue.organisms):
            filename = os.path.join(dir, f'organim_{i}.gen')
            commands = get_organism_commands(organism.start, organism.size, memory)
            with open(filename, 'w') as f:
                for line in commands:
                    line = ''.join(line.tolist())
                    f.write(f'{line}\n')

    # m.memory.update(refresh=True)

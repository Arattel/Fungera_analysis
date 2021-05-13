import pickle
import traceback
import glob
import os
import numpy as np
import modules.common_params.common_headless as c
import modules.memory_classes.memory_headless as m
import modules.queues.queue_headless as q
import modules.organisms.organism_headless as o
from conf.config import Config, default_ancestors
import math
import json
from tqdm import tqdm
from typing import Dict
from copy import deepcopy
from random import randint
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s|%(filename)s|%(lineno)s| %(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    filename='example.log',
)
logger = logging.getLogger(__name__)

c.screen = None


class FungeraHeadless:
    def __init__(self, no_mutations: bool = False):
        self.timer = c.RepeatedTimer(
            c.config['autosave_rate'], self.save_state, (True,)
        )
        np.random.seed(c.config['random_seed'])
        if not os.path.exists('snapshots'):
            os.makedirs('snapshots')
        self.cycle = 0
        self.is_minimal = True
        self.purges = 0
        self.no_mutations = no_mutations
        coords = np.array(c.config['memory_size']) // 2
        ip = np.copy(coords)
        if c.instructions_set_name == 'error_correction':
            ip = ip + 1

        genome_size = self.load_genome_into_memory(
            self.read_genome(default_ancestors[c.instructions_set_name]), coords
        )
        o.organism_class(coords, genome_size, ip=ip)
        if c.config['snapshot_to_load'] != 'new':
            self.load_state()

        self.information_per_site_tables = []
        self.entropy = 0.0

    def run(self):
        try:
            self.input_stream()
        except KeyboardInterrupt:
            curses.endwin()
            self.timer.cancel()
        except Exception:
            curses.endwin()
            self.timer.cancel()

    def read_genome(self, filename):
        with open(filename) as genome_file:
            genome = np.array([list(line.strip()) for line in genome_file])
            return genome

    def load_genome_into_memory(self, genome, address: np.array) -> np.array:

        m.memory.load_genome(genome, address, genome.shape)
        return genome.shape

    def update_position(self, delta):
        m.memory.scroll(delta)
        q.queue.update_all()

    def toogle_minimal(self, memory=None):
        self.is_minimal = not self.is_minimal
        m.memory.clear()
        m.memory = m.memory.toogle() if memory is None else memory.toogle()
        m.memory.update(refresh=True)
        q.queue.toogle_minimal()

    def save_state(self, from_timer=False):
        return_to_full = False
        if not self.is_minimal:
            if from_timer:
                return
            self.toogle_minimal()
            return_to_full = True
        filename = 'snapshots/{}_cycle_{}.snapshot'.format(
            c.config['simulation_name'].lower().replace(' ', '_'), self.cycle
        )
        if c.config['dump_full_snapshots']:
            with open(filename, 'wb') as f:
                state = {
                    'cycle': self.cycle,
                    'memory': m.memory,
                    'queue': q.queue,
                    'information_per_site': self.information_per_site_tables,
                    'entropy': self.entropy
                }
                pickle.dump(state, f)

        metrics = {
            'cycle': self.cycle,
            'information_per_site': self.information_per_site_tables,
            'entropy': self.entropy,
            'number_of_organisms': len(q.queue.organisms),
            'commands_distribution': self.get_commands_distribution(),
            'sizes': self.get_organism_sizes()
        }
        metrics_file = 'snapshots/{}_cycle_{}.snapshot'.format(
            c.config['simulation_name'].lower().replace(' ', '_'), self.cycle
        ) + '2'
        with open(metrics_file, 'wb') as mf:
            pickle.dump(metrics, mf)

    def load_state(self):
        return_to_full = False
        if not self.is_minimal:
            self.toogle_minimal()
            return_to_full = True
        try:
            if (
                    c.config['snapshot_to_load'] == 'last'
                    or c.config['snapshot_to_load'] == 'new'
            ):
                filename = max(glob.glob('snapshots/*'), key=os.path.getctime)
            else:
                filename = c.config['snapshot_to_load']
            with open(filename, 'rb') as f:
                state = pickle.load(f)
                memory = state['memory']
                q.queue = state['queue']
                self.cycle = state['cycle']
        except Exception as e:
            # print(e)
            pass
        if not self.is_minimal or return_to_full:
            self.toogle_minimal(memory)
        else:
            m.memory = memory
            # self.update_info_minimal()

    def make_cycle(self):
        m.memory.update(refresh=True)
        if self.cycle % c.config['random_rate'] == 0 and not self.no_mutations:
            m.memory.cycle()
        if self.cycle % c.config['cycle_gap'] == 0:
            if m.memory.is_time_to_kill():
                q.queue.kill_organisms()
                self.purges += 1
        if not self.is_minimal:
            q.queue.update_all()
        self.cycle += 1
        # self.update_info()

    @staticmethod
    def calculate_entropy(distribution, num_commands):
        entropy = 0
        for key in distribution:
            p = distribution[key]
            log_p = math.log(p, num_commands)
            entropy -= p * log_p
        return entropy

    def get_commands_distribution(self) -> Dict:
        organisms_commands = []
        for organism in q.queue.organisms:
            organism_commands = self.get_organism_commands(
                organism.start,
                organism.size
            )

            organisms_commands.append(organism_commands.flatten())
        try:
            organisms_commands = np.concatenate(organisms_commands)
            commands, counts = np.unique(organisms_commands, return_counts=True)
            command_counts = dict(zip(commands, counts))
            return command_counts
        except ValueError:
            logger.info(f'{organisms_commands}')
            raise ValueError

    def get_organism_sizes(self):
        sizes = []
        for organism in q.queue.organisms:
            sizes.append(organism.size)
        return sizes

    def get_entropy_score(self):
        max_table_size = [max(q.queue.organisms, key=lambda x: x.size[0]).size[0],
                          max(q.queue.organisms, key=lambda x: x.size[1]).size[1]]

        organisms_commands = []

        # Getting command tables
        for organism in q.queue.organisms:
            organisms_commands.append(self.get_organism_commands(
                organism.start,
                organism.size
            ))

        # Getting frequencies
        values_distributions = [[0 for j in range(max_table_size[1])] for i in range(max_table_size[0])]
        for i in range(max_table_size[0]):
            for j in range(max_table_size[1]):
                values = []
                for commands in organisms_commands:
                    if i < commands.shape[0] and j < commands.shape[1]:
                        values.append(commands[i][j])
                values = {x: values.count(x) / len(values) for x in values}
                values_distributions[i][j] = values

        per_site_entropy = np.zeros(max_table_size)
        for i in range(max_table_size[0]):
            for j in range(max_table_size[1]):
                per_site_entropy[i, j] = self.calculate_entropy(values_distributions[i][j], len(c.instructions))

        self.information_per_site_tables = np.array(per_site_entropy)
        return np.sum(per_site_entropy)
        # total_entropy = 0
        # information_tables = []
        # for organism_commands in organisms_commands:
        #     entropy = 0
        #     entropy_table = np.zeros(organism_commands.shape)
        #     max_entropy_per_site = math.log(len(c.instructions), len(c.instructions))
        #     information_per_site = max_entropy_per_site - entropy_table
        #     for i in range(organism_commands.shape[0]):
        #         for j in range(organism_commands.shape[1]):
        #             p = values_distributions[i][j][organism_commands[i][j]]
        #             entropy -= p * math.log(
        #                 p, len(c.instructions)
        #             )
        #             entropy_table[i, j] = -p * math.log(
        #                 p, len(c.instructions)
        #             )
        #
        #     total_entropy += entropy
        #     information_tables.append(entropy_table)
        # information_tables = np.array(information_tables)
        # self.information_per_site_tables = information_tables
        # return total_entropy

    @staticmethod
    def get_organism_commands(start, size):
        return m.memory.memory_map[
               start[0]: start[0] + size[0],
               start[1]: start[1] + size[1],
               ]

    def find_unqiue_genomes(self):
        all_genomes = []
        for organism in q.queue.organisms:
            all_genomes.append(self.get_organism_commands(organism.start, organism.size))

        unique_genomes = []
        indices = set()

        for i, genome in enumerate(all_genomes):
            indentical_indices = set()

            if i not in indices:
                indentical_indices.add(i)
                indices.add(i)
                for j, another_genome in enumerate(all_genomes):
                    if i != j:
                        if another_genome.shape == genome.shape and (another_genome == genome).all():
                            indices.add(j)
                            indentical_indices.add(j)
                unique_genomes.append((genome, len(indentical_indices)))

        return unique_genomes

    def input_stream(self):
        for i in tqdm(range(100000)):
            if len(q.queue.organisms) == 0:
                break
            q.queue.cycle_all()
            self.make_cycle()
            # print(len(q.queue.organisms))


if __name__ == '__main__':
    print(c.instructions)
    print(c.deltas)
    f = FungeraHeadless(no_mutations=True)
    cnt = 0
    while True:
        q.queue.cycle_all()
        f.make_cycle()
        if len(q.queue.organisms) == 0:
            print('iteration ended')
            f.timer.cancel()

            break
        f.entropy = f.get_entropy_score()
        if cnt % 10:
            print(f'Cycle: {f.cycle}')
            print(f'Entropy: {f.entropy}')
            print(f'Num_organims: {len(q.queue.organisms)}')
        cnt += 1

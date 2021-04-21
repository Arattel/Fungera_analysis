import curses
import pickle
import traceback
import glob
import os
import numpy as np
import modules.common as c
import modules.memory as m
import modules.queue as q
import modules.organism as o
import math
import json

from typing import Dict
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s|%(filename)s|%(lineno)s| %(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    filename='example.log',
)
logger = logging.getLogger(__name__)


class Fungera:
    def __init__(self):
        self.timer = c.RepeatedTimer(
            c.config['autosave_rate'], self.save_state, (True,)
        )
        # print(c.config['random_seed'], c.config['simulation_name'], c.config['random_rate'])
        np.random.seed(c.config['random_seed'])
        if not os.path.exists('snapshots'):
            os.makedirs('snapshots')
        self.cycle = 0
        self.is_minimal = False
        self.purges = 0
        self.info_window = c.screen.derived(
            np.array([0, 0]), c.config['info_display_size'],
        )
        genome_size = self.load_genome_into_memory(
            'alloc_child.gen', address=c.config['memory_size'] // 2
        )
        ip = c.config['memory_size'] // 2
        ip[0] += 1
        o.OrganismFull(c.config['memory_size'] // 2, genome_size,ip=ip)
        self.update_info()
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
            print(traceback.format_exc())

    def load_genome_into_memory(self, filename: str, address: np.array) -> np.array:
        with open(filename) as genome_file:
            genome = np.array([list(line.strip()) for line in genome_file])
        m.memory.load_genome(genome, address, genome.shape)
        return genome.shape

    def update_position(self, delta):
        m.memory.scroll(delta)
        q.queue.update_all()
        self.update_info()

    def update_info_full(self):
        self.info_window.erase()
        info = ''
        info += '[{}]           \n'.format(c.config['simulation_name'])
        info += 'Cycle      : {}\n'.format(self.cycle)
        info += 'Position   : {}\n'.format(list(m.memory.position))
        info += 'Total      : {}\n'.format(len(q.queue.organisms))
        info += 'Purges     : {}\n'.format(self.purges)
        info += 'Organism   : {}\n'.format(q.queue.index)
        info += q.queue.get_organism().info()
        self.info_window.print(info)

    def update_info_minimal(self):
        self.info_window.erase()
        info = ''
        info += 'Minimal mode '
        info += '[Running]\n' if c.is_running else '[Paused]\n'
        info += 'Cycle      : {}\n'.format(self.cycle)
        info += 'Total      : {}\n'.format(len(q.queue.organisms))
        if q.queue.organisms:
            entropy = self.get_entropy_score()

            info += f"Entropy: {entropy}\n"
            self.entropy = entropy
            info += f"Commands distribution: {self.get_commands_distribution()}\n"
            info += f"Organism sizes: {self.get_organism_sizes()}\n"
        else:
            info += "Entropy: 0.0"
            raise ValueError
            # info += f'{m.memory.memory_map[organism_bounds]}'
        self.info_window.print(info)

    def update_info(self):
        if not self.is_minimal:
            self.update_info_full()
        else:
            if self.cycle % c.config['cycle_gap'] == 0:
                self.update_info_minimal()

    def toogle_minimal(self, memory=None):
        self.is_minimal = not self.is_minimal
        self.update_info_minimal()
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
        with open(filename, 'wb') as f:
            # TODO: Uncomment later for dumping both state and metrics
            # state = {
            #     'cycle': self.cycle,
            #     'memory': m.memory,
            #     'queue': q.queue,
            #     'information_per_site': self.information_per_site_tables,
            #     'entropy': self.entropy
            # }
            metrics = {
                'cycle': self.cycle,
                'information_per_site': self.information_per_site_tables,
                'entropy': self.entropy,
                'number_of_organisms': len(q.queue.organisms),
                'commands_distribution': self.get_commands_distribution(),
                'sizes': self.get_organism_sizes()
            }
            # pickle.dump(state, f)
            metrics_file = 'snapshots/{}_cycle_{}.snapshot'.format(
                c.config['simulation_name'].lower().replace(' ', '_'), self.cycle
            ) + '2'
            with open(metrics_file, 'wb') as mf:
                pickle.dump(metrics, mf)
        if not self.is_minimal or return_to_full:
            self.toogle_minimal()

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
        except Exception:
            pass
        if not self.is_minimal or return_to_full:
            self.toogle_minimal(memory)
        else:
            m.memory = memory
            self.update_info_minimal()

    def make_cycle(self):
        m.memory.update(refresh=True)
        if self.cycle % c.config['random_rate'] == 0:
            m.memory.cycle()
        if self.cycle % c.config['cycle_gap'] == 0:
            if m.memory.is_time_to_kill():
                q.queue.kill_organisms()
                self.purges += 1
        if not self.is_minimal:
            q.queue.update_all()
        self.cycle += 1
        self.update_info()

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

        self.information_per_site_tables = 1 - np.array(per_site_entropy)
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

    def input_stream(self):
        while True:
            key = c.screen.get_key()
            if key == ord(' '):
                c.is_running = not c.is_running
                if self.is_minimal:
                    self.update_info_minimal()
            elif key == ord('c') and not c.is_running:
                q.queue.cycle_all()
                self.make_cycle()
            elif key == curses.KEY_DOWN and not self.is_minimal:
                self.update_position(c.config['scroll_step'] * c.deltas['down'])
            elif key == curses.KEY_UP and not self.is_minimal:
                self.update_position(c.config['scroll_step'] * c.deltas['up'])
            elif key == curses.KEY_RIGHT and not self.is_minimal:
                self.update_position(c.config['scroll_step'] * c.deltas['right'])
            elif key == curses.KEY_LEFT and not self.is_minimal:
                self.update_position(c.config['scroll_step'] * c.deltas['left'])
            elif key == ord('d') and not self.is_minimal:
                q.queue.select_next()
                self.update_info()
            elif key == ord('a') and not self.is_minimal:
                q.queue.select_previous()
                self.update_info()
            elif key == ord('m'):
                self.toogle_minimal()
            elif key == ord('p'):
                self.save_state()
            elif key == ord('l'):
                self.load_state()
            elif key == ord('k'):
                q.queue.kill_organisms()
            elif key == -1 and c.is_running:
                q.queue.cycle_all()
                self.make_cycle()
            elif len(q.queue.organisms) == 0:
                break


if __name__ == '__main__':
    c.is_running = False
    Fungera().run()

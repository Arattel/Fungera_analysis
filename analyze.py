from fungera_headless import FungeraHeadless
import modules.common_params.common_headless as c
import modules.memory_classes.memory_headless as m
import modules.queues.queue_headless as q
import modules.organisms.organism_headless as o

from tqdm import tqdm
import numpy as  np
from modules.summarization import get_prefixes, get_metrics_for_prefix, get_summarization_cycles, preprocess_records
from modules.caching import GenomeCache
from conf.config import Config
import os
from fungera_headless import FungeraHeadless


def organism_in_queue(queue, organism_id):
    for organism in queue.organisms:
        if organism.organism_id == organism_id:
            return True
    else:
        return False


def life_length(genome, seed, mutation_rate, max_iterations):
    c.is_running = False
    c.screen = None
    c.config['random_seed'] = seed
    c.config['random_rate'] = mutation_rate
    # c.config['cycle_gap'] = 1
    # c.config['memory_size'] = np.array((900, 900))
    print('Recreating memory...')
    m.memory = m.Memory()
    print(m.memory.memory_map.shape)
    q.queue = q.Queue()

    print(genome.shape)

    coords = np.array(c.config['memory_size']) // 2
    ip = np.copy(coords)
    if c.instructions_set_name == 'error_correction':
        ip = ip + 1
    genome_size = f.load_genome_into_memory(
        genome, coords)

    o.organism_class(coords, genome_size, ip=ip)
    alive_time = 0
    for i in range(max_iterations):
        q.queue.cycle_all()
        f.make_cycle()
        is_alive = organism_in_queue(q.queue, organism_id=organism.organism_id)
        q.queue.kill_organisms()
        if not is_alive:
            alive_time = i
            break

    f.timer.cancel()
    return alive_time


def get_expected_life_length(genome, num_iterations):
    life_lengths = []
    for i in tqdm(range(num_iterations)):
        life_lengths.append(life_length(genome, seed=i, mutation_rate=2, max_iterations=50000))
    return sum(life_lengths) / num_iterations


def is_replicator(genome, max_iterations: int = 50000):
    c.is_running = False
    c.screen = None
    c.config['random_rate'] = 1
    c.config['memory_size'] = np.array((500, 500))
    f = FungeraHeadless(no_mutations=True)
    m.memory = m.Memory()
    q.queue = q.Queue()
    coords = np.array(c.config['memory_size']) // 2
    ip = np.copy(coords)
    if c.instructions_set_name == 'error_correction':
        ip = ip + 1
    genome_size = f.load_genome_into_memory(
        genome, coords)

    o.organism_class(coords, genome_size, ip=ip)
    has_child = False
    for i in tqdm(range(max_iterations)):
        q.queue.cycle_all()
        f.make_cycle()
        if len(q.queue.organisms) >= 2:
            has_child = True

            break
    f.timer.cancel()

    if not has_child:
        return False
    elif has_child:
        parent_organism = q.queue.organisms[0]
        child_organism = q.queue.organisms[1]

        parent_organism_code = get_organism_commands(start=parent_organism.start, size=parent_organism.size)
        child_organism_code = get_organism_commands(start=child_organism.start, size=child_organism.size)

        if parent_organism_code.shape != child_organism_code.shape:
            return False
        else:
            return (parent_organism_code == child_organism_code).all()


def get_data_for_snapshot(snapshot_path):
    c.is_running = False
    path = snapshot_path
    c.config['snapshot_to_load'] = path
    f = FungeraHeadless()
    cycle = f.cycle
    entropy = f.get_entropy_score()
    per_site_entropy = f.information_per_site_tables
    genomes = f.find_unqiue_genomes()
    f.timer.cancel()

    records = []
    for genome, num_entries in tqdm(genomes):
        if gc.key_in_cache(genome):
            record = gc.get_key(
                key=genome
            )
            record['num_organisms'] = num_entries
            records.append(record)
        else:
            record = {
                'genome_size': genome.shape,
                'num_organisms': num_entries,
                'expected_life_length': get_expected_life_length(genome, 10),
                'genotype': genome,
                'is_replicator': is_replicator(genome)
            }
            records.append(record)
            del record['num_organisms']
            gc.set_key(key=genome, value=record)

    output_data = {
        'records': records,
        'cycle': cycle,
        'entropy': entropy,
        'path': path,
        'per_site_entropy': per_site_entropy
    }
    return output_data


if __name__ == '__main__':
    gc = GenomeCache(path=Config.caching_path)

    input_dir = c.config['input_dir']
    snapshot_filenames = os.listdir(input_dir)
    prefixes = get_prefixes(snapshot_filenames)
    print('Snapshot_filenames', snapshot_filenames)

    paths = []
    for prefix in tqdm(prefixes):

        metrics = get_metrics_for_prefix(
            prefix=prefix,
            directory=input_dir,
            files=snapshot_filenames
        )
        metrics = preprocess_records(records=metrics)

        if metrics.shape[0] < 50:
            summarization_cycles = metrics.cycles.tolist()
        else:
            summarization_cycles = get_summarization_cycles(
                metrics_df=metrics, max_derivative_points=Config.max_derivative_points,
                smoothing_window=Config.smoothing_window
            )
            print(len(summarization_cycles))
        for cycle in summarization_cycles:
            snapshot_path: str = os.path.join(
                f'{prefix}_cycle_{cycle + 1}.snapshot'
            )
            paths.append(snapshot_path)

    paths = list(map(lambda x: os.path.join(input_dir, x), paths))
    snapshot_filenames = list(map(lambda x: os.path.join(input_dir, x), snapshot_filenames))
    snapshot_filenames = list(filter(lambda x: x.endswith('.snapshot'), snapshot_filenames))

    data = []

    for snapshot in tqdm(paths, desc='Paths:'):
        record_data = get_data_for_snapshot(snapshot_path=snapshot)
        data.append(record_data)
    pickle.dump(data,
                open(c.config['output_file'], 'wb'))

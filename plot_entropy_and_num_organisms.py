import argparse
import pickle
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib

matplotlib.rcParams.update({'font.size': 15})

parser = argparse.ArgumentParser(description='Plot data file into image')

parser.add_argument('-i', '--input',
                    help='Input .data file')

parser.add_argument('-o', '--out',
                    help='Output .png file')

args = parser.parse_args()

input_file = args.input
output_file = args.out

data = None
with open(input_file, 'rb') as file:
    data = pickle.load(file)

records = []
for record in data:
    total_organisms = 0
    for genome in record['records']:
        total_organisms += genome['num_organisms'] if 'num_organisms' in genome else 1
    record = {
        'num_organims': total_organisms,
        'cycle': record['cycle'],
        'entropy': record['entropy']
    }
    records.append(record)

data_df = pd.DataFrame.from_records(records)
data_df = data_df.sort_values(by='cycle')

# Create some mock data

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('time (cycles)')
ax1.set_ylabel('num organisms', color=color)
ax1.plot(data_df.cycle, data_df.num_organims, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('entropy', color=color)  # we already handled the x-label with ax1
ax2.plot(data_df.cycle, data_df.entropy, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(output_file)

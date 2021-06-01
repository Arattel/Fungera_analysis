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

import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('messi.jpg')
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
font_color = (255, 255, 255)
thick = 1
font_size = 0.9


class SnapshotMap:
    def __init__(self, cells=np.array([40, 40]), padding=10, cellsize=np.array([20, 20])):
        size = np.multiply(cells, cellsize)
        size = size + padding * 2
        self.image = np.zeros((*size, 3), np.uint8)
        self.padding = padding
        self.grid = cells
        self.cellsize = cellsize

    def _create_grid(self):
        color = (255, 0, 0)
        thickness = 2

        for i in range(self.grid[0]):
            for j in range(self.grid[1]):
                start = np.array((i * self.cellsize[0], j * self.cellsize[1])) + self.padding
                end = start + self.cellsize

                self.image = cv2.rectangle(self.image, start, end, 10000, 1)

    def show(self):
        plt.imshow(self.image, cmap='Greys')
        plt.show()

    def _putchar(self, coords, char):
        coords = np.multiply(self.cellsize, np.array(coords) + 1)
        offset = np.array([10, -10])
        coords[0] -= self.padding
        coords[1] += self.padding
        coords = coords + offset

        self.image = cv2.putText(self.image, char, coords, font, .5, (255, 255, 255), thick, cv2.LINE_AA)

    def frommatrix(self, matrix):
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                self._putchar((i, j), matrix[i][j])

    def draw_bounds(self, start, end):
        start = np.multiply(np.array(start), self.cellsize) + self.padding
        end = np.multiply(np.array(end), self.cellsize) + self.padding
        self.image = cv2.rectangle(self.image, start, end, (255, 0, 0), -1)
        self.image = cv2.rectangle(self.image, start, end, (255, 255, 255), 1)




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

        snapshot = SnapshotMap(cells=memory.memory_map.shape)

        # print(snapshot.image)
        # snapshot.show()

        for i, organism in enumerate(queue.organisms):
            commands = snapshot.draw_bounds(organism.start, organism.size)
        snapshot.frommatrix(memory.memory_map)
        cv2.imwrite(c.config['output_file'], snapshot.image)

    # m.memory.update(refresh=True)

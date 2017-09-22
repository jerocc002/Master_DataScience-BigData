import multiprocessing
import random
from collections import defaultdict

from copy import copy, deepcopy
from itertools import permutations, combinations
from threading import Thread

import time

from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import euclidean

from optimization.problem import Problem
from optimization.tabu import TabuSearch
from problems.tsp.tsplib import get_cities, get_cities_tup, plot_cities
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
class ClusteringProblem(Problem):

    def __init__(self, samples, k, candidates_by_iteration=100, delta=1):
        self.samples = pd.DataFrame.from_records(data=samples)
        self.k = k
        self.delta = delta
        self.candidates_by_iteration = candidates_by_iteration
        self.draw_process = None
        self.draw_queue = multiprocessing.Queue()
        self.initilized_graph = False

    def get_neighborhood(self, cand):
        # generate candidates
        # candidates = [self.two_opt(cand) for _ in range(0, self.candidates_by_iteration)]
        candidates = []
        while len(candidates) < self.candidates_by_iteration:
            candidates += [self.get_neighbor(cand)]

        return candidates[:self.candidates_by_iteration]

    def get_neighbor(self, cand):
        new_cand = deepcopy(cand)
        i = random.randint(0, self.k-1)
        j = random.randint(0, len(cand[i])-1)

        p = new_cand[i]
        p[j] = p[j] + random.uniform(-self.delta, self.delta)
        new_cand[i] = p
        #new_cand[i] = list(self.samples.iloc[random.randint(0, self.samples.shape[0]-1)].values)

        return new_cand

    def get_initial_solution(self):
        return [list(self.samples.iloc[random.randint(0, self.samples.shape[0])].values) for _ in range(self.k)]

    def fitness(self, cand):
        distances = np.zeros(shape=(self.samples.shape[0], self.k))
        for i in range(self.k):
            result = np.sqrt(((self.samples - cand[i])**2).sum(axis=1))

            distances.T[i] = result.values

        return np.power(distances[np.arange(distances.shape[0]), distances.argmin(axis=1)], 2).sum()

    def plot(self, sol):

        if self.draw_process is None:
            self.draw_process = multiprocessing.Process(None, self._plot, args=(self.draw_queue,))
            self.draw_process.start()

        self.draw_queue.put(sol)

    def _plot(self, queue):

        if not self.initilized_graph:
            plt.ion()
            plt.show()
            fig = plt.figure(1, figsize=(4, 3))
            ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
            self.initilized_graph = True

        while True:
            try:
                sol = queue.get_nowait()
                sol = np.array(sol)

                ax.cla()

                ax.scatter(self.samples.iloc[:, 0], self.samples.iloc[:, 1], self.samples.iloc[:, 2])

                ax.scatter(sol[:, 0], sol[:, 1], sol[:, 2], c='r', marker='x', s=100)
                #plt.plot(*zip(*cities_tups))
                plt.pause(0.01)
            except:
                plt.pause(0.01)



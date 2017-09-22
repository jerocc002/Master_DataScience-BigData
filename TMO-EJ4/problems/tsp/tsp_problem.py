import multiprocessing
import random
from collections import defaultdict

from copy import copy
from itertools import permutations, combinations
from threading import Thread

import time
from scipy.spatial.distance import euclidean

from optimization.problem import Problem
from optimization.tabu import TabuSearch
from problems.tsp.tsplib import get_cities, get_cities_tup, plot_cities
import matplotlib.pyplot as plt

class TSPProblem(Problem):

    def __init__(self, cities, candidates_by_iteration=100, neighborhood=1, feature_penalty=0.1):
        self.cities = cities
        self.candidates_by_iteration = candidates_by_iteration
        self.initilized_graph = False
        self.draw_queue = multiprocessing.Queue()
        self.neighborhood = neighborhood
        self.draw_process = None
        self.penalties = defaultdict(lambda: 0)
        self.feature_penalty = feature_penalty

    def get_neighborhood(self, cand):
        # generate candidates
        # candidates = [self.two_opt(cand) for _ in range(0, self.candidates_by_iteration)]
        candidates = []
        while len(candidates) < self.candidates_by_iteration:
            if self.neighborhood == 1:
                candidates += [self.swap(cand)]
            else:
                candidates += self.k_opt(cand, self.neighborhood-1)

        return candidates[:self.candidates_by_iteration]


    def two_opt(self, cand):
        # swap
        i = 0
        j = 0
        while i == j:
            i, j = random.randint(0, len(cand) - 1), random.randint(0, len(cand) - 1)

        i, j = min(i,j), max(i,j)
        return self.__two_opt(cand, i, j)

    def __two_opt(self, cand, i, j):
        new_cand = copy(cand)
        new_cand[i:j] = new_cand[i:j][::-1]
        return new_cand


    def k_opt(self, cand, k):
        nodes = []
        while len(nodes) < k*2:
            nodes.append(random.randint(0, len(cand) - 1))
            nodes = list(set(nodes))

        nodes = sorted(nodes)
        nodes = list(zip(nodes[0::2], nodes[1::2]))

        if k > 1:
            interchange_routes = self.__interchange_routes(cand, nodes)
        else:
            interchange_routes = [copy(cand)]

        return self.reverse(cand, interchange_routes, nodes)

    def reverse(self, cand, interchange_routes, nodes):
        perms = self.__flat_list([combinations(range(len(nodes)), i) for i in range(1, len(nodes) + 1)])
        result = []
        for p in perms:
            for route in interchange_routes:
                new_cand = route[:]
                for path in p:
                    new_cand[nodes[path][0]:nodes[path][1]] = new_cand[nodes[path][0]:nodes[path][1]][::-1]
                result.append(new_cand)
        result += interchange_routes
        return [list(i) for i in set(tuple(i) for i in result) if list(i) != cand]

    def __interchange_routes(self, cand, routes):
        result = []
        i = 0
        out_paths = []
        for r in routes:
            out_paths.append(cand[i:r[0]])
            i = r[1]+1
        out_paths.append(cand[i:])

        per = permutations(range(len(routes)), len(routes))

        for p in per:
            in_paths = [cand[routes[i][0]:routes[i][1]+1] for i in p]
            new_cand = [out_paths[0]] + self.__flat_list(list(zip(in_paths, out_paths[1:])))
            new_cand = self.__flat_list(new_cand)

            result.append(new_cand)

        return [list(i) for i in set(tuple(i) for i in result)]


    def __flat_list(self, l):
        return [item for sublist in l for item in sublist]


    def swap(self, cand, n=1):
        # swap
        new_cand = copy(cand)
        for i in range(n):
            nodes = []
            while len(nodes) < 2:
                nodes.append(random.randint(0, len(cand) - 1))
                nodes = list(set(nodes))

            i, j = nodes

            new_cand = copy(new_cand)
            new_cand[i], new_cand[j] = new_cand[j], new_cand[i]
        return new_cand




    def get_initial_solution(self):
        return random.sample(range(len(self.cities)), len(self.cities))

    def fitness(self, cand):
        return sum([self.get_feature_cost((src, dst)) + self.feature_penalty * self.penalties[(src, dst)]
                    for src, dst in zip(cand[:-1], cand[1:])]) + \
                    self.get_feature_cost((0, len(cand)-1)) + \
                        self.feature_penalty * self.penalties[(0, len(cand)-1)]

    def get_features(self, sol):
        return [(i, j) for i, j in zip(sol[:-1], sol[1:])] + [(0, len(sol)-1)]

    def get_feature_cost(self, feature):
        i, j = feature
        return euclidean(self.cities[i], self.cities[j])

    def plot_cities(self, cities_tups):

        if self.draw_process is None:
            self.draw_process = multiprocessing.Process(None, self.__plot_cities, args=(self.draw_queue,))
            self.draw_process.start()

        self.draw_queue.put(cities_tups)

    def __plot_cities(self, queue):

        if not self.initilized_graph:
            plt.ion()
            plt.show()
            self.initilized_graph = True

        while True:
            try:
                cities_tups = queue.get_nowait()

                plt.clf()
                plt.scatter(*zip(*cities_tups))
                plt.plot(*zip(*cities_tups))
                plt.pause(0.01)
            except:
                plt.pause(0.01)



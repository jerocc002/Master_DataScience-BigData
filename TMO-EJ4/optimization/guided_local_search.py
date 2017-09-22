from abc import abstractmethod
from collections import defaultdict
from copy import copy

from optimization.base import BaseSearch

class GLS(BaseSearch):

    def __init__(self, local_search, problem, max_iterations, tolerance):
        self.max_iterations = max_iterations
        self.problem = problem
        self.tolerance = tolerance
        self.local_search = local_search


    def stop(self):
        return self.max_iterations <= self.current_iteration or \
               (self.current_iteration - self.last_improving_iteration) >= self.tolerance

    def search(self):
        self.current_iteration = 0
        self.best, self.best_fitness = self.local_search.search()
        current_sol = copy(self.best)
        print("Initial sol %f" % self.best_fitness)

        while not self.stop():
            current_sol, fitness = self.local_search.search(initial_solution=current_sol)

            utilities = []
            for feature in self.problem.get_features(current_sol):
                utility = self.problem.get_feature_cost(feature) / (1 + self.problem.penalties[feature])
                utilities.append((feature, utility))

            penalized_feature = sorted(utilities, key=lambda x: x[1])[0][0]
            penalized_feature = tuple(sorted(list(penalized_feature)))
            self.problem.penalties[penalized_feature] += 1

            if fitness < self.best_fitness:
                self.best = current_sol
                self.best_fitness = fitness
                self.last_improving_iteration = self.current_iteration

                print("Found improving %f" % self.best_fitness)

            self.current_iteration += 1


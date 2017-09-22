from abc import abstractmethod
from copy import copy

from optimization.base import BaseSearch

class ILS(BaseSearch):

    def __init__(self, local_search):
        self.local_search = local_search

    @abstractmethod
    def perturb(self, sol):
        raise Exception("Not implement")

    @abstractmethod
    def accept(self, sol):
        raise Exception("Not implement")

    def search(self):

        self.current_iteration = 0
        self.best, self.best_fitness = self.local_search.search()
        current_sol = copy(self.best)
        print("Initial sol %f" % self.best_fitness)

        while not self.stop():
            current_sol = self.perturb(current_sol)
            current_sol, fitness = self.local_search.search(initial_solution=current_sol)
            current_sol = self.accept(current_sol)

            if fitness < self.best_fitness:
                self.best = current_sol
                self.best_fitness = fitness
                self.last_improving_iteration = self.current_iteration

                print("Found improving %f" % self.best_fitness)

            self.current_iteration += 1


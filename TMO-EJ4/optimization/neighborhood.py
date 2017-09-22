import random
from copy import copy

from optimization.base import BaseSearch


class VND(BaseSearch):

    def __init__(self, local_search, problem, neighborhoods):
        self.local_search = local_search
        self.neighborhoods = neighborhoods
        self.problem = problem


    def search(self, initial_solution=None):

        self.current_iteration = 0
        if initial_solution:
            self.best = initial_solution
        else:
            self.best = self.problem.get_initial_solution()
        self.best_fitness = self.problem.fitness(self.best)
        current_sol = copy(self.best)
        current_fitness = self.best_fitness
        print("Initial sol %f" % self.best_fitness)
        n = 0

        while n < len(self.neighborhoods):
            self.problem.neighborhood = self.neighborhoods[n]
            cand_sol, cand_fitness = self.local_search.search(initial_solution=current_sol)

            if cand_fitness < self.best_fitness:
                self.best = current_sol
                self.best_fitness = cand_fitness
                self.last_improving_iteration = self.current_iteration

                print("Found improving %f in neighborhood %s" % (self.best_fitness, self.neighborhoods[n]))

            if cand_fitness < current_fitness:
                current_sol = cand_sol
                current_fitness = cand_fitness
                n = 0 if n > 0 else n + 1
                if n < len(self.neighborhoods):
                    print("Next neighborhood %s" % self.neighborhoods[n])
            else:
                n += 1
                if n < len(self.neighborhoods):
                    print("Next neighborhood %s" % self.neighborhoods[n])



            self.current_iteration += 1

        return self.best, self.best_fitness


class VNS(BaseSearch):

    def __init__(self, local_search, problem, neighborhoods):
        self.local_search = local_search
        self.neighborhoods = neighborhoods
        self.problem = problem


    def search(self, initial_solution=None):

        self.current_iteration = 0
        if initial_solution:
            self.best = initial_solution
        else:
            self.best = self.problem.get_initial_solution()
        self.best_fitness = self.problem.fitness(self.best)
        current_sol = copy(self.best)
        print("VND. Initial sol %f" % self.best_fitness)
        n = 0

        while n < len(self.neighborhoods):
            self.problem.neighborhood = self.neighborhoods[n]
            print("VND. Get random neighbor")
            cands = self.problem.get_neighborhood(current_sol)
            current_sol = cands[random.randint(0, len(cands)-1)]
            print("VND. Start local search in neighborhood %s" % self.neighborhoods[n])
            current_sol, fitness = self.local_search.search(
                initial_solution=current_sol)

            if fitness < self.best_fitness:
                self.best = current_sol
                self.best_fitness = fitness
                self.last_improving_iteration = self.current_iteration

                print("VND. Found improving %f in neighborhood %s" % (self.best_fitness, self.neighborhoods[n]))
                n = 0 if n > 0 else n + 1
                if n < len(self.neighborhoods):
                    print("VND. Next neighborhood %s" % self.neighborhoods[n])
            else:
                n += 1
                if n < len(self.neighborhoods):
                    print("VND. Next neighborhood %s" % self.neighborhoods[n])

            self.current_iteration += 1

        return self.best, self.best_fitness
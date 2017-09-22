import copy
from abc import abstractmethod


class BaseSearch:
    best = None
    best_fitness = 0
    current_iteration = 0
    last_improving_iteration = 0
    verbose = True
    improved_event = None

    @abstractmethod
    def stop(self):
        raise Exception('Not implemented')

    @abstractmethod
    def search(self, initial_solution=None):


        self.current_iteration = 0
        self.last_improving_iteration = 0
        current_sol, fitness = self.initialize(initial_solution)

        self.best = current_sol
        self.best_fitness = fitness

        if self.improved_event:
            self.improved_event(self)

        if self.verbose:
            print("Initial sol %f" % self.best_fitness)
        while not self.stop():
            current_sol, fitness = self.do_iteration(current_sol)

            if fitness < self.best_fitness:
                self.best = copy.deepcopy(current_sol)
                self.best_fitness = fitness
                self.last_improving_iteration = self.current_iteration

                if self.improved_event:
                    self.improved_event(self)

                if self.verbose:
                    print("Found improving %f" % self.best_fitness)

            self.current_iteration += 1

        return self.best, self.best_fitness

    @abstractmethod
    def initialize(self, initial_solution=None):
        raise Exception("Not implemented")

    @abstractmethod
    def do_iteration(self, current_sol):
        raise Exception("Not implemented")


class SingleSolutionSearch(BaseSearch):
    problem = None

    @abstractmethod
    def select(self, cands):
        raise Exception('Not implemented')

    def initialize(self, initial_solution=None):
        if initial_solution is None:
            self.current_iteration = 0
            current_sol = self.problem.get_initial_solution()
        else:
            current_sol = initial_solution

        return current_sol, self.problem.fitness(current_sol)

    def do_iteration(self, current_sol):
        # get candidates
        cands = self.problem.get_neighborhood(current_sol)

        return self.select(cands)

from copy import copy

from optimization.ils import ILS
from optimization.tabu import TabuSearch
from problems.tsp.tsp_problem import TSPProblem
from problems.tsp.tsplib import get_cities_tup, plot_cities

class ILSTSP(ILS):

    def __init__(self, local_search, max_iterations, perturbation_level=1):
        super(ILSTSP, self).__init__(local_search)

        self.max_iterations = max_iterations
        self.perturbation_level = perturbation_level

    def stop(self):
        return self.current_iteration >= self.max_iterations

    def accept(self, sol):
        return sol

    def perturb(self, sol):
        sol = copy(sol)
        sol = self.local_search.problem.swap(sol, self.perturbation_level)

        return sol

if __name__ == '__main__':
    def improve(caller):
        current = [cities[i] for i in caller.best] + [cities[caller.best[0]]]
        problem.plot_cities(current)

    cities = get_cities_tup(file='berlin52.tsp')
    problem = TSPProblem(cities, candidates_by_iteration=200,
                         neighborhood=2)
    searcher = TabuSearch(problem, max_iterations=100, list_length=10,
                          verbose=True, improved_event=improve)

    ils = ILSTSP(searcher, max_iterations=10,
                 perturbation_level=2)
    ils.search()

    best = [cities[i] for i in searcher.best] + [cities[searcher.best[0]]]
    problem.plot_cities(best)
    print("Best score: %f" % problem.best_fitness)
    print("Finish")



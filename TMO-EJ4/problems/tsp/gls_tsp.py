from optimization.guided_local_search import GLS
from optimization.neighborhood import VND, VNS
from optimization.tabu import TabuSearch
from problems.tsp.tsp_problem import TSPProblem
from problems.tsp.tsplib import get_cities_tup
import numpy as np

if __name__ == '__main__':
    def improve(caller):
        current = [cities[i] for i in caller.best] + [cities[caller.best[0]]]
        problem.plot_cities(current)


    cities = get_cities_tup(file='berlin52.tsp')
    problem = TSPProblem(cities, candidates_by_iteration=200, feature_penalty=12, neighborhood=2)
    searcher = TabuSearch(problem, max_iterations=10, list_length=10, verbose=True, improved_event=improve)

    vnd = GLS(searcher, problem, max_iterations=10000, tolerance=100)
    vnd.search()

    problem.plot_cities([cities[i] for i in vnd.best])

    problem.feature_penalty = 0
    print("Final cost: %f" % problem.fitness(vnd.best))

    best = [cities[i] for i in searcher.best] + [cities[searcher.best[0]]]
    problem.plot_cities(best)

    print("Finish")

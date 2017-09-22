from optimization.tabu import TabuSearch
from problems.tsp.tsp_problem import TSPProblem
from problems.tsp.tsplib import get_cities_tup

if __name__ == '__main__':
    def improve(caller):
        current = [cities[i] for i in caller.best] + [cities[caller.best[0]]]
        problem.plot_cities(current)

    cities = get_cities_tup(file='berlin52.tsp')
    problem = TSPProblem(cities, candidates_by_iteration=200, neighborhood=2)
    searcher = TabuSearch(problem, max_iterations=1000, list_length=10,
                          improved_event=improve)
    searcher.search()

    best = [cities[i] for i in searcher.best] + [cities[searcher.best[0]]]
    problem.plot_cities(best)
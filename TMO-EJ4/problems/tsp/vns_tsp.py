from optimization.neighborhood import VND, VNS
from optimization.tabu import TabuSearch
from problems.tsp.tsp_problem import TSPProblem
from problems.tsp.tsplib import get_cities_tup

if __name__ == '__main__':
    def improve(caller):
        current = [cities[i] for i in caller.best] + [cities[caller.best[0]]]
        problem.plot_cities(current)


    cities = get_cities_tup(file='berlin52.tsp')
    problem = TSPProblem(cities, candidates_by_iteration=300)
    searcher = TabuSearch(problem, max_iterations=100,
                          list_length=10, verbose=True,
                          improved_event=improve)

    vnd = VNS(searcher, problem, [1, 2, 3, 4, 5])
    vnd.search()

    best = [cities[i] for i in searcher.best] + [cities[searcher.best[0]]]
    problem.plot_cities(best)
    print("Finish")

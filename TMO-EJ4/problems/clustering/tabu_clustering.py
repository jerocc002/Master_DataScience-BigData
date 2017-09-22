from sklearn.cluster import KMeans
from sklearn.preprocessing import scale

from optimization.tabu import TabuSearch
from problems.clustering.cluster_problem import ClusteringProblem
from problems.tsp.ils_tsp import ILSTSP
from problems.tsp.tsp_problem import TSPProblem
from problems.tsp.tsplib import get_cities_tup
from sklearn import datasets
import numpy as np
if __name__ == '__main__':
    def improve(caller):
        problem.plot(caller.best)


    #digits = datasets.load_digits()
    #num_clusters = len(np.unique(digits.target))
    #X, y = scale(digits.data), iris.target
    iris = datasets.load_iris()
    X, y = scale(iris.data), iris.target
    num_clusters = 3

    print("Dataset shape: %d,%d" % X.shape)
    problem = ClusteringProblem(X, k=num_clusters, delta=0.1,
                                candidates_by_iteration=100)

    sol = None
    for i in range(100):
        searcher = TabuSearch(problem, max_iterations=10000000,
                              list_length=10, tolerance=1000,
                              improved_event=improve)

        searcher.search(initial_solution=sol)
        sol = searcher.best
        for i in range(10):
            sol = problem.get_neighbor(sol)


import copy

from optimization.base import SingleSolutionSearch

class TabuSearch(SingleSolutionSearch):

    def __init__(self, problem, list_length=100, candidates_by_iteration=10, max_iterations=100,
                 tolerance=10, improved_event=None, verbose=True):
        self.list_length = list_length
        self.candidates_by_iteration = candidates_by_iteration
        self.tabu_list =  []
        self.problem = problem
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.improved_event = improved_event
        self.tolerance=tolerance

    def select(self, cands):
        # excluded candidates into tabu list
        aux_cands = copy.deepcopy(cands)
        cands = [c for c in cands if not c in self.tabu_list]

        # evaluate candidates
        cands = sorted([(c, self.problem.fitness(c)) for c in cands], key=lambda x: x[1], reverse=False)

        # update the best if found improving into candidates
        #if cands[0][1] < self.best_fitness:

        if len(self.tabu_list) == self.list_length:
            self.tabu_list = self.tabu_list[1:]

        if len(cands) == 0:
            return aux_cands[0], self.problem.fitness(aux_cands[0])

        self.tabu_list.append(cands[0][0])

        return cands[0]

    def stop(self):
        return self.max_iterations <= self.current_iteration or \
               (self.current_iteration - self.last_improving_iteration) >= self.tolerance







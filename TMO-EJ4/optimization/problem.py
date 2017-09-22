from abc import abstractmethod


class Problem:

    @abstractmethod
    def get_initial_solution(self):
        '''
        Return a solution generate randomly
        :return: A candidate solution
        '''
        raise Exception('Not implemented')

    @abstractmethod
    def fitness(self, cand):
        '''
        Return the quality of the cand solution
        :param cand: A candidate solution
        :return: Quality of the solution (0 is worst)
        '''
        raise Exception('Not implemented')

    @abstractmethod
    def get_neighborhood(self):
        '''
        Return a list of new candidates from 'cand' transformations
        :param cand: A candidate solution
        :return: list of new solutions
        '''
        raise Exception('Not implemented')
import multiprocessing
import random
import time
import traceback
import pickle as pk
import json
import numpy as np
import matplotlib.pyplot as plt
import copy
import cv2

from optimization.neighborhood import VNS, VND
from optimization.problem import Problem
from optimization.tabu import TabuSearch



class PolygonsProblem(Problem):
    def __init__(self, target_image, num_shapes, candidates_by_iteration=100, max_edges=10,
                 polygon_list=None, delta=1, neighborhood='all', sol_file='sol.png',
                 vnx=''):
        self.target_image = target_image
        self.target_image_diff = target_image.astype('int16')
        self.num_shapes = num_shapes
        self.candidates_by_iteration = candidates_by_iteration
        self.initialized_graph = False
        self.draw_queue = multiprocessing.Queue()
        self.draw_process = None
        self.polygon_list = polygon_list
        self.fitness_count = 0
        self.fitness_time = 0
        self.sol_file = sol_file
        self.max_edges = max_edges
        self.delta = delta
        self.neighborhood = neighborhood
        self.vnx = vnx

    def get_neighborhood(self, cand, neighborhood=None, num_candidates=None):
        if neighborhood is None:
            neighborhood = self.neighborhood
        if num_candidates is None:
            num_candidates = self.candidates_by_iteration
        candidates = []
        while len(candidates) < num_candidates:
            if neighborhood == 'all':
                _neighborhood = random.choice(['move', 'color', 'add', 'remove'])
                candidates += self.get_neighborhood(cand, _neighborhood, 1)
            elif neighborhood == 'move':
                candidates.append(self.__move__neighbor(cand))
            elif neighborhood == 'color':
                candidates.append(self.__color_neighbor(cand))
            elif neighborhood == 'add':
                polygons = cand
                if self.polygon_list:
                    polygons = [cand[i] for i in self.polygon_list]
                if min([len(p) for p, c in polygons]) < self.max_edges:
                    new_cand = self.__add__neighbor(cand)
                    if new_cand is not None:
                        candidates.append(new_cand)
                else:
                    candidates.append(cand)
            elif neighborhood == 'remove':
                polygons = cand
                if self.polygon_list:
                    polygons = [cand[i] for i in self.polygon_list]
                if max([len(p) for p, c in polygons]) > 3:
                    new_cand = self.__remove__neighbor(cand)
                    if new_cand is not None:
                        candidates.append(new_cand)
                else:
                    candidates.append(cand)
        return candidates

    def __remove__neighbor(self, cand):
        # raise Exception("Implementar")
        if self.polygon_list:
            i = random.choice(self.polygon_list)
        else:
            i = random.randint(0, len(cand) - 1)
        cand = copy.deepcopy(cand)
        cand[i][0].remove(random.choice(cand[i][0]))
        cand[i] = cand[i][0], cand[i][1]
        return cand

    def __add__neighbor(self, cand):
        # raise Exception("Implementar")
        if self.polygon_list:
            i = random.choice(self.polygon_list)
        else:
            i = random.randint(0, len(cand) - 1)
        cand = copy.deepcopy(cand)
        if len(cand[i][0]) < self.max_edges:
            cand[i][0].insert(random.randint(0, len(cand[i][0]) - 1),
                              self.get_random_point())
        cand[i] = cand[i][0], cand[i][1]
        return cand

    def __color_neighbor(self, cand):
        # raise Exception("Implementar")
        if self.polygon_list:
            i = random.choice(self.polygon_list)
        else:
            i = random.randint(0, len(cand) - 1)
        cand = copy.deepcopy(cand)
        cand[i] = cand[i][0], self.__perturb_color(cand[i][1])
        return cand

    def __move__neighbor(self, cand):
        # raise Exception("Implementar")
        if self.polygon_list:
            i = random.choice(self.polygon_list)
        else:
            i = random.randint(0, len(cand) - 1)
        cand = copy.deepcopy(cand)
        j = random.randint(0, len(cand[i][0])-1)
        cand[i][0][j] = self.__perturb(*cand[i][0][j])
        return cand

    def __perturb(self, x, y):
        h, w = self.target_image.shape[:2]
        offset = 10
        if random.choice([True, False]):
            x = max(min(x + random.randint(-offset, offset), w), 0)
        else:
            y = max(min(y + random.randint(-offset, offset), h), 0)
        return x, y

    def __perturb_color(self, color):
        offset = 10
        i = random.randint(0, 3)
        color = copy.deepcopy(color)
        color[i] = max(min(color[i] + random.randint(-min(offset, color[i]), min(offset, 255 - color[i])), 255), 0)
        return color

    def get_initial_solution(self):
        # raise Exception("Implementar")
        return [[self.get_random_polygon(), self.get_random_color()] for _ in range(self.num_shapes)]

    def get_random_color(self):
        # raise Exception("Implementar")
        return [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]

    def get_random_polygon(self):
        # raise Exception("Implementar")
        num_edges = random.randint(3, self.max_edges)
        return [self.get_random_point() for _ in range(num_edges)]

    def get_random_point(self):
        # raise Exception("Implementar")
        h, w = self.target_image.shape[:2]
        return random.randint(0, w), random.randint(0, h)

    def get_sol_diff(self, cand):
        h, w = self.target_image.shape[:2]
        sol = self.create_image_from_sol(cand)
        diff = self.target_image_diff - sol
        return diff

    def fitness_new(self, cand):
        start_time = time.time()
        h, w = self.target_image.shape[:2]
        diff = np.abs(self.get_sol_diff(cand)).sum()
        _fitness = (100 * diff / (w * h * 3 * 255)) ** 2
        self.fitness_time += (time.time() - start_time)
        self.fitness_count += 1
        return _fitness

    def fitness_old(self, cand):
        start_time = time.time()
        h, w = self.target_image.shape[:2]
        sol = self.create_image_from_sol(cand)
        diff = np.abs(self.target_image - sol).sum()
        _fitness = (100 * diff / (w * h * 3 * 255)) ** 2
        self.fitness_time += (time.time() - start_time)
        self.fitness_count += 1
        return _fitness

    def fitness(self,cand):
        _fitness = self.fitness_new(cand)
        return _fitness

    def create_image_from_sol(self, cand, to_rgb=True):
        h, w = self.target_image.shape[:2]
        sol = np.zeros((h, w, 4), np.uint8)
        for shape, color in cand:
            overlay = sol.copy()
            pts = np.array(shape)
            pts = pts.reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, 0.8, sol, 0.2, 0, sol)
        if to_rgb:
            sol = cv2.cvtColor(sol, cv2.COLOR_RGBA2RGB)
        return sol

    def plot_sol(self, sol):
        if self.draw_process is None:
            self.draw_process = multiprocessing.Process(None, self.__plot_sol, args=(self.draw_queue,))
            self.draw_process.start()
        if self.draw_queue.qsize() < 1:
            self.draw_queue.put(sol)

    def __plot_sol(self, queue):
        if not self.initialized_graph:
            plt.ion()
            plt.show()
            self.initialized_graph = True
        w, h = self.target_image.shape[:2]
        while True:
            try:
                sol = queue.get_nowait()
                if sol == 'Q':
                    plt.close()
                    return
                sol = self.create_image_from_sol(sol, True)
                cv2.imwrite(self.sol_file, sol)
                try:
                    im = np.concatenate(
                        (cv2.cvtColor(sol, cv2.COLOR_RGB2BGR),
                         cv2.cvtColor(self.target_image, cv2.COLOR_RGB2BGR)),
                        axis=1)
                except:
                    traceback.print_exc()
                plt.clf()
                plt.imshow(im)
                plt.pause(0.0001)
            except:
                plt.pause(0.1)

    def finish(self):
        self.draw_queue.put('Q')


if __name__ == '__main__':
    initial_time = round(time.time())
    improving_list = []
    random.seed(1234)

    # Parameters ########################################################
    #        'init_sol':                  'result/1505728772_mona-lisa-head.png_17.715901.pk',
    par = {
        'view':                      False,
        'init_sol':                  'None',
        'img':                       'data/images/mona-lisa-head.png',
        'num_shapes':                1,
        'candidates_by_iteration':   100,
        'delta':                     50,
        'sol_file':                  'result/'+str(initial_time)+'_mona-lisa-head.png',
        'max_edges':                 7,
        'vnx':                       '',
        'max_len_init_solution':     1000000,
        'max_iter':                  100,
        'list_len':                  2,
        'tolerance':                 50,
        'gopt_max_iter':             10,
        'gopt_list_len':             5,
        'max_time':                  100
    }
    # #################################################### Parameters ###

    def improve(caller):
        global improving_list
        improving_list.append(caller.best)
        if par['view']:
            problem.plot_sol(caller.best)

    if par['init_sol'] != 'None':
        d = pk.load(open(par['init_sol'], 'rb'))
        initial_solution = [(d[i][j][0], d[i][j][1]) for i in range(len(d)) for j in range(len(d[i]))]
        random.seed(5678)
    else:
        initial_solution = None
    current_fitness = 10000000
    i = 1
    img = cv2.imread(par['img'])
    problem = PolygonsProblem(img,
                              num_shapes=par['num_shapes'],
                              candidates_by_iteration=par['candidates_by_iteration'],
                              delta=par['delta'],
                              sol_file=par['sol_file'],
                              max_edges=par['max_edges'],
                              vnx=par['vnx'])
    while initial_solution is None \
            or len(initial_solution) < par['max_len_init_solution'] \
            and problem.fitness_time < par['max_time']:
        problem.num_shapes = i
        problem.polygon_list = [i-1]
        if initial_solution is not None:
            initial_solution.append([problem.get_random_polygon(), problem.get_random_color()])
        searcher = TabuSearch(problem,
                              max_iterations=par['max_iter'],
                              list_length=par['list_len'],
                              improved_event=improve,
                              tolerance=par['tolerance'])
        searcher.search(initial_solution=initial_solution)
        if current_fitness > searcher.best_fitness or initial_solution is None:
            initial_solution = searcher.best
            current_fitness = searcher.best_fitness
            if i > 1:                                                                   # 1:
                problem.polygon_list = None
                # Aplicamos aquí VND (descenso por vecindades) o VNS
                if problem.vnx == 'vnd':
                    vnd = VND(searcher, problem, ['move', 'color', 'add'])    # , 'remove'])
                    vnd.search(initial_solution=initial_solution)
                    if current_fitness > vnd.best_fitness or initial_solution is None:
                        initial_solution = vnd.best
                        current_fitness = vnd.best_fitness
                elif problem.vnx == 'vns':
                    vns = VNS(searcher, problem, ['move', 'color', 'add'])    # , 'remove'])
                    vns.search(initial_solution=initial_solution)
                    if current_fitness > vns.best_fitness or initial_solution is None:
                        initial_solution = vns.best
                        current_fitness = vns.best_fitness
                else:
                    pass
            i += 1
            print("Solution length: %d" % i)
        else:
            initial_solution = initial_solution[:-1]
        print("num fitness per second: %f" % (problem.fitness_count / problem.fitness_time))
        print("fitness time: ========================> %f" % problem.fitness_time)
    print("General optimization")
    problem.polygon_list = None
    searcher = TabuSearch(problem,
                          max_iterations=par['gopt_max_iter'],
                          list_length=par['gopt_list_len'],
                          improved_event=improve)
    searcher.search(initial_solution=initial_solution)
    if par['view']:
        problem.finish()
    else:
        sol = problem.create_image_from_sol(searcher.best)
        cv2.imwrite(par['sol_file'], sol)
    finish_time = time.time()
    pk.dump(improving_list, open(par['sol_file']+'_%f.pk' % searcher.best_fitness, 'wb'))
    output = {
        'parameters': par,
        'result': {
            'Solution length':  i,
            'Time minutes':  (finish_time-initial_time)//60,
            'Best fitness': searcher.best_fitness,
        }
    }
    json.dump(output, open(par['sol_file']+'.json', 'w'))
    print("Finish")

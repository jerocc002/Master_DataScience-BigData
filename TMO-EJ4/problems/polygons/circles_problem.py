import multiprocessing
import random

import time
import traceback

from optimization.problem import Problem
from optimization.tabu import TabuSearch
import numpy as np
import matplotlib.pyplot as plt
import copy
import cv2


class TrianglesProblem(Problem):
    def __init__(self, target_image, num_shapes, candidates_by_iteration=100, triangle_list=None, sol_file='sol.png'):
        self.target_image = target_image
        self.num_shapes = num_shapes
        self.candidates_by_iteration = candidates_by_iteration
        self.initilized_graph = False
        self.draw_queue = multiprocessing.Queue()
        self.draw_process = None
        self.triangle_list = triangle_list
        self.fitness_count = 0
        self.fitness_time = 0
        self.sol_file = sol_file

    def get_neighborhood(self, cand):
        # generate candidates
        candidates = [self.get_neighbor_candidate(cand)
                      for _ in range(self.candidates_by_iteration)]

        return candidates

    def get_neighbor_candidate(self, cand):
        if self.triangle_list:
            i = random.choice(self.triangle_list)
        else:
            i = random.randint(0, len(cand) - 1)

        cand = list(copy.deepcopy(cand))

        if random.choice([True, False]):
            cand[i] = list(cand[i])
            j = random.randint(0, 2)
            c = list(cand[i][0])
            c[j] = self.__perturb(cand[i][0][j])
            cand[i][0] = c
        else:
            cand[i] = cand[i][0], self.__perturb_color(cand[i][1])

        return cand


    def __perturb_color(self, color):
        offset = 10
        i = random.randint(1, 3)
        color = copy.deepcopy(color)
        color[i] = max(min(color[i] + random.randint(-min(offset, color[i]), min(offset, 255 - color[i])), 255), 0)
        #color[i] = random.randint(0, 255)
        return color

    def __perturb(self, c):
        offset = 10
        h, w = self.target_image.shape[:2]
        c = max(min(c + random.randint(-offset, offset), w), 0)
        return c

    def get_initial_solution(self):
        return [[self.get_random_circle(), self.get_random_color()] for _ in range(self.num_shapes)]

    def get_random_color(self):
        return [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255),  random.randint(0, 255)]

    def get_random_circle(self):
        h, w = self.target_image.shape[:2]
        return [random.randint(0, w), random.randint(0, h), random.randint(0, min(w,h))]

    def fitness(self, cand):
        start_time = time.time()
        h, w = self.target_image.shape[:2]
        sol = self.create_image_from_sol(cand)

        diff = abs((self.target_image - sol).sum())

        _fitness = 100 - (100 * (1 - diff / (w * h * 3 * 255)))

        self.fitness_time += (time.time() - start_time)
        self.fitness_count += 1

        return _fitness

    def create_image_from_sol(self, cand, to_rgb=True):
        h, w = self.target_image.shape[:2]
        sol = np.zeros((h, w, 4), np.uint8)
        #cand = [(((0, 0), (50, 0), (50, 20)), cand[0][1])]

        for shape, color in cand:
            overlay = sol.copy()
            pt = shape[:2]
            r = shape[2]

            cv2.circle(overlay, tuple(pt), r, color, -1)

            cv2.addWeighted(overlay, 0.5, sol, 0.5, 0, sol)

        if to_rgb:
            sol = cv2.cvtColor(sol, cv2.COLOR_RGBA2RGB)
        return sol

    def plot_sol(self, sol):

        if self.draw_process is None:
            self.draw_process = multiprocessing.Process(None, self.__plot_sol, args=(self.draw_queue,))
            self.draw_process.start()

        self.draw_queue.put(sol)

    def __plot_sol(self, queue):


        if not self.initilized_graph:
            plt.ion()
            plt.show()
            self.initilized_graph = True

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
                    im = np.concatenate((cv2.cvtColor(sol, cv2.COLOR_RGB2BGR), cv2.cvtColor(self.target_image, cv2.COLOR_RGB2BGR)), axis=1)
                except:
                    traceback.print_exc()

                plt.clf()
                plt.imshow(im)
                #plt.imshow(cv2.cvtColor(sol, cv2.COLOR_RGB2BGR))
                #plt.draw()
                plt.pause(0.0001)

            except:

                plt.pause(0.1)

    def finish(self):
        self.draw_queue.put('Q')



if __name__ == '__main__':
    img = cv2.imread('data/images/mona-lisa-head.png')

    def improve(caller):
        problem.plot_sol(caller.best)

    initial_solucion = []
    i = 1
    current_fitness = 100
    problem = TrianglesProblem(img, num_shapes=i, candidates_by_iteration=1000, triangle_list=[i - 1],
                               sol_file='data/images/little_mondrian_sol.png')

    while len(initial_solucion) < 50:
        problem.num_shapes = i
        problem.triangle_list = [i-1]
        initial_solucion.append([problem.get_random_circle(), problem.get_random_color()])
        searcher = TabuSearch(problem, max_iterations=1000, list_length=10, improved_event=improve, tolerance=100)
        searcher.search(initial_solution=initial_solucion)

        if current_fitness > searcher.best_fitness:
            initial_solucion = searcher.best
            current_fitness = searcher.best_fitness
            i += 1
            print("Solution length: %d" % i)
        else:
            initial_solucion = initial_solucion[:-1]

        print("num fitness per second: %f" % (problem.fitness_count / problem.fitness_time))


    print("General optimization")
    problem = TrianglesProblem(img, num_shapes=i, candidates_by_iteration=1000, triangle_list=None,
                               sol_file='data/images/mona_lisa_head_sol.png')
    searcher = TabuSearch(problem, max_iterations=100000, list_length=100, improved_event=improve)
    searcher.search(initial_solution=initial_solucion)
    problem.finish()

    print("Finish")

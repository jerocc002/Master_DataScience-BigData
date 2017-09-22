import random
import time
from abc import abstractmethod

class BaseGA:
    def __init__(self, mutation_prob=0.1, crossover_prob=0.1, restart_iterations=10000000):
        self.population = []
        self.mutation_prob = mutation_prob
        self.crossover_prob = crossover_prob
        self.restart_iterations = restart_iterations

    @abstractmethod
    def generate_population(self, n=None):
        pass

    @abstractmethod
    def fitness(self, dna):
        pass

    @abstractmethod
    def crossover(self, dna1, dna2):
        pass

    @abstractmethod
    def mutate(self, dna):
        pass

    @abstractmethod
    def select(self, items):
        pass

    @abstractmethod
    def pre(self, population_fitness):
        pass

    @abstractmethod
    def post_generation(self, generation):
        pass

    def run(self, GENERATIONS):



        # Generate initial population. This will create a list of POP_SIZE strings,
        # each initialized to a sequence of random characters.
        self.population = self.generate_population()

        POP_SIZE = len(self.population)

        # Simulate all of the generations.
        best_fitness = float('inf')
        best = None
        no_improving_iterations = 0
        evolution = []

        for generation in range(GENERATIONS):

            weighted_population = []

            # Add individuals and their respective fitness levels to the weighted
            # population list. This will be used to pull out individuals via certain
            # probabilities during the selection phase. Then, reset the population list
            # so we can repopulate it after selection.
            weights = []
            select_population = []
            total_fitness = 0
            improving = False
            valid_population = []
            population_size = len(self.population)

            start = time.time()

            '''
            if config.JOBS > 1:
                l = int(len(self.population) / config.JOBS)
                population_chucked = [self.population[(i*l):((i*l)+l)] for i in range(config.JOBS)]
                population_fitness = Parallel(n_jobs=config.JOBS)(
                    delayed(self.fitness)(chunk) for chunk in population_chucked)
                population_fitness = [item for sublist in population_fitness for item in sublist]
            else:
            '''
            population_fitness = [self.fitness(ind) for ind in self.population]
            self.pre(population_fitness)
            end = time.time()

            for i, fitness_val in enumerate(population_fitness):
                individual = self.population[i]

                if not fitness_val is None:
                    total_fitness += fitness_val
                    select_population.append(individual)
                    weights.append(fitness_val)

                    # print "%s (%f)"%(individual, fitness_val)

                    if best_fitness > fitness_val:
                        best = individual
                        best_fitness = fitness_val
                        no_improving_iterations = 0
                        improving = True


            self.population = valid_population
            missing_individuals = population_size - len(valid_population)
            self.population += self.generate_population(missing_individuals)

            no_improving_iterations += 1
            avg = total_fitness / len(self.population)

            if improving:
                evolution.append({'equation': best, 'fitness': best_fitness, 'mean_population_fitness': avg})

            self.population = []
            select_population[0] = best
            weights[0] = best_fitness


            print("Generation %s... Best: %f  -  Population average: %f" % (generation, best_fitness, avg))

            # Select two random individuals, based on their fitness probabilites, cross
            # their genes over at a random point, mutate them, and add them back to the
            # population for the next iteration.
            for _ in range(int(POP_SIZE / 2)):
                # Selection
                ind1, idx = self.select(select_population, weights)
                ind2, idx = self.select(select_population[:idx] + select_population[idx + 1:],
                                        weights[:idx] + weights[idx + 1:])

                # Crossover
                if random.uniform(0, 1) < self.crossover_prob:
                    ind1, ind2 = self.crossover(ind1, ind2)

                # Mutate and add back into the population.
                if random.uniform(0, 1) < self.mutation_prob:
                    ind1 = self.mutate(ind1)
                self.population.append(ind1)

                if random.uniform(0, 1) < self.mutation_prob:
                    ind2 = self.mutate(ind2)
                self.population.append(ind2)

            if no_improving_iterations >= self.restart_iterations:
                print("Restarting")
                no_improving_iterations = 0
                self.population = self.generate_population() + [best]

            self.post_generation(generation)

        return best, best_fitness, evolution

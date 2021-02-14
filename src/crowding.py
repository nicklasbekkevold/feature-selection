import numpy as np

from common import (Diversity, Fitness, create_offsping,
                    generate_initial_population)


class Crowding:

    @staticmethod
    def match(parent_1, parent_2, child_1, child_2):

        def distance(chromosome_a, chromosome_b):
            d = 0
            for gene_a, gene_b in zip(chromosome_a, chromosome_b):
                if gene_a != gene_b:
                    d += 1
            return d

        d_1 = distance(parent_1, child_1) + distance(parent_2, child_2)
        d_2 = distance(parent_1, child_2) + distance(parent_2, child_1)
        if d_1 < d_2:
            return ((parent_1, child_1), (parent_2, child_2))
        return ((parent_1, child_2), (parent_2, child_1))

    @staticmethod
    def crowding_step(population, crossover_rate, mutation_rate, population_size, fitness_function):
        old_population = list(population)
        new_population = []
        while len(new_population) < population_size:
            parent_1, parent_2 = np.random.choice(old_population, 2, True)
            child_1, child_2 = create_offsping(parent_1, parent_2, crossover_rate, mutation_rate)
            matches = Crowding.match(parent_1, parent_2, child_1, child_2)
            for parent, child in matches:
                if fitness_function(parent) < fitness_function(child):
                    new_population.append(child)
                elif fitness_function(parent) == fitness_function(child):
                    new_population.append(np.random.choice([parent, child]))
                else:
                    new_population.append(parent)

        return new_population

    @staticmethod
    def run(crossover_rate, mutation_rate, n_genes, population_size, fitness_function, generations):
        history = {
            'max': [],
            'average_fitness': [],
            'entropy': [],
            'baseline': fitness_function(''.join('1' for _ in range(n_genes))),
            'generations': [],
        }
        population = generate_initial_population(n_genes, population_size)

        for _ in range(generations):
            history['max'].append(fitness_function(max(population, key=fitness_function)))
            history['average_fitness'].append(Fitness.average_fitness(population, fitness_function))
            history['entropy'].append(Diversity.entropy(population))
            history['generations'].append(population)
            population = Crowding.crowding_step(population, crossover_rate, mutation_rate, population_size, fitness_function)

        return population, history

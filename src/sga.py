import numpy as np

from common import (Diversity, Fitness, create_offsping,
                    generate_initial_population)


class SGA:

    @staticmethod
    def rank_selection(population_size, population, fitness_function, replace=False):
        ranking = sorted(population, key=fitness_function)
        rank_sum = len(ranking) * (len(ranking) + 1) / 2
        probabilities = [rank / rank_sum for rank, _ in enumerate(ranking, 1)]
        return np.random.choice(ranking, population_size, p=probabilities, replace=replace)

    @staticmethod
    def mu_lambda_selection(population_size, offspring, fitness_function):
        return sorted(offspring, key=fitness_function, reverse=True)[:population_size]

    @staticmethod
    def generation(population, crossover_rate, mutation_rate, population_size, fitness_function):
        offspring = []
        while len(offspring) < 3 * population_size:
            parent_a, parent_b = SGA.rank_selection(2, population, fitness_function)
            offspring_a, offspring_b = create_offsping(parent_a, parent_b, crossover_rate, mutation_rate)
            offspring += [offspring_a, offspring_b]
        return SGA.mu_lambda_selection(population_size, offspring, fitness_function)

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
            population = SGA.generation(population, crossover_rate, mutation_rate, population_size, fitness_function)

        return population, history

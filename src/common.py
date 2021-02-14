import math
import random

import numpy as np

from LinReg import LinReg


def generate_initial_population(n_genes, population_size):
    return [''.join(random.choice(['0', '1']) for _ in range(n_genes)) for _ in range(population_size)]


def mutate(chromosome, mutation_rate):
    mutated_chromosome = ''
    for gene in chromosome:
        if random.random() <= mutation_rate:
            mutated_chromosome += '0' if gene == '1' else '1'
            continue
        mutated_chromosome += gene
    return mutated_chromosome


def crossover(chromosome_a, chromosome_b, crossover_rate):
    if random.random() <= crossover_rate:
        crossover_point = random.randrange(1, len(chromosome_a) - 1)
        offspring_a = chromosome_a[:crossover_point:] + chromosome_b[crossover_point:]
        offspring_b = chromosome_b[:crossover_point:] + chromosome_a[crossover_point:]
        return offspring_a, offspring_b
    return chromosome_a, chromosome_b


def create_offsping(parent_a, parent_b, crossover_rate, mutation_rate):
    offspring_a, offspring_b = crossover(parent_a, parent_b, crossover_rate)
    offspring_a = mutate(offspring_a, mutation_rate)
    offspring_b = mutate(offspring_b, mutation_rate)
    return offspring_a, offspring_b


class Diversity:

    @staticmethod
    def entropy(population):
        N = len(population)
        occurences = [0 for _ in range(len(population[0]))]
        for chromosome in population:
            for locus, gene in enumerate(chromosome):
                occurences[locus] += int(gene)
        information = [n_i / N * math.log2(n_i / N) if n_i != 0 else 0 for n_i in occurences]
        return -sum(information)


class Fitness:

    data = np.array(np.loadtxt('src/dataset.txt', delimiter=','))
    lin_reg = LinReg()
    fitness_memo = {}

    @staticmethod
    def sine_fitness(chromosome):
        upper_bound = 128
        scaling_factor = 2 ** -(len(chromosome) - math.log2(upper_bound))
        real_value = int(chromosome, 2)
        return math.sin(real_value * scaling_factor)

    @staticmethod
    def rmse_fitness(chromosome):
        x = Fitness.data[:, :-1]
        y = Fitness.data[:, -1]
        x = Fitness.lin_reg.get_columns(x, chromosome)
        if chromosome not in Fitness.fitness_memo:
            Fitness.fitness_memo[chromosome] = -Fitness.lin_reg.get_fitness(x, y)
        return Fitness.fitness_memo[chromosome]

    @staticmethod
    def average_fitness(population, fitness_function):
        fitnesses = [fitness_function(chromosome) for chromosome in population]
        return sum(fitnesses) / len(fitnesses)

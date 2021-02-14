import json
from typing import Callable

from common import Fitness


def get_parameters(filename):
    parameters = Parameters()
    with open(f'src/{filename}.json', 'r') as f:
        ga_parameter = json.load(f)
        for attr, value in ga_parameter.items():
            setattr(parameters, attr, value)

        if parameters.fitness_function == 'sine':
            parameters.fitness_function = Fitness.sine_fitness
        elif parameters.fitness_function == 'rmse':
            parameters.fitness_function = Fitness.rmse_fitness
        else:
            parameters.fitness_function = Fitness.sine_fitness
    return parameters


class Parameters:

    generations: int
    n_genes: int
    population_size: int
    fitness_function: Callable[[str], float]
    crossover_rate: float
    mutation_rate: float

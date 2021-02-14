from common import Fitness
from crowding import Crowding
from parameters import get_parameters
from sga import SGA
from utils import plot_diversity, plot_history, plot_sine, print_summary

if __name__ == "__main__":
    parameters = get_parameters('solution')

    population, sga_history = SGA.run(
        parameters.crossover_rate,
        parameters.mutation_rate,
        parameters.n_genes,
        parameters.population_size,
        parameters.fitness_function,
        parameters.generations,
    )
    print_summary(sga_history, 'SGA')
    plot_history(sga_history, 'sga')

    population, crowding_history = Crowding.run(
        parameters.crossover_rate,
        parameters.mutation_rate,
        parameters.n_genes,
        parameters.population_size,
        parameters.fitness_function,
        parameters.generations,
    )
    print_summary(crowding_history, 'Crowding')
    plot_history(crowding_history, 'crowding')

    plot_diversity(sga_history, crowding_history, 'diversity')

    if parameters.fitness_function is Fitness.sine_fitness:
        plot_sine(sga_history['generations'], 'sga')
        plot_sine(crowding_history['generations'], 'crowding')

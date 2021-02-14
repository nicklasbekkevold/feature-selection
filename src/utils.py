# type: ignore
import math

import matplotlib.pyplot as plt
import numpy as np

from common import Fitness


def print_summary(history, algorithm):
    max, average_fitness, entropy, baseline, _ = history.keys()
    print(algorithm, 'summary statistics')
    print(f'Max:             {history[max][-1]:>10.3f}')
    print(f'Average fitness: {history[average_fitness][-1]:>10.3f}')
    print(f'Entropy:         {history[entropy][-1]:>10.3f}')
    print(f'Baseline:        {history[baseline]:>10.3f}')
    print('---------------------------')


def plot_history(history, filename):
    fig, ax1 = plt.subplots()
    ax1.set_title(filename.upper())
    baseline = history['baseline']

    color = 'tab:blue'
    ax1.set_xlabel('generation')
    ax1.set_ylabel('Average fitness', color=color)
    ax1.plot(history['average_fitness'], color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.plot(np.full(len(history['average_fitness']), baseline), linestyle='--', color='black', label='Baseline')

    ax2 = ax1.twinx()

    color = 'tab:orange'
    ax2.set_ylabel('entropy', color=color)
    ax2.plot(history['entropy'], color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    ax1.legend()
    fig.savefig(f'src/results/{filename}.png')
    plt.close()


def plot_sine(generations, filename):
    plt.title(f'Population plot ({filename.upper()})')
    plt.xlabel('x')
    plt.ylabel('sin(x)')

    x = np.linspace(0, 128, 1000, endpoint=True)
    plt.plot(x, np.sin(x), label='sin(x)')

    upper_bound = 128
    scaling_factor = 2 ** -(len(generations[0][0]) - math.log2(upper_bound))

    population = [chromosome for population in generations for chromosome in population]  # flatten generation
    chromosomes = [int(chromosome, 2) * scaling_factor for chromosome in population]
    fitnesses = [Fitness.sine_fitness(chromosome) for chromosome in population]
    plt.scatter(chromosomes, fitnesses, color='tab:orange')

    plt.tight_layout()
    plt.savefig(f'src/results/{filename}_sine.png')
    plt.close()


def plot_diversity(sga, crowding, filename):
    plt.title('Diveristy')
    plt.xlabel('generations')
    plt.ylabel('entropy')

    plt.plot(crowding['entropy'], label='Crowding', color='tab:blue')
    plt.plot(sga['entropy'], label='Simple Genetic Algorithm', color='tab:orange')

    plt.legend()
    plt.tight_layout()
    plt.savefig(f'src/results/{filename}.png')
    plt.close()

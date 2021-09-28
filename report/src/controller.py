import generation
import numpy as np


class Controller():
    """
    Controls object instantiation, algorithm evolution, and network training.

    Args:
        pop_size [int]: Number of phenotypes in each generation.
        num_gens [int]: Total number of generations for training.

    Attributes:
        X [np array]: Data for training and validating the neural network.
        y [np array]: True corresponding values for the neural network.
        size [int]: Number of phenotypes in each generation.
        gens [int]: Total number of generations for training.
        pop [Generation]: Populations of phenotypes as the generation object.

    Methods:
        __init__ -> Passes data for the networks and runs the evolution methods.
    """


    def __init__(self, X, y, pop_size=30, num_gens=100):
        """ Gets the data for the networks and runs the evolution methods."""
        self.X = X
        self.y = y
        self.size = pop_size # Population size for each generation.
        self.gens = num_gens # Total number of generations.
        self.main()

    def main(self):
        """ Instances generation 0, evolves the population, plots values."""
        self.pop = generation.Generation(self.X, self.y, pop_size=self.size)
        self.evolve()
        self.print_best_net()
        self.plot_results()

    def evolve(self):
        """ Evolves the network given the number of generations."""
        for i in range(self.gens):
            self.pop.calc_fitness()
            self.pop.next_generation()
        self.pop.calc_fitness()
        self.pop.print_gen_info()

    def print_best_net(self):
        """ Prints the hyperparameters of the best performing network."""
        print("#############################################")
        print("Best Network Configuration DNA:\n"
                + self.pop.generation[0].dna)
        print("#############################################")
        print("---------------------------------------------------")
        print(self.pop.generation[0].layers)
        print(self.pop.generation[0].funct)
        print(self.pop.generation[0].lr)
        print("END")

    def plot_results(self):
        """ Plots all fitness values accross every generation."""
        self.pop.plot_total_fitness()
        self.pop.plot_fitness()
        self.pop.generation[0].network.plot_error()

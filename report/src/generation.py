import random
import phenotype
import numpy as np
import matplotlib.pyplot as plt


class Generation():
    """
    Collection of phenotypes with fitnesses and methods for reproduction.

    Args:
        X [np array]: Data for training and validating the neural network.
        y [np array]: True corresponding values for the neural network.
        pop_size [int]: Number of phenotypes in the generation.

    Attributes:
        ID [int]: Value for keeping track of generation number.
        X [np array]: Data for training and validating the neural network.
        y [np array]: True corresponding values for the neural network.
        size [int]: Number of phenotypes in the generation.
        generation [list]: Array of all phenotypes in the generation.
        total_history [list]: The fitness totals for every generation.
        best_history [list]: The best fitness for each generation.
        worst_history [list]: The worst fitness for each generation.
        avg_history [list]: The average fitness for each generation.
        fitness [list]: Fitness values for each phenotype in the generation.
        total [float]: Total fitness for the current generation.
        best [float]: Best fitness for the current generation.
        worst [float]: Worst fitness for the current generation.
        avg [float]: Average fitness for the current generation.
        scaled_fitness [list]: Scaled fitness values for the whole generation.
        new_generation [list]: Temporary list to replace with the current gen.
        roulette [list]: Values for helping pick parents based on fitness.

    Methods:
        __init__ -> Initializes Generation 0, and lal data tracking attributes.
        calc_fitness -> Instances and trains all networks and gets fitness.
        next_generation -> Creates a new generation from elitism/reproduction.
        print_gen_info -> Prints summary information of each generation.
    """


    def __init__(self, X, y, pop_size=30):
        """ Initializes Generation 0, and lal data tracking attributes."""
        self.ID = 0
        self.X = X
        self.y = y
        self.size = pop_size
        self.generation = self.init_generation()
        self.init_fitness_history()
        plt.style.use('ggplot')

    def init_generation(self):
        """ Initializes generation 0 with a population of random phenotypes."""
        gen = []
        for i in range(self.size):
            gen.append(phenotype.Phenotype(self.X, self.y))

        return gen

    def init_fitness_history(self):
        """ Creates lists for plotting the fitness over several generations."""
        self.total_history = []
        self.best_history = []
        self.worst_history = []
        self.avg_history = []

    def calc_fitness(self):
        """ Instances and trains all networks and gets fitness from error."""
        self.gen_init_text()

        for i in range(self.size):
            phenotype = self.generation[i]
            phenotype.init_network()
            phenotype.get_fitness()
            self.print_train_info(i)

        self.organize()

    def gen_init_text(self):
        """ YOUR TEXT HERE"""
        s = "# GENERATION " + str(self.ID) + " INITIALIZED #"
        t = '#'
        for i in range(len(s) - 1):
            t += '#'
        print(t)
        print(s)
        print(t)

    def print_train_info(self, i):
        """ Prints information on Network training to terminal."""
        print("Network %s has finished training.\r" %str(i+1), end="")

    def organize(self):
        """ Updates attributes and prepares for reproduction."""
        self.sort_generation()
        self.get_fitness_extrema()
        self.update_fitness_histories()
        self.print_gen_info()

    def sort_generation(self):
        """ Sorts generation in decending order by total fitness"""
        self.generation.sort(key=lambda x: x.fitness)
        self.fitness = [x.fitness for x in self.generation]

    def get_fitness_extrema(self):
        """ Finds best, worst, and average fitness for rescaling."""
        self.total = sum(self.fitness)
        self.best = self.fitness[0]
        self.worst = self.fitness[-1]
        self.avg = np.mean(self.fitness)
        self.rescale()

    def rescale(self):
        """ Rescales the fitness to improve selection/reproduction."""
        c = 1.2
        if (self.avg - self.worst) <= (self.best - self.avg) / (c - 1):
            a = (c - 1) / (self.best - self.avg)
            b = (self.best - (c * self.avg)) / (self.best - self.avg)

        else:
            a = 1 / (self.avg - self.worst)
            b = -self.worst / (self.avg - self.worst)

        self.scaled_fitness = [(a*fit + b) for fit in self.fitness]

    def update_fitness_histories(self):
        """ Appends all fitnesses to their respective lists for plotting."""
        self.total_history.append(self.total)
        self.best_history.append(self.best)
        self.worst_history.append(self.worst)
        self.avg_history.append(self.avg)

    def print_gen_info(self):
        """ Prints summary information of each generation to terminal."""
        print("---------------------------------------------------")
        print("Summary:")
        print()
        print("Total Generation Fitness: " + str(self.total))
        print("Best Generation Fitness: " + str(self.best))
        print("Worst Generation Fitness: " + str(self.worst))
        print("Average Generation Fitness: " + str(self.avg))
        print("---------------------------------------------------")
        print()

    def next_generation(self):
        """ Creates a new generation based on elitism and reproduction."""
        self.new_generation = []
        self.elitism()
        self.repopulate()
        self.update_generation()

    def elitism(self):
        """ Carries over top 2 of the previous gen, and has them reproduce."""
        elite1 = self.generation[0]
        elite2 = self.generation[1]

        zygote1, zygote2 = elite1.reproduce(elite2)
        child1 = self.birth(zygote1)
        child2 = self.birth(zygote2)

        self.new_generation.append(elite1)
        self.new_generation.append(elite2)
        self.new_generation.append(child1)
        self.new_generation.append(child2)

    def repopulate(self):
        """ Instance a new generation of phenotypes based on their fitness."""
        self.roulette = self.create_roulette()
        self.sex()

    def create_roulette(self):
        """ Create a roulette wheel for selecting parents based on fitness."""
        norms = [x/self.size for x in self.scaled_fitness]
        return [sum(norms[:i]) for i in range(1, self.size + 1)]

    def sex(self):
        """ Select parents and call the reproduce methods to get new DNA."""
        n = (self.size - 4) // 2
        for i in range(n):
            parent1 = self.select_parent()
            parent2 = self.select_parent()

            zygote1, zygote2 = parent1.reproduce(parent2)
            child1 = self.birth(zygote1)
            child2 = self.birth(zygote2)

            self.new_generation.append(child1)
            self.new_generation.append(child2)

    def select_parent(self):
        """ Selects parents randomly based off of roulette wheel selection."""
        r = random.random()

        for i in range(self.size):
            if r <= self.roulette[i]:
                return self.generation[i]

    def birth(self, zygote):
        """ Instance a new phenotype based on the parent's DNA."""
        return phenotype.Phenotype(self.X, self.y, dna=zygote)

    def update_generation(self):
        """ Replaces the old generation with the new one and increment by 1."""
        self.generation = self.new_generation
        self.ID += 1

    def plot_total_fitness(self):
        """ Plot the fitness totals over all generations."""
        plt.title("Total Fitness for each Generation")
        plt.plot(self.total_history, color="black")
        plt.xlabel("Number of Generations")
        plt.ylabel("Fitness")
        plt.show()

    def plot_fitness(self):
        """ Plot the best, average, and worst fitnesses over all generations."""
        plt.plot(self.best_history, color="red")
        plt.plot(self.avg_history, color="green")
        plt.plot(self.worst_history, color="blue")
        plt.title("Other Fitness histories for each Generation")
        plt.legend(["Best Fitness", "Average Fitness", "Worst Fitness"])
        plt.xlabel("Number of Generations")
        plt.ylabel("Fitness")
        plt.show()

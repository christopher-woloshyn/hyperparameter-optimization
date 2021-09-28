import random
import deepmlp as mlp

class Phenotype():
    """
    Instructions for instancing a neural network with certain hyperparameters.

    Args:
        self [Phenotype]: Individual network with certain hyperparameters.
        X [np array]: Data for training and validating the neural network.
        y [np array]: True corresponding values for the neural network.
        dna [str]: a binary encoding of the hyperparameters in the network.
        max_depth [int]: Maximum number of hidden layers allowed.
        global_epochs [int]: Fixed number of epochs each network will train on.

    Attributes:
        X [np array]: Data for training and validating the neural network.
        y [np array]: True corresponding values for the neural network.
        epochs [int]: The number of epochs the network will train for.
        dna [str]: A string of 0s and 1s representing all hyperparameters.
        genes [list]: A separation of the dna representing each hyperparameter.
        fitness [float]: The error value from the last epoch in training.

    Methods:
        __init__ -> Instances one individual of the population.
        init_network -> Initializes the network based on phenotype DNA.
        get_fitness -> Trains the network and gets error from the last epoch.
        reproduce -> Splits and combines the DNA from self and one other 'mate'.
    """


    def __init__(self, X, y, dna='', max_depth=4, global_epochs=200, ):
        """ Instances one individual of the population."""
        self.X = X
        self.y = y
        self.dna = dna
        self.max_depth = max_depth
        self.epochs = global_epochs
        self.genes = self.get_genes()

    def get_genes(self):
        """ Gets genes from DNA, either offspring DNA or random DNA."""
        if self.dna == '':
            self.generate_dna()

        return self.separate_dna()

    def generate_dna(self):
        """ Generates a random set of DNA; meant for generation 0."""
        dna_length = 3*self.max_depth + 13 # 10 to learn rate, 3 to active func.
        for i in range(dna_length):
            self.dna += random.choice(['1', '0'])

    def separate_dna(self):
        """ Separates DNA into genes that encode certain parameters."""
        genes = []
        for i in range(self.max_depth):
            genes.append(self.dna[3*i:3*(i+1)])

        hid_layers = 3*self.max_depth
        genes.append(self.dna[hid_layers:hid_layers+10])
        genes.append(self.dna[-3:])

        return genes

    def init_network(self):
        """ Initializes the network based on phenotype DNA."""
        self.layers = self.get_layers()
        self.network = mlp.DeepMLP(self.X, self.y, self.layers, seed=True)

    def get_layers(self):
        """ Converts the first section of dna to layers parameter."""
        nodes = []
        nodes.append(len(self.X[0]))

        for i in range(self.max_depth):
            n = int(self.genes[i], 2)
            if n != 0:
                nodes.append(n)

        nodes.append(len(self.y[0]))

        return nodes

    def get_fitness(self):
        """ Trains the network for a fixed number of epochs, and gets error."""
        self.lr = self.get_learn_rate()
        self.funct = self.get_activation()
        self.network.train(function=self.funct,
                            epochs=self.epochs, learn_rate=self.lr)
        self.fitness = self.network.valid_errs[-1]

    def get_learn_rate(self):
        """ Converts the second to last gene to the network learning rate."""
        return (int(self.genes[-2], 2) + 1) / 40960 #(between ~0 and .0.025)

    def get_activation(self):
        """ Converts last gene into a string of the activation function."""
        s = self.genes[-1]
        if  s == '000':
            return 'sigmoid'
        elif s == '001':
            return 'ReLU'
        elif s == '010':
            return 'leaky'
        elif s == '011':
            return 'softplus'
        elif s == '100':
            return 'swish'
        elif s == '101':
            return 'e_swish'
        else:
            return 'sigmoid'

    def reproduce(self, mate):
        """ Splits and combines the DNA from self and one other 'mate'."""
        s = random.choice(range(len(self.genes)))
        zygote1 = self.mutate(self.dna[:s] + mate.dna[s:])
        zygote2 = self.mutate(mate.dna[:s] + self.dna[s:])

        return zygote1, zygote2

    def mutate(self, string, prob=0.05):
        """ Scans the DNA string, randomly mutates with certain probability."""
        mutated = ''
        for i in range(len(string)):
            if random.random() < prob:
                mutated += str(1 - int(string[i]))
            else:
                mutated += string[i]

        return mutated

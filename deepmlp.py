import numpy as np
import matplotlib.pyplot as plt
from activations import *


class DeepMLP:
    """
    A multi-layer perceptron neural network for deep learning.

    Args:
        self [DeepMLP]: A multi-layer neural network for deep learning.
        X [np array]: Data for training and validating the neural network.
        y [np array]: True corresponding values for the neural network.
        layers [list]: A list indicating the dimension of each network layer.
        DEBUG=False [bool]: Indicates if debugging text will print to terminal.

    Attributes:
        DEBUG [bool]:
        data [list]: A list of both the training data and true values.
        depth [int]: Total number of layers in the network
        layers [list]: A list indicating the dimension of each network layer.
        w [list]: A list of the weights matrices for each layer of the network.
        b [list]: A list of the bias vectors for each layer of the network.
        nab_w [list]: A list of the weights gradients used for gradient descent.
        nab_b [list]: A list of the bias gradients used for gradient descent.
        train_errs [list]: A list of the training error from each epoch.
        valid_errs [list]: A list of the validation error from each epoch.
        functions [dict]: A dictionary containing all activation functions.
        activation [str]: Activation function for each neuron in the network.

    Methods:
        __init__ -> Instances the Deep Multi-layer Perceptron object.
        train -> Adjusts the weights/biases through stochastic gradient descent.
        predict -> Makes a prediction using the trained weights/biases.
    """


    def __init__(self, X, y, layers, seed=False, DEBUG=False):
        """ Instances the Deep Multi-layer Perceptron object."""
        self.DEBUG = DEBUG

        if seed:
            np.random.seed(1)

        if self.DEBUG:
            np.random.seed(1)

        self.data = [X, y]
        self.depth = len(layers)
        self.layers = layers
        self.w, self.b = self.init_weights()
        self.nab_w, self.nab_b = self.init_weights()

        self.train_errs = []
        self.valid_errs = []

        self.functions = {
        "sigmoid": sigmoid,
        "ReLU": ReLU,
        "leaky": leaky_ReLU,
        "softplus": softplus,
        "swish": swish,
        "e_swish": e_swish,
        }

        plt.style.use('ggplot')
        #self.print_initialization()

    def init_weights(self, nabla=False):
        """ Creates random weights and biases for each layer of nodes."""
        w = []
        b = []
        if not nabla:
            for i in range(self.depth-1):
                w.append(\
                (4 * np.random.rand(self.layers[i], self.layers[i+1]) - 2) / 3)
                b.append(\
                (4 * np.random.rand(self.layers[i+1]) - 2) / 3)

        else:
            for i in range(self.depth-1):
                w.append(np.zeros((self.layers[i], self.layers[i+1])))
                b.append(np.zeros(self.layers[i+1]))

        return w, b

    def train(self, function='sigmoid', epochs=100, learn_rate=0.001, split=.7):
        """
        Adjusts the weights/biases through stochastic gradient descent.

        Args:
            self [DeepMLP]: ANN object that contains the weights and biases.
            epochs [int]: Number of passes to adjust the weights and biases.
            function [str]: Activation function for each neuron in the network.
            learn_rate [float]: Rate at which gradient descent is applied.
            split [float]: Split between training and validation data.
        """
        self.data = self.shuffle()
        self.activation = function
        train, valid, train_true, valid_true = self.split_data(split)
        epoch = 0

        while epoch < epochs: #self.error_change():38,
            E_train = self.calc_err(train, train_true, function, back_prop=True)
            E_valid = self.calc_err(valid, valid_true, function)
            self.update(learn_rate, E_train, E_valid)
            epoch += 1

            if self.DEBUG:
                self.print_progress(epoch)
                self.print_weights()

        if self.DEBUG:
            self.print_weights(final=True)
            self.print_biases(final=True)

    def calc_err(self, X, y, function, back_prop=False):
        """ Calculates activation error and will back prop if specified."""
        E = 0
        for i in range(len(X)):
            act = self.forward_prop(X[i], function)
            err = act[-1] - y[i]
            E += np.dot(err, err) / 2

            if back_prop:
                self.back_prop(err, act, function)

        return E

    def shuffle(self):
        """ Shuffles the training and valid data so they stay corresponding."""
        assert len(self.data[0]) == len(self.data[1])
        p = np.random.permutation(len(self.data[0]))
        return [self.data[0][p], self.data[1][p]]

    def split_data(self, split):
        """ Splits shuffled data into training data and validation data."""
        s = round(len(self.data[0]) * split)
        N = len(self.data[0])
        train = self.data[0][0:s]
        valid = self.data[0][s:N]
        train_true = self.data[1][0:s]
        valid_true = self.data[1][s:N]
        return train, valid, train_true, valid_true

    def matmul(self, a, b):
        """ Chunked matmul which converts datatypes and prevents overflows."""
        a = np.array([a])
        print(a)
        print(b)
        print(a.dtype)
        print(b.dtype)
        out = np.empty((a.shape[0], b.shape[1]), dtype=np.int8)
        for i in range(a.shape[0]):
            a_conv = a[i].astype(np.float64)
            temp = np.dot(a_conv, b)
            temp[temp>127] = 127
            out[i] = temp.astype(np.float64)
        print(out)
        print()
        return out
        # Not in use, but potentially needed along with gradient clipping.

    def error_change(self):
        """ Checks to see if the validation error increases after each epoch."""
        try:
            if self.valid_errs[-1] - self.valid_errs[-2] <= 0:
                return True
            else:
                return False
        except:
            return True

    def forward_prop(self, x, function):
        """ Produces activations for each layer of the network."""
        actives = [x]
        for i in range(self.depth-1):
            actives.append(self.activate(actives[-1], i, function))
        return actives # list of all activations; last entry is output.

    def back_prop(self, error, act, function):
        """ Calculates the gradient and acquires partial derivative matrices"""
        err = error
        for i in range(self.depth-1):
            partial = err * self.activate(act[-(2+i)], -(1+i), function, deriv=True)
            self.nab_w[-(1+i)] += np.outer(act[-(2+i)], partial)
            self.nab_b[-(1+i)] += partial
            err = np.dot(self.w[-(1+i)], partial)

    def activate(self, inp, iter, func, deriv=False):
        """ Applies the linear basis and activation functions to the data."""
        linear_basis = np.dot(inp, self.w[iter]) + self.b[iter]
        return self.functions[func](linear_basis, deriv)

    def update(self, rate, E_train, E_valid):
        """ Adjusts weights/biases based on gradient matrices and learn rate."""
        self.w = [self.w[i] - rate * self.nab_w[i] for i in range(self.depth-1)]
        self.b = [self.b[i] - rate * self.nab_b[i] for i in range(self.depth-1)]
        self.train_errs.append(E_train)
        self.valid_errs.append(E_valid)
        self.nab_w, self.nab_b = self.init_weights(nabla=True)

    def print_initialization(self):
        """ Prints initial information about the network."""
        print('Network Initialized')
        print('===================')
        print()
        print('Input Dimension:', self.layers[0])
        for i in range(self.depth - 2):
            print('Hidden #' + str(i+1) + ' Dimension:', self.layers[i+1])

        print('Output Dimention:', self.layers[-1])
        print()

        if self.DEBUG:
            self.print_weights()
            self.print_biases()

    def print_weights(self, final=False):
        """ Prints the current weights for each layer of the network."""
        if final:
            print('Final Network Weights:')
            print('----------------------')
        else:
            print('Current Network Weights:')
            print('------------------------')

        if self.depth == 2:
            print('Input to Output:\n', self.w[0])
            print()

        else:
            print('Input to Hidden #1:\n', self.w[0])
            print()

            for i in range(self.depth - 3):
                print('Hidden #' + str(i+1) +
                    ' to Hidden #' + str(i+2) + ':\n', self.w[i+1])
                print()

            print('Hidden #' + str(self.depth - 2) +
            ' to Output:\n', self.w[-1])
            print()

    def print_biases(self, final=False):
        """ Prints the current biases for each layer of the network."""
        if final:
            print('Final Network Biases:')
            print('---------------------')
        else:
            print('Current Network Biases:')
            print('-----------------------')

        if self.depth == 2:
            print('Input to Output:\n', self.b[0])
            print()

        else:
            print('Input to Hidden #1:\n', self.b[0])
            print()

            for i in range(self.depth - 3):
                print('Hidden #' + str(i+1) +
                    ' to Hidden #' + str(i+2) + ':\n', self.b[i+1])
                print()

            print('Hidden #' + str(self.depth - 2) +
            ' to Output:\n', self.b[-1])
            print()

    def print_progress(self, epoch):
        """ Prints the new error values after each epoch. Only for DEBUG."""
        print("Epoch Count: " + str(epoch))
        print("Error on Training Data: " + str(self.train_errs[-1]))
        print("Error on Validation Data: " + str(self.valid_errs[-1]))
        print()

    def plot_error(self):
        """ Plots the training and validation errors over all epochs."""
        if self.activation == 'sigmoid':
            f = 'Sigmoid'
        elif self.activation == 'ReLU':
            f = 'ReLU'
        elif self.activation == 'leaky':
            f = 'Leaky ReLU'
        elif self.activation == 'softplus':
            f = 'Softplus'
        elif self.activation == 'swish':
            f = 'Swish'
        else:
            f = 'E-swish'

        plt.plot(self.train_errs, color="red", label="Training Error")
        plt.plot(self.valid_errs, color="blue", label="Validation Error")
        plt.title('Multilayer Perceptron:\n' + f)
        plt.xlabel("No. of Epochs")
        plt.ylabel("Squared Error")
        plt.legend()
        plt.show()

    def adjacency_matrix(self):
        """ Creats an adjacency matrix of the whole network topology."""
        block = []
        for i in range(self.depth):
            row = []
            for j in range(self.depth):
                if j - i == 1:
                    row.append(self.w[i])
                elif j - i == -1:
                    row.append(self.w[j].T)
                else:
                    row.append(np.zeros((self.layers[i], self.layers[j])))
            block.append(row)
        adj_matrix = np.block(block)

        if self.DEBUG:
            print(adj_matrix)

        plt.imshow(adj_matrix)
        plt.show()

    def predict(self, X):
        """ Makes a prediction using the trained weights/biases."""
        print()
        out = []
        for i in range(len(X)):
            out.append(self.forward_prop(X[i], self.activation)[-1])

        return out

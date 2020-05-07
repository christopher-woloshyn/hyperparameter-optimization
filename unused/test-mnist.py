import numpy as np
from viznet import deepmlp as mlp
from mnist_pre import *

X, y = mnist_preprocess('data/mnist_test.csv')
layers = [784, 16, 16, 10]

net = dmlp.DeepMLP(X, y, layers, DEBUG=True)
net.train(epochs=10, function='swish', learn_rate=0.01)
net.plot_error()

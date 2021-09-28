import controller
import numpy as np

def get_data():
    """ Creates the input data array (based on XOR gate)."""
    data = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
    ])
    return data

def get_exp():
    """ Creates the expected output data (based on XOR gate)."""
    exp = np.array([
    [0],
    [1],
    [1],
    [0],
    [0],
    [1],
    [1],
    [0],
    [0],
    [1],
    [1],
    [0],
    [0],
    [1],
    [1],
    [0],
    [0],
    [1],
    [1],
    [0],
    [0],
    [1],
    [1],
    [0],
    [0],
    [1],
    [1],
    [0],
    [0],
    [1],
    [1],
    [0],
    ])
    return exp

def main():
    X, y = get_data(), get_exp()
    controller.Controller(X, y, pop_size=50, num_gens=100)

main()

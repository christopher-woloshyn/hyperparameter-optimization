import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def mnist_preprocess(path):
    """Preprocess the MNIST data to be ready for the Neural Network."""
    print('Preprocessing MNIST Dataset...')
    data = pd.read_csv(path)
    inp = inputs(data)
    out = outputs(data) * (1/255)
    print(inp * (1/255))
    print()
    return inp, out

def inputs(data):
    """Extracts the input vector from the dataframe."""
    info = []
    for i in range(len(data)):
        info.append(data.iloc[i][1:] / 255)
    return np.array(info)

def outputs(data):
    """Extracts the expected output vector from the dataframe."""
    truth = []
    for i in range(len(data)):
        num = int(data['label'][i])
        truth.append([1 if j == num else 0 for j in range(10)])
    return np.array(truth)

def show_number(num, n=28):
    """Visualized the number using matplotlib."""
    matrix = []
    for slice in range(n):
        row = num[slice*n:(slice+1)*n]
        matrix.append(row)

    plt.imshow(matrix, cmap='gray')
    plt.show()

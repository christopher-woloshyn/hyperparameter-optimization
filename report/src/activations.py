import numpy as np

# Logistic activations
def sigmoid(x, deriv=False):
    """ Sigmoid activation function. Squishes values between 0 and 1."""
    if deriv:
        return sigmoid(x) * (1 - sigmoid(x))

    return 1 / (1 + np.exp(-x))

# Linear Rectifiers
def ReLU(x, deriv=False):
    """ Rectified linear activation; the maximum between 0 and its input."""
    out = []
    for val in x:
        if deriv:
            if max(0, val) == 0:
                out.append(0)
            else:
                out.append(1)
        else:
            out.append(max(0, val))

    return out

def leaky_ReLU(x, deriv=False, a=0.01):
    """Rectified linear activation with a small slope for values less than 0."""
    out = []
    for val in x:
        if deriv:
            if val > 0:
                out.append(1)
            else:
                out.append(a)
        else:
            if val > 0:
                out.append(val)
            else:
                out.append(a*val)

    return out

# Continuous "Linear" Rectifiers
def softplus(x, deriv=False):
    """A smooth version of ReLU and anti-derivative for the sigmoid function."""
    if deriv:
        return sigmoid(x)

    return np.log(1 + np.exp(x))

def swish(x, deriv=False, B=1):
    """A close continuous approximation of ReLU, recently published."""
    if deriv:
        return sigmoid(B*x) + B*swish(x) * (1 - sigmoid(B*x))

    return x * sigmoid(B*x)

def e_swish(x, deriv=False, B=1.5):
    """A variation of the swish function with different parameters."""
    if deriv:
        return B*sigmoid(x) + e_swish(x) * (1 - sigmoid(x))

    return B*x * sigmoid(x)

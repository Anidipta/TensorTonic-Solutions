import numpy as np, math

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    return 1 / (1 + np.exp(-np.array(x, dtype=float)))
import itertools
import numpy as np


def generate(n_dims: int) -> np.ndarray:
    shape = (2,) * n_dims
    data = np.zeros(shape)
    for indices in itertools.product([0, 1], repeat=n_dims):
        data[indices] = np.sum(indices) / n_dims
    return data


def generate_custom(n_dims: int, value_fn) -> np.ndarray:
    shape = (2,) * n_dims
    data = np.zeros(shape)
    for indices in itertools.product([0, 1], repeat=n_dims):
        data[indices] = value_fn(indices)
    return data

import numpy as np
from numpy.typing import NDArray


def emd_efecto(u: NDArray[np.float32], v: NDArray[np.float32]) -> float:
    return float(np.sum(np.abs(u - v)))


def emd_causal(u: NDArray[np.float64], v: NDArray[np.float64]) -> float:
    from pyemd import emd

    if not all(isinstance(arr, np.ndarray) for arr in [u, v]):
        raise TypeError("u and v must be numpy arrays.")

    n: int = u.size
    coste: NDArray[np.float64] = np.empty((n, n))

    for i in range(n):
        coste[i, :i] = [hamming_distance(i, j) for j in range(i)]
        coste[:i, i] = coste[i, :i]
    np.fill_diagonal(coste, 0)

    mat_costes: NDArray[np.float64] = np.array(coste, dtype=np.float64)
    return emd(u, v, mat_costes)


def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def count_bits(n: int) -> int:
    return bin(n).count("1")

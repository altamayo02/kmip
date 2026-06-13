import numpy as np
from itertools import product, islice
from math import ceil
from typing import Generator

from src.models.system import System


def assign_elements_to_blocks(elements, k: int):
    """Assign actual elements to k labeled blocks (blocks may be empty)."""
    n = len(elements)
    if n == 0:
        yield [[] for _ in range(k)]
        return
    total = k ** n
    for code in range(total):
        blocks = [[] for _ in range(k)]
        remaining = code
        for e in elements:
            blocks[remaining % k].append(e)
            remaining //= k
        yield blocks


def all_k_partitions(
    alcance_indices, mecanismo_indices, k: int
) -> Generator:
    """Yield every k-partition of the given alcance and mecanismo indices.

    Each partition is a tuple of (frozenset(mech), frozenset(alc)) pairs
    for each of the k blocks. Blocks where both mech and alc are empty
    are excluded.
    """
    alc_assign = assign_elements_to_blocks(list(alcance_indices), k)
    mech_assign = assign_elements_to_blocks(list(mecanismo_indices), k)

    total = k ** (len(alcance_indices) + len(mecanismo_indices))
    for alc_blocks, mech_blocks in islice(
        product(alc_assign, mech_assign), 0, ceil(total / 2)
    ):
        if any(not a and not b for a, b in zip(alc_blocks, mech_blocks)):
            continue
        yield tuple(
            (frozenset(mech_blocks[i]), frozenset(alc_blocks[i]))
            for i in range(k)
        )


def marginal_from_cube(cube, initial_state, keep_dims):
    """Marginalize cube over dims NOT in keep_dims, return probability at
    the initial state."""
    if not keep_dims:
        return float(np.mean(cube.data))
    marginalize = np.array(
        [d for d in cube.dims if d not in keep_dims], dtype=np.int8
    )
    if marginalize.size:
        mc = cube.marginalizar(marginalize)
    else:
        mc = cube
    if mc.dims.size == 0:
        return float(mc.data)
    inicial = tuple(int(initial_state[j]) for j in mc.dims)
    return float(mc.data[inicial[::-1]])


def k_partition_distribution(system: System, k_partition) -> np.ndarray:
    """Marginal distribution vector under a k-partition for the given system."""
    dist = np.zeros(len(system.ncubos), dtype=np.float32)
    for pos_idx, cube in enumerate(system.ncubos):
        future_idx = cube.indice
        for mech_block, alc_block in k_partition:
            if future_idx in alc_block:
                dist[pos_idx] = marginal_from_cube(
                    cube, system.estado_inicial, set(mech_block)
                )
                break
    return dist


def normalize_partition(partition):
    """Normalize a k-partition for deduplication (stable ordering)."""
    return tuple(
        sorted(
            (tuple(sorted(m)), tuple(sorted(a)))
            for m, a in partition
        )
    )


def count_k_partitions(m: int, n: int, k: int) -> int:
    """Exact count of valid k-partitions for m alcance + n mecanismo elements."""
    from math import comb
    total = k ** (m + n)
    term = 0
    for j in range(1, k + 1):
        sign = -1 if j % 2 else 1
        term += sign * comb(k, j) * ((k - j) ** (m + n))
    return total + term

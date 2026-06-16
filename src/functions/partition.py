import numpy as np

from src.models.system import System


def _marginal_from_cube(cube, initial_state: np.ndarray, keep_dims: set) -> float:
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
    num_nodes = len(system.ncubos)
    dist = np.zeros(num_nodes, dtype=np.float32)
    for pos_idx, cube in enumerate(system.ncubos):
        future_idx = cube.indice
        for mech_block, alc_block in k_partition:
            if future_idx in alc_block:
                dist[pos_idx] = _marginal_from_cube(
                    cube, system.estado_inicial, set(mech_block)
                )
                break
    return dist

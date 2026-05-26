import numpy as np


def nested_cube(coords: tuple, n_dims: int, factor: float = 0.35) -> np.ndarray:
    pos = np.zeros(3)
    for i, c in enumerate(coords):
        axis = i % 3
        sign = -1 if (c == 0) else 1
        if i < 3:
            pos[axis] = sign
        else:
            pos[axis] += sign * (factor ** (i - 2))
    return pos


def isometric(coords: tuple) -> np.ndarray:
    x = coords[0] - 0.5 if len(coords) > 0 else 0
    y = coords[1] - 0.5 if len(coords) > 1 else 0
    z = coords[2] - 0.5 if len(coords) > 2 else 0
    return np.array([
        (x - y) * np.cos(np.pi / 6),
        (x + y) * np.sin(np.pi / 6) - z,
        0,
    ])


def stereographic(point_4d: tuple, w_factor: float = 2.0) -> np.ndarray:
    x, y, z, w = point_4d
    scale = 1 / (1 - w / w_factor) if w != w_factor else 1
    return np.array([x * scale, y * scale, z * scale])


def random_matrix(n_dims: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((3, n_dims))


def offset_alternating(coords: tuple, delta: float = 0.5) -> np.ndarray:
    pos = np.zeros(3)
    axes = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    remaining = len(coords) - 3
    for i, c in enumerate(coords):
        if i < 3:
            pos += (2 * c - 1) * axes[i]
        else:
            pos += (2 * c - 1) * axes[i % 3] * (delta ** (i - 2))
    return pos

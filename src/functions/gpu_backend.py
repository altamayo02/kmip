"""
GPU backend for k-partition evaluation.

Uses CuPy for GPU-accelerated batch evaluation of partitions.
Preserves exact same logic as the CPU path (lookup table + L1 EMD).

Two operations:
  1. precompute_marginals_table  -- precompute all marginal probabilities p(j|M)
  2. eval_masks_gpu              -- batch evaluate partitions from bitmask arrays
"""

import time

import numpy as np

import os
import warnings

HAS_CUPY = False
HAS_GPU = False
_HAS_HEADERS = False

# Auto-detect CUDA headers from nvidia pip packages BEFORE importing CuPy
_CUDA_HEADER_CANDIDATES = [
    os.path.join(
        os.environ.get("APPDATA", ""),
        "Python",
        f"Python{os.sys.version_info.major}{os.sys.version_info.minor}",
        "site-packages",
        "nvidia",
        "cuda_runtime",
        "include",
    ),
]

if not os.environ.get("CUDA_PATH"):
    for candidate in _CUDA_HEADER_CANDIDATES:
        if os.path.isfile(os.path.join(candidate, "cuda_runtime.h")):
            # candidate = ...\nvidia\cuda_runtime\include
            # CUDA_PATH should be parent of include = ...\nvidia\cuda_runtime
            os.environ["CUDA_PATH"] = os.path.dirname(candidate)
            _HAS_HEADERS = True
            break
else:
    _HAS_HEADERS = True

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import cupy as cp
    HAS_CUPY = True

    if _HAS_HEADERS:
        try:
            _test = cp.array([1.0, 2.0])
            _test + _test
            _ = cp.mean(_test)
            HAS_GPU = True
        except Exception:
            pass

except ImportError:
    pass


# ── Precomputation ──────────────────────────────────────────────────────

def precompute_marginals_table(system, use_gpu: bool = False) -> np.ndarray:
    """Precompute p(j | M) for all cubes j and all mech subsets M.

    Returns:
        table[n_cubes, 2^n_dims] where table[cube_idx, mech_mask] = marginal probability
        of the initial state for that cube, marginalizing over dims NOT in mask.
    """
    n = len(system.ncubos)
    m = len(system.dims_ncubos)
    initial = system.estado_inicial
    table = np.empty((n, 1 << m), dtype=np.float32)

    for j, cube in enumerate(system.ncubos):
        data = cube.data.astype(np.float64)
        for mask in range(1 << m):
            if mask == 0:
                table[j, mask] = float(np.mean(data))
                continue

            keep_dims = [d for d in range(m) if mask & (1 << d)]
            marg_axes = tuple(
                m - 1 - d for d in range(m) if d not in keep_dims
            )

            if not marg_axes:
                idx = tuple(int(initial[d]) for d in keep_dims)
                table[j, mask] = float(data[idx[::-1]])
            else:
                result = data.mean(axis=marg_axes)
                if result.ndim == 0:
                    table[j, mask] = float(result)
                else:
                    idx = tuple(int(initial[d]) for d in keep_dims)
                    table[j, mask] = float(result[idx[::-1]])

    return table


# ── GPU batch evaluation ─────────────────────────────────────────────

def _eval_gpu_batch(masks_gpu, k, m, precomputed_gpu, intact_gpu):
    """Evaluate one batch; all arrays already on GPU.

    Returns (best_emd, best_relative_idx).
    """
    B = masks_gpu.shape[0]
    mech_masks = masks_gpu[:, :k]
    alc_masks = masks_gpu[:, k:]

    group_for_alc = cp.zeros((B, m), dtype=cp.int32)
    for j in range(m):
        bit_j = 1 << j
        for g in range(k):
            has_j = (alc_masks[:, g] & bit_j) != 0
            group_for_alc[has_j, j] = g

    dist = cp.zeros((B, m), dtype=cp.float32)
    for j in range(m):
        g = group_for_alc[:, j]
        mech_mask = mech_masks[cp.arange(B), g]
        dist[:, j] = precomputed_gpu[j, mech_mask]

    emd = cp.sum(cp.abs(dist - intact_gpu[None, :]), axis=1)
    best_val = float(cp.min(emd))
    best_rel = int(cp.argmin(emd))
    return best_val, best_rel


def eval_masks_gpu(
    masks_host: np.ndarray,
    k: int,
    m: int,
    precomputed_host: np.ndarray,
    intact_host: np.ndarray,
) -> tuple[float, int]:
    """Evaluate one batch from host arrays. Returns (best_emd, best_rel_idx)."""
    return _eval_gpu_batch(
        cp.array(masks_host), k, m,
        cp.array(precomputed_host), cp.array(intact_host),
    )

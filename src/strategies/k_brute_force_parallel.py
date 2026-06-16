"""
Parallel KBruteForce: evaluate all k-partitions using multiprocessing + GPU.

Uses unlabeled partition enumeration (no group-label redundancy) instead of
the old code-based encoding.  Precomputed marginal lookup table eliminates
~90% of CPU compute cost per partition.

Never materializes the full partition list — streams via generators.
"""

import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from src.config import Config
from src.strategies.base import SIA
from src.middlewares.slogger import SafeLogger
from src.middlewares.profile import profiler_manager, profile
from src.functions.emd import emd_efecto
from src.functions.format import fmt_kparticion
from src.functions.partitions import (
    all_k_partitions_unlabeled,
    count_k_partitions_unlabeled,
    partition_bitmasks,
)
from src.functions.gpu_backend import (
    HAS_GPU,
    _eval_gpu_batch,
    precompute_marginals_table,
)
from src.solution import Solution


LABEL = "KBruteForceParallel"
TAG_STRATEGY = f"{LABEL}_strategy"
TAG_ANALYSIS = f"{LABEL}_analysis"

SEQUENTIAL_THRESHOLD = 100000
GPU_THRESHOLD = 500000
GPU_BATCH_SIZE = 500000


# ── Lookup-table based evaluation ───────────────────────────────────────

def _partition_dist(precomputed, intact, mech_masks, alc_masks):
    n = len(intact)
    emd_sum = 0.0
    for j in range(n):
        for g, mask in enumerate(alc_masks):
            if mask & (1 << j):
                mech_mask = mech_masks[g]
                break
        emd_sum += abs(precomputed[j, mech_mask] - intact[j])
    return emd_sum


# ── Worker globals (set by initializer) ──────────────────────────────────

_worker_precomputed: np.ndarray | None = None
_worker_intact: np.ndarray | None = None


def _worker_init(precomputed, intact):
    global _worker_precomputed, _worker_intact
    _worker_precomputed = precomputed
    _worker_intact = intact


# ── Lazy chunk generators (never materialise all partitions) ────────────

def _cpu_chunks(partitions, m, chunk_size):
    """Yield (start_idx, [(mask_tuple, partition_obj), ...]) lazily."""
    chunk: list = []
    start = 0
    for p in partitions:
        mm, am = partition_bitmasks(p, m)
        chunk.append((mm + am, p))
        if len(chunk) >= chunk_size:
            yield start, chunk
            start += len(chunk)
            chunk = []
    if chunk:
        yield start, chunk


def _gpu_batches(partitions, m, k, batch_size):
    """Yield (ndarray_of_masks, list_of_partitions) batches."""
    batch = np.empty((batch_size, 2 * k), dtype=np.int64)
    parts: list = []
    row = 0
    for p in partitions:
        parts.append(p)
        mm, am = partition_bitmasks(p, m)
        batch[row] = mm + am
        row += 1
        if row >= batch_size:
            yield batch, parts
            batch = np.empty((batch_size, 2 * k), dtype=np.int64)
            parts = []
            row = 0
    if row > 0:
        yield batch[:row], parts


def _eval_chunk(chunk_args: tuple) -> tuple[float, int, object]:
    """Evaluate a chunk of partitions via the lookup table.

    chunk_args = (start_idx, [(mask_tuple, partition_obj), ...])

    Returns (best_emd, global_idx, best_partition).
    """
    start_idx, chunk = chunk_args
    best_emd = float("inf")
    best_rel = -1
    best_part = None

    for rel_idx, (masks, part) in enumerate(chunk):
        mech_masks = masks[:len(masks)//2]
        alc_masks = masks[len(masks)//2:]
        emd = _partition_dist(_worker_precomputed, _worker_intact, mech_masks, alc_masks)
        if emd < best_emd - 1e-15:
            best_emd = emd
            best_rel = rel_idx
            best_part = part
            if emd == 0.0:
                break

    return best_emd, start_idx + best_rel, best_part


# ── Class ────────────────────────────────────────────────────────────────

class KBruteForceParallel(SIA):
    def __init__(self, tpm: np.ndarray, config: Config, k: int = 3,
                 n_workers: int | None = None, use_gpu: bool = True):
        super().__init__(tpm, config)
        profiler_manager.start_session(
            f"K{k}ForcePar{len(tpm[1])}{config.pagina_muestra}"
        )
        self.k = k
        self.n_workers = n_workers
        self.use_gpu = HAS_GPU and use_gpu
        self.distancia_metrica = emd_efecto
        self.logger = SafeLogger(TAG_STRATEGY)

    @profile(context={"type": TAG_ANALYSIS})
    def aplicar_estrategia(
        self, estado_inicial: str, condiciones: str, alcance: str, mecanismo: str
    ):
        self.sia_preparar_subsistema(estado_inicial, condiciones, alcance, mecanismo)

        futuros = self.sia_subsistema.indices_ncubos
        presentes = self.sia_subsistema.dims_ncubos
        intact = self.sia_dists_marginales
        m, n = futuros.size, presentes.size

        if self.k > m + n:
            return [
                Solution(
                    estrategia=LABEL,
                    perdida=float("inf"),
                    distribucion_subsistema=intact,
                    distribucion_particion=intact,
                    tiempo_ejecucion=time.time() - self.sia_tiempo_inicio,
                    particion=f"k={self.k} > m+n={m+n} (no v\u00e1lido)",
                )
            ]

        total = count_k_partitions_unlabeled(m, n, self.k)
        if total == 0:
            return [
                Solution(
                    estrategia=LABEL,
                    perdida=float("inf"),
                    distribucion_subsistema=intact,
                    distribucion_particion=intact,
                    tiempo_ejecucion=time.time() - self.sia_tiempo_inicio,
                    particion="sin soluci\u00f3n",
                )
            ]

        # ── Precompute marginal lookup table ─────────────────────────
        subsistema = self.sia_subsistema
        precomputed = precompute_marginals_table(subsistema)

        best_emd = float("inf")
        best_part = None
        mech_list = list(presentes)
        alc_list = list(futuros)

        # ── GPU batch evaluation (streaming) ─────────────────────────
        if self.use_gpu and total > GPU_THRESHOLD and HAS_GPU:
            import cupy as cp
            precomputed_gpu = cp.array(precomputed)
            intact_gpu = cp.array(intact)

            gen = all_k_partitions_unlabeled(mech_list, alc_list, self.k)
            for batch_arr, batch_parts in _gpu_batches(
                gen, m, self.k, GPU_BATCH_SIZE
            ):
                masks_gpu = cp.array(batch_arr)
                bat_emd, bat_rel = _eval_gpu_batch(
                    masks_gpu, self.k, m, precomputed_gpu, intact_gpu
                )
                if bat_emd < best_emd - 1e-15:
                    best_emd = bat_emd
                    best_part = batch_parts[bat_rel]
                    if best_emd == 0.0:
                        break

        # ── Sequential (small, lazy generator) ───────────────────────
        elif total <= SEQUENTIAL_THRESHOLD:
            gen = all_k_partitions_unlabeled(mech_list, alc_list, self.k)
            for p in gen:
                mm, am = partition_bitmasks(p, m)
                emd_val = _partition_dist(precomputed, intact, mm, am)
                if emd_val < best_emd - 1e-15:
                    best_emd = emd_val
                    best_part = p
                    if best_emd == 0.0:
                        break

        # ── Parallel CPU workers (streaming chunks) ──────────────────
        else:
            n_workers = self.n_workers or _default_workers()
            chunk_size = max(500, total // (n_workers * 4))

            gen = all_k_partitions_unlabeled(mech_list, alc_list, self.k)
            chunk_gen = _cpu_chunks(gen, m, chunk_size)

            with ProcessPoolExecutor(
                max_workers=n_workers,
                initializer=_worker_init,
                initargs=(precomputed, intact),
            ) as executor:
                for result_emd, _result_idx, result_part in executor.map(
                    _eval_chunk, chunk_gen
                ):
                    if result_emd < best_emd - 1e-15:
                        best_emd = result_emd
                        best_part = result_part
                        if result_emd == 0.0:
                            executor.shutdown(wait=False, cancel_futures=True)
                            break

        if best_part is None:
            return [
                Solution(
                    estrategia=LABEL,
                    perdida=float("inf"),
                    distribucion_subsistema=intact,
                    distribucion_particion=intact,
                    tiempo_ejecucion=time.time() - self.sia_tiempo_inicio,
                    particion="sin soluci\u00f3n",
                )
            ]

        fmt = fmt_kparticion(best_part)
        dist = subsistema.distribucion_marginal()
        return [
            Solution(
                estrategia=LABEL,
                perdida=best_emd,
                distribucion_subsistema=intact,
                distribucion_particion=dist if best_emd > 0 else intact,
                tiempo_ejecucion=time.time() - self.sia_tiempo_inicio,
                particion=fmt,
            )
        ]


def _default_workers():
    import os
    return os.cpu_count() or 4

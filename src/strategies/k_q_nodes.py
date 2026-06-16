import random
import time
from typing import Union

import numpy as np

from src.config import Config
from src.middlewares.slogger import SafeLogger
from src.functions.emd import emd_efecto
from src.functions.labels import ABECEDARY
from src.functions.partition import k_partition_distribution
from src.middlewares.profile import profiler_manager, profile
from src.functions.format import fmt_kparticion
from src.functions.gpu_backend import HAS_GPU, HAS_CUPY
from src.strategies.base import SIA
from src.solution import Solution


LABEL = "KQNodes"
TAG_STRATEGY = f"{LABEL}_strategy"
TAG_ANALYSIS = f"{LABEL}_analysis"
EFECTO = 1
ACTUAL = 0
INFTY_POS = float("inf")
EPS = 1e-12


class KQNodes(SIA):
    def __init__(
        self,
        tpm: np.ndarray,
        config: Config,
        k: int = 3,
        use_refinement: bool = True,
        use_multi_seed: bool = False,
        use_gpu: bool = True,
    ):
        super().__init__(tpm, config, k=k)
        profiler_manager.start_session(
            f"K{k}QNodes{len(tpm[1])}{config.pagina_muestra}"
        )
        self.k = k
        self.use_refinement = use_refinement
        self.use_multi_seed = use_multi_seed
        self.use_gpu = HAS_GPU and use_gpu
        self.early_stopping = True

        self.m: int
        self.n: int

        self.indices_alcance: np.ndarray
        self.indices_mecanismo: np.ndarray

        self.memo_evaluate = {}

        self._intact: np.ndarray | None = None
        self._mech_pos: dict[int, int] | None = None
        self._alc_pos: dict[int, int] | None = None
        self._marg_cache: dict | None = None
        self._use_gpu_eval: bool = False

        self.logger = SafeLogger(TAG_STRATEGY)

    @profile(context={"type": TAG_ANALYSIS})
    def aplicar_estrategia(
        self,
        estado_inicial: str,
        condicion: str,
        alcance: str,
        mecanismo: str,
        _skip_prep: bool = False,
    ):
        if not _skip_prep:
            self.sia_preparar_subsistema(estado_inicial, condicion, alcance, mecanismo)

        futuro = tuple(
            (EFECTO, idx_efecto)
            for idx_efecto in self.sia_subsistema.indices_ncubos
        )
        presente = tuple(
            (ACTUAL, idx_actual)
            for idx_actual in self.sia_subsistema.dims_ncubos
        )

        self.m = self.sia_subsistema.indices_ncubos.size
        self.n = self.sia_subsistema.dims_ncubos.size

        self.indices_alcance = self.sia_subsistema.indices_ncubos
        self.indices_mecanismo = self.sia_subsistema.dims_ncubos

        if self.use_gpu:
            self._setup_gpu_cache()
            self._use_gpu_eval = True
        else:
            self._marg_cache = None
            self._intact = None
            self._use_gpu_eval = False

        all_vertices = list(presente + futuro)

        if self.k == 1:
            return [
                Solution(
                    estrategia=LABEL,
                    perdida=0.0,
                    distribucion_subsistema=self.sia_dists_marginales,
                    distribucion_particion=self.sia_dists_marginales,
                    tiempo_ejecucion=time.time() - self.sia_tiempo_inicio,
                    particion="(trivial, k=1)",
                )
            ]

        if self.k > len(all_vertices):
            return [
                Solution(
                    estrategia=LABEL,
                    perdida=float("inf"),
                    distribucion_subsistema=self.sia_dists_marginales,
                    distribucion_particion=self.sia_dists_marginales,
                    tiempo_ejecucion=time.time() - self.sia_tiempo_inicio,
                    particion=f"k={self.k} > total nudos={len(all_vertices)} (invalido)",
                )
            ]

        if self.k == 2:
            return self._search_k2(all_vertices)

        best_groups, best_emd, best_dist = self._k_partition_algorithm(all_vertices)

        kp = self._to_kpartition_tuple(best_groups)
        fmt = fmt_kparticion(kp)

        return [
            Solution(
                estrategia=LABEL,
                perdida=best_emd,
                distribucion_subsistema=self.sia_dists_marginales,
                distribucion_particion=best_dist,
                tiempo_ejecucion=time.time() - self.sia_tiempo_inicio,
                particion=fmt,
            )
        ]

    # ── Main algorithm ──────────────────────────────────────────────────────

    def _k_partition_algorithm(self, vertices):
        if self.k >= len(vertices):
            groups = [[v] for v in vertices]
            emd, dist = self._evaluate_k_partition(groups)
            return groups, emd, dist

        if self.use_multi_seed:
            best_groups = None
            best_emd = INFTY_POS
            best_dist = None

            seeds = self._select_seeds_deterministic(vertices)
            groups, emd, dist = self._run_with_seeds(vertices, seeds)
            if emd < best_emd:
                best_emd = emd
                best_groups = groups
                best_dist = dist
            if best_emd == 0:
                return best_groups, best_emd, best_dist

            for _ in range(min(5, len(vertices) - 1)):
                seeds = self._select_seeds_random(vertices)
                groups, emd, dist = self._run_with_seeds(vertices, seeds)
                if emd < best_emd:
                    best_emd = emd
                    best_groups = groups
                    best_dist = dist
                if best_emd == 0:
                    break

            return best_groups, best_emd, best_dist
        else:
            seeds = self._select_seeds_deterministic(vertices)
            return self._run_with_seeds(vertices, seeds)

    def _run_with_seeds(self, vertices, seeds):
        groups, _ = self._greedy_assign(vertices, seeds)

        emd, dist = self._evaluate_k_partition(groups)
        if emd == 0 and self.early_stopping:
            return groups, emd, dist

        if self.use_refinement:
            groups, emd, dist = self._refine(groups)

        return groups, emd, dist

    # ── Seed selection ─────────────────────────────────────────────────────

    def _select_seeds_deterministic(self, vertices):
        if len(vertices) <= self.k:
            return list(vertices)[: self.k]

        scores = []
        for v in vertices:
            group = [v]
            emd, _ = self._evaluate_k_partition(
                [group, [x for x in vertices if x != v]]
            )
            scores.append((emd, v))

        scores.sort(key=lambda x: -x[0])
        return [v for _, v in scores[: self.k]]

    def _select_seeds_random(self, vertices):
        k_actual = min(self.k, len(vertices))
        return random.sample(vertices, k_actual)

    # ── Greedy assignment ──────────────────────────────────────────────────

    def _greedy_assign(self, all_vertices, seeds):
        groups = [[s] for s in seeds]

        used = set(seeds)
        remaining = [v for v in all_vertices if v not in used]

        for v in remaining:
            best_emd = INFTY_POS
            best_group = -1
            best_dist = None

            for g_idx in range(self.k):
                groups[g_idx].append(v)
                emd, dist = self._evaluate_k_partition(groups)
                groups[g_idx].pop()

                if emd < best_emd:
                    best_emd = emd
                    best_group = g_idx
                    best_dist = dist

            groups[best_group].append(v)

            if best_emd == 0 and self.early_stopping:
                break

        return groups, remaining

    # ── Iterative refinement ───────────────────────────────────────────────

    def _refine(self, groups):
        emd, dist = self._evaluate_k_partition(groups)
        if emd == 0:
            return groups, emd, dist

        improved = True
        max_iterations = 10
        iteration = 0

        while improved and iteration < max_iterations:
            improved = False
            iteration += 1

            for g_src in range(self.k):
                if len(groups[g_src]) <= 1:
                    continue

                for v in list(groups[g_src]):
                    if len(groups[g_src]) <= 1:
                        break

                    for g_dst in range(self.k):
                        if g_dst == g_src:
                            continue

                        groups[g_src].remove(v)
                        groups[g_dst].append(v)

                        new_emd, new_dist = self._evaluate_k_partition(groups)

                        if new_emd < emd - EPS:
                            emd = new_emd
                            dist = new_dist
                            improved = True
                            if emd == 0 and self.early_stopping:
                                return groups, emd, dist
                            break
                        else:
                            groups[g_dst].remove(v)
                            groups[g_src].append(v)

        return groups, emd, dist

    # ── Evaluation ─────────────────────────────────────────────────────────

    def _evaluate_k_partition(self, groups):
        normalized = tuple(
            sorted(tuple(sorted(g)) for g in groups)
        )

        if normalized in self.memo_evaluate:
            return self.memo_evaluate[normalized]

        if self._use_gpu_eval:
            emd, dist = self._eval_fast(groups)
        else:
            kp = self._to_kpartition_tuple(groups)
            dist = k_partition_distribution(self.sia_subsistema, kp)
            emd = emd_efecto(dist, self.sia_dists_marginales)

        self.memo_evaluate[normalized] = (emd, dist)
        return emd, dist

    def _to_kpartition_tuple(self, groups):
        parts = []
        for g in groups:
            mech = frozenset(idx for time, idx in g if time == ACTUAL)
            alc = frozenset(idx for time, idx in g if time == EFECTO)
            parts.append((mech, alc))
        return tuple(parts)

    # ── GPU-accelerated evaluation ──────────────────────────────────────────

    def _setup_gpu_cache(self):
        """Set up lazy marginal cache + position maps for fast evaluation.
        
        If CuPy is available, cube data is uploaded to GPU for accelerated
        marginalization via cp.mean (up to 5-10x faster on GTX 1650 for
        large tensors).
        """
        system = self.sia_subsistema
        mech_dims = self.indices_mecanismo
        alc_indices = self.indices_alcance

        self._mech_pos = {int(idx): pos for pos, idx in enumerate(mech_dims)}
        self._alc_pos = {int(idx): pos for pos, idx in enumerate(alc_indices)}
        self._intact = self.sia_dists_marginales
        self._marg_cache = {}
        self._system_for_cache = system
        self._cube_gpu = None

        if HAS_CUPY:
            import cupy as cp
            self._cp = cp
            self._cube_gpu = [
                cp.array(cube.data.astype(np.float32))
                for cube in system.ncubos
            ]
        else:
            self._cp = None
            self._cube_gpu = None

    def _get_marginal(self, cube_idx: int, mask: int) -> float:
        key = (cube_idx, mask)
        val = self._marg_cache.get(key)
        if val is not None:
            return val

        system = self._system_for_cache
        cube = system.ncubos[cube_idx]
        mech_dims = self.indices_mecanismo
        initial = system.estado_inicial

        keep_pos = [d for d in range(len(mech_dims)) if mask & (1 << d)]
        keep_set = set(int(mech_dims[d]) for d in keep_pos)

        if self._cube_gpu is not None:
            cp = self._cp
            data_gpu = self._cube_gpu[cube_idx]
            if not keep_set:
                val = float(cp.mean(data_gpu).get())
                self._marg_cache[key] = val
                return val

            marg_dims = [d for d in cube.dims if d not in keep_set]
            if marg_dims:
                cube_len = len(cube.dims)
                ejes = tuple(
                    cube_len - 1 - pos
                    for pos, d in enumerate(cube.dims)
                    if d in marg_dims
                )
                result = cp.mean(data_gpu, axis=ejes)
            else:
                result = data_gpu

            if result.ndim == 0:
                val = float(result.get())
            else:
                remaining_dims = [d for d in cube.dims if d not in marg_dims]
                idx = tuple(int(initial[d]) for d in remaining_dims)
                val = float(result[idx[::-1]].get())
        else:
            if not keep_set:
                val = float(np.mean(cube.data))
                self._marg_cache[key] = val
                return val

            marginalize = np.array(
                [d for d in cube.dims if d not in keep_set],
                dtype=np.int8,
            )
            if marginalize.size:
                mc = cube.marginalizar(marginalize)
            else:
                mc = cube
            if mc.dims.size == 0:
                val = float(mc.data)
            else:
                inicial = tuple(int(initial[idx]) for idx in mc.dims)
                val = float(mc.data[inicial[::-1]])

        self._marg_cache[key] = val
        return val

    def _get_dist(self, mech_masks, alc_masks):
        dist = np.empty(self.m, dtype=np.float32)
        for j in range(self.m):
            for g in range(self.k):
                if alc_masks[g] & (1 << j):
                    dist[j] = self._get_marginal(j, mech_masks[g])
                    break
        return dist

    def _masks_from_partition(self, groups):
        kp = self._to_kpartition_tuple(groups)
        mech_masks = [0] * self.k
        alc_masks = [0] * self.k
        for g, (mech_set, alc_set) in enumerate(kp):
            for idx in mech_set:
                pos = self._mech_pos.get(int(idx))
                if pos is not None:
                    mech_masks[g] |= 1 << pos
            for idx in alc_set:
                pos = self._alc_pos.get(int(idx))
                if pos is not None:
                    alc_masks[g] |= 1 << pos
        return mech_masks, alc_masks

    def _eval_fast(self, groups):
        mech_masks, alc_masks = self._masks_from_partition(groups)
        dist = self._get_dist(mech_masks, alc_masks)
        emd = float(np.sum(np.abs(dist - self._intact)))
        return emd, dist

    # ── k=2 fallback ───────────────────────────────────────────────────────

    def _search_k2(self, vertices):
        seeds = self._select_seeds_deterministic(vertices)[:2]
        groups, emd, dist = self._run_with_seeds(vertices, seeds)

        kp = self._to_kpartition_tuple(groups)
        fmt = fmt_kparticion(kp)

        return [
            Solution(
                estrategia=LABEL,
                perdida=emd,
                distribucion_subsistema=self.sia_dists_marginales,
                distribucion_particion=dist,
                tiempo_ejecucion=time.time() - self.sia_tiempo_inicio,
                particion=fmt,
            )
        ]

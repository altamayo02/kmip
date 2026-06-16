import math
import time
from itertools import combinations, permutations

import numpy as np

from src.config import Config
from src.functions.emd import emd_efecto
from src.functions.format import fmt_kparticion
from src.functions.partitions import (
    all_k_partitions_unlabeled,
    count_k_partitions_unlabeled,
)
from src.solution import Solution
from src.strategies.base import SIA
from src.strategies.geometric_base import GeometricBase

LABEL = "KGeometric"


class KGeometric(SIA, GeometricBase):
    def __init__(self, tpm: np.ndarray, config: Config, k: int = 3):
        super().__init__(tpm, config)
        self.k = k

    def aplicar_estrategia(
        self,
        estado_inicial: str,
        condicion: str,
        alcance: str,
        mecanismo: str,
    ):
        self.sia_preparar_subsistema(estado_inicial, condicion, alcance, mecanismo)
        start = time.time()

        if self.k < 2:
            return [
                Solution(
                    estrategia=LABEL,
                    perdida=float("inf"),
                    distribucion_subsistema=self.sia_dists_marginales,
                    distribucion_particion=self.sia_dists_marginales,
                    tiempo_ejecucion=time.time() - start,
                    particion="k must be >= 2",
                )
            ]

        if self.k == 2:
            n_dims = len(self.sia_subsistema.dims_ncubos)
            if n_dims <= 14:
                return self._search_k2(start)
            return self._search_k2_selection(start)
        return self._search_kgt2(start)

    def _search_k2(self, start: float):
        """Exact bipartition search via geometric paths (like GeometricSIA)."""
        self._compute_geometric_data()

        candidatos = self._identify_bipartitions()
        mejor_valor = float("inf")
        mejores_mip = []
        mejores_dist = None

        for presentes_idx, futuros_idx in candidatos:
            presentes = self.sia_subsistema.dims_ncubos[presentes_idx]
            futuros = self.sia_subsistema.indices_ncubos[futuros_idx]
            dist = self.sia_subsistema.bipartir(
                futuros, presentes
            ).distribucion_marginal()
            emd = emd_efecto(dist, self.sia_dists_marginales)

            if emd < mejor_valor - 1e-12:
                mejor_valor = emd
                mejores_mip = [(presentes_idx, futuros_idx)]
                mejores_dist = dist
            elif abs(emd - mejor_valor) < 1e-12:
                mejores_mip.append((presentes_idx, futuros_idx))

        return self._build_k2_solutions(
            mejores_mip, mejor_valor, mejores_dist, start
        )

    def _search_kgt2(self, start: float):
        """Approximate k-partition search: families A + B, shared cache."""
        # ── Setup ──────────────────────────────────────────────────────
        futuros = self.sia_subsistema.indices_ncubos
        presentes = self.sia_subsistema.dims_ncubos
        u_indices = list(presentes)
        v_indices = list(futuros)
        u = len(u_indices)
        v = len(v_indices)

        if self.k > u + v:
            return [
                Solution(
                    estrategia=LABEL,
                    perdida=float("inf"),
                    distribucion_subsistema=self.sia_dists_marginales,
                    distribucion_particion=self.sia_dists_marginales,
                    tiempo_ejecucion=time.time() - start,
                    particion=(
                        f"k={self.k} > u+v={u+v} (no v\u00e1lido)"
                    ),
                )
            ]

        initial = self.sia_subsistema.estado_inicial

        cube_data = []
        cube_dims = []
        for idx in range(v):
            nc = self.sia_subsistema.ncubos[idx]
            d = nc.data
            if d.itemsize > 4:
                compact = np.ascontiguousarray(d.astype(np.float32))
                object.__setattr__(nc, "data", compact)
                d = compact
            cube_data.append(d)
            cube_dims.append(list(nc.dims))

        cube_mean = np.array(
            [float(d.mean()) for d in cube_data], dtype=np.float32
        )

        self.tpm = None

        def _marg_value(data, dims, marg_dims):
            n = len(dims)
            index = []
            for a in range(n):
                d = dims[n - 1 - a]
                if d in marg_dims:
                    index.append(slice(None))
                else:
                    index.append(int(initial[d]))
            return float(data[tuple(index)].mean())

        all_mech = frozenset(u_indices)
        all_alc = frozenset(v_indices)
        intact = self.sia_dists_marginales

        # ── Decide: exact enumeration vs heuristic ──────────────────────
        total_partitions = count_k_partitions_unlabeled(u, v, self.k)
        if total_partitions <= 500_000:
            return self._search_exact(
                u_indices, v_indices, total_partitions,
                cube_data, cube_dims, initial, intact, start,
            )

        # ── Build influence matrix directly (no hypercube) ──────────
        influence = self._build_influence_matrix(u, v)

        # ── Primary: geometric clustering ────────────────────────────
        cluster_result = self._search_geometric_clustering(
            u_indices, v_indices,
            cube_data, cube_dims, initial, intact,
            influence, start,
        )

        # ── Families A+B fallback ─────────────────────────────────────
        family_b_feasible = (
            u >= self.k - 1
            and v >= self.k - 1
            and math.comb(u, self.k - 1)
                * math.comb(v, self.k - 1)
                * math.factorial(self.k - 1)
            <= 500_000
        )
        need_sizes = set(range(self.k))
        if family_b_feasible and u >= 1:
            need_sizes.add(u - 1)
        dist_cache = {}
        for s in sorted(need_sizes):
            for mech_combo in combinations(u_indices, s):
                sel_mech = frozenset(mech_combo)
                if sel_mech in dist_cache:
                    continue
                dist = np.zeros(v, dtype=np.float32)
                for j in range(v):
                    dist[j] = _marg_value(
                        cube_data[j], cube_dims[j], sel_mech
                    )
                dist_cache[sel_mech] = dist

        best_emd = float("inf")
        best_partitions: list[tuple] = []
        best_dist: np.ndarray | None = None

        if cluster_result is not None:
            best_emd, best_partitions, best_dist = cluster_result

        def _update_best(emd, kp, dist):
            nonlocal best_emd, best_partitions, best_dist
            if emd < best_emd - 1e-12:
                best_emd = emd
                best_partitions = [kp]
                best_dist = dist
            elif abs(emd - best_emd) < 1e-12:
                best_partitions.append(kp)

        if best_emd > 0:
            self._run_family_A(
                u, v, u_indices, v_indices,
                all_mech, all_alc, intact, cube_mean,
                dist_cache, _update_best,
            )

        if best_emd > 0 and family_b_feasible:
            self._run_family_B(
                u, v, u_indices, v_indices,
                all_mech, all_alc, intact, cube_mean,
                dist_cache, _update_best,
            )

        soluciones = []
        for kp in best_partitions:
            fmt = fmt_kparticion(kp)
            soluciones.append(
                Solution(
                    estrategia=LABEL,
                    perdida=best_emd,
                    distribucion_subsistema=self.sia_dists_marginales,
                    distribucion_particion=best_dist,
                    tiempo_ejecucion=time.time() - start,
                    particion=fmt,
                )
            )
        return soluciones

    def _run_family_A(
        self, u, v, u_indices, v_indices,
        all_mech, all_alc, intact, cube_mean,
        dist_cache, update_best,
    ):
        for combo in combinations(range(u + v), self.k - 1):
            sel_mech = frozenset(
                u_indices[i] for i in combo if i < u
            )
            sel_alc = frozenset(
                v_indices[j - u] for j in combo if j >= u
            )

            groups = []
            for i in sel_mech:
                groups.append((frozenset({i}), frozenset()))
            for j in sel_alc:
                groups.append((frozenset(), frozenset({j})))

            rem_mech = all_mech - sel_mech
            rem_alc = all_alc - sel_alc

            if not rem_mech and not rem_alc:
                continue

            groups.append((rem_mech, rem_alc))
            kp = tuple(groups)

            dist = dist_cache[sel_mech].copy()
            for j in sel_alc:
                dist[j] = cube_mean[j]

            emd = emd_efecto(dist, intact)
            update_best(emd, kp, dist)
            if emd == 0:
                break

    def _run_family_B(
        self, u, v, u_indices, v_indices,
        all_mech, all_alc, intact, cube_mean,
        dist_cache, update_best,
    ):
        k1 = self.k - 1
        for mech_combo in combinations(range(u), k1):
            mech_set = frozenset(u_indices[i] for i in mech_combo)
            for alc_combo in combinations(range(v), k1):
                alc_vals = [v_indices[j] for j in alc_combo]
                for perm in permutations(range(k1)):
                    groups = []
                    alc_override = {}
                    for t in range(k1):
                        m = u_indices[mech_combo[t]]
                        a = alc_vals[perm[t]]
                        groups.append((frozenset({m}), frozenset({a})))
                        alc_override[a] = m

                    rem_mech = all_mech - mech_set
                    rem_alc = all_alc - frozenset(alc_vals)
                    if not rem_mech and not rem_alc:
                        continue
                    groups.append((rem_mech, rem_alc))
                    kp = tuple(groups)

                    dist = dist_cache[mech_set].copy()
                    for a, m in alc_override.items():
                        dist[a] = dist_cache[all_mech - frozenset({m})][a]

                    emd = emd_efecto(dist, intact)
                    update_best(emd, kp, dist)
                    if emd == 0:
                        return

    def _build_influence_matrix(self, u, v):
        """Transition cost when only one mech flips (level-1 geometric, O(u×v), no hypercube)."""
        dims = self.sia_subsistema.dims_ncubos
        initial = self.sia_subsistema.estado_inicial[dims]
        flat = [nc.data.ravel() for nc in self.sia_subsistema.ncubos]

        def _to_int(state):
            return int("".join(str(int(b)) for b in reversed(state)), 2)

        init_int = _to_int(initial)
        influence = np.zeros((u, v), dtype=np.float32)
        for i in range(u):
            flip = initial.copy()
            flip[i] = 1 - flip[i]
            flip_int = _to_int(flip)
            for j in range(v):
                influence[i, j] = 0.5 * abs(
                    float(flat[j][init_int]) - float(flat[j][flip_int])
                )
        return influence

    def _search_geometric_clustering(
        self, u_indices, v_indices,
        cube_data, cube_dims, initial, intact,
        influence, start,
    ):
        """Primary k-partition search: cluster mechs by influence, assign alcs."""
        u = len(u_indices)
        v = len(v_indices)
        k = self.k

        clusters = [[i] for i in range(u)]

        def _marg_value(data, dims, marg_dims):
            n = len(dims)
            index = []
            for a in range(n):
                d = dims[n - 1 - a]
                if d in marg_dims:
                    index.append(slice(None))
                else:
                    index.append(int(initial[d]))
            return float(data[tuple(index)].mean())

        def _evaluate(mech_groups):
            n_groups = len(mech_groups)
            padded = list(mech_groups)
            while len(padded) < k:
                padded.append([])

            alc_assignment = []
            for j in range(v):
                best_cost = -float("inf")
                best_g = 0
                for g, mg in enumerate(padded):
                    if not mg:
                        continue
                    cost = float(np.sum(influence[[i for i in mg], j]))
                    if cost > best_cost:
                        best_cost = cost
                        best_g = g
                alc_assignment.append(best_g)

            groups = []
            for g in range(k):
                mech_set = (
                    frozenset(u_indices[i] for i in padded[g])
                    if padded[g] else frozenset()
                )
                alc_set = frozenset(
                    v_indices[j] for j in range(v)
                    if alc_assignment[j] == g
                )
                groups.append((mech_set, alc_set))
            kp = tuple(groups)

            dist = np.zeros(v, dtype=np.float32)
            for pos in range(v):
                g = alc_assignment[pos]
                mech_set = (
                    frozenset(u_indices[i] for i in padded[g])
                    if padded[g] else frozenset()
                )
                dist[pos] = _marg_value(
                    cube_data[pos], cube_dims[pos], mech_set
                )

            emd = emd_efecto(dist, intact)
            return emd, kp, dist

        best_emd = float("inf")
        best_partitions = []
        best_dist = None

        def _update(emd, kp, dist):
            nonlocal best_emd, best_partitions, best_dist
            if emd < best_emd - 1e-12:
                best_emd = emd
                best_partitions = [kp]
                best_dist = dist
            elif abs(emd - best_emd) < 1e-12:
                best_partitions.append(kp)

        if u <= k:
            emd, kp, dist = _evaluate(clusters)
            _update(emd, kp, dist)

        while len(clusters) > 1:
            best_sim = -float("inf")
            best_pair = None
            for i in range(len(clusters)):
                cent_i = np.mean(influence[clusters[i]], axis=0)
                for j in range(i + 1, len(clusters)):
                    cent_j = np.mean(influence[clusters[j]], axis=0)
                    sim = -float(np.sum(np.abs(cent_i - cent_j)))
                    if sim > best_sim:
                        best_sim = sim
                        best_pair = (i, j)

            i, j = best_pair
            clusters[i].extend(clusters[j])
            clusters.pop(j)

            n_groups = len(clusters)
            if n_groups <= k:
                emd, kp, dist = _evaluate(clusters)
                _update(emd, kp, dist)
                if best_emd == 0:
                    break

        if best_partitions:
            return (best_emd, best_partitions, best_dist)
        return None

    def _search_exact(
        self, u_indices, v_indices, total,
        cube_data, cube_dims, initial, intact, start,
    ):
        """Exact k-partition search via full unlabeled enumeration (small space)."""
        v = len(v_indices)

        def _marg_value(data, dims, marg_dims):
            n = len(dims)
            index = []
            for a in range(n):
                d = dims[n - 1 - a]
                if d in marg_dims:
                    index.append(slice(None))
                else:
                    index.append(int(initial[d]))
            return float(data[tuple(index)].mean())

        best_emd = float("inf")
        best_partitions = []
        best_dist = None

        for kp in all_k_partitions_unlabeled(u_indices, v_indices, self.k):
            dist = np.zeros(v, dtype=np.float32)
            for pos in range(v):
                future_idx = v_indices[pos]
                data = cube_data[pos]
                dims = cube_dims[pos]
                for mech_block, alc_block in kp:
                    if future_idx in alc_block:
                        marg_set = frozenset(
                            d for d in dims if d not in mech_block
                        )
                        dist[pos] = _marg_value(data, dims, marg_set)
                        break

            emd = emd_efecto(dist, intact)

            if emd < best_emd - 1e-12:
                best_emd = emd
                best_partitions = [kp]
                best_dist = dist
            elif abs(emd - best_emd) < 1e-12:
                best_partitions.append(kp)

        soluciones = []
        for kp in best_partitions:
            fmt = fmt_kparticion(kp)
            soluciones.append(
                Solution(
                    estrategia=LABEL,
                    perdida=best_emd,
                    distribucion_subsistema=self.sia_dists_marginales,
                    distribucion_particion=best_dist,
                    tiempo_ejecucion=time.time() - start,
                    particion=fmt,
                )
            )
        return soluciones

    # ── Internals: k=2 helpers ──────────────────────────────────────────

    def _identify_bipartitions(self):
        """Candidate bipartitions from geometric path costs."""
        key = (tuple(self._caminos[0][0]), tuple(self._estado_final))
        costos = self._tabla_transiciones[key]
        candidatos = []
        n_vars = len(costos)

        for idx in range(n_vars):
            presentes = list(range(len(self._estado_final)))
            futuros = [i for i in range(n_vars) if i != idx]
            candidatos.append([presentes, futuros])

        es_par = len(self._caminos) % 2 == 0
        mitad = len(self._caminos) // 2 if es_par else (
            len(self._caminos) // 2
        ) + 1

        for nivel in range(1, mitad):
            costo_min = 1e5
            mejores_presentes = []
            mejores_futuros = []
            for estado in self._caminos[nivel]:
                costo = 0
                presentes = []
                futuros = []
                actual = self._tabla_transiciones.get(
                    (tuple(self._caminos[0][0]), tuple(estado)), None
                )
                comp_estado = (1 - np.array(estado)).tolist()
                complementario = self._tabla_transiciones.get(
                    (tuple(self._caminos[0][0]), tuple(comp_estado)),
                    None,
                )
                for idx, val in enumerate(estado):
                    if val == self._caminos[0][0][idx]:
                        presentes.append(idx)
                if actual is not None and complementario is not None:
                    for idx in self._idx_ncubos:
                        if actual[idx] <= complementario[idx]:
                            futuros.append(idx)
                            costo += actual[idx]
                        else:
                            costo += complementario[idx]
                if costo < costo_min:
                    costo_min = costo
                    mejores_presentes = presentes
                    mejores_futuros = futuros
            candidatos.append([mejores_presentes, mejores_futuros])

        return candidatos

    def _build_k2_solutions(
        self, mip_list, mejor_valor, mejores_dist, start
    ):
        soluciones = []
        for presentes_idx, futuros_idx in mip_list:
            presentes_set = set(presentes_idx)
            futuros_set = set(futuros_idx)
            all_mech = set(range(len(self.sia_subsistema.dims_ncubos)))
            all_alc = set(
                range(len(self.sia_subsistema.indices_ncubos))
            )

            presentes_actual = set(
                int(x)
                for x in self.sia_subsistema.dims_ncubos[list(presentes_set)]
            )
            futuros_actual = set(
                int(x)
                for x in self.sia_subsistema.indices_ncubos[list(futuros_set)]
            )
            rest_mech_actual = set(
                int(x)
                for x in self.sia_subsistema.dims_ncubos[
                    list(all_mech - presentes_set)
                ]
            )
            rest_alc_actual = set(
                int(x)
                for x in self.sia_subsistema.indices_ncubos[
                    list(all_alc - futuros_set)
                ]
            )

            particion = (
                (frozenset(presentes_actual), frozenset(futuros_actual)),
                (frozenset(rest_mech_actual), frozenset(rest_alc_actual)),
            )
            fmt = fmt_kparticion(particion)
            soluciones.append(
                Solution(
                    estrategia=LABEL,
                    perdida=mejor_valor,
                    distribucion_subsistema=self.sia_dists_marginales,
                    distribucion_particion=mejores_dist,
                    tiempo_ejecucion=time.time() - start,
                    particion=fmt,
                )
            )
        return soluciones

    def _search_k2_selection(self, start: float):
        """Approximate bipartition via selection (fast fallback for large n)."""
        futuros = self.sia_subsistema.indices_ncubos
        presentes = self.sia_subsistema.dims_ncubos
        u_indices = list(presentes)
        v_indices = list(futuros)
        u = len(u_indices)
        v = len(v_indices)

        initial = self.sia_subsistema.estado_inicial

        cube_data = []
        cube_dims = []
        for idx in range(v):
            nc = self.sia_subsistema.ncubos[idx]
            cube_data.append(nc.data)
            cube_dims.append(list(nc.dims))

        cube_mean = [float(d.mean()) for d in cube_data]

        def _marg_value(data, dims, marg_dims):
            n = len(dims)
            index = []
            for a in range(n):
                d = dims[n - 1 - a]
                if d in marg_dims:
                    index.append(slice(None))
                else:
                    index.append(int(initial[d]))
            return float(data[tuple(index)].mean())

        intact = self.sia_dists_marginales
        best_emd = float("inf")
        best_dist = None
        best_groups = None

        for i in range(u):
            sel_mech = frozenset({u_indices[i]})
            dist = np.zeros(v, dtype=np.float32)
            for j in range(v):
                dist[j] = _marg_value(cube_data[j], cube_dims[j], sel_mech)
            emd = emd_efecto(dist, intact)
            if emd < best_emd - 1e-12:
                best_emd = emd
                best_dist = dist
                best_groups = (
                    (frozenset({u_indices[i]}), frozenset()),
                    (frozenset(u_indices) - {u_indices[i]}, frozenset(v_indices)),
                )

        for j in range(v):
            sel_alc = frozenset({v_indices[j]})
            dist = cube_mean[j]
            full = np.zeros(v, dtype=np.float32)
            for jj in range(v):
                if jj == j:
                    full[jj] = cube_mean[jj]
                else:
                    full[jj] = float(
                        cube_data[jj][
                            tuple(int(initial[d]) for d in reversed(cube_dims[jj]))
                        ]
                    )
            emd = emd_efecto(full, intact)
            if emd < best_emd - 1e-12:
                best_emd = emd
                best_dist = full
                best_groups = (
                    (frozenset(), frozenset({v_indices[j]})),
                    (frozenset(u_indices), frozenset(v_indices) - {v_indices[j]}),
                )

        if best_groups is None:
            return [
                Solution(
                    estrategia=LABEL,
                    perdida=float("inf"),
                    distribucion_subsistema=self.sia_dists_marginales,
                    distribucion_particion=self.sia_dists_marginales,
                    tiempo_ejecucion=time.time() - start,
                    particion="sin soluci\u00f3n",
                )
            ]

        fmt = fmt_kparticion(best_groups)
        return [
            Solution(
                estrategia=LABEL,
                perdida=best_emd,
                distribucion_subsistema=self.sia_dists_marginales,
                distribucion_particion=best_dist,
                tiempo_ejecucion=time.time() - start,
                particion=fmt,
            )
        ]

    # ── Internals: geometric node scoring ──────────────────────────────

    def _compute_node_scores(self):
        """Score each node by geometric transition cost importance."""
        key = (tuple(self._caminos[0][0]), tuple(self._estado_final))
        costos = self._tabla_transiciones.get(key, [])

        fut_scores = {j: costos[j] for j in range(len(costos))}

        pres_scores = {}
        for nivel in range(1, len(self._caminos)):
            for estado in self._caminos[nivel]:
                for i in range(len(estado)):
                    if estado[i] != self._caminos[0][0][i]:
                        k = (
                            tuple(self._caminos[0][0]),
                            tuple(estado),
                        )
                        if k in self._tabla_transiciones:
                            c = sum(
                                v for v in self._tabla_transiciones[k]
                                if v is not None
                            )
                            pres_scores[i] = max(
                                pres_scores.get(i, 0), c
                            )

        self._node_scores = {
            "present": pres_scores,
            "future": fut_scores,
        }

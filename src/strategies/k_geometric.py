import time
from itertools import combinations
from typing import Any, Optional

import numpy as np

from src.config import Config
from src.strategies.base import SIA
from src.models.bnb_optimizer import BnBOptimizer, Node
from src.models.system import System
from src.middlewares.slogger import SafeLogger
from src.middlewares.profile import profiler_manager, profile
from src.functions.emd import emd_efecto
from src.functions.format import fmt_kparticion
from src.solution import Solution

LABEL = "KGeometric"
TAG_STRATEGY = f"{LABEL}_strategy"
TAG_ANALYSIS = f"{LABEL}_analysis"
EFECTO = 1
ACTUAL = 0


def k_partition_distribution(system: System, k_partition) -> np.ndarray:
    num_nodes = len(system.ncubos)
    dist = np.zeros(num_nodes, dtype=np.float32)

    for future_idx in system.indices_ncubos:
        cube = system.ncubos[future_idx]
        for mech_block, alc_block in k_partition:
            if future_idx in alc_block:
                keep = set(mech_block)
                if not keep:
                    dist[future_idx] = float(np.mean(cube.data))
                else:
                    marginalize = np.array(
                        [d for d in cube.dims if d not in keep],
                        dtype=np.int8,
                    )
                    if marginalize.size:
                        mc = cube.marginalizar(marginalize)
                    else:
                        mc = cube
                    if mc.dims.size == 0:
                        dist[future_idx] = float(mc.data)
                    else:
                        inicial = tuple(
                            int(system.estado_inicial[j]) for j in mc.dims
                        )
                        dist[future_idx] = float(mc.data[inicial[::-1]])
                break
    return dist


class KGeometric(SIA):
    def __init__(
        self,
        tpm: np.ndarray,
        config: Config,
        k: int = 3,
        mode: str = "geometric",
    ):
        super().__init__(tpm, config)
        profiler_manager.start_session(
            f"K{k}{mode}{len(tpm[1])}{config.pagina_muestra}"
        )
        self.k = k
        self.mode = mode
        self.logger = SafeLogger(TAG_STRATEGY)
        self._geometric_candidates: list[tuple[frozenset, frozenset]] = []

    @profile(context={"type": TAG_ANALYSIS})
    def aplicar_estrategia(
        self,
        estado_inicial: str,
        condicion: str,
        alcance: str,
        mecanismo: str,
    ):
        self.sia_preparar_subsistema(estado_inicial, condicion, alcance, mecanismo)

        futuros = self.sia_subsistema.indices_ncubos
        presentes = self.sia_subsistema.dims_ncubos
        m, n = futuros.size, presentes.size

        if self.k > m + n:
            return [
                Solution(
                    estrategia=LABEL,
                    perdida=float("inf"),
                    distribucion_subsistema=self.sia_dists_marginales,
                    distribucion_particion=self.sia_dists_marginales,
                    tiempo_ejecucion=time.time() - self.sia_tiempo_inicio,
                    particion=f"k={self.k} > m+n={m+n} (no v\u00e1lido)",
                )
            ]

        if self.mode == "geometric":
            self._precompute_geometric_candidates(futuros, presentes)

        solver = BnBOptimizer(direction="min")
        result = solver.solve(
            initial_state=(tuple(), frozenset(presentes), frozenset(futuros)),
            branch_fn=self._branch,
            bound_fn=self._bound,
            is_complete_fn=self._is_complete,
            estimate_fn=self._estimate,
            disable_pruning=True,
        )

        if result is None:
            return [
                Solution(
                    estrategia=LABEL,
                    perdida=float("inf"),
                    distribucion_subsistema=self.sia_dists_marginales,
                    distribucion_particion=self.sia_dists_marginales,
                    tiempo_ejecucion=time.time() - self.sia_tiempo_inicio,
                    particion="sin soluci\u00f3n",
                )
            ]

        state, valor = result
        groups, _, _ = state

        dist = k_partition_distribution(self.sia_subsistema, groups)

        fmt = fmt_kparticion(groups)
        return [
            Solution(
                estrategia=LABEL,
                perdida=float(valor),
                distribucion_subsistema=self.sia_dists_marginales,
                distribucion_particion=dist,
                tiempo_ejecucion=time.time() - self.sia_tiempo_inicio,
                particion=fmt,
            )
        ]

    def _precompute_geometric_candidates(
        self,
        futuros: np.ndarray,
        presentes: np.ndarray,
    ):
        n_estado = len(self.sia_subsistema.estado_inicial)
        estado_inicial_list = self.sia_subsistema.estado_inicial.tolist()
        estado_final_list = (1 - self.sia_subsistema.estado_inicial).tolist()

        flat_data = []
        for ncubo in self.sia_subsistema.ncubos:
            flat_data.append(ncubo.data.ravel())

        caminos: dict[int, list[list[int]]] = {
            0: [estado_inicial_list]
        }
        transiciones: dict[tuple[tuple[int, ...], tuple[int, ...]], list[float]] = {}

        def _hamming(a, b):
            return sum(x != y for x, y in zip(a, b))

        def _compute_transition(estado_a, estado_b):
            key = (tuple(estado_a), tuple(estado_b))
            if key in transiciones:
                return
            distancia = _hamming(estado_a, estado_b)
            factor = 1 / (2**distancia)

            estado_a_int = int("".join(map(str, estado_a[::-1])), 2)
            estado_b_int = int("".join(map(str, estado_b[::-1])), 2)
            diffs = [
                abs(flat[estado_a_int] - flat[estado_b_int])
                for flat in flat_data
            ]

            if distancia > 1:
                for i in range(len(estado_a)):
                    if estado_a[i] != estado_b[i]:
                        intermedio = estado_b.copy()
                        intermedio[i] = estado_a[i]
                        _compute_transition(estado_a, intermedio)
                        temp_key = (tuple(estado_a), tuple(intermedio))
                        if temp_key in transiciones:
                            diffs = [
                                d + (transiciones[temp_key][j] if transiciones[temp_key][j] is not None else 0)
                                for j, d in enumerate(diffs)
                            ]
            transiciones[key] = [factor * d for d in diffs]

        for nivel in range(1, n_estado + 1):
            visitados = set()
            caminos[nivel] = []
            for estado_anterior in caminos[nivel - 1]:
                for i in range(n_estado):
                    if estado_anterior[i] != estado_final_list[i]:
                        nuevo = estado_anterior.copy()
                        nuevo[i] = estado_final_list[i]
                        t = tuple(nuevo)
                        if t not in visitados:
                            caminos[nivel].append(nuevo)
                            _compute_transition(estado_inicial_list, nuevo)
                            visitados.add(t)

        key_opt = tuple(estado_inicial_list), tuple(estado_final_list)
        costos = transiciones.get(key_opt, [0.0] * max(len(futuros), 0))

        candidatos_presentes_futuros = []

        n_vars = len(costos) if costos else len(futuros) + len(presentes)

        for idx in range(n_vars):
            cand_presentes = list(range(len(estado_inicial_list)))
            cand_futuros = [i for i in range(n_vars) if i != idx]
            candidatos_presentes_futuros.append([cand_presentes, cand_futuros])

        es_par = len(caminos) % 2 == 0
        mitad = len(caminos) // 2 if es_par else (len(caminos) // 2) + 1

        for nivel in range(1, mitad):
            mejor_costo = 1e5
            mejores_presentes = []
            mejores_futuros = []
            for estado in caminos[nivel]:
                costo = 0
                presentes_local = [
                    idx for idx, val in enumerate(estado)
                    if val == estado_inicial_list[idx]
                ]
                actual = transiciones.get(
                    (tuple(estado_inicial_list), tuple(estado)), None
                )
                comp_estado = (1 - np.array(estado)).tolist()
                complementario = transiciones.get(
                    (tuple(estado_inicial_list), tuple(comp_estado)), None
                )
                if actual is not None and complementario is not None:
                    futuros_local = []
                    for idx in range(n_vars):
                        if actual[idx] <= complementario[idx]:
                            futuros_local.append(idx)
                            costo += actual[idx]
                        else:
                            costo += complementario[idx]
                    if costo < mejor_costo:
                        mejor_costo = costo
                        mejores_presentes = presentes_local
                        mejores_futuros = futuros_local
            if mejores_presentes or mejores_futuros:
                candidatos_presentes_futuros.append(
                    [mejores_presentes, mejores_futuros]
                )

        dims_mech = presentes
        idxs_alc = futuros

        seen = set()
        for cand_presentes, cand_futuros in candidatos_presentes_futuros:
            mech_set = frozenset(dims_mech[cand_presentes]) if cand_presentes else frozenset()
            alc_set = frozenset(idxs_alc[cand_futuros]) if cand_futuros else frozenset()
            if not mech_set and not alc_set:
                continue
            key = (mech_set, alc_set)
            if key not in seen:
                seen.add(key)
                self._geometric_candidates.append(key)

    def _get_candidates(
        self,
        rem_mech: frozenset,
        rem_alc: frozenset,
    ):
        if self.mode == "exhaustive":
            return self._all_subsets(rem_mech, rem_alc)
        return [
            (m, a) for m, a in self._geometric_candidates
            if m.issubset(rem_mech) and a.issubset(rem_alc)
            and (m or a)
            and not (m == rem_mech and a == rem_alc)
        ]

    def _all_subsets(
        self,
        rem_mech: frozenset,
        rem_alc: frozenset,
    ):
        elems = [(0, m) for m in rem_mech] + [(1, a) for a in rem_alc]
        subsets = []
        for r in range(1, len(elems)):
            for combo in combinations(elems, r):
                mech = frozenset(x[1] for x in combo if x[0] == 0)
                alc = frozenset(x[1] for x in combo if x[0] == 1)
                if not mech and not alc:
                    continue
                if mech == rem_mech and alc == rem_alc:
                    continue
                subsets.append((mech, alc))
        return subsets

    def _compute_loss(self, groups, rem_mech, rem_alc):
        complete = groups + ((rem_mech, rem_alc),)
        dist = k_partition_distribution(self.sia_subsistema, complete)
        return float(emd_efecto(dist, self.sia_dists_marginales))

    def _bound(self, node: Node) -> float:
        groups, rem_mech, rem_alc = node.state
        return self._compute_loss(groups, rem_mech, rem_alc)

    def _is_complete(self, node: Node) -> bool:
        groups, rem_mech, rem_alc = node.state
        return (
            len(groups) == self.k
            and not rem_mech
            and not rem_alc
            and all(m or a for m, a in groups)
        )

    def _estimate(
        self,
        initial_state: Any,
    ) -> tuple[Any, float]:
        groups, rem_mech, rem_alc = initial_state

        for _ in range(self.k - len(groups) - 1):
            candidates = self._get_candidates(rem_mech, rem_alc)
            if not candidates:
                best = (rem_mech, rem_alc)
            else:
                valid = [
                    c for c in candidates
                    if len(rem_mech) - len(c[0]) + len(rem_alc) - len(c[1])
                    >= self.k - len(groups) - 1
                ]
                if not valid:
                    best = (rem_mech, rem_alc)
                else:
                    best = min(
                        valid,
                        key=lambda c: self._compute_loss(
                            groups + (c,), rem_mech - c[0], rem_alc - c[1]
                        ),
                    )
            groups = groups + (best,)
            rem_mech -= best[0]
            rem_alc -= best[1]

        groups = groups + ((rem_mech, rem_alc),)

        if not all(m or a for m, a in groups):
            return (initial_state, float("inf"))

        final_loss = self._compute_loss(groups, frozenset(), frozenset())
        return (groups, frozenset(), frozenset()), float(final_loss)

    def _branch(self, node: Node) -> list[Node]:
        groups, rem_mech, rem_alc = node.state
        if len(groups) >= self.k:
            return []

        if len(groups) == self.k - 1:
            if not rem_mech and not rem_alc:
                return []
            return [
                Node(
                    state=(groups + ((rem_mech, rem_alc),), frozenset(), frozenset()),
                    bound=0.0,
                )
            ]

        candidates = self._get_candidates(rem_mech, rem_alc)
        children = []
        for mech, alc in candidates:
            new_groups = groups + ((mech, alc),)
            new_mech = rem_mech - mech
            new_alc = rem_alc - alc
            children.append(
                Node(state=(new_groups, new_mech, new_alc), bound=0.0)
            )
        return children

import time
from typing import Union

import numpy as np

from src.config import Config
from src.middlewares.slogger import SafeLogger
from src.functions.emd import emd_efecto
from src.functions.labels import ABECEDARY
from src.middlewares.profile import profiler_manager, profile
from src.functions.format import fmt_biparticion_q
from src.strategies.base import SIA
from src.solution import Solution


LABEL = "Q-Nodes"
TAG_STRATEGY = f"{LABEL}_strategy"
TAG_ANALYSIS = f"{LABEL}_analysis"
EFECTO = 1
ACTUAL = 0
INFTY_POS: float = float("inf")
LAST_IDX = -1


class QNodes(SIA):
    def __init__(self, tpm: np.ndarray, config: Config):
        super().__init__(tpm, config)
        profiler_manager.start_session(
            f"NET{len(tpm[1])}{config.pagina_muestra}"
        )
        self.m: int
        self.n: int
        self.etiquetas = [tuple(s.lower() for s in ABECEDARY), ABECEDARY]
        self.vertices: set[tuple]
        self.clave_submodular = [], []
        self.memoria_delta = {}
        self.memoria_grupo_candidato = {}

        self.indices_alcance: np.ndarray
        self.indices_mecanismo: np.ndarray

        self.logger = SafeLogger(TAG_STRATEGY)

    @profile(context={"type": TAG_ANALYSIS})
    def aplicar_estrategia(
        self,
        estado_inicial: str,
        condicion: str,
        alcance: str,
        mecanismo: str,
    ):
        self.sia_preparar_subsistema(estado_inicial, condicion, alcance, mecanismo)

        futuro = tuple(
            (EFECTO, idx_efecto) for idx_efecto in self.sia_subsistema.indices_ncubos
        )

        presente = tuple(
            (ACTUAL, idx_actual) for idx_actual in self.sia_subsistema.dims_ncubos
        )

        self.m = self.sia_subsistema.indices_ncubos.size
        self.n = self.sia_subsistema.dims_ncubos.size

        self.indices_alcance = self.sia_subsistema.indices_ncubos
        self.indices_mecanismo = self.sia_subsistema.dims_ncubos

        vertices = list(presente + futuro)
        self.vertices = set(presente + futuro)
        mip = self._algorithm(vertices)

        fmt_mip = fmt_biparticion_q(list(mip), self._nodes_complement(mip))
        perdida_mip, dist_marginal_mip = self.memoria_grupo_candidato[mip]

        return Solution(
            estrategia=LABEL,
            perdida=perdida_mip,
            distribucion_subsistema=self.sia_dists_marginales,
            distribucion_particion=dist_marginal_mip,
            tiempo_ejecucion=time.time() - self.sia_tiempo_inicio,
            particion=fmt_mip,
        )

    @profile(context={"type": TAG_ANALYSIS})
    def _algorithm(self, vertices: list[tuple[int, int]]):
        for i in range(len(vertices) - 1):
            omegas_ciclo = [vertices[0]]
            deltas_ciclo = vertices[1:]

            emd_particion_candidata = INFTY_POS
            dist_particion_candidata = None

            for j in range(len(deltas_ciclo) - 1):
                emd_local = 1e5
                indice_mip = 0

                for k in range(len(deltas_ciclo)):
                    emd_union, emd_delta, dist_marginal_delta = self._funcion_submodular(
                        deltas_ciclo[k], omegas_ciclo
                    )

                    emd_iteracion = emd_union - emd_delta

                    if emd_iteracion < emd_local:
                        if emd_delta == 0:
                            clave = (
                                tuple(deltas_ciclo[k])
                                if isinstance(deltas_ciclo[k], list)
                                else (deltas_ciclo[k],)
                            )
                            self.memoria_grupo_candidato[clave] = (
                                emd_delta,
                                dist_marginal_delta,
                            )
                            return clave

                        emd_local = emd_iteracion
                        indice_mip = k
                        emd_particion_candidata = emd_delta
                        dist_particion_candidata = dist_marginal_delta

                omegas_ciclo.append(deltas_ciclo[indice_mip])
                deltas_ciclo.pop(indice_mip)

            self.memoria_grupo_candidato[
                tuple(
                    deltas_ciclo[LAST_IDX]
                    if isinstance(deltas_ciclo[LAST_IDX], list)
                    else deltas_ciclo
                )
            ] = (emd_particion_candidata, dist_particion_candidata)

            par_candidato = (
                [omegas_ciclo[LAST_IDX]]
                if isinstance(omegas_ciclo[LAST_IDX], tuple)
                else omegas_ciclo[LAST_IDX]
            ) + (
                deltas_ciclo[LAST_IDX]
                if isinstance(deltas_ciclo[LAST_IDX], list)
                else deltas_ciclo
            )

            omegas_ciclo.pop()
            omegas_ciclo.append(par_candidato)

            vertices = omegas_ciclo

        return min(
            self.memoria_grupo_candidato,
            key=lambda k: self.memoria_grupo_candidato[k][0],
        )

    def _funcion_submodular(
        self, deltas: Union[tuple, list[tuple]], omegas: list[Union[tuple, list[tuple]]]
    ):
        vector_delta_marginal = None
        self.clave_submodular = [], []

        clave_delta_actual, clave_delta_efecto = self._definir_clave(deltas)
        clave_delta = tuple(clave_delta_actual), tuple(clave_delta_efecto)

        idxs_alcance_delta = self.clave_submodular[EFECTO]
        dims_mecanismo_delta = self.clave_submodular[ACTUAL]

        if clave_delta not in self.memoria_delta:
            particion_delta = self.sia_subsistema.bipartir(
                np.array(idxs_alcance_delta, dtype=np.int8),
                np.array(dims_mecanismo_delta, dtype=np.int8),
            )
            vector_delta_marginal = particion_delta.distribucion_marginal()
            emd_delta = emd_efecto(vector_delta_marginal, self.sia_dists_marginales)
            self.memoria_delta[clave_delta] = emd_delta, vector_delta_marginal
        else:
            emd_delta, vector_delta_marginal = self.memoria_delta[clave_delta]

        for omega in omegas:
            self._definir_clave(omega)

        idxs_alcance_union = self.clave_submodular[EFECTO]
        dims_mecanismo_union = self.clave_submodular[ACTUAL]

        particion_union = self.sia_subsistema.bipartir(
            np.array(idxs_alcance_union, dtype=np.int8),
            np.array(dims_mecanismo_union, dtype=np.int8),
        )
        vector_union_marginal = particion_union.distribucion_marginal()
        emd_union = emd_efecto(vector_union_marginal, self.sia_dists_marginales)

        return emd_union, emd_delta, vector_delta_marginal

    def _definir_clave(
        self,
        conjunto: Union[tuple[int, int], list[tuple[int, int]]],
    ):
        if isinstance(conjunto, tuple):
            tiempo, indice = conjunto
            self.clave_submodular[tiempo].append(indice)
        else:
            for tiempo, indice in conjunto:
                self.clave_submodular[tiempo].append(indice)
        self.clave_submodular[ACTUAL].sort()
        self.clave_submodular[EFECTO].sort()
        return self.clave_submodular

    def _nodes_complement(self, nodes: list[tuple[int, int]]):
        return list(set(self.vertices) - set(nodes))

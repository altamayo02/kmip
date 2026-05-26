import time
from typing import List, Dict, Tuple

import numpy as np
from concurrent.futures import ThreadPoolExecutor

from src.config import Config
from src.strategies.base import SIA
from src.models.system import System
from src.middlewares.slogger import SafeLogger
from src.middlewares.profile import profiler_manager, profile
from src.functions.emd import emd_efecto
from src.functions.labels import ABECEDARY
from src.functions.format import fmt_biparticion_q
from src.solution import Solution


LABEL = "Geometric"
TAG_STRATEGY = f"{LABEL}_strategy"
TAG_ANALYSIS = f"{LABEL}_analysis"
ACTUAL = 0
EFECTO = 1


class GeometricSIA(SIA):
    def __init__(self, tpm: np.ndarray, config: Config):
        super().__init__(tpm, config)
        profiler_manager.start_session(
            f"NET{len(tpm[1])}{config.pagina_muestra}"
        )
        self.etiquetas = [tuple(s.lower() for s in ABECEDARY), ABECEDARY]
        self.logger = SafeLogger(TAG_STRATEGY)
        self.tabla_transiciones: dict = {}
        self.vertices: set[tuple]
        self.tabla: dict[int, list[tuple[int, int]]] = {}
        self.memoria_particiones: dict[tuple[int, int], tuple[float, float]] = {}

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
            (EFECTO, efecto) for efecto in self.sia_subsistema.indices_ncubos
        )
        presente = tuple(
            (ACTUAL, actual) for actual in self.sia_subsistema.dims_ncubos
        )

        self._flat_data = []
        for idx, ncubo in enumerate(self.sia_subsistema.ncubos):
            self._flat_data.append(ncubo.data.ravel())

        self.vertices = set(presente + futuro)
        dims = self.sia_subsistema.dims_ncubos
        self.estado_inicial = self.sia_subsistema.estado_inicial[dims]
        self.estado_final = 1 - self.estado_inicial
        mip = self._find_mip()

        fmt_mip = fmt_biparticion_q(list(mip), self._nodes_complement(mip))

        return Solution(
            estrategia=LABEL,
            perdida=self.memoria_particiones[mip][0],
            distribucion_subsistema=self.sia_dists_marginales,
            distribucion_particion=self.memoria_particiones[mip][1],
            tiempo_ejecucion=time.time() - self.sia_tiempo_inicio,
            particion=fmt_mip,
        )

    def _nodes_complement(self, nodes: list[tuple[int, int]]):
        return list(set(self.vertices) - set(nodes))

    def _find_mip(self):
        self.sia_logger.critic("empieza.")
        estado_inicial = self.estado_inicial
        estado_final = self.estado_final
        self.idx_ncubos = list(range(len(self.sia_subsistema.indices_ncubos)))
        self.caminos: Dict[int, List[List[int]]] = {0: [estado_inicial.tolist()]}
        self.tabla_transiciones[
            tuple(self.caminos[0][0]), tuple(self.caminos[0][0])
        ] = [0.0 for _ in range(len(self.sia_subsistema.indices_ncubos))]
        for nivel in range(1, len(estado_inicial) + 1):
            self._calcular_costos_nivel(estado_final, nivel)
        candidatos = self._identificar_particiones_optimas()
        for idx, (presentes, futuros) in enumerate(candidatos):
            presentes = self.sia_subsistema.dims_ncubos[presentes]
            futuros = self.sia_subsistema.indices_ncubos[futuros]
            dist = (
                self.sia_subsistema.bipartir(futuros, presentes).distribucion_marginal()
            )
            emd = emd_efecto(dist, self.sia_dists_marginales)
            key = [(0, nodo) for nodo in presentes]
            key.extend([(1, nodo) for nodo in futuros])
            self.memoria_particiones[tuple(key)] = (emd, dist)
        return min(
            self.memoria_particiones, key=lambda k: self.memoria_particiones[k][0]
        )

    def _calcular_costos_nivel(self, estado_final: np.ndarray, nivel):
        n = len(estado_final)
        visitados: set[tuple] = set()
        self.caminos[nivel] = []
        for estado_anterior in self.caminos[nivel - 1]:
            estado_actual = np.array(estado_anterior)
            for i in range(n):
                if estado_actual[i] != estado_final[i]:
                    nuevo_estado = estado_actual.copy()
                    nuevo_estado[i] = estado_final[i]
                    nuevo_estado_tuple = tuple(nuevo_estado)
                    if nuevo_estado_tuple not in visitados:
                        self.caminos[nivel].append(nuevo_estado.tolist())
                        self._calcular_costo(
                            self.caminos[0][0],
                            nuevo_estado.tolist(),
                            self.idx_ncubos,
                        )
                        visitados.add(nuevo_estado_tuple)

    def _calcular_costo(
        self,
        estado_inicial: list,
        estado_final: list,
        ncubos: list[int],
    ):
        key = tuple(estado_inicial), tuple(estado_final)
        if key not in self.tabla_transiciones:
            self.tabla_transiciones[key] = [None] * len(
                self.sia_subsistema.indices_ncubos
            )
        distancia_hamming = self._hamming(estado_inicial, estado_final)
        factor = 1 / (2**distancia_hamming)

        estado_ini_int = int("".join(map(str, estado_inicial[::-1])), 2)
        estado_fin_int = int("".join(map(str, estado_final[::-1])), 2)

        diffs = np.abs(
            np.array([flat[estado_ini_int] for flat in self._flat_data])
            - np.array([flat[estado_fin_int] for flat in self._flat_data])
        )
        self.tabla_transiciones[key] = diffs.tolist()

        if distancia_hamming > 1:
            for i in range(len(estado_inicial)):
                if estado_inicial[i] != estado_final[i]:
                    nuevo_estado = estado_final.copy()
                    nuevo_estado[i] = estado_inicial[i]
                    nuevo_estado_tuple = tuple(nuevo_estado)
                    temp_key = tuple(estado_inicial), nuevo_estado_tuple
                    for n in ncubos:
                        if self.tabla_transiciones[temp_key][n] is not None:
                            self.tabla_transiciones[key][n] = (
                                self.tabla_transiciones[key][n]
                                + self.tabla_transiciones[temp_key][n]
                            )
        tmp = []
        for i, n in enumerate(self.tabla_transiciones[key]):
            if n is not None:
                tmp.append(factor * n)
            else:
                tmp.append(n)
        self.tabla_transiciones[key] = tmp

    def _identificar_particiones_optimas(self):
        key = tuple(self.caminos[0][0]), tuple(self.estado_final)
        costos: list = self.tabla_transiciones[key]
        candidatos = []
        n_vars = len(costos)
        for idx in range(n_vars):
            presentes = [i for i in range(len(self.estado_final))]
            futuros = [i for i in range(n_vars) if i != idx]
            candidatos.append([presentes, futuros])
        es_par = len(self.caminos) % 2 == 0
        mitad = len(self.caminos) // 2 if es_par else (len(self.caminos) // 2) + 1
        for nivel in range(1, mitad):
            costo_candidato_nivel = 1e5
            presentes_nivel = []
            futuros_nivel = []
            for estado in self.caminos[nivel]:
                costo_candidato = 0
                presentes = []
                futuros = []
                actual = self.tabla_transiciones.get(
                    (tuple(self.caminos[0][0]), tuple(estado)), None
                )
                estado_complementario = (1 - np.array(estado)).tolist()
                complementario = self.tabla_transiciones.get(
                    (tuple(self.caminos[0][0]), tuple(estado_complementario)), None
                )
                for idx, i in enumerate(estado):
                    if i == self.caminos[0][0][idx]:
                        presentes.append(idx)
                if actual is not None and complementario is not None:
                    for idx, _ in enumerate(self.idx_ncubos):
                        if actual[idx] <= complementario[idx]:
                            futuros.append(idx)
                            costo_candidato += actual[idx]
                        else:
                            costo_candidato += complementario[idx]
                if costo_candidato < costo_candidato_nivel:
                    costo_candidato_nivel = costo_candidato
                    presentes_nivel = presentes
                    futuros_nivel = futuros
            candidatos.append([presentes_nivel, futuros_nivel])
        return candidatos

    def _hamming(self, a: List[int], b: List[int]) -> int:
        return sum(x != y for x, y in zip(a, b))

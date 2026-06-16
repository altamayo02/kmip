import time
from itertools import islice
from math import ceil

import numpy as np

from src.config import Config
from src.strategies.base import SIA
from src.models.system import System
from src.middlewares.slogger import SafeLogger
from src.middlewares.profile import profiler_manager, profile
from src.functions.emd import emd_efecto
from src.functions.format import fmt_kparticion
from src.functions.partitions import all_k_partitions_unlabeled
from src.solution import Solution


LABEL = "KBruteForce"
TAG_STRATEGY = f"{LABEL}_strategy"
TAG_ANALYSIS = f"{LABEL}_analysis"


class KBruteForce(SIA):
    def __init__(self, tpm: np.ndarray, config: Config, k: int = 3):
        super().__init__(tpm, config)
        profiler_manager.start_session(
            f"K{k}Force{len(tpm[1])}{config.pagina_muestra}"
        )
        self.k = k
        self.early_stopping = True
        self.distancia_metrica = emd_efecto
        self.logger = SafeLogger(TAG_STRATEGY)

    @profile(context={"type": TAG_ANALYSIS})
    def aplicar_estrategia(
        self, estado_inicial: str, condiciones: str, alcance: str, mecanismo: str
    ):
        self.sia_preparar_subsistema(estado_inicial, condiciones, alcance, mecanismo)

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
                    particion=f"k={self.k} > m={m} (no v\u00e1lido)",
                )
            ]

        small_phi = np.inf
        mejores: list[dict] = []
        subsistema = self.sia_subsistema

        for kp in all_k_partitions_unlabeled(
            list(presentes), list(futuros), self.k
        ):
            part_dist = _k_partition_distribution(subsistema, kp)
            emd_value = self.distancia_metrica(part_dist, self.sia_dists_marginales)

            if emd_value < small_phi:
                small_phi = emd_value
                mejores = [{"dist": part_dist, "particion": kp}]
                if emd_value == 0 and self.early_stopping:
                    break
            elif emd_value == small_phi:
                mejores.append({"dist": part_dist, "particion": kp})

        seen = set()
        soluciones = []
        for mejor in mejores:
            seen.add(mejor["particion"])
            fmt = fmt_kparticion(mejor["particion"])
            soluciones.append(
                Solution(
                    estrategia=LABEL,
                    perdida=small_phi,
                    distribucion_subsistema=self.sia_dists_marginales,
                    distribucion_particion=mejor["dist"],
                    tiempo_ejecucion=time.time() - self.sia_tiempo_inicio,
                    particion=fmt,
                )
            )

        return soluciones


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


def _k_partition_distribution(system: System, k_partition) -> np.ndarray:
    """Marginal distribution vector under a k-partition for the given system."""
    dist = np.zeros(len(system.ncubos), dtype=np.float32)
    for pos_idx, cube in enumerate(system.ncubos):
        future_idx = cube.indice
        for mech_block, alc_block in k_partition:
            if future_idx in alc_block:
                dist[pos_idx] = _marginal_from_cube(
                    cube, system.estado_inicial, set(mech_block)
                )
                break
    return dist

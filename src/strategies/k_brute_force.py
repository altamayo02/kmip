import time

import numpy as np

from src.config import Config
from src.strategies.base import SIA
from src.middlewares.slogger import SafeLogger
from src.middlewares.profile import profiler_manager, profile
from src.functions.emd import emd_efecto
from src.functions.format import fmt_kparticion
from src.solution import Solution
from src.strategies.k_partition_utils import (
    all_k_partitions,
    k_partition_distribution,
    normalize_partition,
)


LABEL = "KBruteForce"
TAG_STRATEGY = f"{LABEL}_strategy"
TAG_ANALYSIS = f"{LABEL}_analysis"


class KBruteForce(SIA):
    def __init__(self, tpm: np.ndarray, config: Config, k: int = 2):
        super().__init__(tpm, config)
        profiler_manager.start_session(
            f"KForce{len(tpm[1])}{config.pagina_muestra}_k{k}"
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

        for kp in all_k_partitions(futuros, presentes, self.k):
            part_dist = k_partition_distribution(subsistema, kp)
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
            norm = normalize_partition(mejor["particion"])
            if norm in seen:
                continue
            seen.add(norm)
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

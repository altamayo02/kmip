import time
import collections
from collections.abc import Iterable, Mapping, MutableMapping, Sequence

import numpy as np

# Python 3.10+ moved these aliases to collections.abc; pyphi still imports from collections.
if not hasattr(collections, "Iterable"):
    setattr(collections, "Iterable", Iterable)
if not hasattr(collections, "Mapping"):
    setattr(collections, "Mapping", Mapping)
if not hasattr(collections, "MutableMapping"):
    setattr(collections, "MutableMapping", MutableMapping)
if not hasattr(collections, "Sequence"):
    setattr(collections, "Sequence", Sequence)

import pyphi
from src.config import Config
from src.middlewares.slogger import SafeLogger
from src.middlewares.profile import profiler_manager, profile
from src.functions.labels import ABECEDARY
from src.functions.format import fmt_kparticion
from src.functions.emd import emd_efecto
from src.functions.partitions import (
    all_k_partitions_unlabeled,
    count_k_partitions_unlabeled,
    set_partitions,
)
from src.strategies.base import SIA
from src.solution import Solution


LABEL = "Pyphi"
TAG_STRATEGY = f"{LABEL}_strategy"
TAG_ANALYSIS = f"{LABEL}_analysis"


class Phi(SIA):
    def __init__(self, tpm: np.ndarray, config: Config, k: int = 2):
        super().__init__(tpm, config)
        profiler_manager.start_session(
            f"Phi{len(tpm[1])}{config.pagina_muestra}"
        )
        self.k = k
        self.logger = SafeLogger(TAG_STRATEGY)

    @profile(context={"type": TAG_ANALYSIS})
    def aplicar_estrategia(
        self,
        estado_inicial: str,
        condiciones: str,
        alcance: str,
        mecanismo: str,
    ):
        self.sia_preparar_subsistema(estado_inicial, condiciones, alcance, mecanismo)

        futuros = self.sia_subsistema.indices_ncubos
        presentes = self.sia_subsistema.dims_ncubos
        m, n = futuros.size, presentes.size
        if self.k > m + n:
            return [Solution(
                estrategia=LABEL,
                perdida=float("inf"),
                distribucion_subsistema=self.sia_dists_marginales,
                distribucion_particion=self.sia_dists_marginales,
                tiempo_ejecucion=time.time() - self.sia_tiempo_inicio,
                particion=f"k={self.k} > m+n={m+n} (no v\u00e1lido)",
            )]

        if self.k == 2 or self.k == 3:
            return self._run_pyphi_mip(estado_inicial, alcance, mecanismo)
        return self._run_enumerate_emd()

    def _run_pyphi_mip(self, estado_inicial, alcance, mecanismo):
        if self.k == 2:
            pyphi.config.PARTITION_TYPE = 'BI'
        elif self.k == 3:
            pyphi.config.PARTITION_TYPE = 'TRI'

        n_nodes = len(self.tpm[1])
        node_labels = pyphi.labels.NodeLabels(
            tuple(ABECEDARY[:n_nodes]),
            tuple(range(n_nodes)),
        )

        network = pyphi.Network(self.tpm, node_labels=node_labels)

        subsystem = pyphi.Subsystem(
            network=network,
            state=np.array([int(b) for b in estado_inicial]),
            nodes=range(n_nodes),
        )

        efecto_mip = subsystem.effect_mip(
            mechanism=tuple(
                i for i, b in enumerate(mecanismo) if b == "1"
            ),
            purview=tuple(
                i for i, b in enumerate(alcance) if b == "1"
            ),
        )

        perdida = float(efecto_mip.phi)
        particion = self._format_mip(efecto_mip)

        return [Solution(
            estrategia=LABEL,
            perdida=perdida,
            distribucion_subsistema=np.array([0.0]),
            distribucion_particion=np.array([0.0]),
            tiempo_ejecucion=time.time() - self.sia_tiempo_inicio,
            particion=particion,
            quiere_hablar=True,
        )]

    def _run_enumerate_emd(self):
        start = time.time()
        sistemas = self.sia_subsistema
        n_nodes = len(sistemas.ncubos)
        intacto = self.sia_dists_marginales

        total = sum(1 for _ in set_partitions(list(range(n_nodes)), self.k))
        if total > 500_000:
            return [Solution(
                estrategia=LABEL,
                perdida=float("inf"),
                distribucion_subsistema=intacto,
                distribucion_particion=intacto,
                tiempo_ejecucion=time.time() - start,
                particion=f"k={self.k}: demasiadas particiones ({total})",
            )]

        mejor_perdida = float("inf")
        mejores_particiones = []
        mejor_dist = None

        for blocks in set_partitions(list(range(n_nodes)), self.k):
            kp = tuple((frozenset(block), frozenset(block)) for block in blocks)
            dist = _k_partition_distribution_ncubos(sistemas, kp)
            emd = emd_efecto(dist, intacto)

            if emd < mejor_perdida - 1e-12:
                mejor_perdida = emd
                mejores_particiones = [kp]
                mejor_dist = dist
            elif abs(emd - mejor_perdida) < 1e-12:
                mejores_particiones.append(kp)

        soluciones = []
        for kp in mejores_particiones:
            soluciones.append(Solution(
                estrategia=LABEL,
                perdida=mejor_perdida,
                distribucion_subsistema=intacto,
                distribucion_particion=mejor_dist,
                tiempo_ejecucion=time.time() - start,
                particion=fmt_kparticion(kp),
            ))
        return soluciones

    def _format_mip(self, mip):
        try:
            k_partition = tuple(
                (frozenset(part.mechanism), frozenset(part.purview))
                for part in mip.partition.parts
            )
            return fmt_kparticion(k_partition)
        except Exception:
            return "PyPhi partition"


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


def _k_partition_distribution_ncubos(system, k_partition) -> np.ndarray:
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

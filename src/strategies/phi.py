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

from pyphi import Network, Subsystem
from pyphi.labels import NodeLabels
from src.config import Config
from src.middlewares.slogger import SafeLogger
from src.middlewares.profile import profiler_manager, profile
from src.functions.labels import ABECEDARY
from src.functions.format import fmt_kparticion
from src.strategies.base import SIA
from src.solution import Solution


LABEL = "Pyphi"
TAG_STRATEGY = f"{LABEL}_strategy"
TAG_ANALYSIS = f"{LABEL}_analysis"


class Phi(SIA):
    def __init__(self, tpm: np.ndarray, config: Config):
        super().__init__(tpm, config)
        profiler_manager.start_session(
            f"Phi{len(tpm[1])}{config.pagina_muestra}"
        )
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

        n_nodes = len(self.tpm[1])
        node_labels = NodeLabels(
            tuple(ABECEDARY[:n_nodes]),
            tuple(range(n_nodes)),
        )

        network = Network(self.tpm, node_labels=node_labels)

        subsystem = Subsystem(
            network=network,
            state=np.array([int(b) for b in estado_inicial]),
            nodes=range(n_nodes),
        )

        efecto_mip = subsystem.effect_mip(
            mechanism=tuple(
                i
                for i, b in enumerate(mecanismo)
                if b == "1"
            ),
            purview=tuple(
                i
                for i, b in enumerate(alcance)
                if b == "1"
            ),
        )

        perdida = float(efecto_mip.phi)
        particion = self._format_mip(efecto_mip)

        distribucion_subsistema = np.array([0.0])
        distribucion_particion = np.array([0.0])

        return [Solution(
            estrategia=LABEL,
            perdida=perdida,
            distribucion_subsistema=distribucion_subsistema,
            distribucion_particion=distribucion_particion,
            tiempo_ejecucion=time.time() - self.sia_tiempo_inicio,
            particion=particion,
            quiere_hablar=True,
        )]

    def _format_mip(self, mip):
        try:
            k_partition = tuple(
                (frozenset(part.mechanism), frozenset(part.purview))
                for part in mip.partition.parts
            )
            return fmt_kparticion(k_partition)
        except Exception:
            return "PyPhi partition"

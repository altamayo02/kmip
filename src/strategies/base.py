import time
from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as NDArray

from src.config import Config
from src.middlewares.slogger import SafeLogger
from src.models.system import System


class SIA(ABC):
    TAG_PREPARACION = "sia_preparation"

    def __init__(self, tpm: np.ndarray, config: Config) -> None:
        self.tpm = tpm
        self.config = config
        self.sia_logger = SafeLogger(self.TAG_PREPARACION)

        self.sia_subsistema: System
        self.sia_dists_marginales: NDArray[np.float32]
        self.sia_tiempo_inicio: float = 0.0

    @abstractmethod
    def aplicar_estrategia(self):
        pass

    def sia_preparar_subsistema(
        self,
        estado_inicial: str,
        condicion: str,
        alcance: str,
        mecanismo: str,
    ):
        if self._chequear_parametros(estado_inicial, condicion, alcance, mecanismo):
            raise Exception(
                "El estado inicial tiene una dimensión diferente "
                "con las condiciones, alcance o mecanismo."
            )

        dims_condicionadas = np.array(
            [ind for ind, bit in enumerate(condicion) if bit == "0"], dtype=np.int8
        )
        dims_alcance = np.array(
            [ind for ind, bit in enumerate(alcance) if bit == "0"], dtype=np.int8
        )
        dims_mecanismo = np.array(
            [ind for ind, bit in enumerate(mecanismo) if bit == "0"], dtype=np.int8
        )
        dims_estado_inicial = np.array(
            [int(ind) for ind in estado_inicial],
            dtype=np.int8,
        )

        completo = System(self.tpm, dims_estado_inicial)

        candidato = completo.condicionar(dims_condicionadas)
        self.sia_logger.critic("Sistema Candidato creado.")

        subsistema = candidato.substraer(dims_alcance, dims_mecanismo)
        self.sia_logger.critic("Subsistema creado.")

        self.sia_subsistema = subsistema
        self.sia_dists_marginales = subsistema.distribucion_marginal()
        self.sia_tiempo_inicio = time.time()

    def _chequear_parametros(
        self, estado_inicial: str, candidato: str, futuro: str, presente: str
    ):
        return not (
            len(self.tpm[1])
            == len(estado_inicial)
            == len(candidato)
            == len(futuro)
            == len(presente)
        )

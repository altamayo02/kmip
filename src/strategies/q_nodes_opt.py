import numpy as np

from src.config import Config
from src.strategies.q_nodes import QNodes

DTYPE_INT = np.int8
DTYPE_FLT = np.float32


class QNodes_Opt(QNodes):
    def __init__(self, tpm: np.ndarray, config: Config):
        super().__init__(tpm, config)
        self.LABEL = "QNodes_Opt"
        self.TAG_STRATEGY = f"{self.LABEL}_strategy"
        self.TAG_ANALYSIS = f"{self.LABEL}_analysis"

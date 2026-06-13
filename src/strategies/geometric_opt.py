import numpy as np

from src.config import Config
from src.strategies.geometric import GeometricSIA

DTYPE_INT = np.int8
DTYPE_FLT = np.float32


class GeometricSIA_Opt(GeometricSIA):
    def __init__(self, tpm: np.ndarray, config: Config):
        super().__init__(tpm, config)
        self.LABEL = "Geometric_Opt"
        self.TAG_STRATEGY = f"{self.LABEL}_strategy"
        self.TAG_ANALYSIS = f"{self.LABEL}_analysis"

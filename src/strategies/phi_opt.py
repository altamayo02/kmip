import numpy as np

from src.config import Config
from src.strategies.phi import Phi

DTYPE_INT = np.int8


class Phi_Opt(Phi):
    def __init__(self, tpm: np.ndarray, config: Config):
        super().__init__(tpm, config)
        self.LABEL = "Pyphi_Opt"
        self.TAG_STRATEGY = f"{self.LABEL}_strategy"
        self.TAG_ANALYSIS = f"{self.LABEL}_analysis"

import numpy as np

from src.config import Config
from src.strategies.k_brute_force import KBruteForce

DTYPE_INT = np.int8
DTYPE_FLT = np.float32


class KBruteForce_Opt(KBruteForce):
    def __init__(self, tpm: np.ndarray, config: Config, k: int = 2):
        super().__init__(tpm, config, k)
        self.LABEL = "KBruteForce_Opt"
        self.TAG_STRATEGY = f"{self.LABEL}_strategy"
        self.TAG_ANALYSIS = f"{self.LABEL}_analysis"

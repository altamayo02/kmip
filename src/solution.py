from dataclasses import dataclass

import numpy as np


@dataclass
class Solution:
    estrategia: str
    perdida: float
    distribucion_subsistema: np.ndarray
    distribucion_particion: np.ndarray
    particion: str
    tiempo_ejecucion: float = 0.0
    quiere_hablar: bool = True

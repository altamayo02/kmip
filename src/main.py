from typing import Optional
from src.config import Config
from src.loader import TpmLoader
from src.strategies.brute_force import BruteForce
from src.strategies.q_nodes import QNodes
from src.strategies.geometric import GeometricSIA
from src.strategies.phi import Phi
from strategies.k_brute_force import KBruteForce
from src.presentation import mostrar_solucion
from src.strategies.base import SIA


def iniciar(
    estado_inicial: str = "100",
    condiciones: str = "111",
    alcance: str = "111",
    mecanismo: str = "111",
    estrategia: str = "QNodes",
    pagina: str = "A",
    config: Optional[Config] = None,
		k: int = 2
):
    if config is None:
        config = Config(pagina_muestra=pagina)

    gestor_sistema = TpmLoader.cargar(len(estado_inicial), pagina)

    estrategias: dict[str, type[SIA]] = {
        "BruteForce": BruteForce,
        "QNodes": QNodes,
        "Phi": Phi,
        "Geometric": GeometricSIA,
        "KBruteForce": KBruteForce,
    }

    if estrategia not in estrategias:
        raise ValueError(
            f"Estrategia '{estrategia}' no reconocida. "
            f"Opciones: {list(estrategias.keys())}"
        )
    
    if estrategias[estrategia] is KBruteForce:
      analizador = estrategias[estrategia](gestor_sistema, config, k)
    else:
      analizador = estrategias[estrategia](gestor_sistema, config)
    soluciones = analizador.aplicar_estrategia(
        estado_inicial,
        condiciones,
        alcance,
        mecanismo,
    )
    mostrar_solucion(soluciones, config)
    return soluciones

from typing import Optional
from src.config import Config
from src.loader import TpmLoader
from src.strategies.brute_force import BruteForce
from src.strategies.q_nodes import QNodes
from src.strategies.geometric import GeometricSIA
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
):
    if config is None:
        config = Config(pagina_muestra=pagina)

    gestor_sistema = TpmLoader.cargar(len(estado_inicial), pagina)

    estrategias: dict[str, type[SIA]] = {
        "BruteForce": BruteForce,
        "QNodes": QNodes,
        "Geometric": GeometricSIA,
    }

    if estrategia not in estrategias:
        raise ValueError(
            f"Estrategia '{estrategia}' no reconocida. "
            f"Opciones: {list(estrategias.keys())}"
        )

    analizador = estrategias[estrategia](gestor_sistema, config)
    soluciones = analizador.aplicar_estrategia(
        estado_inicial,
        condiciones,
        alcance,
        mecanismo,
    )
    mostrar_solucion(soluciones, config)
    return soluciones

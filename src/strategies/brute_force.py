import time
from itertools import product, chain, combinations, islice
from typing import Generator, Tuple, Union

import pandas as pd
import numpy as np
from numpy.typing import NDArray

from src.config import Config
from src.strategies.base import SIA
from src.models.system import System
from src.solution import Solution
from src.middlewares.slogger import SafeLogger
from src.middlewares.profile import profile, profiler_manager
from src.functions.emd import emd_efecto
from src.functions.labels import literales
from src.functions.format import fmt_biparticion_fuerza_bruta


LABEL = "BruteForce"
TAG_STRATEGY = f"{LABEL}_strategy"
TAG_ANALYSIS = f"{LABEL}_analysis"
TAG_FULL_ANALYSIS = f"{LABEL}_full_analysis"


class BruteForce(SIA):
    def __init__(self, tpm: np.ndarray, config: Config):
        super().__init__(tpm, config)
        profiler_manager.start_session(
            f"Force{len(tpm[1])}{config.pagina_muestra}"
        )
        self.early_stopping = True
        self.distancia_metrica = emd_efecto
        self.logger = SafeLogger(TAG_STRATEGY)

    @profile(context={"type": TAG_ANALYSIS})
    def aplicar_estrategia(
        self, estado_inicial: str, condiciones: str, alcance: str, mecanismo: str
    ):
        self.sia_preparar_subsistema(estado_inicial, condiciones, alcance, mecanismo)

        futuros = self.sia_subsistema.indices_ncubos
        presentes = self.sia_subsistema.dims_ncubos
        m, n = futuros.size, presentes.size

        small_phi = np.inf
        mejores: list[dict] = []

        for subalcance, submecanismo in _biparticiones(
            futuros, presentes, (1 << m) * (1 << n)
        ):
            subsistema = self.sia_subsistema
            arr_alcance = np.array(subalcance, dtype=np.int8)
            arr_mecanismo = np.array(submecanismo, dtype=np.int8)

            particion = subsistema.bipartir(arr_alcance, arr_mecanismo)

            part_marg_dist = particion.distribucion_marginal()
            emd_value = self.distancia_metrica(
                part_marg_dist, self.sia_dists_marginales
            )

            if emd_value < small_phi:
                small_phi = emd_value
                mejores = [
                    {
                        "dist": part_marg_dist,
                        "prim": (submecanismo, subalcance),
                        "dual": (
                            set(presentes) - set(submecanismo),
                            set(futuros) - set(subalcance),
                        ),
                    }
                ]
                if emd_value == 0.0 and self.early_stopping:
                    break
            elif emd_value == small_phi:
                mejores.append(
                    {
                        "dist": part_marg_dist,
                        "prim": (submecanismo, subalcance),
                        "dual": (
                            set(presentes) - set(submecanismo),
                            set(futuros) - set(subalcance),
                        ),
                    }
                )

        soluciones = []
        for mejor in mejores:
            fmt = fmt_biparticion_fuerza_bruta(
                [mejor["prim"][0], mejor["prim"][1]],
                [mejor["dual"][0], mejor["dual"][1]],
            )
            soluciones.append(
                Solution(
                    estrategia=LABEL,
                    perdida=small_phi,
                    distribucion_subsistema=self.sia_dists_marginales,
                    distribucion_particion=mejor["dist"],
                    tiempo_ejecucion=time.time() - self.sia_tiempo_inicio,
                    particion=fmt,
                )
            )

        return soluciones

    @profile(context={"type": TAG_FULL_ANALYSIS})
    def analizar_completamente_una_red(self) -> None:
        import os
        output_dir = f"review/resolver/N{len(self.tpm[1])}{self.config.pagina_muestra}"
        os.makedirs(output_dir, exist_ok=True)

        initial_state = self.sia_subsistema.estado_inicial
        system = System(self.tpm, initial_state)
        self._analizar_candidatos(system)

    def _analizar_candidatos(self, sistema: System) -> None:
        cantidad = len(self.tpm[1])
        dim_candidatas = _generar_candidatos(cantidad)

        for dimensiones in dim_candidatas:
            self._procesar_candidato(sistema, np.array(dimensiones, dtype=np.int8))

    def _procesar_candidato(
        self, completo: System, condiciones: NDArray[np.int8]
    ) -> None:
        candidato = completo.condicionar(condiciones)
        nombre = literales(np.setdiff1d(candidato.dims_ncubos, condiciones))
        self._procesar_subsistema(candidato, nombre)

    def _procesar_subsistema(
        self, mecanismo_removido: System, nombre_candidato: str
    ) -> None:
        import os
        output_dir = f"review/resolver/N{len(self.tpm[1])}{self.config.pagina_muestra}"
        os.makedirs(output_dir, exist_ok=True)
        results_file = f"{output_dir}/{nombre_candidato}.xlsx"

        with pd.ExcelWriter(results_file) as writer:
            for alcance_removido, sub_present in _generar_subsistemas(
                mecanismo_removido.dims_ncubos
            ):
                if not self._deberia_omitir_subsistema(
                    alcance_removido, mecanismo_removido
                ):
                    self._analizar_subsistema(
                        mecanismo_removido,
                        np.array(alcance_removido, dtype=np.int8),
                        np.array(sub_present, dtype=np.int8),
                        writer,
                    )

    def _deberia_omitir_subsistema(
        self, alcance_removido: tuple, candidate: System
    ) -> bool:
        return len(alcance_removido) == candidate.indices_ncubos.size

    def _analizar_subsistema(
        self,
        candidato: System,
        alcance_removido: NDArray[np.int8],
        mecanismo_removido: NDArray[np.int8],
        writer: pd.ExcelWriter,
    ) -> None:
        subsistema = candidato.substraer(alcance_removido, mecanismo_removido)
        dist_marginal = subsistema.distribucion_marginal()

        nombre_subsistema = self._get_nombre_subsistema(
            candidato, alcance_removido, mecanismo_removido
        )
        resultado = self._analizar_particiones(dist_marginal, subsistema)
        resultado.to_excel(writer, sheet_name=nombre_subsistema)

    def _analizar_particiones(
        self, distribucion: NDArray[np.float32], subsistema: System
    ) -> pd.DataFrame:
        m, n = subsistema.indices_ncubos.size, subsistema.dims_ncubos.size

        llave_presente = [f"{number:0{n}b}" for number in range(1 << n)]
        llave_futuro = [f"{number:0{m}b}" for number in range(1 << m - 1)]

        resultados = pd.DataFrame(
            columns=llave_futuro,
            index=llave_presente,
            dtype=np.float32,
        )

        for alcance, mecanismo in _generar_particiones(m, n):
            sub_alcance = np.array([i for i, bit in enumerate(alcance) if bit])
            sub_mecanismo = np.array([i for i, bit in enumerate(mecanismo) if bit])

            particion = subsistema.bipartir(
                np.array(sub_alcance, dtype=np.int8),
                np.array(sub_mecanismo, dtype=np.int8),
            )

            dist_parte_marginal = particion.distribucion_marginal()
            emd_value = self.distancia_metrica(dist_parte_marginal, distribucion)

            etiqueta_mecanismo = "".join(map(str, mecanismo.astype(int)))
            etiqueta_alcance = "".join(map(str, alcance.astype(int)))

            resultados.loc[etiqueta_mecanismo, etiqueta_alcance] = emd_value

        return resultados

    def _get_nombre_subsistema(
        self,
        candidato: System,
        sub_alcance: NDArray[np.int8],
        sub_mecanismo: NDArray[np.int8],
    ) -> str:
        futuro_removido = np.setdiff1d(candidato.dims_ncubos, sub_alcance)
        presente_removido = np.setdiff1d(candidato.dims_ncubos, sub_mecanismo)
        return f"{literales(futuro_removido)}|{literales(presente_removido)}"


def _subconjuntos(arr: np.ndarray):
    return chain.from_iterable(combinations(arr, r) for r in range(len(arr) + 1))


def _biparticiones(
    alcances: np.ndarray,
    mecanismos: np.ndarray,
    total=None,
):
    if total is None:
        total = (1 << alcances.size) * (1 << mecanismos.size)
    return islice(
        product(_subconjuntos(alcances), _subconjuntos(mecanismos)), 1, total - 1
    )


def _generar_candidatos(n_vars: int):
    return (combo for r in range(n_vars) for combo in combinations(range(n_vars), r))


def _generar_subsistemas(vars: tuple):
    tiempos = [combo for r in range(len(vars) + 1) for combo in combinations(vars, r)]
    return product(tiempos, tiempos)


def _generar_particiones(
    m: int,
    n: int,
) -> Union[Generator[Tuple[np.ndarray, np.ndarray], None, None], np.ndarray]:
    if m < 1:
        raise ValueError(f"Alcance trivial: Future no debe tener {m} elementos")

    m_combinations = 1 << (m - 1)
    n_combinations = 1 << n

    m_bits = np.empty((m_combinations, m), dtype=np.uint8)
    n_bits = np.empty((n_combinations, n), dtype=np.uint8)

    m_indices = np.arange(m_combinations, dtype=np.uint32)[:, np.newaxis]
    n_indices = np.arange(n_combinations, dtype=np.uint32)[:, np.newaxis]

    m_shifts = np.arange(m - 1, -1, -1, dtype=np.uint8)
    n_shifts = np.arange(n - 1, -1, -1, dtype=np.uint8)

    m_bits = (m_indices >> m_shifts) & 1
    n_bits = (n_indices >> n_shifts) & 1

    def partition_generator():
        m_row = m_bits[0]
        for j in range(1, n_combinations):
            yield m_row, n_bits[j]
        for i in range(1, m_combinations):
            m_row = m_bits[i]
            for j in range(n_combinations):
                yield m_row, n_bits[j]

    return partition_generator()

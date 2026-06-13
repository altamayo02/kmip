import time
import os
import copy
from itertools import islice, product, chain, combinations
from math import ceil
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from src.config import Config
from src.strategies.brute_force import BruteForce, _subconjuntos, _biparticiones
from src.models.system import System
from src.models.ncube import NCube
from src.solution import Solution
from src.functions.emd import emd_efecto
from src.functions.format import fmt_biparticion_fuerza_bruta

DTYPE_INT = np.int8
DTYPE_FLT = np.float32

MIN_PARALLEL = 8192


def _serializar(subsistema):
    """Serializa subsistema para enviar a procesos workers."""
    cubes = []
    for nc in subsistema.ncubos:
        cubes.append((int(nc.indice), nc.dims.copy(), nc.data.copy()))
    return cubes, subsistema.estado_inicial.copy()


def _worker_proc(args):
    """Worker: evalua particiones y devuelve solo los mejores locales."""
    cubes_serial, estado_inicial, dist_marginal, chunk = args

    ncubos = tuple(
        NCube(indice=idx, dims=dims, data=data)
        for idx, dims, data in cubes_serial
    )
    system = System.__new__(System)
    system.estado_inicial = estado_inicial
    system.ncubos = ncubos
    system.memo = {}

    best_emd = float("inf")
    best_results = []

    for al, me in chunk:
        arr_a = np.array(al, dtype=DTYPE_INT)
        arr_m = np.array(me, dtype=DTYPE_INT)
        part = system.bipartir(arr_a, arr_m)
        pmd = part.distribucion_marginal()
        emd = float(np.sum(np.abs(pmd - dist_marginal)))

        if emd < best_emd:
            best_emd = emd
            best_results = [(emd, pmd.copy(), list(al), list(me))]
            if emd == 0:
                break
        elif emd == best_emd:
            best_results.append((emd, pmd.copy(), list(al), list(me)))

    return best_results


class BruteForce_Opt(BruteForce):
    def __init__(self, tpm: np.ndarray, config: Config):
        super().__init__(tpm, config)
        self.LABEL = "BruteForce_Opt"
        self.TAG_STRATEGY = f"{self.LABEL}_strategy"
        self.TAG_ANALYSIS = f"{self.LABEL}_analysis"
        self.TAG_FULL_ANALYSIS = f"{self.LABEL}_full_analysis"
        self._n_workers = max(1, (os.cpu_count() or 4))

    def aplicar_estrategia(
        self, estado_inicial: str, condiciones: str, alcance: str, mecanismo: str
    ):
        self.sia_preparar_subsistema(estado_inicial, condiciones, alcance, mecanismo)

        futuros = self.sia_subsistema.indices_ncubos
        presentes = self.sia_subsistema.dims_ncubos
        m, n = futuros.size, presentes.size

        espacio = list(_biparticiones(futuros, presentes, (1 << m) * (1 << n)))
        n_total = len(espacio)

        if n_total < MIN_PARALLEL or self._n_workers <= 1:
            return super().aplicar_estrategia(
                estado_inicial, condiciones, alcance, mecanismo
            )

        return self._procesar_paralelo(espacio, futuros, presentes)

    def _procesar_paralelo(self, espacio, futuros, presentes):
        chunk_size = max(1, len(espacio) // self._n_workers)
        chunks = [espacio[i:i + chunk_size] for i in range(0, len(espacio), chunk_size)]

        cubes_ser, estado_arr = _serializar(self.sia_subsistema)
        dist_marginal = self.sia_dists_marginales

        futuros_set = set(map(int, futuros))
        presentes_set = set(map(int, presentes))

        small_phi = float("inf")
        mejores = []

        args_list = [(cubes_ser, estado_arr, dist_marginal, ch) for ch in chunks]

        with ProcessPoolExecutor(max_workers=self._n_workers) as pool:
            futures = {pool.submit(_worker_proc, a): a for a in args_list}

            for f in as_completed(futures):
                try:
                    batch = f.result()
                except Exception:
                    continue

                for emd, p_dist, al, me in batch:
                    if emd < small_phi:
                        small_phi = emd
                        mejores = [{
                            "dist": p_dist,
                            "prim": (me, al),
                            "dual": (
                                presentes_set - set(me),
                                futuros_set - set(al),
                            ),
                        }]
                        if emd == 0:
                            break
                    elif emd == small_phi:
                        mejores.append({
                            "dist": p_dist,
                            "prim": (me, al),
                            "dual": (
                                presentes_set - set(me),
                                futuros_set - set(al),
                            ),
                        })

                if small_phi == 0:
                    break

        soluciones = []
        for mejor in mejores:
            fmt = fmt_biparticion_fuerza_bruta(
                [mejor["prim"][0], mejor["prim"][1]],
                [mejor["dual"][0], mejor["dual"][1]],
            )
            soluciones.append(
                Solution(
                    estrategia=self.LABEL,
                    perdida=small_phi,
                    distribucion_subsistema=self.sia_dists_marginales,
                    distribucion_particion=mejor["dist"],
                    tiempo_ejecucion=time.time() - self.sia_tiempo_inicio,
                    particion=fmt,
                )
            )

        return soluciones

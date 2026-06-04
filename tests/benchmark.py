import csv
import os
import sys
import time
import multiprocessing as mp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import Config
from src.loader import TpmLoader
from src.strategies.brute_force import BruteForce
from src.strategies.geometric import GeometricSIA
from src.strategies.q_nodes import QNodes


TIMEOUT_SEC = 120
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
RESULTS_CSV = os.path.join(RESULTS_DIR, "benchmark_resultados.csv")

ESTRATEGIAS = {
    "BruteForce": BruteForce,
    "Geometric": GeometricSIA,
    "QNodes": QNodes,
}

def _pagina_unica(N):
    ruta = os.path.join(
        os.path.dirname(__file__), "..", "data", "samples", f"N{N}A.csv"
    )
    return "A" if os.path.isfile(ruta) else None


def _deshabilitar_profiler():
    import src.middlewares.profile as pm

    pm.profiler_manager.enabled = False


def _ejecutar_estrategia(cola, N, pagina, nombre_estrategia):
    try:
        _deshabilitar_profiler()

        tpm = TpmLoader.cargar(N, pagina)
        config = Config(pagina_muestra=pagina)
        Cls = ESTRATEGIAS[nombre_estrategia]
        estado_inicial = "1" + "0" * (N - 1)
        condiciones = "1" * N
        alcance = "1" * N
        mecanismo = "1" * N

        analizador = Cls(tpm, config)
        inicio = time.perf_counter()
        resultado = analizador.aplicar_estrategia(
            estado_inicial, condiciones, alcance, mecanismo
        )
        elapsed = time.perf_counter() - inicio
        solucion = resultado[0] if isinstance(resultado, list) else resultado
        cola.put(("OK", elapsed, float(solucion.perdida)))
    except Exception as e:
        import traceback

        tb = traceback.format_exc()
        cola.put(("ERROR", 0.0, f"{type(e).__name__}: {e}\n{tb}"))


def _medir(N, pagina, nombre_estrategia):
    cola = mp.Queue()
    proceso = mp.Process(
        target=_ejecutar_estrategia, args=(cola, N, pagina, nombre_estrategia)
    )
    proceso.start()
    proceso.join(timeout=TIMEOUT_SEC)

    if proceso.is_alive():
        proceso.terminate()
        proceso.join()
        return "TIMEOUT", TIMEOUT_SEC, 0.0

    try:
        estado, valor, extra = cola.get_nowait()
    except Exception:
        return "ERROR", 0.0, 0.0

    return estado, valor, extra


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    todas_las_filas = []

    for N in [2, 3, 4, 5, 6, 8]:
        pagina = _pagina_unica(N)
        if not pagina:
            continue
        print(f"\n=== N = {N} ===")

        for nombre_estrat in ["BruteForce", "Geometric", "QNodes"]:
            estado, t, perdida = _medir(N, pagina, nombre_estrat)
            m = N
            n = N
            todas_las_filas.append(
                [N, nombre_estrat, m, n, pagina, f"{t:.6f}", f"{perdida}", estado]
            )
            if estado == "OK":
                print(
                    f"  {nombre_estrat:12s} | {t:9.4f}s | phi={perdida:.4f}"
                )
            else:
                print(
                    f"  {nombre_estrat:12s} | ERROR: {extra[:120]}"
                )

    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "N",
                "estrategia",
                "m",
                "n",
                "pagina",
                "tiempo_seg",
                "perdida",
                "estado",
            ]
        )
        writer.writerows(todas_las_filas)

    print(f"\nResultados guardados en: {RESULTS_CSV}")


if __name__ == "__main__":
    mp.freeze_support()
    main()

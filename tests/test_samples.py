import csv
import os
import sys
import time
import multiprocessing as mp
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import Config
from src.loader import TpmLoader
from src.strategies.geometric import GeometricSIA
from src.strategies.q_nodes import QNodes


TIMEOUT_SEC = 300
SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "samples")
RESULTS_CSV = os.path.join(os.path.dirname(__file__), "resultados_samples.csv")

ESTRATEGIAS = {
    "Geometric": GeometricSIA,
    "QNodes": QNodes,
}


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
        cola.put(("OK", elapsed, float(solucion.perdida), solucion.particion))
    except Exception as e:
        import traceback

        tb = traceback.format_exc()
        cola.put(("ERROR", 0.0, 0.0, f"{type(e).__name__}: {e}\n{tb}"))


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
        return "TIMEOUT", TIMEOUT_SEC, 0.0, ""

    try:
        estado, valor, extra, particion = cola.get_nowait()
    except Exception:
        return "ERROR", 0.0, 0.0, ""

    return estado, valor, extra, particion


def _normalizar_particion(s):
    import re
    lines = s.strip().split("\n")
    if len(lines) < 2:
        return s

    top_parts = re.findall(r"⎛(.*?)⎞", lines[0])
    bottom_parts = re.findall(r"⎝(.*?)⎠", lines[1])

    sides = []
    for mech_str, purv_str in zip(top_parts, bottom_parts):
        mech = frozenset()
        purv = frozenset()
        mech_str = mech_str.strip()
        purv_str = purv_str.strip()
        if mech_str and mech_str != "∅":
            mech = frozenset(sorted(n.strip() for n in mech_str.split(",") if n.strip()))
        if purv_str and purv_str != "∅":
            purv = frozenset(sorted(n.strip() for n in purv_str.split(",") if n.strip()))
        sides.append((mech, purv))

    return frozenset(sides)


def _obtener_samples():
    samples = []
    for fname in sorted(os.listdir(SAMPLES_DIR)):
        if not fname.endswith(".csv"):
            continue
        if not fname.startswith("N"):
            continue
        base = fname[:-4]
        N_str = base[1:].rstrip("ABC")
        variante = base[len(N_str) + 1:]
        N = int(N_str)
        if N > 15:
            continue
        samples.append((N, variante, fname))
    return samples


def main():
    samples = _obtener_samples()

    if not samples:
        print("No se encontraron samples en data/samples/")
        return

    print(f"Se procesaran {len(samples)} samples (N <= 15)")

    todas_las_filas = []

    for N, variante, fname in samples:
        print(f"\n=== N{N}{variante} ===")
        resultados_estrategias = {}

        for nombre_estrat in ["Geometric", "QNodes"]:
            estado, t, perdida, particion = _medir(N, variante, nombre_estrat)
            resultados_estrategias[nombre_estrat] = {
                "estado": estado,
                "tiempo": t,
                "perdida": perdida,
                "particion": particion,
            }

            if estado == "OK":
                print(f"  {nombre_estrat:10s} | {t:9.4f}s | perdida={perdida:.6f} | particion={particion}")
            else:
                extra_preview = str(particion)[:120] if particion else ""
                print(f"  {nombre_estrat:10s} | {estado}: {extra_preview}")

        geo = resultados_estrategias["Geometric"]
        qn = resultados_estrategias["QNodes"]

        particiones_diferentes = "No"
        phi_diferente = "No"

        if geo["estado"] == "OK" and qn["estado"] == "OK":
            if _normalizar_particion(geo["particion"]) != _normalizar_particion(qn["particion"]):
                particiones_diferentes = "Si"
            if abs(geo["perdida"] - qn["perdida"]) > 1e-12:
                phi_diferente = "Si"

        todas_las_filas.append(
            [
                N,
                variante,
                f"{geo['tiempo']:.6f}",
                f"{geo['perdida']}",
                geo["particion"],
                geo["estado"],
                f"{qn['tiempo']:.6f}",
                f"{qn['perdida']}",
                qn["particion"],
                qn["estado"],
                particiones_diferentes,
                phi_diferente,
            ]
        )

    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "N",
                "variante",
                "geometric_tiempo_seg",
                "geometric_perdida",
                "geometric_particion",
                "geometric_estado",
                "qnodes_tiempo_seg",
                "qnodes_perdida",
                "qnodes_particion",
                "qnodes_estado",
                "particiones_diferentes",
                "phi_diferente",
            ]
        )
        writer.writerows(todas_las_filas)

    print(f"\nResultados guardados en: {RESULTS_CSV}")


if __name__ == "__main__":
    mp.freeze_support()
    main()

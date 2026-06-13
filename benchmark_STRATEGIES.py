"""
Benchmark para comparar estrategias originales vs optimizadas.
Carga TPMs existentes de data/samples/ y prueba escalando N.
Maneja errores (pyphi estados no reachables) sin crashear.
"""

import time
import sys
import os
from typing import Callable

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.middlewares.profile import profiler_manager
profiler_manager.enabled = False

from src.config import Config
from src.loader import TpmLoader
from src.strategies.brute_force import BruteForce
from src.strategies.brute_force_opt import BruteForce_Opt
from src.strategies.geometric import GeometricSIA
from src.strategies.geometric_opt import GeometricSIA_Opt
from src.strategies.k_brute_force import KBruteForce
from src.strategies.k_brute_force_opt import KBruteForce_Opt
from src.strategies.q_nodes import QNodes
from src.strategies.q_nodes_opt import QNodes_Opt
from src.strategies.phi import Phi
from src.strategies.phi_opt import Phi_Opt


WARMUP = 1
ITERATIONS = 3
TIMEOUT = 300


def medir_estrategia(
    nombre: str,
    clase: type,
    tpm: np.ndarray,
    config: Config,
    estado: str,
    condicion: str,
    alcance: str,
    mecanismo: str,
    init_kwargs: dict = None,
    iteraciones: int = ITERATIONS,
) -> dict:
    base = {
        "nombre": nombre,
        "media": float("nan"),
        "std": float("nan"),
        "min": float("nan"),
        "max": float("nan"),
        "resultados": None,
        "error": None,
    }
    kwargs = init_kwargs or {}
    try:
        inst = clase(tpm, config, **kwargs)
        _ = inst.aplicar_estrategia(estado, condicion, alcance, mecanismo)
        inst = clase(tpm, config, **kwargs)

        tiempos = []
        resultados = None
        for i in range(iteraciones):
            t0 = time.perf_counter()
            soluciones = inst.aplicar_estrategia(estado, condicion, alcance, mecanismo)
            t1 = time.perf_counter()
            tiempos.append(t1 - t0)
            if resultados is None:
                resultados = soluciones

        tiempos_arr = np.array(tiempos)
        base.update({
            "tiempos": tiempos,
            "media": float(np.mean(tiempos_arr)),
            "std": float(np.std(tiempos_arr)),
            "min": float(np.min(tiempos_arr)),
            "max": float(np.max(tiempos_arr)),
            "resultados": resultados,
        })
    except Exception as e:
        base["error"] = str(e)[:120]
    return base


def validar_resultados(ref, test, nombre_ref, nombre_test):
    if not ref or not test:
        return False
    if len(ref) != len(test):
        return False
    for r, t in zip(ref, test):
        if abs(r.perdida - t.perdida) > 1e-6:
            return False
    return True


# --- Acumulador global de resultados ---
_RESUMEN = []


def ejecutar_benchmark(
    n_nodos: int,
    pagina: str,
    estrategias: list,
    solo_estado: str = None,
):
    global _RESUMEN
    try:
        tpm = TpmLoader.cargar(n_nodos, pagina)
    except FileNotFoundError:
        print(f"  [SKIP] TPM N{n_nodos}{pagina}.csv no encontrada")
        return

    config = Config(pagina_muestra=f"{n_nodos}{pagina}", profiler_habilitado=False)
    condicion = "1" * n_nodos
    alcance = "1" * n_nodos
    mecanismo = "1" * n_nodos
    estado = solo_estado or ("1" + "0" * (n_nodos - 1))

    print(f"  Estado: {estado}")

    for entry in estrategias:
        if len(entry) == 3:
            nombre_orig, clase_orig, clase_opt = entry
            init_kwargs = {}
        else:
            nombre_orig, clase_orig, clase_opt, init_kwargs = entry

        if not clase_opt:
            continue

        if "k=" in nombre_orig:
            k_val = init_kwargs.get("k", 2)
            n_parts = k_val ** (n_nodos * 2) // 2
            if n_parts > 1000000:
                print(f"  [SKIP] {nombre_orig}: {n_parts} particiones (max 1M)")
                _RESUMEN.append({
                    "N": n_nodos, "pagina": pagina,
                    "estrategia": nombre_orig + " [SKIP]",
                    "orig_ms": None, "opt_ms": None, "speedup": None,
                    "orig_error": f"{n_parts} particiones", "opt_error": None,
                })
                continue

        r_orig = medir_estrategia(
            nombre_orig, clase_orig, tpm, config,
            estado, condicion, alcance, mecanismo,
            init_kwargs=init_kwargs,
        )
        r_opt = medir_estrategia(
            clase_opt.__name__, clase_opt, tpm, config,
            estado, condicion, alcance, mecanismo,
            init_kwargs=init_kwargs,
        )

        label_orig = r_orig["nombre"]
        label_opt = r_opt["nombre"]

        if r_orig["error"]:
            print(f"  {label_orig}: ERROR - {r_orig['error']}")
        else:
            _m(r_orig)

        if r_opt["error"]:
            print(f"  {label_opt}: ERROR - {r_opt['error']}")
        else:
            _m(r_opt)

        ok = False
        ratio = float("nan")
        if not r_orig["error"] and not r_opt["error"]:
            ok = validar_resultados(
                r_orig["resultados"], r_opt["resultados"],
                nombre_orig, clase_opt.__name__,
            )
            ratio = r_orig["media"] / r_opt["media"] if r_opt["media"] > 0 else float("inf")
            status = f"OK {ratio:.2f}x" if ok else "FALLO"
            print(f"    >> {status}")

        _RESUMEN.append({
            "N": n_nodos,
            "pagina": pagina,
            "estrategia": nombre_orig,
            "orig_ms": r_orig["media"] * 1000 if not r_orig["error"] else None,
            "opt_ms": r_opt["media"] * 1000 if not r_opt["error"] else None,
            "speedup": ratio if ok else None,
            "orig_error": r_orig["error"],
            "opt_error": r_opt["error"],
        })


def _m(r):
    ms = r["media"] * 1000
    s = r["std"] * 1000
    phi = r["resultados"][0].perdida if r["resultados"] else None
    phi_str = f" phi={phi:.6f}" if phi is not None else ""
    print(f"  {r['nombre']}: {ms:.1f}ms ±{s:.1f}ms{phi_str}")


def mostrar_resumen():
    global _RESUMEN
    if not _RESUMEN:
        return
    print("\n" + "=" * 70)
    print("RESUMEN DE SPEEDUP (orig / opt)")
    print("=" * 70)
    print(f"{'N':>3} {'Estrategia':<20} {'Orig (ms)':>10} {'Opt (ms)':>10} {'Speedup':>8}")
    print("-" * 70)
    for r in _RESUMEN:
        n = f"N{r['N']}{r['pagina']}"
        orig = f"{r['orig_ms']:.1f}" if r['orig_ms'] is not None else "ERROR"
        opt = f"{r['opt_ms']:.1f}" if r['opt_ms'] is not None else "ERROR"
        sp = f"{r['speedup']:.2f}x" if r['speedup'] else "-"
        print(f"{n:>3} {r['estrategia']:<20} {orig:>10} {opt:>10} {sp:>8}")


def main():
    global _RESUMEN
    _RESUMEN = []

    print("=" * 70)
    print("BENCHMARK DE ESTRATEGIAS - Escalando N")
    print("=" * 70)
    print(f"Warmup: {WARMUP}, Iteraciones: {ITERATIONS}")

    disponibles = sorted(os.listdir("data/samples"))
    print(f"TPMs: {[f.replace('.csv','') for f in disponibles]}")

    estrategias = [
        ("BruteForce", BruteForce, BruteForce_Opt),
        ("Geometric", GeometricSIA, GeometricSIA_Opt),
    ]
    for k_val in range(2, 6):
        estrategias.append(
            (f"KBruteForce(k={k_val})", KBruteForce, KBruteForce_Opt, {"k": k_val})
        )
    estrategias += [
        ("QNodes", QNodes, QNodes_Opt),
        ("PyPhi", Phi, Phi_Opt),
    ]

    casos = [
        (3, "A"),
        (4, "A"),
        (5, "A"),
        (6, "A"),
        (8, "A"),
    ]

    for n, pag in casos:
        print(f"\n--- N={n}{pag} ---")
        ejecutar_benchmark(n, pag, estrategias)

    mostrar_resumen()


if __name__ == "__main__":
    main()

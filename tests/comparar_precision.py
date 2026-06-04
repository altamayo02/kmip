import csv
import os
import sys
import time
import multiprocessing as mp

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import Config
from src.loader import TpmLoader
from src.strategies.geometric import GeometricSIA


TIMEOUT_SEC = 600  # 10 min for N=18
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
RESULTS_CSV = os.path.join(RESULTS_DIR, "comparacion_precision_N16-18.csv")


def _deshabilitar_profiler():
    import src.middlewares.profile as pm
    pm.profiler_manager.enabled = False


def _parchear_distribucion_marginal(dtype):
    import src.models.system as sys_mod

    def distribucion_marginal(self):
        d = np.empty(self.indices_ncubos.size, dtype=dtype)
        for i, nc in enumerate(self.ncubos):
            p = nc.data
            if nc.dims.size:
                p = nc.data[tuple(self.estado_inicial[j] for j in nc.dims)[::-1]]
            d[i] = p
        return d

    sys_mod.System.distribucion_marginal = distribucion_marginal


def _encontrar_estado_inicial(tpm_f64):
    tpm_f16 = tpm_f64.astype(np.float16)
    diff = np.abs(tpm_f64 - tpm_f16.astype(np.float64)).sum(axis=1)
    peor_fila = int(diff.argmax())
    bits = f"{peor_fila:0{tpm_f64.shape[1]}b}"
    return bits[::-1]


def _ejecutar(cola, N, pagina, dtype_name, estado_inicial):
    try:
        _deshabilitar_profiler()
        dtype = {"float64": np.float64, "float32": np.float32, "float16": np.float16}[
            dtype_name
        ]

        tpm = TpmLoader.cargar(N, pagina).astype(dtype)
        _parchear_distribucion_marginal(dtype)

        config = Config(pagina_muestra=pagina)
        solver = GeometricSIA(tpm, config)
        condiciones = "1" * N
        alcance = "1" * N
        mecanismo = "1" * N

        inicio = time.perf_counter()
        sol = solver.aplicar_estrategia(
            estado_inicial, condiciones, alcance, mecanismo
        )
        elapsed = time.perf_counter() - inicio

        cola.put(
            (
                "OK",
                elapsed,
                float(sol.perdida),
                sol.distribucion_subsistema,
                sol.distribucion_particion,
            )
        )
    except Exception as e:
        import traceback
        cola.put(
            (
                "ERROR",
                0.0,
                0.0,
                None,
                None,
                f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
            )
        )


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    configs = [
        (15, "B"),
        (16, "A"),
        (17, "A"),
        (18, "A"),
    ]

    todas_filas = []

    for N, pagina in configs:
        print(f"\n======= N = {N}{pagina} =======")

        tpm_f64 = TpmLoader.cargar(N, pagina)
        estado_inicial = _encontrar_estado_inicial(tpm_f64)
        print(f"  estado_inicial = {estado_inicial}")

        resultados = {}

        for dtype_name in ["float64", "float32", "float16"]:
            print(f"  --- {dtype_name} ... ", end="", flush=True)

            cola = mp.Queue()
            proc = mp.Process(
                target=_ejecutar, args=(cola, N, pagina, dtype_name, estado_inicial)
            )
            proc.start()
            proc.join(timeout=TIMEOUT_SEC)

            if proc.is_alive():
                proc.terminate()
                proc.join()
                print("TIMEOUT")
                resultados[dtype_name] = {
                    "phi": None, "tiempo": TIMEOUT_SEC,
                    "dist_sub": None, "dist_part": None, "estado": "TIMEOUT",
                }
                continue

            try:
                estado, elapsed, phi, dist_sub, dist_part = cola.get_nowait()
            except Exception:
                print("ERROR (vacio)")
                continue

            if estado == "OK":
                print(f"phi={phi:.6f}  t={elapsed:.4f}s")
                resultados[dtype_name] = {
                    "phi": phi, "tiempo": elapsed,
                    "dist_sub": dist_sub, "dist_part": dist_part, "estado": "OK",
                }
            else:
                print(f"ERROR: {str(dist_part)[:120]}")
                resultados[dtype_name] = {
                    "phi": None, "tiempo": 0.0,
                    "dist_sub": None, "dist_part": None, "estado": "ERROR",
                }

        # Print summary for this N
        ref = "float64"
        if resultados[ref]["phi"] is not None:
            for dt in ["float64", "float32", "float16"]:
                r = resultados[dt]
                if r["phi"] is None:
                    continue
                diff_abs = r["phi"] - resultados[ref]["phi"] if dt != ref else 0
                diff_rel = (diff_abs / resultados[ref]["phi"] * 100
                            if resultados[ref]["phi"] != 0 else 0)
                print(f"    {dt:8s} | phi={r['phi']:.10f} | {r['tiempo']:.4f}s"
                      f" | d={diff_abs:+.2e} ({diff_rel:+.4f}%)")

        # Marginal comparison
        if all(resultados[dt]["dist_sub"] is not None
               for dt in ["float64", "float32", "float16"]):
            ref_dist = resultados["float64"]["dist_sub"]
            for dt in ["float32", "float16"]:
                d = resultados[dt]["dist_sub"]
                max_a = float(np.max(np.abs(d.astype(np.float64) - ref_dist.astype(np.float64))))
                print(f"    marg {dt:8s} | max|diff|={max_a:.2e}")

        # Write to overall CSV
        for dt in ["float64", "float32", "float16"]:
            r = resultados[dt]
            if r["phi"] is None:
                todas_filas.append([N, pagina, dt, "", r["tiempo"], "", "", r["estado"]])
                continue
            da = dr = ""
            if dt != ref and resultados[ref]["phi"] is not None:
                da = r["phi"] - resultados[ref]["phi"]
                dr = da / resultados[ref]["phi"] * 100 if resultados[ref]["phi"] != 0 else 0
            todas_filas.append([
                N, pagina, dt, f"{r['phi']:.10f}", f"{r['tiempo']:.4f}",
                f"{da:+.2e}" if da != "" else "",
                f"{dr:+.4f}" if dr != "" else "",
                r["estado"],
            ])

    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "N", "pagina", "dtype", "phi", "tiempo_seg",
            "diff_abs_vs_float64", "diff_rel_pct_vs_float64", "estado",
        ])
        writer.writerows(todas_filas)

    print(f"\nResultados guardados en: {RESULTS_CSV}")


if __name__ == "__main__":
    mp.freeze_support()
    main()

import csv
import os
import sys
import time
import multiprocessing as mp

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import Config
from src.loader import TpmLoader
from src.strategies.brute_force import BruteForce
from src.strategies.geometric import GeometricSIA


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
RESULTS_CSV = os.path.join(RESULTS_DIR, "analisis_precision_completo.csv")


def _noop_profile():
    import src.middlewares.profile as pm

    pm.profiler_manager.enabled = False


# ─── A. TPM ERROR DISTRIBUTION ────────────────────────────────────


def analisis_tpm(N, pagina):
    f64 = TpmLoader.cargar(N, pagina)
    filas = []
    for label, arr in [("float32", f64.astype(np.float32)), ("float16", f64.astype(np.float16))]:
        diff = np.abs(f64 - arr.astype(np.float64))
        flat = diff.ravel()
        f = f64.ravel()
        rel = flat / np.maximum(f, 1e-30)
        exactos = np.count_nonzero(diff == 0)
        total = diff.size
        filas.append([
            "TPM", N, pagina, label,
            f"{diff.max():.2e}", f"{diff.mean():.2e}",
            f"{np.median(flat):.2e}", f"{np.percentile(flat, 99):.2e}",
            f"{np.percentile(flat, 99.9):.2e}", f"{rel.max():.4f}",
            f"{exactos/total*100:.1f}", f"{exactos}/{total}",
            "",
        ])
        print(f"    {label:8s}: max|d|={diff.max():.2e}  mean|d|={diff.mean():.2e}  "
              f"p99={np.percentile(flat,99):.2e}  max|d|/|v|={rel.max():.4f}  "
              f"exactos={exactos/total*100:.1f}%")
    return filas


# ─── B. GAP ANALYSIS (full enumeration) ───────────────────────────


def _ejecutar_gaps(cola, N, pagina, dtype_name, estado_inicial):
    try:
        _noop_profile()
        dtype = {"float64": np.float64, "float32": np.float32}[dtype_name]
        tpm = TpmLoader.cargar(N, pagina).astype(dtype)
        config = Config(pagina_muestra=pagina)

        from src.strategies.base import SIA
        class Gapper(SIA):
            def aplicar_estrategia(self):
                pass

        g = Gapper(tpm, config)
        g.sia_preparar_subsistema(estado_inicial, "1"*N, "1"*N, "1"*N)

        from src.strategies.brute_force import _biparticiones
        from src.functions.emd import emd_efecto

        phis = []
        futuros = g.sia_subsistema.indices_ncubos
        presentes = g.sia_subsistema.dims_ncubos
        m, n = futuros.size, presentes.size

        for suba, subm in _biparticiones(futuros, presentes, (1 << m) * (1 << n)):
            aa = np.array(suba, dtype=np.int8) if suba else np.array([], dtype=np.int8)
            am = np.array(subm, dtype=np.int8) if subm else np.array([], dtype=np.int8)
            p = g.sia_subsistema.bipartir(aa, am)
            d = p.distribucion_marginal()
            phis.append(emd_efecto(d, g.sia_dists_marginales))

        phis.sort()
        gaps = [phis[i+1] - phis[i] for i in range(len(phis)-1)]
        cola.put(("OK", phis, gaps))
    except Exception as e:
        import traceback
        cola.put(("ERROR", [], [], f"{type(e).__name__}: {e}"))


def analisis_gaps(N, pagina, estado_inicial):
    n_parts = (1 << (2*N)) - 2
    print(f"  Enumerando {n_parts} particiones...", flush=True)
    filas = []
    for dt in ["float64", "float32"]:
        print(f"    {dt}...", end=" ", flush=True)
        cola = mp.Queue()
        proc = mp.Process(
            target=_ejecutar_gaps, args=(cola, N, pagina, dt, estado_inicial)
        )
        proc.start()
        timeout = 600
        proc.join(timeout=timeout)
        if proc.is_alive():
            proc.terminate()
            proc.join()
            print("TIMEOUT")
            continue
        try:
            est, phis, gaps = cola.get_nowait()
        except Exception:
            print("ERROR")
            continue
        if est != "OK" or not gaps:
            print("ERROR")
            continue
        mg = min(gaps)
        print(f"{len(phis)} phis | gap_min={mg:.2e}  mean={np.mean(gaps):.2e}  "
              f"median={np.median(gaps):.2e}")
        filas.append([
            "GAPS", N, pagina, dt,
            f"{mg:.2e}", f"{np.mean(gaps):.2e}",
            f"{np.median(gaps):.2e}", f"{max(gaps):.2e}",
            "", "", "", f"{len(phis)}",
            "",
        ])
    return filas


# ─── C. MIP RANKING COMPARISON ─────────────────────────────────────


def _ejecutar_mip(cola, N, pagina, dtype_name, estado_inicial):
    try:
        _noop_profile()
        dtype = {"float64": np.float64, "float32": np.float32}[dtype_name]
        tpm = TpmLoader.cargar(N, pagina).astype(dtype)
        config = Config(pagina_muestra=pagina)
        bf = BruteForce(tpm, config)
        sol = bf.aplicar_estrategia(estado_inicial, "1"*N, "1"*N, "1"*N)
        cola.put(("OK", float(sol.perdida), sol.particion))
    except Exception as e:
        import traceback
        cola.put(("ERROR", 0.0, f"{type(e).__name__}: {e}"))


def analisis_mip(N, pagina, estado_inicial):
    print(f"  Comparando MIP float64 vs float32...", end=" ", flush=True)
    r = {}
    for dt in ["float64", "float32"]:
        cola = mp.Queue()
        proc = mp.Process(
            target=_ejecutar_mip, args=(cola, N, pagina, dt, estado_inicial)
        )
        proc.start()
        proc.join(timeout=120)
        if proc.is_alive():
            proc.terminate()
            proc.join()
            r[dt] = None
            continue
        try:
            est, phi, part = cola.get_nowait()
        except Exception:
            r[dt] = None
            continue
        r[dt] = (phi, part) if est == "OK" else None

    if r["float64"] and r["float32"]:
        diff = abs(r["float64"][0] - r["float32"][0])
        match = "SI" if r["float64"][1] == r["float32"][1] else "NO"
        print(f"phi64={r['float64'][0]:.6f}  phi32={r['float32'][0]:.6f}  "
              f"diff={diff:.2e}  misma_MIP={match}")
        return [[
            "MIP", N, pagina, "float32_vs_float64",
            f"{diff:.2e}", "", "", "", "", "", match, "", ""
        ]]
    print("ERROR")
    return []


# ─── D. GEOMETRIC: ALL PHIS (via memoria_particiones) ──────────────


def _ejecutar_geo_phis(cola, N, pagina, dtype_name, estado_inicial):
    try:
        _noop_profile()
        dtype = {"float64": np.float64, "float32": np.float32, "float16": np.float16}[dtype_name]
        tpm = TpmLoader.cargar(N, pagina).astype(dtype)
        config = Config(pagina_muestra=pagina)

        import src.models.system as sys_mod

        def dm(self):
            d = np.empty(self.indices_ncubos.size, dtype=dtype)
            for i, nc in enumerate(self.ncubos):
                p = nc.data
                if nc.dims.size:
                    p = nc.data[tuple(self.estado_inicial[j] for j in nc.dims)[::-1]]
                d[i] = p
            return d
        sys_mod.System.distribucion_marginal = dm

        geo = GeometricSIA(tpm, config)
        geo.aplicar_estrategia(estado_inicial, "1"*N, "1"*N, "1"*N)

        phis = [v[0] for v in geo.memoria_particiones.values()]
        cola.put(("OK", phis))
    except Exception as e:
        import traceback
        cola.put(("ERROR", []))


# ─── MAIN ──────────────────────────────────────────────────────────


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    todas = []

    # A. TPM errors
    print("=" * 60)
    print("A. ERROR EN TPM")
    print("=" * 60)
    for N, p in [(15, "B"), (16, "A"), (17, "A"), (18, "A")]:
        print(f"  N={N}{p}:")
        todas.extend(analisis_tpm(N, p))

    # B. Gaps via full enumeration (small N)
    print("\n" + "=" * 60)
    print("B. GAPS ENTRE PHIS (todas las particiones)")
    print("=" * 60)
    for N, p in [(4, "A"), (5, "A")]:
        print(f"  N={N}{p}:")
        est = "1" + "0" * (N - 1)
        todas.extend(analisis_gaps(N, p, est))

    # C. MIP comparison (BruteForce, small N)
    print("\n" + "=" * 60)
    print("C. MIP float64 vs float32 (BruteForce)")
    print("=" * 60)
    for N, p in [(4, "A"), (5, "A"), (6, "A")]:
        print(f"  N={N}{p}:")
        est = "1" + "0" * (N - 1)
        todas.extend(analisis_mip(N, p, est))

    # D. Geometric: phi candidates for N15-18
    print("\n" + "=" * 60)
    print("D. PHIS DE CANDIDATOS Geometric (memoria_particiones)")
    print("=" * 60)
    for N, p in [(15, "B"), (16, "A")]:
        print(f"  N={N}{p}:")
        tpm64 = TpmLoader.cargar(N, p)
        bits = f"{np.argmax(np.abs(tpm64 - tpm64.astype(np.float16).astype(np.float64)).sum(axis=1)):0{N}b}"
        est = bits[::-1]
        print(f"    estado={est}")

        for dt in ["float64", "float32", "float16"]:
            print(f"    {dt}...", end=" ", flush=True)
            cola = mp.Queue()
            proc = mp.Process(
                target=_ejecutar_geo_phis, args=(cola, N, p, dt, est)
            )
            proc.start()
            proc.join(timeout=120)
            if proc.is_alive():
                proc.terminate()
                proc.join()
                print("TIMEOUT")
                continue
            try:
                est2, phis = cola.get_nowait()
            except Exception:
                print("ERROR")
                continue
            if est2 != "OK" or not phis:
                print("ERROR")
                continue
            phis.sort()
            gaps = [phis[i+1] - phis[i] for i in range(len(phis)-1)]
            print(f"{len(phis)} candidatos | gap_min={min(gaps):.2e}  "
                  f"mean={np.mean(gaps):.2e}  phi_min={min(phis):.6f}")
            todas.append([
                "GEO_CAND", N, p, dt,
                f"{min(gaps):.2e}", f"{np.mean(gaps):.2e}",
                f"{np.median(gaps):.2e}", f"{max(gaps):.2e}",
                "", "", "", f"{len(phis)}",
                f"phi_min={min(phis):.6f}",
            ])

    # Save CSV
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "tipo", "N", "pagina", "precision",
            "max_abs_diff", "mean_abs_diff", "median_abs_diff",
            "p99_abs", "p999_abs", "max_rel_diff",
            "pct_exactos", "detalle", "nota",
        ])
        w.writerows(todas)

    print(f"\nResultados: {RESULTS_CSV}")


if __name__ == "__main__":
    mp.freeze_support()
    main()

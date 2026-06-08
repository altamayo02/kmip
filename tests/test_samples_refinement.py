"""
Test refinement conjecture against real TPM samples from data/samples/.

For each sample (N=3..6), exhaustively enumerate k-partitions for k=2,3,4
and check whether optimal k-partitions always refine optimal (k-1)-partitions.

Usage:
    uv run python tests/test_samples_refinement.py
"""

import sys
import os
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.loader import TpmLoader
from src.models.system import System
from src.functions.emd import emd_efecto
from tests.test_kpartition_refinement import (
    all_k_partitions, k_partition_distribution, refines, count_k_partitions
)


def optimal_k_partitions(system, k):
    """Return (best_emd, list_of_best_partitions, total_evaluated) for a real TPM."""
    m = len(system.indices_ncubos)
    n = len(system.dims_ncubos)
    intact_dist = system.distribucion_marginal()
    best_emd = float("inf")
    best_parts = []
    total = 0
    for kp in all_k_partitions(m, n, k):
        total += 1
        dist = k_partition_distribution(system, kp)
        emd = emd_efecto(dist, intact_dist)
        if emd < best_emd - 1e-12:
            best_emd = emd
            best_parts = [kp]
        elif abs(emd - best_emd) < 1e-12:
            best_parts.append(kp)
    return best_emd, best_parts, total


def _influence_matrix(tpm):
    n = tpm.shape[1]
    num_states = 1 << n
    infl = np.zeros((n, n), dtype=np.float64)
    for f in range(n):
        for p in range(n):
            total = 0.0
            cnt = 0
            for s in range(num_states):
                bit = (s >> (n - 1 - p)) & 1
                s0 = s & ~(1 << (n - 1 - p))
                s1 = s | (1 << (n - 1 - p))
                v0 = float(tpm[min(s0, num_states - 1), f]) if s0 < num_states else float(tpm[s, f])
                v1 = float(tpm[min(s1, num_states - 1), f]) if s1 < num_states else float(tpm[s, f])
                total += abs(v1 - v0)
                cnt += 1
            infl[p, f] = total / cnt if cnt else 0.0
    return infl


def _fmt_block(mech, alc, labels):
    m = ",".join(labels[i] for i in sorted(mech))
    a = ",".join(labels[i] for i in sorted(alc))
    return f"M={{{m}}}->A={{{a}}}"


def display_violation(name, tpm, system, k_low, k_high, low_parts, high_parts, low_emd, high_emd):
    """Display a clear estado-nodo representation of the violation."""
    n = tpm.shape[1]
    labels = [chr(65 + i) for i in range(n)]
    infl = _influence_matrix(tpm)

    print()
    print("=" * 74)
    print(f"  VIOLACION: {name}")
    print(f"  k={k_low} EMD={low_emd:.6f}  |  k={k_high} EMD={high_emd:.6f}")
    print("=" * 74)

    # ── TPM ──
    print()
    print("  TPM (estado presente -> P(Y_f = 1)):")
    print()
    header = "  estado  " + "  ".join(f"{l}_f" for l in labels)
    print(f"  {header}")
    print(f"  " + "-" * len(header))
    for s in range(1 << n):
        bits = [(s >> (n - 1 - i)) & 1 for i in range(n)]
        estado = "".join(str(b) for b in bits)
        vals = "  ".join(f"{tpm[s, c]:.4f}" for c in range(n))
        marker = " <-- inicial" if all(b == 1 for b in bits) else ""
        print(f"  {estado}     {vals}{marker}")
    print()

    # ── Influence matrix ──
    print("  Influencia causal promedio |ΔP|:")
    header2 = "      " + "  ".join(f"{l:>6s}" for l in labels)
    print(f"  {header2}")
    for p in range(n):
        vals = "  ".join(f"{infl[p, f]:.4f}" for f in range(n))
        print(f"  {labels[p]}_p: {vals}")
    print()

    # ── Optimal partitions ──
    for k, parts, emd in [(k_low, low_parts, low_emd), (k_high, high_parts, high_emd)]:
        print(f"  k={k} (EMD={emd:.6f})  |  {len(parts)} optima(s):")
        for pi, part in enumerate(parts[:3]):
            blocks_str = "  ".join(_fmt_block(m, a, labels) for m, a in part)
            print(f"    [{pi}] {blocks_str}")
        print()

    # ── Crossing ──
    hp = high_parts[0]
    low_block_map = {}
    if low_parts:
        lp = low_parts[0]
        for bj, (m, a) in enumerate(lp):
            for x in m:
                low_block_map[("m", x)] = bj
            for x in a:
                low_block_map[("a", x)] = bj

    print("  Cruce en detalle:")
    print()
    for bi, (mech, alc) in enumerate(hp):
        if not mech or not alc:
            continue
        m0 = next(iter(mech))
        a0 = next(iter(alc))
        bm = low_block_map.get(("m", m0))
        ba = low_block_map.get(("a", a0))
        if bm is not None and ba is not None and bm != ba:
            mstr = ",".join(labels[m] for m in sorted(mech))
            astr = ",".join(labels[a] for a in sorted(alc))
            print(f"    Bloque B{bi}: M={{{mstr}}} -> A={{{astr}}}")
            print(f"      mecanismo {labels[m0]} esta en B{bm} de k={k_low}")
            print(f"      alcance   {labels[a0]} esta en B{ba} de k={k_low}")
            print(f"      -> CRUCE: mecanismo y alcance del mismo bloque")
            print(f"         de k={k_high} estan en BLOQUES DISTINTOS de k={k_low}")

            # Show the marginalization difference
            print()
            print(f"    Consecuencia en la marginalizacion de {labels[a0]}_f:")
            # Under k_low
            if ba is not None:
                low_mech = [m for m, _ in low_parts[0]][ba]
                print(f"    k={k_low}: {labels[a0]}_f marginaliza sobre padres FUERA de M={{{','.join(labels[m] for m in sorted(low_mech))}}}")
            # Under k_high
            high_mech = mech
            print(f"    k={k_high}: {labels[a0]}_f marginaliza sobre padres FUERA de M={{{','.join(labels[m] for m in sorted(high_mech))}}}")
            print()

    # ── Marginalization comparison ──
    print("  Comparacion de marginales para el nodo que cruza:")
    print()
    # Find a crossing node
    for bi, (mech, alc) in enumerate(hp):
        if not mech or not alc:
            continue
        m0 = next(iter(mech))
        a0 = next(iter(alc))
        bm = low_block_map.get(("m", m0))
        ba = low_block_map.get(("a", a0))
        if bm is not None and ba is not None and bm != ba:
            future_idx = a0
            # Under k_low: get the mechanism for the block containing a0
            for l_mech, l_alc in low_parts[0]:
                if a0 in l_alc:
                    low_mech_set = set(l_mech)
                    break
            # Under k_high: get the mechanism for the block containing a0
            high_mech_set = set(mech)
            
            # Print the TPM column
            print(f"    Columna {labels[future_idx]}_f en la TPM:")
            print(f"    estado  P({labels[future_idx]}_f=1 | estado)")
            for s in range(1 << n):
                bits = [(s >> (n - 1 - i)) & 1 for i in range(n)]
                estado = "".join(str(b) for b in bits)
                print(f"    {estado}     {tpm[s, future_idx]:.4f}")
            
            print()
            print(f"    Marginal k={k_low} (mecanismo={{{','.join(labels[m] for m in sorted(low_mech_set))}}}):")
            # Marginalize over dims NOT in low_mech_set
            other_dims = [p for p in range(n) if p not in low_mech_set]
            rows_low = {}
            for combination in range(1 << len(low_mech_set)):
                assgn_mech = {}
                for ii, p in enumerate(sorted(low_mech_set)):
                    assgn_mech[p] = (combination >> (len(low_mech_set) - 1 - ii)) & 1
                
                vals = []
                for s in range(1 << n):
                    bits = [(s >> (n - 1 - i)) & 1 for i in range(n)]
                    match = all(bits[p] == assgn_mech[p] for p in low_mech_set)
                    if match:
                        vals.append(tpm[s, future_idx])
                
                if vals:
                    mstr = ",".join(f"{labels[p]}={assgn_mech[p]}" for p in sorted(low_mech_set))
                    avg = np.mean(vals)
                    print(f"      P({labels[future_idx]}_f | {mstr}) = {avg:.4f}  (promedio {len(vals)} estados)")

            print()
            print(f"    Marginal k={k_high} (mecanismo={{{','.join(labels[m] for m in sorted(high_mech_set))}}}):")
            other_dims_h = [p for p in range(n) if p not in high_mech_set]
            for combination in range(1 << len(high_mech_set)):
                assgn_mech = {}
                for ii, p in enumerate(sorted(high_mech_set)):
                    assgn_mech[p] = (combination >> (len(high_mech_set) - 1 - ii)) & 1
                
                vals = []
                for s in range(1 << n):
                    bits = [(s >> (n - 1 - i)) & 1 for i in range(n)]
                    match = all(bits[p] == assgn_mech[p] for p in high_mech_set)
                    if match:
                        vals.append(tpm[s, future_idx])
                
                if vals:
                    mstr = ",".join(f"{labels[p]}={assgn_mech[p]}" for p in sorted(high_mech_set))
                    avg = np.mean(vals)
                    print(f"      P({labels[future_idx]}_f | {mstr}) = {avg:.4f}  (promedio {len(vals)} estados)")

            print()
            break  # just one crossing node


def test_sample(n, variant):
    """Test a single TPM sample for refinement violations."""
    tpm = TpmLoader.cargar(n, variant)
    init = np.ones(n, dtype=np.int8)
    system = System(tpm, init)
    labels = [chr(65 + i) for i in range(n)]

    results = {}
    for k in range(2, n + 1):
        total = count_k_partitions(n, n, k)
        if total > 50000:
            print(f"  {n}{variant}: skipping k={k} ({total} partitions > 50k limit)")
            break
        emd, parts, _ = optimal_k_partitions(system, k)
        results[k] = {"emd": emd, "parts": parts}

    if len(results) < 2:
        return None

    ks = sorted(results.keys())
    for i in range(1, len(ks)):
        k_high, k_low = ks[i], ks[i - 1]
        ok = all(
            any(refines(kp, km1p) for km1p in results[k_low]["parts"])
            for kp in results[k_high]["parts"]
        )
        if not ok:
            return {
                "name": f"N{n}{variant}",
                "n": n,
                "variant": variant,
                "tpm": tpm,
                "system": system,
                "k_low": k_low,
                "k_high": k_high,
                "low_emd": results[k_low]["emd"],
                "high_emd": results[k_high]["emd"],
                "low_parts": results[k_low]["parts"],
                "high_parts": results[k_high]["parts"],
            }
    return None


def main():
    print("=" * 74)
    print("  Refinement Conjecture Test: Real TPM Samples")
    print("=" * 74)

    samples_to_test = []
    for n in range(3, 7):
        for variant in ["A", "B", "C"]:
            try:
                tpm = TpmLoader.cargar(n, variant)
                samples_to_test.append((n, variant, tpm.shape))
            except FileNotFoundError:
                pass

    print(f"\n  Found {len(samples_to_test)} testable samples:")
    for n, v, shape in samples_to_test:
        print(f"    N{n}{v}: {shape}")

    violations = []
    for n, v, shape in samples_to_test:
        print(f"\n  Testing N{n}{v}... ", end="")
        result = test_sample(n, v)
        if result:
            violations.append(result)
            print("VIOLATION found!")
        else:
            print("ok (refinement holds)")

    print(f"\n{'=' * 74}")
    print(f"  RESULTS: {len(violations)} violations out of {len(samples_to_test)} samples")
    print(f"{'=' * 74}")

    for v in violations:
        display_violation(
            v["name"], v["tpm"], v["system"],
            v["k_low"], v["k_high"],
            v["low_parts"], v["high_parts"],
            v["low_emd"], v["high_emd"],
        )


if __name__ == "__main__":
    main()

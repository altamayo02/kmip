"""
Test: k-partición seleccionando k-1 nodos de (u+v).

Cada selección de k-1 items de (presentes ∪ futuros) forma una k-partición:
  - item < u (presente) → grupo (mech={i}, alc={})
  - item >= u (futuro)  → grupo (mech={}, alc={j})
  - sink                → (mech=resto_presentes, alc=resto_futuros)

Verifica contra Geometric (k=2) y KBruteForce (k>2).
"""

import sys
import os
import time
import math
from itertools import combinations

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.functions.emd import emd_efecto
from src.models.system import System
from src.strategies.geometric import GeometricSIA
from src.config import Config

from test_kpartition_refinement import (
    all_k_partitions,
    k_partition_distribution,
    count_k_partitions,
)

from src.strategies.k_brute_force_parallel import KBruteForceParallel
from src.strategies.k_brute_force import KBruteForce
from loader import TpmLoader


def enumerate_selection_partitions(
    system: System,
    k: int,
):
    """Enumerate all k-partitions formed by selecting k-1 nodes from (u+v).

    Yields (groups_tuple, emd) for each valid partition.
    """
    u = len(system.dims_ncubos)       # present dims (mech candidates)
    v = len(system.indices_ncubos)    # future nodes (alc candidates)
    all_mech = frozenset(range(u))
    all_alc = frozenset(range(v))
    intact = system.distribucion_marginal()

    if k > u + v:
        return

    for combo in combinations(range(u + v), k - 1):
        sel_mech = frozenset(i for i in combo if i < u)
        sel_alc = frozenset(j - u for j in combo if j >= u)

        groups = []
        for i in sel_mech:
            groups.append((frozenset({i}), frozenset()))
        for j in sel_alc:
            groups.append((frozenset(), frozenset({j})))

        rem_mech = all_mech - sel_mech
        rem_alc = all_alc - sel_alc

        if not rem_mech and not rem_alc:
            continue

        groups.append((rem_mech, rem_alc))

        groups_tuple = tuple(groups)
        dist = k_partition_distribution(system, groups_tuple)
        emd = emd_efecto(dist, intact)

        yield groups_tuple, emd


def best_selection_partition(system: System, k: int):
    """Return (best_emd, best_groups, total_evaluated)."""
    best_emd = float("inf")
    best_groups = None
    total = 0

    for groups_tuple, emd in enumerate_selection_partitions(system, k):
        total += 1
        if emd < best_emd - 1e-12:
            best_emd = emd
            best_groups = groups_tuple

    return best_emd, best_groups, total


# ── Verification against Geometric ────────────────────────────────────────

def test_selection_vs_geometric(trials=50, max_n=4, seed=42):
    """Compare selection algorithm (k=2) vs GeometricSIA optimal bipartition."""
    rng = np.random.default_rng(seed)
    ge_ok = 0
    se_ok = 0
    total = 0
    ge_better = []
    se_better = []

    for _ in range(trials):
        n = rng.integers(3, max_n + 1)
        initial = np.ones(n, dtype=np.int8)
        tpm = rng.random((1 << n, n), dtype=np.float64)
        system = System(tpm, initial)
        config = Config()

        # Geometric
        geo = GeometricSIA(tpm, config)
        geo_sols = geo.aplicar_estrategia(
            "1" * n, "1" * n, "1" * n, "1" * n
        )
        geo_best = min(s.perdida for s in geo_sols)

        # Selection k=2
        sel_best, _, sel_total = best_selection_partition(system, 2)

        total += 1
        if abs(sel_best - geo_best) < 1e-12:
            se_ok += 1
        if abs(geo_best - sel_best) < 1e-12:
            ge_ok += 1

        diff = sel_best - geo_best
        if diff > 1e-12:
            ge_better.append(diff)
        elif diff < -1e-12:
            se_better.append(-diff)

    print(f"\n-- Selection vs Geometric (k=2, {total} trials) --")
    print(f"  Selection matches Geometric optimal: {se_ok}/{total} ({se_ok/max(1,total)*100:.1f}%)")
    print(f"  Selection found optimal:             {se_ok}/{total}")
    if ge_better:
        print(f"  Geometric better (mean={np.mean(ge_better):.6f}, max={np.max(ge_better):.6f})")
    if se_better:
        print(f"  Selection better (mean={np.mean(se_better):.6f}, max={np.max(se_better):.6f})")

    return se_ok, total


# ── Verification against brute force (k>2) ───────────────────────────────

def test_selection_vs_bruteforce(
    trials=50,
    max_n=4,
    max_k=4,
    max_partitions=50000,
    seed=42,
):
    """Compare selection algorithm (k>2) vs exhaustive brute force."""
    rng = np.random.default_rng(seed)
    total = 0
    optimal_count = 0
    gaps = []

    trial = 0
    while trial < trials:
        n = rng.integers(3, max_n + 1)
        initial = np.ones(n, dtype=np.int8)
        tpm = rng.random((1 << n, n), dtype=np.float64)
        system = System(tpm, initial)

        k = rng.integers(3, min(n, max_k) + 1)

        # Brute force
        bf_emd, bf_parts, bf_total = optimal_k_partitions(system, k)
        if bf_total > max_partitions:
            continue

        # Selection
        sel_emd, sel_groups, sel_total = best_selection_partition(system, k)
        if sel_emd == float("inf"):
            continue

        trial += 1
        total += 1
        gap = sel_emd - bf_emd
        gaps.append(gap)
        if abs(gap) < 1e-12:
            optimal_count += 1

        if trial % 10 == 0:
            print(f"  [{trial}/{trials}] ...")

    print(f"\n-- Selection vs Brute Force (k>2, {total} trials) --")
    print(f"  Selection found optimal: {optimal_count}/{total} ({optimal_count/max(1,total)*100:.1f}%)")
    if gaps:
        print(f"  Gap mean={np.mean(gaps):.6f} max={np.max(gaps):.6f}")
        nonzero = sum(1 for g in gaps if g > 1e-12)
        print(f"  Non-zero gaps: {nonzero}/{total}")

    return optimal_count, total


# ── Single-case debug ────────────────────────────────────────────────────

def debug_case(n=3, k=3, seed=42):
    """Print all selection partitions for a single case."""
    rng = np.random.default_rng(seed)
    initial = np.zeros(n, dtype=np.int8)
    initial[0] = 1
    tpm = TpmLoader.cargar(n)
    system = System(tpm, initial)
    intact = system.distribucion_marginal()

    print(f"\n-- Debug: n={n}, k={k}, seed={seed} --")
    print(f"  u={len(system.dims_ncubos)}, v={len(system.indices_ncubos)}")
    print(f"  Total selections: C({len(system.dims_ncubos)+len(system.indices_ncubos)}, {k-1})")

    best_emd = float("inf")
    best_groups = None
    total = 0

    for groups_tuple, emd in enumerate_selection_partitions(system, k):
        total += 1
        desc = "  ".join(
            f"({set(m)},{set(a)})" for m, a in groups_tuple
        )
        marker = " <-- BEST" if emd < best_emd - 1e-12 else ""
        if emd < best_emd - 1e-12:
            best_emd = emd
            best_groups = groups_tuple
        #print(f"  [{total:3d}] EMD={emd:.8f}  {desc}{marker}")

    config = Config()
    strat = KBruteForceParallel(tpm, config, k=k, use_gpu=True)
    sols = strat.aplicar_estrategia(
        "1" + "0" * (n-1), "1" * n, "1" * n, "1" * n
    )
    bf_emd = sols[0].perdida
    bf_total = count_k_partitions(
        len(system.dims_ncubos), len(system.indices_ncubos), k
    )
    print(f"\n  Selection best: {best_emd:.8f} (evaluated {total})")
    print(f"  Brute force:    {bf_emd:.8f} ({bf_total} valid partitions)")
    print(f"  Gap: {best_emd - bf_emd:.8f}")
    for sol in sols:
      print(sol.particion)

    return best_emd, bf_emd


# ── Larger system benchmarks (no BF comparison) ──────────────────────────

def benchmark_larger(trials=20, max_n=6, seed=42):
    """Benchmark selection on larger systems; compare only against itself."""
    rng = np.random.default_rng(seed)

    print(f"\n-- Benchmark: larger systems ({trials} trials, max-n={max_n}) --")
    print(f"  {'n':>3} {'k':>3} {'C(u+v,k-1)':>12} {'count':>6} {'emd':>10} {'time(s)':>10}")
    print(f"  {'-'*43}")

    total_count = 0
    total_time = 0.0

    for _ in range(trials):
        n = rng.integers(5, max_n + 1)
        initial = np.ones(n, dtype=np.int8)
        tpm = rng.random((1 << n, n), dtype=np.float64)
        system = System(tpm, initial)

        k = rng.integers(3, n + 1)

        u = len(system.dims_ncubos)
        v = len(system.indices_ncubos)
        theoretical = math.comb(u + v, k - 1)

        start = time.perf_counter()
        best_emd, _, evaluated = best_selection_partition(system, k)
        elapsed = time.perf_counter() - start

        total_count += evaluated
        total_time += elapsed

        if best_emd == float("inf"):
            print(f"  {n:>3} {k:>3} {theoretical:>12,} {'N/A':>6} {'inf':>10} {elapsed:>10.4f}")
        else:
            print(f"  {n:>3} {k:>3} {theoretical:>12,} {evaluated:>6,} {best_emd:>10.6f} {elapsed:>10.4f}")

    print(f"  {'-'*43}")
    print(f"  Total evaluated: {total_count:,} | Total time: {total_time:.2f}s")


def test_k_variation(trials=10, n=5, max_k=5, seed=42):
    """For fixed n, test all k from 3 to max_k."""
    rng = np.random.default_rng(seed)

    print(f"\n-- k variation: n={n} ({trials} trials, max-k={max_k}) --")
    print(f"  {'k':>3} {'C(u+v,k-1)':>12} {'optimal':>9} {'time(s)':>10} {'bf_match':>10}")
    print(f"  {'-'*46}")

    for _ in range(trials):
        initial = np.ones(n, dtype=np.int8)
        tpm = rng.random((1 << n, n), dtype=np.float64)
        system = System(tpm, initial)

        print(f"  Trial {_+1}:")
        for k in range(3, min(max_k, n) + 1):
            u = len(system.dims_ncubos)
            v = len(system.indices_ncubos)
            theoretical = math.comb(u + v, k - 1)

            start = time.perf_counter()
            best_emd, _, evaluated = best_selection_partition(system, k)
            elapsed = time.perf_counter() - start

            # BF only if feasible (n <= 4 or small k)
            bf_str = ""
            if n <= 4:
                bf_emd, _, bf_total = optimal_k_partitions(system, k)
                bf_str = "OK" if abs(best_emd - bf_emd) < 1e-12 else f"GAP={best_emd - bf_emd:.6f}"
            else:
                bf_str = "N/A"

            print(f"    {k:>3} {theoretical:>12,} {best_emd:>9.6f} {elapsed:>10.4f} {bf_str:>10}")

    return


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Test k-partition via node selection"
    )
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--max-n", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--kvar", action="store_true")
    args = parser.parse_args()

    if args.debug:
        debug_case(n=args.n, k=args.k, seed=args.seed)
    elif args.benchmark:
        benchmark_larger(trials=20, max_n=args.max_n, seed=args.seed)
    elif args.kvar:
        test_k_variation(trials=10, n=args.n, max_k=args.k, seed=args.seed)
    else:
        test_selection_vs_geometric(trials=args.trials, max_n=args.max_n, seed=args.seed)
        test_selection_vs_bruteforce(trials=args.trials, max_n=args.max_n, seed=args.seed)


if __name__ == "__main__":
    main()

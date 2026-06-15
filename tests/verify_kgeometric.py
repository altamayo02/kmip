"""
Verificación: KGeometric (B&B) vs fuerza bruta para k-particiones.

Genera TPMs pequeños (n=3..5), compara optimalidad de B&B
en modo exhaustive y geometric contra el óptimo real.
"""

import sys
import os
import time
import argparse
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.functions.emd import emd_efecto
from src.models.system import System
from src.strategies.k_geometric import KGeometric
from src.config import Config

from test_kpartition_refinement import (
    all_k_partitions,
    k_partition_distribution as bf_k_part_dist,
    count_k_partitions,
)


def brute_force_optimal(system: System, k: int):
    m = len(system.indices_ncubos)
    n = len(system.dims_ncubos)
    intact = system.distribucion_marginal()
    best_emd = float("inf")
    best_parts = []
    total = 0

    for kp in all_k_partitions(m, n, k):
        total += 1
        dist = bf_k_part_dist(system, kp)
        emd = emd_efecto(dist, intact)
        if emd < best_emd - 1e-12:
            best_emd = emd
            best_parts = [kp]
        elif abs(emd - best_emd) < 1e-12:
            best_parts.append(kp)

    return best_emd, best_parts, total


def run_trial(
    tpm: np.ndarray,
    initial_state: np.ndarray,
    k: int,
    max_partitions: int = 50000,
):
    system = System(tpm, initial_state)

    bf_count = count_k_partitions(
        len(system.indices_ncubos),
        len(system.dims_ncubos),
        k,
    )
    if bf_count > max_partitions:
        return None

    bf_emd, bf_parts, bf_total = brute_force_optimal(system, k)

    config = Config()
    config_str = "1" * len(initial_state)

    results = {}

    for mode in ("exhaustive", "geometric"):
        strat = KGeometric(tpm, config, k=k, mode=mode)
        start = time.time()
        sols = strat.aplicar_estrategia(
            estado_inicial=config_str,
            condicion=config_str,
            alcance=config_str,
            mecanismo=config_str,
        )
        elapsed = time.time() - start

        if sols and sols[0].perdida != float("inf"):
            bb_emd = sols[0].perdida
            gap = bb_emd - bf_emd if bf_emd != float("inf") else 0.0
            is_optimal = abs(gap) < 1e-12
        else:
            bb_emd = float("inf")
            gap = float("inf")
            is_optimal = False

        results[mode] = {
            "emd": bb_emd,
            "gap": gap,
            "optimal": is_optimal,
            "time": elapsed,
            "nodes": strat._nodes_explored if hasattr(strat, "_nodes_explored") else -1,
        }

    return {
        "n": len(initial_state),
        "k": k,
        "bf_emd": bf_emd,
        "bf_total": bf_total,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Verify KGeometric B&B against brute force"
    )
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--max-n", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-partitions", type=int, default=50000)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    summary = {
        "trials": 0,
        "by_n": defaultdict(lambda: {"trials": 0, "exhaustive_ok": 0, "geometric_ok": 0}),
        "exhaustive_gaps": [],
        "geometric_gaps": [],
        "times": {"exhaustive": [], "geometric": []},
    }

    print(f"{'='*60}")
    print(f"  KGeometric Verification")
    print(f"  Trials: {args.trials} | Max-N: {args.max_n} | Seed: {args.seed}")
    print(f"{'='*60}\n")

    trial = 0
    while trial < args.trials:
        n = rng.integers(3, args.max_n + 1)
        initial = np.ones(n, dtype=np.int8)
        tpm = rng.random((1 << n, n), dtype=np.float64)

        max_k = min(n, 4)
        k = rng.integers(2, max_k + 1)

        trial_result = run_trial(tpm, initial, k, args.max_partitions)
        if trial_result is None:
            continue

        trial += 1
        summary["trials"] += 1
        nn = trial_result["n"]
        summary["by_n"][nn]["trials"] += 1

        for mode in ("exhaustive", "geometric"):
            r = trial_result["results"][mode]
            if r["optimal"]:
                summary["by_n"][nn][f"{mode}_ok"] += 1
            summary[f"{mode}_gaps"].append(r["gap"])
            summary["times"][mode].append(r["time"])

        if args.verbose:
            ex_ok = trial_result["results"]["exhaustive"]["optimal"]
            ge_ok = trial_result["results"]["geometric"]["optimal"]
            ex_gap = trial_result["results"]["exhaustive"]["gap"]
            ge_gap = trial_result["results"]["geometric"]["gap"]
            print(
                f"  [{trial:3d}] n={nn} k={trial_result['k']} "
                f"BF={trial_result['bf_emd']:.6f} "
                f"({trial_result['bf_total']} parts) "
                f"ex={'OK' if ex_ok else f'{ex_gap:.6f}'} "
                f"ge={'OK' if ge_ok else f'{ge_gap:.6f}'}"
            )

        if trial % 10 == 0:
            print(f"  [{trial}/{args.trials}] ...")

    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    print(f"  Total trials: {summary['trials']}")

    for nn in sorted(summary["by_n"], key=int):
        s = summary["by_n"][nn]
        print(f"\n  N={nn} ({s['trials']} trials):")
        print(f"    Exhaustive optimal:  {s['exhaustive_ok']}/{s['trials']} "
              f"({s['exhaustive_ok']/max(1,s['trials'])*100:.1f}%)")
        print(f"    Geometric optimal:   {s['geometric_ok']}/{s['trials']} "
              f"({s['geometric_ok']/max(1,s['trials'])*100:.1f}%)")

    print(f"\n  Gap analysis:")
    for mode in ("exhaustive", "geometric"):
        gaps = [g for g in summary[f"{mode}_gaps"] if g != float("inf")]
        if gaps:
            print(f"    {mode:12s}: mean={np.mean(gaps):.6f} "
                  f"max={np.max(gaps):.6f} "
                  f"non-zero={sum(1 for g in gaps if g > 1e-12)}/{len(gaps)}")
        times = summary["times"][mode]
        if times:
            print(f"    {'':12s}  time: mean={np.mean(times):.4f}s "
                  f"max={np.max(times):.4f}s")

    return 0


if __name__ == "__main__":
    main()

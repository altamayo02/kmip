"""
Test the refinement conjecture: does every optimal k-partition always refine
some optimal (k-1)-partition?

Enumerates exhaustively all k-partitions for small N (3..6) and various
TPM classes. Reports violations of the refinement property along with
their EMD values.

Usage:
    uv run python tests/test_kpartition_refinement.py
    uv run python tests/test_kpartition_refinement.py --trials 200 --max-n 6 --seed 42
"""

import sys
import os
import argparse
import json
import time
from collections import defaultdict
from itertools import combinations

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.functions.emd import emd_efecto
from src.models.system import System


# ── Set partition generation ────────────────────────────────────────────
def _set_partitions(elements: list, k: int):
    """Generate all set partitions of `elements` into exactly `k` nonempty blocks.
    
    Each yielded value is a list of `k` lists (blocks). Block order follows
    the order in which blocks first appear (canonical).
    """
    n = len(elements)
    if n < k or k <= 0:
        return
    if k == 1:
        yield [list(elements)]
        return
    if n == k:
        yield [[e] for e in elements]
        return

    def _rec(remaining, blocks):
        if not remaining:
            if len(blocks) == k:
                yield blocks
            return
        e = remaining[0]
        rest = remaining[1:]
        for i in range(len(blocks)):
            copy = [b[:] for b in blocks]
            copy[i].append(e)
            yield from _rec(rest, copy)
        if len(blocks) < k:
            yield from _rec(rest, blocks + [[e]])

    yield from _rec(elements, [])


def _assignments(elements: list, k: int):
    """Yield all ways to assign each element to one of k labeled blocks.
    
    Each yielded value is a list of k lists (blocks), some possibly empty.
    This generates exactly k^n assignments.
    """
    n = len(elements)
    if n == 0:
        yield [[] for _ in range(k)]
        return
    total = k ** n
    for code in range(total):
        blocks = [[] for _ in range(k)]
        remaining = code
        for e in elements:
            bidx = remaining % k
            blocks[bidx].append(e)
            remaining //= k
        yield blocks


def all_k_partitions(m: int, n: int, k: int):
    """Yield every k-partition of `m` alcance and `n` mecanismo indices.
    
    Includes degenerate partitions where some blocks have empty mechanism
    or empty alcance (as permitted in IIT).  Two blocks that are both
    empty in mechanism AND alcance are excluded.
    
    Each partition is a tuple of `k` pairs:
        ((mechanism_block, alcance_block), ...)
    where each block is a frozenset of ints.
    """
    alcance_elems = list(range(m))
    mecanismo_elems = list(range(n))

    for alc_blocks in _assignments(alcance_elems, k):
        for mech_blocks in _assignments(mecanismo_elems, k):
            if any(len(alc_blocks[i]) == 0 and len(mech_blocks[i]) == 0
                   for i in range(k)):
                continue
            yield tuple(
                (frozenset(mech_blocks[i]), frozenset(alc_blocks[i]))
                for i in range(k)
            )


def count_k_partitions(m: int, n: int, k: int) -> int:
    """Exact count of k-partitions = k^(m+n) - sigma_{j=0}^{k-1} C(k,j) f(j)
    
    where f(j) = (k - j)^m * (k - j)^n for blocks that are both empty.
    By inclusion-exclusion: k^(m+n) - sum_{j=1}^{k} (-1)^(j+1) C(k,j) (k-j)^m (k-j)^n
    
    Simplified: k^(m+n) - (number where at least one block is M=∅ AND A=∅)
    """
    from math import comb as _comb
    total = k ** (m + n)
    # Inclusion-exclusion to subtract cases with empty-empty blocks
    term = 0
    for j in range(1, k + 1):
        sign = -1 if j % 2 else 1  # (-1)^(j+1) → negative for odd j
        term += sign * _comb(k, j) * ((k - j) ** (m + n))
    return total + term


# ── k-partition distribution ────────────────────────────────────────────
def _marginal_from_cube(cube, initial_state: np.ndarray, keep_dims: set) -> float:
    """Marginalize NCube over dims NOT in keep_dims, return probability of
    the initial state in the resulting distribution."""
    if not keep_dims:
        return float(np.mean(cube.data))
    marginalize = np.array(
        [d for d in cube.dims if d not in keep_dims], dtype=np.int8
    )
    if marginalize.size:
        mc = cube.marginalizar(marginalize)
    else:
        mc = cube
    if mc.dims.size == 0:
        return float(mc.data)
    inicial = tuple(int(initial_state[j]) for j in mc.dims)
    return float(mc.data[inicial[::-1]])


def k_partition_distribution(system: System, k_partition) -> np.ndarray:
    """Return the marginal distribution vector under a k-partition.
    
    For each future node, find which block's alcance it belongs to,
    marginalize its NCube over mechanism dims NOT in that block's
    mechanism, and extract the probability at the initial state.
    """
    num_nodes = len(system.ncubos)
    dist = np.zeros(num_nodes, dtype=np.float32)

    for future_idx in system.indices_ncubos:
        cube = system.ncubos[future_idx]
        for mech_block, alc_block in k_partition:
            if future_idx in alc_block:
                dist[future_idx] = _marginal_from_cube(
                    cube, system.estado_inicial, set(mech_block)
                )
                break
    return dist


# ── Optimal k-partitions ────────────────────────────────────────────────
def optimal_k_partitions(system: System, k: int):
    """Enumerate all k-partitions and return those with minimum EMD.
    
    Returns (emd_min, list_of_partitions, total_evaluated).
    """
    m = len(system.indices_ncubos)
    n = len(system.dims_ncubos)

    intact_dist = system.distribucion_marginal()
    best_emd = float("inf")
    best_partitions = []
    total = 0

    for kp in all_k_partitions(m, n, k):
        total += 1
        part_dist = k_partition_distribution(system, kp)
        emd = emd_efecto(part_dist, intact_dist)
        if emd < best_emd - 1e-12:
            best_emd = emd
            best_partitions = [kp]
        elif abs(emd - best_emd) < 1e-12:
            best_partitions.append(kp)

    return best_emd, best_partitions, total


# ── Refinement check ────────────────────────────────────────────────────
def refines(k_partition, km1_partition) -> bool:
    """Does k_partition refine km1_partition?
    
    k_partition refines km1_partition iff every block of k_partition is
    a subset of some block of km1_partition.  Multiple fine blocks may
    map to the same coarse block (that is the nature of refinement).
    """
    elem_to_block = {}
    for bidx, (mech, alc) in enumerate(km1_partition):
        for m in mech:
            elem_to_block[("m", m)] = bidx
        for a in alc:
            elem_to_block[("a", a)] = bidx

    for mech, alc in k_partition:
        if not mech and not alc:
            return False

        first = ("m", next(iter(mech))) if mech else ("a", next(iter(alc)))
        target = elem_to_block.get(first)
        if target is None:
            return False

        for m in mech:
            if elem_to_block.get(("m", m)) != target:
                return False
        for a in alc:
            if elem_to_block.get(("a", a)) != target:
                return False

    return True


def partition_exists_in(partition, partition_list) -> bool:
    """Check if partition is structurally equivalent to any in the list."""
    for other in partition_list:
        if len(partition) != len(other):
            continue
        match = True
        for b1, b2 in zip(partition, other):
            m1, a1 = b1
            m2, a2 = b2
            if set(m1) != set(m2) or set(a1) != set(a2):
                match = False
                break
        if match:
            return True
    return False


def all_optimal_refine(opt_k_partitions, opt_km1_partitions) -> bool:
    """True iff every optimal k-partition refines some optimal (k-1)-partition."""
    for kp in opt_k_partitions:
        if not any(refines(kp, km1p) for km1p in opt_km1_partitions):
            return False
    return True


# ── TPM generators ──────────────────────────────────────────────────────
def tpm_random_uniform(n: int, rng: np.random.Generator) -> np.ndarray:
    """Fully random TPM — each entry ~ Uniform(0,1)."""
    return rng.random((1 << n, n), dtype=np.float64)


def tpm_random_binary(n: int, rng: np.random.Generator) -> np.ndarray:
    """Binary deterministic TPM — each entry 0 or 1."""
    return rng.integers(0, 2, size=(1 << n, n), dtype=np.int32).astype(np.float64)


def tpm_sparse(n: int, rng: np.random.Generator, p_active: float = 0.3) -> np.ndarray:
    """Sparse TPM — most entries 0, fraction p_active non-zero ~Uniform(0,1)."""
    tpm = np.zeros((1 << n, n), dtype=np.float64)
    mask = rng.random((1 << n, n)) < p_active
    tpm[mask] = rng.random(mask.sum(), dtype=np.float64)
    return tpm


def tpm_clustered(
    n: int, rng: np.random.Generator, cluster_size: int = 2, intra: float = 0.9
) -> np.ndarray:
    """TPM with clustered structure — nodes in the same cluster have
    strong causal connections, cross-cluster connections are weak."""
    num_states = 1 << n
    tpm = np.zeros((num_states, n), dtype=np.float64)
    clusters = [list(range(c, min(c + cluster_size, n))) for c in range(0, n, cluster_size)]

    for s_idx in range(num_states):
        state_bits = [(s_idx >> (n - 1 - i)) & 1 for i in range(n)]
        for node in range(n):
            p = 0.0
            for cl in clusters:
                if node in cl:
                    for parent in cl:
                        w = intra if parent != node else 1.0
                        p += w * state_bits[parent]
                else:
                    for parent in cl:
                        if rng.random() < 0.15:
                            p += rng.uniform(0.0, 0.15) * state_bits[parent]
            p = min(max(p / (n * 1.0), 0.0), 1.0)
            tpm[s_idx, node] = p if rng.random() < 0.85 else float(state_bits[node])

    return tpm


def tpm_asymmetric(
    n: int, rng: np.random.Generator, asymmetry: float = 0.2
) -> np.ndarray:
    """TPM with asymmetric connections designed to potentially create
    cases where k-partitions cross optimal (k-1)-partition boundaries."""
    num_states = 1 << n
    tpm = np.zeros((num_states, n), dtype=np.float64)

    strong_pairs = list(combinations(range(n), 2))
    rng.shuffle(strong_pairs)

    for s_idx in range(num_states):
        state_bits = [(s_idx >> (n - 1 - i)) & 1 for i in range(n)]
        for node in range(n):
            p = 0.0
            denom = 0.0
            p += 1.0 * state_bits[node]
            denom += 1.0
            for other in range(n):
                if other == node:
                    continue
                w = rng.uniform(0.0, asymmetry) if rng.random() < 0.5 else rng.uniform(0.8, 1.0)
                p += w * state_bits[other]
                denom += w
            tpm[s_idx, node] = p / denom

    return tpm


# ── Single trial ────────────────────────────────────────────────────────
TPM_GENERATORS = {
    "uniform": tpm_random_uniform,
    "binary": tpm_random_binary,
    "sparse": tpm_sparse,
    "asymmetric": tpm_asymmetric,
}


def run_trial(
    n: int,
    rng: np.random.Generator,
    tpm_type: str,
    tpm_gen_kwargs: dict = None,
    k_values: tuple = (2, 3, 4),
    max_partitions: int = 50_000,
    verbose: bool = False,
):
    """Generate a TPM, enumerate k-partitions, check refinement conjecture.
    
    Returns a dict with results or None if skipped.
    """
    gen = TPM_GENERATORS.get(tpm_type)
    if gen is None:
        raise ValueError(f"Unknown TPM type: {tpm_type}")

    kwargs = tpm_gen_kwargs or {}
    tpm = gen(n, rng, **kwargs)

    initial_state = np.ones(n, dtype=np.int8)
    system = System(tpm, initial_state)

    intact_dist = system.distribucion_marginal()

    k_values = [k for k in k_values if k <= n and count_k_partitions(n, n, k) <= max_partitions]

    if not k_values:
        return None

    results = {}
    for k in k_values:
        best_emd, best_parts, total = optimal_k_partitions(system, k)
        results[k] = {
            "best_emd": best_emd,
            "num_optimal": len(best_parts),
            "total_evaluated": total,
            "partitions": best_parts,
        }

    refinements_ok = {}
    for i in range(1, len(k_values)):
        k_larger = k_values[i]
        k_smaller = k_values[i - 1]
        ok = all_optimal_refine(
            results[k_larger]["partitions"],
            results[k_smaller]["partitions"],
        )
        refinements_ok[f"{k_smaller}to{k_larger}"] = ok

    violation_detail = None
    if not all(refinements_ok.values()):
        for k_larger, k_smaller in [
            (k_values[i], k_values[i - 1]) for i in range(1, len(k_values))
        ]:
            if not refinements_ok[f"{k_smaller}to{k_larger}"]:
                violation_detail = _describe_violation(
                    results[k_larger]["partitions"],
                    results[k_smaller]["partitions"],
                    k_smaller,
                    k_larger,
                    system,
                )

    return {
        "n": n,
        "tpm_type": tpm_type,
        "initial_state": tuple(int(x) for x in initial_state),
        "seed_used": rng.bit_generator.state["state"]["state"],
        "results": {str(k): v for k, v in results.items()},
        "refinements_ok": refinements_ok,
        "violation": violation_detail,
    }


def _describe_violation(
    opt_k, opt_km1, k_smaller, k_larger, system
) -> dict:
    """Describe a refinement violation in detail."""
    for kp in opt_k:
        non_refining_count = sum(1 for _ in opt_km1 if not refines(kp, _))
        refining_count = len(opt_km1) - non_refining_count
        if non_refining_count == len(opt_km1):
            return {
                "type": f"opt-{k_larger}-does-not-refine-any-opt-{k_smaller}",
                "k_larger_partition": _partition_to_dict(kp),
                "opt_km1_count": len(opt_km1),
            }
    return None


def _partition_to_dict(partition):
    """Convert partition to JSON-serializable dict."""
    return [
        {"mechanism": sorted(int(x) for x in mech), "alcance": sorted(int(x) for x in alc)}
        for mech, alc in partition
    ]


# ── Main runner ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Test k-partition refinement conjecture"
    )
    parser.add_argument(
        "--trials", type=int, default=500,
        help="Number of random TPMs to test (default: 500)"
    )
    parser.add_argument(
        "--max-n", type=int, default=5,
        help="Maximum number of nodes (default: 5, max 6)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--tpm-types", nargs="+",
        default=["uniform", "binary", "sparse", "asymmetric"],
        help="TPM generator types to test"
    )
    parser.add_argument(
        "--max-partitions", type=int, default=50_000,
        help="Skip k where partition count exceeds this (default: 50000)"
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output results as JSON"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print per-trial details"
    )
    args = parser.parse_args()

    max_n = min(args.max_n, 6)
    rng = np.random.default_rng(args.seed)

    summary = {
        "config": {
            "trials": args.trials,
            "max_n": max_n,
            "seed": args.seed,
            "tpm_types": args.tpm_types,
        },
        "by_type": defaultdict(lambda: {"trials": 0, "violations": 0}),
        "by_n": defaultdict(lambda: {"trials": 0, "violations": 0}),
        "total_violations": 0,
        "violations": [],
    }

    print(f"{'='*70}")
    print(f"  k-partition Refinement Conjecture Test")
    print(f"  Trials: {args.trials}  |  Max N: {max_n}  |  Seed: {args.seed}")
    print(f"  TPM types: {', '.join(args.tpm_types)}")
    print(f"{'='*70}\n")

    start_wall = time.time()
    trial_idx = 0

    while trial_idx < args.trials:
        n = rng.integers(3, max_n + 1)
        tpm_type = rng.choice(args.tpm_types)

        subseed = rng.integers(0, 2**31)
        trial_rng = np.random.default_rng(subseed)

        k_vals = tuple(k for k in (2, 3, 4) if k <= n)

        trial_result = run_trial(
            n=n,
            rng=trial_rng,
            tpm_type=tpm_type,
            k_values=k_vals,
            max_partitions=args.max_partitions,
            verbose=args.verbose,
        )

        if trial_result is None:
            continue

        trial_idx += 1

        ttype = trial_result["tpm_type"]
        nn = trial_result["n"]
        ro = trial_result["refinements_ok"]
        has_violation = not all(ro.values())

        summary["by_type"][ttype]["trials"] += 1
        summary["by_n"][str(nn)]["trials"] += 1

        if has_violation:
            summary["total_violations"] += 1
            summary["by_type"][ttype]["violations"] += 1
            summary["by_n"][str(nn)]["violations"] += 1
            summary["violations"].append(trial_result)
            marker = " ! VIOLATION"
        else:
            marker = ""

        if args.verbose or has_violation or trial_idx % 50 == 0:
            emd_info = "; ".join(
                f"k={k}: EMD={trial_result['results'][str(k)]['best_emd']:.6f}"
                f" ({trial_result['results'][str(k)]['num_optimal']} opt)"
                for k in sorted(int(kk) for kk in trial_result["results"])
            )
            print(
                f"  [{trial_idx:4d}/{args.trials}] N={nn} {ttype:12s} | "
                f"{emd_info} | refines={ro}{marker}"
            )

    elapsed = time.time() - start_wall

    # ── Final report ────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  RESULTS")
    print(f"{'='*70}")
    print(f"  Total trials:        {args.trials}")
    print(f"  Total violations:    {summary['total_violations']}")
    print(f"  Violation rate:      {summary['total_violations']/max(1,args.trials)*100:.2f}%")
    print(f"  Wall time:           {elapsed:.2f}s")
    print()

    print(f"  By N:")
    for nn in sorted(summary["by_n"], key=int):
        s = summary["by_n"][nn]
        rate = s["violations"] / max(1, s["trials"]) * 100
        print(f"    N={nn}: {s['violations']}/{s['trials']} violations ({rate:.2f}%)")

    print(f"\n  By TPM type:")
    for ttype in sorted(summary["by_type"]):
        s = summary["by_type"][ttype]
        rate = s["violations"] / max(1, s["trials"]) * 100
        print(f"    {ttype:12s}: {s['violations']}/{s['trials']} violations ({rate:.2f}%)")

    if summary["violations"]:
        print(f"\n{'='*70}")
        print(f"  VIOLATION DETAILS ({len(summary['violations'])} total)")
        print(f"{'='*70}")
        for i, v in enumerate(summary["violations"][:10]):
            print(f"\n  --- Violation {i+1} ---")
            print(f"  N={v['n']}, type={v['tpm_type']}, seed={v['seed_used']}")
            print(f"  Initial state: {v['initial_state']}")
            for k_str, res in v["results"].items():
                k = int(k_str)
                print(f"    k={k}: best_emd={res['best_emd']:.8f}, "
                      f"opt_count={res['num_optimal']}, evaluated={res['total_evaluated']}")
                for pi, part in enumerate(res["partitions"][:3]):
                    pd = _partition_to_dict(part)
                    print(f"      opt[{pi}]: {json.dumps(pd)}")
            print(f"  Refinement checks: {v['refinements_ok']}")
            if v.get("violation"):
                print(f"  Violation detail: {v['violation']}")

        if len(summary["violations"]) > 10:
            print(f"\n  ... and {len(summary['violations']) - 10} more violations")

    if args.json:
        output = {
            "summary": {
                "total_trials": args.trials,
                "total_violations": summary["total_violations"],
                "violation_rate": summary["total_violations"] / max(1, args.trials),
                "wall_time_s": elapsed,
                "by_n": {k: dict(v) for k, v in summary["by_n"].items()},
                "by_type": {k: dict(v) for k, v in summary["by_type"].items()},
            },
            "violations": [
                {
                    "n": v["n"],
                    "tpm_type": v["tpm_type"],
                    "seed_used": v["seed_used"],
                    "results": {
                        k: {
                            "best_emd": res["best_emd"],
                            "num_optimal": res["num_optimal"],
                            "total_evaluated": res["total_evaluated"],
                            "partitions": [_partition_to_dict(p) for p in res["partitions"]],
                        }
                        for k, res in v["results"].items()
                    },
                    "refinements_ok": v["refinements_ok"],
                }
                for v in summary["violations"]
            ],
        }

        print(f"\n{json.dumps(output, indent=2)}")

    return summary["total_violations"] == 0


if __name__ == "__main__":
    main()

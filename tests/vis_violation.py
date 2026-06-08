"""
Visualize a k-partition refinement violation.

Generates random TPMs until a violation is found, then displays:
  - Causal influence matrix (avg |ΔP| per present→future pair)
  - Optimal k=2 / k=3 partitions with preserved vs cut connections
  - Exact crossing: which mechanism→alcance pairs are mismatched
  - Summary diagram showing the crossing
  - DOT file for Graphviz visualization

Usage:
    uv run python tests/vis_violation.py
    uv run python tests/vis_violation.py --n 3 --seed 123 --trials 200
    uv run python tests/vis_violation.py --replay N4_uniform_seed999.json
"""

import sys
import os
import json
import argparse
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.functions.emd import emd_efecto
from src.models.system import System
from tests.test_kpartition_refinement import (
    TPM_GENERATORS,
    all_k_partitions,
    k_partition_distribution,
    optimal_k_partitions,
    refines,
    count_k_partitions,
    _partition_to_dict,
)


# ── Causal influence ────────────────────────────────────────────────────
def _influence_matrix(tpm: np.ndarray) -> np.ndarray:
    """Average |P(Y_f=1|X_p=1) - P(Y_f=1|X_p=0)| over all other vars."""
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
                v0 = float(tpm[s0, f] if s0 < num_states else tpm[s, f])
                v1 = float(tpm[s1, f] if s1 < num_states else tpm[s, f])
                total += abs(v1 - v0)
                cnt += 1
            infl[p, f] = total / cnt if cnt else 0.0
    return infl


# ── Finding a violation ─────────────────────────────────────────────────
def find_violation(
    n: int,
    tpm_type: str,
    seed: int,
    max_trials: int = 200,
    verbose: bool = True,
) -> dict | None:
    """Search for a refinement violation by generating random TPMs."""
    rng_master = np.random.default_rng(seed)
    labels = [chr(65 + i) for i in range(n)]

    for trial in range(1, max_trials + 1):
        subseed = int(rng_master.integers(0, 2**31))
        trial_rng = np.random.default_rng(subseed)

        gen = TPM_GENERATORS[tpm_type]
        tpm = gen(n, trial_rng)
        init = np.ones(n, dtype=np.int8)
        system = System(tpm, init)

        results = {}
        for k in range(2, n + 1):
            total = count_k_partitions(n, n, k)
            if total > 50000:
                break
            best_emd, best_parts, _ = optimal_k_partitions(system, k)
            results[k] = {"emd": best_emd, "parts": best_parts}

        if len(results) < 2:
            continue

        ks = sorted(results.keys())
        for i in range(1, len(ks)):
            k_high, k_low = ks[i], ks[i - 1]
            ok = all(
                any(refines(kp, km1p) for km1p in results[k_low]["parts"])
                for kp in results[k_high]["parts"]
            )
            if not ok:
                if verbose:
                    print(f"  Found violation: trial {trial}, seed={subseed}")
                    print(f"    k={k_low} EMD={results[k_low]['emd']:.6f}")
                    print(f"    k={k_high} EMD={results[k_high]['emd']:.6f}")

                return {
                    "n": n,
                    "tpm_type": tpm_type,
                    "seed": subseed,
                    "tpm": tpm.tolist(),
                    "labels": labels,
                    "initial_state": [1] * n,
                    "influence": _influence_matrix(tpm).tolist(),
                    "results": {
                        str(k): {
                            "emd": float(results[k]["emd"]),
                            "partitions": [
                                [
                                    (sorted(int(x) for x in m), sorted(int(x) for x in a))
                                    for m, a in p
                                ]
                                for p in results[k]["parts"]
                            ],
                        }
                        for k in ks if k in results
                    },
                }

    return None


# ── Replay from saved JSON ──────────────────────────────────────────────
def replay(path: str) -> dict | None:
    with open(path) as f:
        return json.load(f)


# ── Text display ────────────────────────────────────────────────────────
def _fmt_block(mech, alc, labels):
    m = ",".join(labels[i] for i in sorted(mech))
    a = ",".join(labels[i] for i in sorted(alc))
    return f"M={{{m}}} -> A={{{a}}}"


def display(info: dict):
    n = info["n"]
    labels = info["labels"]
    infl = np.array(info["influence"])
    results = info["results"]

    print()
    print("=" * 76)
    print(f"  k-PARTITION REFINEMENT VIOLATION")
    print(f"  N={n}  |  Type: {info['tpm_type']}  |  Seed: {info['seed']}")
    print(f"  Initial state: {info['initial_state']}")
    print("=" * 76)

    # ── 1. Influence matrix ─────────────────────────────────────────────
    print()
    print("  1. CAUSAL INFLUENCE MATRIX")
    print("     avg |P(Y_f=1 | X_p=1) - P(Y_f=1 | X_p=0)|")
    print()
    header = f"        " + "  ".join(f"{labels[f]:>6s}" for f in range(n))
    print(f"  {header}")
    print(f"        " + "------" * n)
    for p in range(n):
        vals = "  ".join(
            f"\033[91m{infl[p,f]:.4f}\033[0m" if infl[p,f] > 0.1
            else f"\033[93m{infl[p,f]:.4f}\033[0m" if infl[p,f] > 0.02
            else f"{infl[p,f]:.4f}"
            for f in range(n)
        )
        print(f"  {labels[p]}_p:  {vals}")
    print()

    # ── 2. Each partition ───────────────────────────────────────────────
    print("  2. OPTIMAL PARTITIONS")
    print()
    for k_str in sorted(results, key=int):
        k = int(k_str)
        res = results[k_str]
        print(f"  k={k}  |  EMD = {res['emd']:.6f}")
        for pi, part in enumerate(res["partitions"]):
            for bi, (mech, alc) in enumerate(part):
                arrow = "  ".join(
                    f"\033[92m{labels[m]}->{labels[a]}\033[0m"
                    for m in mech for a in alc
                    if infl[m, a] > 0.02
                )
                cuts = "  ".join(
                    f"\033[91m{labels[m]}->{labels[a]}\033[0m"
                    for m in mech for a in range(n) if a not in alc and infl[m, a] > 0.02
                )
                print(f"     B{bi}: {_fmt_block(mech, alc, labels)}")
                if arrow:
                    print(f"          preserve: {arrow}")
                if cuts:
                    print(f"          cut:      {cuts}")
        print()

    # ── 3. Crossing analysis ────────────────────────────────────────────
    print("  3. CROSSING ANALYSIS")
    print()

    ks = sorted(int(k) for k in results)
    for i in range(1, len(ks)):
        k_low, k_high = ks[i - 1], ks[i]
        low_parts = results[str(k_low)]["partitions"]
        high_parts = results[str(k_high)]["partitions"]

        for hpi, hp in enumerate(high_parts):
            if any(refines(hp, lp) for lp in low_parts):
                continue

            print(f"  k={k_high} optimal partition [{hpi}] crosses k={k_low}:")
            print()

            # Build k_low element-to-block map
            for lpi, lp in enumerate(low_parts):
                elem_to_block = {}
                for bj, (m, a) in enumerate(lp):
                    for x in m:
                        elem_to_block[("m", x)] = bj
                    for x in a:
                        elem_to_block[("a", x)] = bj

                for bi, (mech, alc) in enumerate(hp):
                    if not mech or not alc:
                        continue
                    m0 = next(iter(mech))
                    a0 = next(iter(alc))
                    bm = elem_to_block.get(("m", m0))
                    ba = elem_to_block.get(("a", a0))
                    if bm is not None and ba is not None and bm != ba:
                        # Mismatch! Cross-block pair
                        mech_str = ",".join(labels[m] for m in sorted(mech))
                        alc_str = ",".join(labels[a] for a in sorted(alc))
                        print(
                            f"     Block B{bi}: M={{{mech_str}}} -> A={{{alc_str}}}"
                        )
                        print(
                            f"       In k={k_low} opt[{lpi}]: m={labels[m0]} in B{bm},"
                            f" a={labels[a0]} in B{ba}"
                        )
                        print(
                            f"       k={k_low} pairs B{bm} (mechanism) with B{bm}"
                            f" (alcance), but k={k_high} pairs B{bi} (mechanism)"
                        )
                        print(
                            f"       with B{bi} (alcance), crossing the"
                            f" k={k_low} boundary."
                        )
                        print()
            break  # one crossing detail is enough

    # ── 4. Summary diagram ──────────────────────────────────────────────
    print("  4. PARTITION DIAGRAM")
    print()
    for k_str in sorted(results, key=int):
        k = int(k_str)
        res = results[k_str]
        if not res["partitions"]:
            continue
        part = res["partitions"][0]

        mech_line = "M: "
        alc_line = "A: "
        sep_line = "   "
        block_labels = []

        for bi, (mech, alc) in enumerate(part):
            mech_line += " ".join(labels[m] for m in sorted(mech)) + " "
            alc_line += " ".join(labels[a] for a in sorted(alc)) + " "
            sep_line += "-" * (len(mech) + len(alc)) + " "
            block_labels.append(f"[B{bi}]" + " " * (len(mech) + len(alc) - 4))

        print(f"  k={k}  (EMD={res['emd']:.6f})")
        print(f"    {mech_line}")
        print(f"    {sep_line}")
        print(f"    {alc_line}")
        print(f"    {' '.join(block_labels)}")

        # Show which connections are cut
        cut_list = []
        for bi, (mech, alc) in enumerate(part):
            for m in mech:
                for f in range(n):
                    if f not in alc and infl[m, f] > 0.02:
                        cut_list.append(f"{labels[m]}_p->{labels[f]}_f({infl[m,f]:.3f})")

        if cut_list:
            print(f"    cuts: {', '.join(cut_list[:8])}")
        print()

    # ── 5. Implication ──────────────────────────────────────────────────
    print()
    print("  5. IMPLICATION")
    print()
    print("  The hierarchical algorithm (recurse on optimal bipartition)")
    print("  would commit to a fixed pairing of mechanism -> alcance")
    print("  at each level.  The k=3 crossing shows that a better")
    print("  global k=3 partition may assign mechanism indices to")
    print("  alcance indices differently than any optimal k=2 partition,")
    print("  making the recursive approach miss it entirely.")
    print()


# ── DOT generation ──────────────────────────────────────────────────────
def generate_dot(info: dict) -> str:
    n = info["n"]
    labels = info["labels"]
    infl = np.array(info["influence"])
    results = info["results"]

    c2 = ["#e41a1c", "#377eb8"]
    c3 = ["#4daf4a", "#984ea3", "#ff7f00"]
    c4 = ["#ffff33", "#a65628", "#f781bf", "#999999"]

    lines = [
        "digraph KMIP {",
        "  rankdir=LR; splines=true; compound=true;",
        '  node [fontname="Arial", style=filled, fillcolor=white];',
        '  edge [fontname="Arial", fontsize=8];',
        "",
        '  subgraph cluster_p { label="Present"; style=filled; fillcolor="#FFFACD";',
    ]
    for i in range(n):
        lines.append(f'    p_{i} [label="{labels[i]}_p", shape=box];')
    lines.append("  }")

    lines.append('  subgraph cluster_f { label="Future"; style=filled; fillcolor="#E0FFFF";')
    for i in range(n):
        lines.append(f'    f_{i} [label="{labels[i]}_f", shape=box];')
    lines.append("  }")

    mx = infl.max() or 1.0
    for p in range(n):
        for f in range(n):
            w = infl[p, f]
            if w > 0.01:
                pw = max(0.3, 3.0 * w / mx)
                lines.append(
                    f'  p_{p} -> f_{f} [label="{w:.3f}", penwidth={pw:.1f}];'
                )

    # k=2 coloring (outer peripheries)
    if "2" in results and results["2"]["partitions"]:
        for bi, (m, a) in enumerate(results["2"]["partitions"][0]):
            col = c2[bi % 2]
            for x in m:
                lines.append(f'  p_{x} [peripheries=2, color="{col}", fontcolor="{col}"];')
            for x in a:
                lines.append(f'  f_{x} [peripheries=2, color="{col}", fontcolor="{col}"];')

    # k=3 coloring (fill)
    if "3" in results and results["3"]["partitions"]:
        for bi, (m, a) in enumerate(results["3"]["partitions"][0]):
            col = c3[bi % 3]
            for x in m:
                lines.append(
                    f'  p_{x} [style="filled,bold", fillcolor="{col}20",'
                    f' color="{col}", fontcolor="{col}"];'
                )
            for x in a:
                lines.append(
                    f'  f_{x} [style="filled,bold", fillcolor="{col}20",'
                    f' color="{col}", fontcolor="{col}"];'
                )

    lines.append("}")
    return "\n".join(lines)


# ── Main ────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Visualize k-partition violation")
    ap.add_argument("--n", type=int, default=4, choices=[3, 4, 5])
    ap.add_argument("--tpm-type", default="sparse", choices=list(TPM_GENERATORS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--trials", type=int, default=500)
    ap.add_argument("--replay", type=str, default=None,
                    help="Replay from saved JSON file")
    ap.add_argument("--json", type=str, default=None,
                    help="Save violation data to JSON file")
    ap.add_argument("--dot", action="store_true", default=False,
                    help="Generate DOT file")
    ap.add_argument("--save", type=str, default=None,
                    help="Auto-save if a violation found (or search anyway)")
    args = ap.parse_args()

    if args.replay:
        with open(args.replay) as f:
            info = json.load(f)
        display(info)
        return

    # Quick search for violation
    print(f"Searching for a k-partition violation (N={args.n}, {args.tpm_type})...")
    info = find_violation(
        n=args.n,
        tpm_type=args.tpm_type,
        seed=args.seed,
        max_trials=args.trials,
        verbose=True,
    )

    if info is None:
        print(f"  No violation found in {args.trials} trials.")
        return

    display(info)

    # Save JSON
    if args.json:
        with open(args.json, "w") as f:
            json.dump(info, f, indent=2)
        print(f"  Saved: {args.json}")

    # DOT
    if args.dot:
        dot = generate_dot(info)
        dot_path = f"results/vis_N{info['n']}_{info['tpm_type']}_seed{info['seed']}.dot"
        os.makedirs("results", exist_ok=True)
        with open(dot_path, "w", encoding="utf-8") as f:
            f.write(dot)
        print(f"  DOT file: {dot_path}")

    # Auto-save
    if args.save and not args.json:
        path = args.save
        with open(path, "w") as f:
            json.dump(info, f, indent=2)
        print(f"  Saved: {path}")


if __name__ == "__main__":
    main()

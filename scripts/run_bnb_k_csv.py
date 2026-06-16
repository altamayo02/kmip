#!/usr/bin/env python
"""
Branch and Bound k-partition runner for CSV TPM files.

Usage:
  uv run python -X utf8 scripts/run_bnb_k_csv.py --dataset N6A --data-dir data/samples --k 5 --initial-state 100000 --objective final_phi --mode exact --partition-space nodes
  uv run python -X utf8 scripts/run_bnb_k_csv.py --dataset N6A --data-dir data/samples --k 5 --initial-state 100000 --objective final_phi --mode heuristic --generators selection --beam-width 50
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from math import comb, factorial

_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.strategies.branch_and_bound_k import (
    branch_and_bound_k_from_state_node_tpm,
    BnBConfig,
    SearchReport,
    VariableCodec,
    ensure_state_node_tpm,
    load_tpm_csv,
    parse_initial_state,
    export_search_tree_mermaid,
    clear_caches,
    _reset_nid,
    make_context,
    count_set_partitions,
)

_BUILTIN_PRINT = print


def pflush(*args, **kwargs):
    kwargs.setdefault("flush", True)
    _BUILTIN_PRINT(*args, **kwargs)


def build_result_dict(result: SearchReport) -> dict:
    return {
        "dataset": result.dataset_name,
        "csv_path": result.csv_path,
        "n_nodes": result.n_nodes,
        "n_search_vars": result.n_search_vars,
        "target_k": result.target_k,
        "initial_state": result.initial_state_str,
        "objective": result.objective,
        "mode": result.mode,
        "partition_space": result.partition_space,
        "generators": list(result.generators),
        "optimality_certified": result.optimality_certified,
        "termination_reason": result.termination_reason,
        "best_partition_labels": result.best_partition_labels_str,
        "best_partition_masks": list(result.best_partition),
        "best_accumulated_loss": result.best_accumulated_loss,
        "best_final_phi": result.best_final_phi,
        "incumbent_source": result.incumbent_source,
        "metrics": {
            "nodes_created": result.nodes_created,
            "nodes_expanded": result.nodes_expanded,
            "nodes_pruned_by_bound": result.nodes_pruned_by_bound,
            "nodes_pruned_by_dominance": result.nodes_pruned_by_dominance,
            "complete_nodes_found": result.complete_nodes_found,
            "incumbent_updates": result.incumbent_updates,
            "runtime_seconds": result.runtime_seconds,
        },
    }


def save_results(result: SearchReport, output_dir: str, label: str):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = build_result_dict(result)
    jp = out / f"{label}.json"
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    pflush(f"  Saved JSON: {jp}")

    if result.all_nodes:
        mmd = export_search_tree_mermaid(
            result.all_nodes, max_nodes=100,
            codec=VariableCodec.from_node_count(result.n_nodes, result.partition_space)
            if result.n_nodes else None
        )
        mp = out / f"{label}_tree.mmd"
        with open(mp, "w", encoding="utf-8") as f:
            f.write(mmd)
        pflush(f"  Saved Mermaid: {mp}")
    else:
        pflush("  (beam/enumeration mode: no tree to export)")


def count_stirling(n: int, k: int) -> int:
    """S(n,k) Stirling numbers of the second kind."""
    if n < k:
        return 0
    if k == 1 or n == k:
        return 1
    total = 0
    for j in range(k + 1):
        sign = 1 if (k - j) % 2 == 0 else -1
        total += sign * comb(k, j) * (j ** n)
    return abs(total) // factorial(k)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Branch and Bound k-partition runner for CSV TPM files"
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset", type=str, help="Dataset name like N3A, N17A")
    src.add_argument("--file", type=str, help="Direct path to CSV file")
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--k", type=int, default=5, dest="target_k")
    p.add_argument("--initial-state", type=str, default="ones")
    p.add_argument("--metric", type=str, default="emd_effect",
                   choices=["emd_effect", "emd_causal"])
    p.add_argument("--objective", type=str, default="final_phi",
                   choices=["final_phi", "accumulated_path"])
    p.add_argument("--partition-space", type=str, default="mech_alc",
                   choices=["mech_alc", "node_pairs"],
                   help="Partition space (default: mech_alc). "
                        "mech_alc: separates mechanism/purview (full k-MIP space). "
                        "node_pairs: forces a/A together (experimental, restricted).")
    p.add_argument("--mode", type=str, default="heuristic",
                   choices=["exact", "heuristic"])
    p.add_argument("--generators", type=str, default="selection",
                   help="Comma-separated: selection, bruteforce")
    p.add_argument("--beam-width", type=int, default=50)
    p.add_argument("--top-l", type=int, default=5, dest="top_l_per_generator")
    p.add_argument("--max-expansion", type=int, default=None,
                   dest="max_expansion_candidates_per_node",
                   help="Max children per node in accumulated_path mode")
    p.add_argument("--max-nodes", type=int, default=None)
    p.add_argument("--timeout-seconds", type=float, default=None)
    p.add_argument("--output-dir", type=str, default="results/bnb")
    p.add_argument("--verbose", action="store_true", default=False)
    return p.parse_args(argv)


def main():
    global print
    print = pflush
    args = parse_args()

    # Resolve CSV
    if args.file:
        csv_path = Path(args.file)
        dset = csv_path.stem
    else:
        csv_path = Path(args.data_dir) / f"{args.dataset}.csv"
        dset = args.dataset
    if not csv_path.exists():
        print(f"ERROR: TPM file not found: {csv_path.resolve()}")
        sys.exit(1)
    csvs = str(csv_path.resolve())

    # Load
    print(f"Loading: {csvs}")
    raw = load_tpm_csv(csvs)
    sn = ensure_state_node_tpm(raw)
    n_nodes = sn.shape[1]
    n_sv = n_nodes if args.partition_space == "node_pairs" else 2 * n_nodes
    print(f"  shape: {sn.shape}  n_nodes: {n_nodes}  n_search_vars: {n_sv}")
    print(f"  partition_space: {args.partition_space}  objective: {args.objective}  mode: {args.mode}")
    if args.partition_space == "node_pairs":
        print("  WARNING: node_pairs forces a/A to stay together. "
              "This is an experimental restricted space that does NOT span "
              "the full mechanism/purview partition space.")

    # Initial state
    try:
        init = parse_initial_state(args.initial_state, n_nodes)
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    istr = "".join(str(int(b)) for b in init)
    print(f"  initial_state: {istr}")

    if args.target_k > n_sv:
        print(f"ERROR: k={args.target_k} > n_search_vars={n_sv}")
        sys.exit(1)

    # Check viability for exact final_phi
    if args.objective == "final_phi" and args.mode == "exact":
        total_parts = count_stirling(n_sv, args.target_k)
        print(f"  total_set_partitions: {total_parts}")
        if total_parts > 500_000:
            print(f"ERROR: Too many set partitions ({total_parts}). "
                  f"Use --mode heuristic or reduce k/n.")
            sys.exit(1)

    gens = tuple(g.strip() for g in args.generators.split(","))
    config = BnBConfig(
        target_k=args.target_k,
        partition_space=args.partition_space,
        objective=args.objective,
        mode=args.mode,
        generators=gens,
        top_l_per_generator=args.top_l_per_generator,
        beam_width=args.beam_width,
        max_nodes=args.max_nodes,
        timeout_seconds=args.timeout_seconds,
    )

    # Set max_expansion_candidates_per_node ONLY from CLI (default = 0 for final_phi)
    if args.max_expansion_candidates_per_node is not None:
        config.max_expansion_candidates_per_node = args.max_expansion_candidates_per_node
    elif args.objective == "accumulated_path":
        config.max_expansion_candidates_per_node = 100  # default for BnB mode
    else:
        config.max_expansion_candidates_per_node = 0  # auto-detect (enum or beam)

    # For heuristic, set beam width
    if args.mode == "heuristic" and args.objective == "final_phi":
        config.max_expansion_candidates_per_node = 0
        if n_nodes >= 10:
            config.beam_width = min(config.beam_width, 30)
        if n_nodes >= 17:
            config.beam_width = min(config.beam_width, 10)

    # Determine algorithm name
    algo_name = (
        "exact_enumeration" if args.objective == "final_phi" and args.mode == "exact" else
        "beam_search" if args.objective == "final_phi" and args.mode == "heuristic" else
        "accumulated_path_bnb"
    )

    print(f"  generators: {config.generators}")
    print(f"  beam_width: {config.beam_width}")
    print(f"  max_expansion_per_node: {config.max_expansion_candidates_per_node}")
    print()

    # Run
    print("Running...")
    clear_caches()
    _reset_nid()
    start = time.time()
    result = branch_and_bound_k_from_state_node_tpm(
        state_node_tpm=sn, target_k=args.target_k,
        initial_state=init, metric=args.metric,
        config=config, verbose=args.verbose,
        dataset_name=dset, csv_path=csvs,
    )
    elapsed = time.time() - start

    # Recompute certification
    result.mode = args.mode
    if args.objective == "final_phi" and args.mode == "exact":
        result.optimality_certified = True
        result.termination_reason = "exhausted_all_final_partitions"
    elif args.objective == "final_phi":
        result.optimality_certified = False
    else:
        result.optimality_certified = (
            args.objective == "accumulated_path"
            and args.mode == "exact"
            and result.termination_reason == "queue_exhausted"
            and config.max_nodes is None
            and config.timeout_seconds is None
            and config.max_expansion_candidates_per_node == 0
        )

    # Print result
    print()
    print("=" * 65)
    print("  RESULTS")
    print("=" * 65)
    print(f"  Dataset:              {dset}")
    print(f"  CSV:                  {csvs}")
    print(f"  n_nodes:              {n_nodes}")
    print(f"  n_search_vars:        {n_sv}")
    print(f"  target_k:             {args.target_k}")
    print(f"  initial_state:        {istr}")
    print(f"  partition_space:      {args.partition_space}")
    print(f"  objective:            {args.objective}")
    print(f"  mode:                 {args.mode}")
    print(f"  generators:           {','.join(result.generators)}")
    print(f"  algorithm:            {algo_name}")
    print(f"  incumbent_source:     {result.incumbent_source}")
    print(f"  optimality_certified: {'YES' if result.optimality_certified else 'NO'}")
    print()
    print(f"  Best partition ({len(result.best_partition)} blocks):")
    print(f"    {result.best_partition_labels_str}")
    print()
    assert len(result.best_partition) == args.target_k, (
        f"BUG: result has {len(result.best_partition)} blocks, expected {args.target_k}"
    )
    if args.objective == "final_phi":
        print(f"  Best final phi:        {result.best_final_phi:.6f}")
        print(f"  Path accumulated loss: {result.best_accumulated_loss:.6f} (diagnostic)")
    else:
        print(f"  Best accumulated loss: {result.best_accumulated_loss:.6f}")
        print(f"  Final phi:             {result.best_final_phi:.6f} (diagnostic)")
    print()
    print("  Metrics:")
    print(f"    partitions evaluated:  {result.nodes_created}")
    if result.nodes_expanded:
        print(f"    partial_candidates:    {result.nodes_expanded}")
    if result.complete_nodes_found:
        print(f"    complete_partitions:   {result.complete_nodes_found}")
    print(f"    incumbent_updates:     {result.incumbent_updates}")
    print(f"    runtime:               {elapsed:.4f}s")
    print(f"    termination_reason:    {result.termination_reason}")

    # Save
    label = f"{dset}_k{args.target_k}_{args.objective}_{istr}_{args.partition_space}"
    try:
        save_results(result, args.output_dir, label)
    except Exception as e:
        print(f"  WARNING: save failed: {e}")
    print()
    print("Done.")


if __name__ == "__main__":
    main()

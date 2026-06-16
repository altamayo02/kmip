"""
Demo: Branch and Bound for k-partitions with real TPM and label output.

Usage:
    uv run python -X utf8 tests/demo_bnb_state_node_tpm.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.loader import TpmLoader
from src.strategies.branch_and_bound_k import (
    branch_and_bound_k_from_state_node_tpm,
    BnBConfig,
    ensure_state_node_tpm,
)

if __name__ == "__main__":
    # ── Load TPM ──────────────────────────────────────────────────────────
    raw = TpmLoader.cargar(3, "A")
    state_node = ensure_state_node_tpm(raw)
    n_nodes = state_node.shape[1]
    initial_state = np.ones(n_nodes, dtype=np.int8)

    print("=" * 65)
    print(f"  TPM N={n_nodes} | target_k=5 | initial_state={''.join(map(str, initial_state))}")
    print(f"  state_node_tpm shape: {state_node.shape}")
    print("=" * 65)

    # ── Run k=5 ───────────────────────────────────────────────────────────
    result = branch_and_bound_k_from_state_node_tpm(
        state_node_tpm=state_node,
        target_k=5,
        initial_state=initial_state,
        metric="emd_effect",
        config=BnBConfig(
            target_k=5,
            M_worst_per_block=5,
            upper_frontier_width=10,
        ),
        verbose=False,
    )

    # ── Report ────────────────────────────────────────────────────────────
    print()
    print("  Best partition (masks): ", result.best_partition)
    print("  Best partition (labels):", result.best_partition_labels_str)
    print(f"  Accumulated loss:        {result.best_accumulated_loss:.6f}")
    print(f"  Source:                  {result.incumbent_source}")
    print(f"  Nodes created:           {result.nodes_created}")
    print(f"  Nodes expanded:          {result.nodes_expanded}")
    print(f"  Pruned (bound):          {result.nodes_pruned_by_bound}")
    print(f"  Pruned (dominance):      {result.nodes_pruned_by_dominance}")
    print(f"  Runtime:                 {result.runtime_seconds:.4f}s")
    print()

    # ── Path ──────────────────────────────────────────────────────────────
    print("  Path:")
    for step in result.best_path:
        print(f"    Step {step['step']}: {step['parent_partition']} -> {step['child_partition']}")
        print(f"      {step['split']}")
        print(f"      C={step['accumulated_loss_after_split']:.6f}")
    print()

    # ── Path with labels ──────────────────────────────────────────────────
    print("  Path (labels):")
    print(result.best_path_labels_str)
    print()

    # ── Verification ──────────────────────────────────────────────────────
    assert len(result.best_partition) == 5, f"Expected 5 blocks, got {len(result.best_partition)}"
    assert len(result.best_path) == 4, f"Expected 4 steps, got {len(result.best_path)}"
    path_sum = sum(step["delta_phi"] for step in result.best_path)
    assert abs(result.best_accumulated_loss - path_sum) < 1e-10
    print("  Verification: OK (k=5 reached, path sum matches accumulated loss)")

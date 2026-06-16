"""
Integration test for BB with real TPM via the standard wrapper.

Usage:
    uv run python -X utf8 tests/test_bnb_real_tpm.py
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


def run_k2_test():
    print("=" * 60)
    print("  BB with loader TPM (N=3, k=2)")
    print("=" * 60)

    raw = TpmLoader.cargar(3, "A")
    state_node = ensure_state_node_tpm(raw)
    init = np.ones(3, dtype=np.int8)

    result = branch_and_bound_k_from_state_node_tpm(
        state_node_tpm=state_node,
        target_k=2,
        initial_state=init,
        metric="emd_effect",
        config=BnBConfig(target_k=2, partition_space="nodes", objective="final_phi"),
        verbose=True,
    )

    print()
    print(f"  Best partition (labels): {result.best_partition_labels_str}")
    print(f"  Best loss:               {result.best_accumulated_loss:.6f}")
    print()

    assert len(result.best_partition) == 2
    assert result.best_accumulated_loss >= 0
    return result


def run_k3_test():
    print("=" * 60)
    print("  BB with loader TPM (N=3, k=3)")
    print("=" * 60)

    raw = TpmLoader.cargar(3, "A")
    state_node = ensure_state_node_tpm(raw)
    init = np.ones(3, dtype=np.int8)

    result = branch_and_bound_k_from_state_node_tpm(
        state_node_tpm=state_node,
        target_k=3,
        initial_state=init,
        metric="emd_effect",
        verbose=True,
    )

    print()
    print(f"  Best partition (labels): {result.best_partition_labels_str}")
    print(f"  Best loss:               {result.best_accumulated_loss:.6f}")
    print()

    assert len(result.best_partition) == 3
    assert result.best_accumulated_loss >= 0
    return result


def run_k5_test():
    print("=" * 60)
    print("  BB with loader TPM (N=3, k=5)")
    print("=" * 60)

    raw = TpmLoader.cargar(3, "A")
    state_node = ensure_state_node_tpm(raw)
    init = np.ones(3, dtype=np.int8)

    result = branch_and_bound_k_from_state_node_tpm(
        state_node_tpm=state_node,
        target_k=5,
        initial_state=init,
        metric="emd_effect",
        verbose=False,
    )

    print()
    print(f"  Best partition (masks):  {result.best_partition}")
    print(f"  Best partition (labels): {result.best_partition_labels_str}")
    print(f"  Best loss:               {result.best_accumulated_loss:.6f}")
    print(f"  Path steps:              {len(result.best_path)}")
    print(f"  Nodes created:           {result.nodes_created}")
    print(f"  Nodes expanded:          {result.nodes_expanded}")
    print(f"  Runtime:                 {result.runtime_seconds:.4f}s")
    print()

    assert len(result.best_partition) == 5
    assert len(result.best_path) == 4
    assert result.best_partition_labels_str
    assert "a" in result.best_partition_labels_str or "A" in result.best_partition_labels_str
    return result


if __name__ == "__main__":
    r2 = run_k2_test()
    r3 = run_k3_test()
    r5 = run_k5_test()
    print("All real TPM tests passed!")

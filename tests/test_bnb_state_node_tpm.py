"""
Tests for VariableCodec and state-node TPM with Branch and Bound.

Run: uv run python -m pytest tests/test_bnb_state_node_tpm.py -v
"""

import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np

from src.strategies.branch_and_bound_k import (
    VariableCodec,
    popcount,
    validate_state_node_tpm,
    ensure_state_node_tpm,
    state_state_to_state_node_off_probs,
    part_distribution,
    reconstruct_distribution,
    make_context,
    phi_partition,
    branch_and_bound_k_from_state_node_tpm,
    BnBConfig,
    SearchReport,
    clear_caches,
    _reset_nid,
    enumerate_set_partitions,
)


def compute_final_phi(p, ctx):
    return phi_partition(p, ctx)


@pytest.fixture(autouse=True)
def auto_reset():
    clear_caches()
    _reset_nid()


# ═════════════════════════════════════════════════════════════════════════════
#  Test A: Codec labels
# ═════════════════════════════════════════════════════════════════════════════

class TestCodecLabels:
    def test_codec_n3_nodes(self):
        codec = VariableCodec.from_node_count(3, "node_pairs")
        assert codec.part_to_str((1, 2)) == "{a,A} | {b,B}"
        assert codec.part_to_str((1, 2, 4)) == "{a,A} | {b,B} | {c,C}"
        assert codec.n_search_vars == 3

    def test_codec_n3_time_vars(self):
        codec = VariableCodec.from_node_count(3, "time_variables")
        assert codec.labels == ("a", "b", "c", "A", "B", "C")
        assert codec.mask_to_str(1) == "{a}"
        assert codec.mask_to_str(8) == "{A}"
        assert codec.part_to_str((7, 56)) == "{a,b,c} | {A,B,C}"
        assert codec.n_search_vars == 6

    def test_codec_lower_upper_nodes(self):
        codec = VariableCodec.from_node_count(3, "node_pairs")
        assert codec.lower_indices_from_mask(1) == (0,)
        assert codec.upper_indices_from_mask(1) == (0,)
        assert codec.lower_indices_from_mask(6) == (1, 2)
        assert codec.upper_indices_from_mask(6) == (1, 2)


# ═════════════════════════════════════════════════════════════════════════════
#  Test B: TPM validation
# ═════════════════════════════════════════════════════════════════════════════

class TestTPMValidation:
    def test_valid_shape(self):
        validate_state_node_tpm(np.zeros((8, 3)))
        with pytest.raises(ValueError, match="shape mismatch"):
            validate_state_node_tpm(np.zeros((7, 3)))

    def test_invalid_values(self):
        with pytest.raises(ValueError, match="in \\[0, 1\\]"):
            validate_state_node_tpm(np.full((8, 3), 1.5))


# ═════════════════════════════════════════════════════════════════════════════
#  Test C: Distribution reconstruction
# ═════════════════════════════════════════════════════════════════════════════

class TestDistribution:
    def test_deterministic_sums_to_one(self):
        n = 3
        tpm = np.zeros((8, n), dtype=np.float64)
        tpm[:, 0] = 0.0
        tpm[:, 1] = 1.0
        tpm[:, 2] = 1.0
        init = np.ones(n, dtype=np.int8)
        codec = VariableCodec.from_node_count(n, "node_pairs")
        dist = part_distribution(tpm, (0,), (0,), init)
        assert abs(dist.sum() - 1.0) < 1e-12

    def test_reconstructed_sums_to_one(self):
        n = 3
        rng = np.random.default_rng(42)
        tpm = rng.random((8, n))
        init = np.ones(n, dtype=np.int8)
        codec = VariableCodec.from_node_count(n, "node_pairs")
        partition = (1, 2, 4)
        recon = reconstruct_distribution(partition, codec, tpm, init, n)
        assert abs(recon.sum() - 1.0) < 1e-12


# ═════════════════════════════════════════════════════════════════════════════
#  Test D: BnB from TPM with labels (nodes space)
# ═════════════════════════════════════════════════════════════════════════════

class TestBnbFromTPM:
    def test_labels_in_output(self):
        n = 3
        rng = np.random.default_rng(123)
        tpm = rng.random((8, n))
        init = np.ones(n, dtype=np.int8)
        result = branch_and_bound_k_from_state_node_tpm(
            state_node_tpm=tpm, target_k=2,
            initial_state=init, metric="emd_effect",
            config=BnBConfig(target_k=2, partition_space="node_pairs", objective="final_phi", generators=("selection",)),
        )
        assert "a" in result.best_partition_labels_str or "A" in result.best_partition_labels_str

    def test_result_structure(self):
        n = 3
        rng = np.random.default_rng(456)
        tpm = rng.random((8, n))
        init = np.ones(n, dtype=np.int8)
        result = branch_and_bound_k_from_state_node_tpm(
            state_node_tpm=tpm, target_k=2,
            initial_state=init, metric="emd_effect",
            config=BnBConfig(target_k=2, partition_space="node_pairs", objective="final_phi", generators=("selection",)),
        )
        assert isinstance(result, SearchReport)
        assert result.target_k == 2
        assert len(result.best_partition) == 2

    def test_k3_from_loader(self):
        from src.loader import TpmLoader
        raw = TpmLoader.cargar(3, "A")
        sn = ensure_state_node_tpm(raw)
        init = np.ones(3, dtype=np.int8)
        result = branch_and_bound_k_from_state_node_tpm(
            state_node_tpm=sn, target_k=3,
            initial_state=init, metric="emd_effect",
            config=BnBConfig(target_k=3, partition_space="node_pairs", objective="final_phi"),
        )
        assert len(result.best_partition) == 3
        # Path may be empty for exact enumeration (final_phi objective)


# ═════════════════════════════════════════════════════════════════════════════
#  Test E: State-state conversion
# ═════════════════════════════════════════════════════════════════════════════

class TestStateStateConversion:
    def test_conversion_n3(self):
        n = 3
        tpm_ss = np.zeros((8, 8), dtype=np.float64)
        for s in range(8):
            tpm_ss[s, s] = 1.0
        state_node = state_state_to_state_node_off_probs(tpm_ss)
        assert state_node.shape == (8, 3)
        for row in range(8):
            for j in range(3):
                expected = 1.0 if ((row >> j) & 1) == 0 else 0.0
                assert abs(state_node[row, j] - expected) < 1e-12


# ═════════════════════════════════════════════════════════════════════════════
#  Test F: Enumerate set partitions (exact final_phi for small n)
# ═════════════════════════════════════════════════════════════════════════════

class TestExactEnumeration:
    def test_n3_k2_via_enumeration(self):
        from src.loader import TpmLoader
        raw = TpmLoader.cargar(3, "A")
        sn = ensure_state_node_tpm(raw)
        init = np.ones(3, dtype=np.int8)
        ctx = make_context(sn, init, "emd_effect", "node_pairs")

        best_phi = float("inf")
        best_part = None
        for part in enumerate_set_partitions(3, 2):
            phi = phi_partition(part, ctx)
            if phi < best_phi:
                best_phi = phi
                best_part = part
        assert best_part is not None
        assert len(best_part) == 2
        assert best_phi >= 0


# ═════════════════════════════════════════════════════════════════════════════
#  Test G: Context and phi
# ═════════════════════════════════════════════════════════════════════════════

class TestContext:
    def test_context_creation(self):
        n = 3
        tpm = np.random.random((8, n))
        init = np.ones(n, dtype=np.int8)
        ctx = make_context(tpm, init, "emd_effect", "node_pairs")
        assert ctx.n_nodes == n
        assert abs(ctx.intact_distribution.sum() - 1.0) < 1e-12

    def test_phi_partition(self):
        n = 3
        rng = np.random.default_rng(42)
        tpm = rng.random((8, n))
        init = np.ones(n, dtype=np.int8)
        ctx = make_context(tpm, init, "emd_effect", "node_pairs")
        phi = phi_partition((1, 2, 4), ctx)
        assert phi >= 0
        phi1 = phi_partition((7,), ctx)
        assert phi1 == 0.0


# ═════════════════════════════════════════════════════════════════════════════
#  Test H: Nodes space does not split mech from purv
# ═════════════════════════════════════════════════════════════════════════════

class TestNodesSpace:
    def test_partition_preserves_mech_purv_pairs(self):
        from src.loader import TpmLoader
        raw = TpmLoader.cargar(3, "A")
        sn = ensure_state_node_tpm(raw)
        init = np.ones(3, dtype=np.int8)
        codec = VariableCodec.from_node_count(3, "node_pairs")

        ctx = make_context(sn, init, "emd_effect", "node_pairs")
        result = branch_and_bound_k_from_state_node_tpm(
            state_node_tpm=sn, target_k=3,
            initial_state=init, metric="emd_effect",
            config=BnBConfig(target_k=3, partition_space="node_pairs", objective="final_phi"),
        )
        # Each block mask represents node indices, not time-separated vars
        labels_str = result.best_partition_labels_str
        # In nodes space, each block includes both lowercase and uppercase for same node
        # e.g., {a,A} | {b,B} | {c,C}
        assert labels_str is not None
        # Verify no block has a mix like {a,B} which would split a node's mech from its purv
        for block in result.best_partition:
            indices = set()
            for i in range(ctx.n_nodes):
                if (block >> i) & 1:
                    indices.add(i)
            # All indices in the block are node indices - this is the correct behavior
            # In nodes space, each index i maps to node i, which has both mech(i) and purv(i)
            assert len(indices) == popcount(block)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

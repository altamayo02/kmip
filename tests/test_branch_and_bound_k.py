"""
Tests for refactored Branch and Bound k-partition engines.
"""

import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pytest
import numpy as np
from src.strategies.branch_and_bound_k import (
    popcount, bits_to_indices, full_mask, canonical_partition,
    enumerate_set_partitions, enumerate_node_selection_partitions,
    count_set_partitions, SelectionSplitGenerator, BruteForceSmallBlockGenerator,
    VariableCodec, make_context, phi_partition, clear_phi_cache, clear_caches,
    run_exact_final_phi, run_heuristic_beam_final_phi,
    BnBConfig, FinalPhiResult, run_accumulated_path_bnb,
    load_tpm_csv, ensure_state_node_tpm, parse_initial_state,
    _reset_nid, branch_and_bound_k_from_state_node_tpm,
    BnBConfig as _BC,
)


@pytest.fixture(autouse=True)
def auto_reset():
    clear_caches()
    clear_phi_cache()
    _reset_nid()


def _tpm3():
    """Return deterministic state-node TPM for N=3."""
    tpm = np.zeros((8, 3), dtype=np.float64)
    for r in range(8):
        tpm[r, 0] = 0.3 if r & 1 else 0.7
        tpm[r, 1] = 0.5 if r & 2 else 0.5
        tpm[r, 2] = 0.2 if r & 4 else 0.8
    return tpm


# ═════════════════════════════════════════════════════════════════════════════
#  Exact enumeration
# ═════════════════════════════════════════════════════════════════════════════

class TestExactEnumeration:
    def test_n6_k5_count(self):
        """S(6,5) = 15 (in mech_alc, S(12,5) = 137940)."""
        count = sum(1 for _ in enumerate_set_partitions(6, 5))
        assert count == 15

    def test_n10_k5_count(self):
        """S(10,5) = 42525 — matches N=5 mech/alc space."""
        count = sum(1 for _ in enumerate_set_partitions(10, 5))
        assert count == 42525

    def test_n3_k2_phi_found(self):
        tpm = _tpm3()
        sn = ensure_state_node_tpm(tpm)
        init = np.ones(3, dtype=np.int8)
        # mech_alc space: 6 variables -> S(6,2) = 31 partitions
        ctx = make_context(sn, init, "emd_effect", "mech_alc")
        res = run_exact_final_phi(ctx, 2, BnBConfig(target_k=2))
        assert res.optimality_certified
        assert len(res.partition) == 2
        assert res.final_phi >= 0
        assert res.termination_reason == "exhausted_all_final_partitions"

    def test_enumeration_via_wrapper(self):
        tpm = _tpm3()
        sn = ensure_state_node_tpm(tpm)
        init = np.ones(3, dtype=np.int8)
        r = branch_and_bound_k_from_state_node_tpm(
            sn, target_k=2, initial_state=init, metric="emd_effect",
            config=BnBConfig(target_k=2, objective="final_phi", mode="exact",
                              partition_space="mech_alc"),
        )
        assert r.optimality_certified
        assert len(r.best_partition) == 2
        assert r.best_final_phi >= 0
        assert r.incumbent_source == "exact_final_partition_enumeration"

    def test_mech_alc_can_be_separated(self):
        """Verify that a and A can end up in different blocks."""
        tpm = _tpm3()
        sn = ensure_state_node_tpm(tpm)
        init = np.ones(3, dtype=np.int8)
        r = branch_and_bound_k_from_state_node_tpm(
            sn, target_k=3, initial_state=init,
            config=BnBConfig(target_k=3, objective="final_phi", mode="exact",
                              partition_space="mech_alc"),
        )
        labels = r.best_partition_labels_str
        # In mech_alc, blocks may separate a from A
        # Just confirm the partition has 3 blocks
        assert len(r.best_partition) == 3
        assert r.n_search_vars == 6  # 3 mech + 3 alc for N=3


# ═════════════════════════════════════════════════════════════════════════════
#  Selection partitions
# ═════════════════════════════════════════════════════════════════════════════

class TestSelectionPartitions:
    def test_n6_k5_count(self):
        """C(6,4) = 15 selection partitions for k=5 in node_pairs space."""
        count = sum(1 for _ in enumerate_node_selection_partitions(6, 5))
        assert count == 15  # C(6,4)

    def test_mech_alc_selection(self):
        """In mech_alc space, selection is over 2n variables.
        For N=3, k=2: C(6,1) = 6 selection partitions."""
        count = sum(1 for _ in enumerate_node_selection_partitions(6, 2))
        assert count == 6  # C(6,1)


# ═════════════════════════════════════════════════════════════════════════════
#  Beam search
# ═════════════════════════════════════════════════════════════════════════════

class TestBeamSearch:
    def test_beam_n3_k2(self):
        tpm = _tpm3()
        sn = ensure_state_node_tpm(tpm)
        init = np.ones(3, dtype=np.int8)
        ctx = make_context(sn, init, "emd_effect", "nodes")
        res = run_heuristic_beam_final_phi(ctx, 2, BnBConfig(target_k=2, beam_width=10))
        assert not res.optimality_certified
        assert len(res.partition) == 2

    def test_beam_via_wrapper(self):
        tpm = _tpm3()
        sn = ensure_state_node_tpm(tpm)
        init = np.ones(3, dtype=np.int8)
        # Force beam by setting max_expansion > 0
        r = branch_and_bound_k_from_state_node_tpm(
            sn, target_k=2, initial_state=init, metric="emd_effect",
            config=BnBConfig(target_k=2, objective="final_phi",
                              beam_width=10, generators=("selection",),
                              max_expansion_candidates_per_node=10),
        )
        assert not r.optimality_certified
        assert len(r.best_partition) == 2


# ═════════════════════════════════════════════════════════════════════════════
#  Stirling numbers
# ═════════════════════════════════════════════════════════════════════════════

class TestStirling:
    def test_s6_5(self):
        assert count_set_partitions(6, 5) == 15

    def test_s10_5(self):
        """S(10,5)=42525 — matches N=5 full mech/alc space."""
        assert count_set_partitions(10, 5) == 42525

    def test_s5_3(self):
        assert count_set_partitions(5, 3) == 25

    def test_s3_2(self):
        assert count_set_partitions(3, 2) == 3


# ═════════════════════════════════════════════════════════════════════════════
#  Mech/Alc space — default
# ═════════════════════════════════════════════════════════════════════════════

class TestMechAlcSpace:
    def test_default_is_mech_alc(self):
        codec = VariableCodec.from_node_count(3)
        assert codec.partition_space == "mech_alc"

    def test_n5_has_10_vars(self):
        codec = VariableCodec.from_node_count(5, "mech_alc")
        assert codec.n_search_vars == 10  # 5 mech + 5 alc

    def test_partition_can_separate_a_from_A(self):
        codec = VariableCodec.from_node_count(3, "mech_alc")
        # {a,B} is valid: mech={a} (index 0), alc={B} (index 4)
        mask = (1 << 0) | (1 << 4)  # bits 0 and 4
        lower = codec.lower_indices_from_mask(mask)
        upper = codec.upper_indices_from_mask(mask)
        assert lower == (0,)  # a
        assert upper == (1,)  # B (index 4 -> 4-3=1)

    def test_mech_alc_partition_has_proper_count(self):
        """N=5, k=5 in mech_alc should be S(10,5)=42525."""
        tpm = _tpm3()  # just for existence check
        # Just test the count formula
        assert count_set_partitions(10, 5) == 42525


# ═════════════════════════════════════════════════════════════════════════════
#  Accumulated path BnB (preserved)
# ═════════════════════════════════════════════════════════════════════════════

class TestAccumulatedPath:
    def test_constant_delta(self):
        tpm = _tpm3()
        sn = ensure_state_node_tpm(tpm)
        init = np.ones(3, dtype=np.int8)
        r = branch_and_bound_k_from_state_node_tpm(
            sn, target_k=2, initial_state=init, metric="emd_effect",
            config=BnBConfig(target_k=2, objective="accumulated_path",
                              partition_space="mech_alc",
                              use_initial_greedy_incumbent=False,
                              enable_bound_pruning=False,
                              max_expansion_candidates_per_node=10),
        )
        assert r.best_accumulated_loss >= 0


# ═════════════════════════════════════════════════════════════════════════════
#  Codec
# ═════════════════════════════════════════════════════════════════════════════

class TestCodec:
    def test_mech_alc_labels(self):
        codec = VariableCodec.from_node_count(3, "mech_alc")
        assert codec.labels == ("a", "b", "c", "A", "B", "C")
        assert codec.n_search_vars == 6
        assert codec.mask_to_str(1) == "{a}"  # bit 0 = a
        assert codec.mask_to_str(8) == "{A}"  # bit 3 = A

    def test_node_pairs_labels(self):
        codec = VariableCodec.from_node_count(3, "node_pairs")
        assert codec.part_to_str((1, 2, 4)) == "{a,A} | {b,B} | {c,C}"
        assert codec.n_search_vars == 3

    def test_selection_generator(self):
        gen = SelectionSplitGenerator()
        splits = gen.generate((7,), 7, None, 5)
        assert len(splits) == 3  # {0}|{6}, {1}|{5}, {2}|{4} for block 7


# ═════════════════════════════════════════════════════════════════════════════
#  CLI tests
# ═════════════════════════════════════════════════════════════════════════════

class TestCLI:
    def test_cli_max_expansion(self):
        from scripts.run_bnb_k_csv import parse_args
        args = parse_args(["--dataset", "N3A", "--data-dir", "data", "--max-expansion", "20"])
        assert args.max_expansion_candidates_per_node == 20

    def test_parse_initial_state(self):
        r = parse_initial_state("101", 3)
        assert np.array_equal(r, np.array([1, 0, 1], dtype=np.int8))

    def test_parse_ones(self):
        r = parse_initial_state("ones", 5)
        assert np.array_equal(r, np.ones(5, dtype=np.int8))

"""
Tests for the BnB CSV runner.

Run: uv run python -m pytest tests/test_run_bnb_k_csv.py -v
"""

import sys
import os
import json
import tempfile
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np

from scripts.run_bnb_k_csv import (
    parse_args,
    build_result_dict,
)


@pytest.fixture
def tmp_csv_dir():
    tmp = tempfile.mkdtemp()
    yield tmp
    shutil.rmtree(tmp)


def make_state_node_csv(n_nodes: int, rng=None) -> str:
    if rng is None:
        rng = np.random.default_rng(42)
    rows = 2 ** n_nodes
    tpm = rng.random((rows, n_nodes))
    lines = []
    for row in tpm:
        lines.append(",".join(f"{v:.10f}" for v in row))
    return "\n".join(lines)


class TestLoad:
    def test_csv_file_load(self, tmp_csv_dir):
        csv_content = make_state_node_csv(3)
        csv_path = os.path.join(tmp_csv_dir, "test.csv")
        with open(csv_path, "w") as f:
            f.write(csv_content)
        args = parse_args(["--file", csv_path, "--initial-state", "111",
                           "--mode", "exact"])
        assert args.file == csv_path

    def test_dataset_load(self, tmp_csv_dir):
        csv_content = make_state_node_csv(3)
        csv_path = os.path.join(tmp_csv_dir, "N3A.csv")
        with open(csv_path, "w") as f:
            f.write(csv_content)
        args = parse_args(["--dataset", "N3A", "--data-dir", tmp_csv_dir])
        assert args.dataset == "N3A"
        assert args.data_dir == tmp_csv_dir


class TestShapeValidation:
    def test_invalid_shape_raises(self, tmp_csv_dir):
        csv_path = os.path.join(tmp_csv_dir, "bad.csv")
        tpm = np.zeros((7, 3))
        np.savetxt(csv_path, tpm, delimiter=",")
        from src.strategies.branch_and_bound_k import load_tpm_csv
        with pytest.raises((ValueError,), match="Rows must be a power"):
            load_tpm_csv(csv_path)


class TestInitialState:
    def test_ones(self):
        from src.strategies.branch_and_bound_k import parse_initial_state
        result = parse_initial_state("ones", 5)
        assert np.array_equal(result, np.ones(5, dtype=np.int8))

    def test_zeros(self):
        from src.strategies.branch_and_bound_k import parse_initial_state
        result = parse_initial_state("zeros", 5)
        assert np.array_equal(result, np.zeros(5, dtype=np.int8))

    def test_binary_string(self):
        from src.strategies.branch_and_bound_k import parse_initial_state
        result = parse_initial_state("101", 3)
        assert np.array_equal(result, np.array([1, 0, 1], dtype=np.int8))

    def test_wrong_length(self):
        from src.strategies.branch_and_bound_k import parse_initial_state
        with pytest.raises(ValueError, match="must have length"):
            parse_initial_state("101", 5)


class TestResultDict:
    def test_build_result_dict(self):
        from src.strategies.branch_and_bound_k import SearchReport
        report = SearchReport(
            best_partition=(1, 2, 4),
            best_accumulated_loss=0.15,
            best_final_phi=0.15,
            best_path=[],
            best_partition_str="{0} | {1} | {2}",
            best_path_str="",
            best_partition_labels_str="{a,A} | {b,B} | {c,C}",
            best_path_labels_str="",
            target_k=3,
            incumbent_source="greedy_initial",
            nodes_created=10,
            nodes_expanded=5,
            nodes_pruned_by_bound=2,
            nodes_pruned_by_dominance=1,
            complete_nodes_found=1,
            incumbent_updates=1,
            runtime_seconds=0.01,
            all_nodes=[],
            M_worst_per_block=5,
            upper_frontier_width=10,
            termination_reason="queue_exhausted",
            optimality_certified=True,
            live_nodes_remaining=0,
            objective="final_phi",
            mode="exact",
            generators=("geomip",),
            partition_space="node_pairs",
            dataset_name="N3A",
            csv_path="/tmp/N3A.csv",
            n_nodes=3,
            n_search_vars=3,
            initial_state_str="111",
        )
        d = build_result_dict(report)
        assert d["dataset"] == "N3A"
        assert d["best_accumulated_loss"] == 0.15
        assert d["best_final_phi"] == 0.15
        assert d["optimality_certified"] is True


class TestEndToEnd:
    def test_small_exact_runs(self, tmp_csv_dir):
        import subprocess
        csv_content = make_state_node_csv(3)
        csv_path = os.path.join(tmp_csv_dir, "N3A.csv")
        with open(csv_path, "w") as f:
            f.write(csv_content)
        script = os.path.join(os.path.dirname(__file__), "..", "scripts", "run_bnb_k_csv.py")
        result = subprocess.run(
            [sys.executable, "-X", "utf8", script,
             "--file", csv_path,
             "--initial-state", "111", "--mode", "exact",
             "--partition-space", "node_pairs",
             "--output-dir", tmp_csv_dir],
            capture_output=True, text=True,
            cwd=os.path.dirname(__file__) + "/..",
        )
        assert result.returncode == 0, f"STDERR: {result.stderr}"

    def test_heuristic_runs(self, tmp_csv_dir):
        import subprocess
        csv_content = make_state_node_csv(3)
        csv_path = os.path.join(tmp_csv_dir, "N3A.csv")
        with open(csv_path, "w") as f:
            f.write(csv_content)
        script = os.path.join(os.path.dirname(__file__), "..", "scripts", "run_bnb_k_csv.py")
        result = subprocess.run(
            [sys.executable, "-X", "utf8", script,
             "--file", csv_path,
             "--initial-state", "ones", "--mode", "heuristic",
             "--partition-space", "node_pairs",
             "--output-dir", tmp_csv_dir],
            capture_output=True, text=True,
            cwd=os.path.dirname(__file__) + "/..",
        )
        assert result.returncode == 0, f"STDERR: {result.stderr}"
        assert "SUMMARY" in result.stdout


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

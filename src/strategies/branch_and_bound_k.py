"""
Branch and Bound for k-partitions.

Three independent engines:
  - exact_final_phi: enumerate all set partitions, pick min phi(P_k)
  - heuristic_beam_final_phi: beam search evaluating phi(P_current) at each step
  - accumulated_path_bnb: tree search minimizing sum of incremental phis

Partition spaces:
  - "nodes" (default): blocks are sets of nodes (mech+purv paired per node)
  - "time_variables": each time-indexed variable is independent
"""

import heapq
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Optional

import numpy as np


# ═════════════════════════════════════════════════════════════════════════════
#  Bitmask utilities
# ═════════════════════════════════════════════════════════════════════════════

def popcount(mask: int) -> int:
    return mask.bit_count()


def min_bit_index(mask: int) -> int:
    return (mask & -mask).bit_length() - 1


def bits_to_indices(mask: int) -> tuple[int, ...]:
    return tuple(i for i in range(mask.bit_length()) if mask >> i & 1)


def indices_to_mask(indices: tuple[int, ...]) -> int:
    m = 0
    for i in indices:
        m |= 1 << i
    return m


def full_mask(n: int) -> int:
    return (1 << n) - 1


def canonical_partition(partition: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(sorted(partition, key=min_bit_index))


def apply_split(parent_partition: tuple[int, ...], block_mask: int,
                left_mask: int, right_mask: int) -> tuple[int, ...]:
    new_blocks: list[int] = []
    for b in parent_partition:
        if b == block_mask:
            new_blocks.append(left_mask)
            new_blocks.append(right_mask)
        else:
            new_blocks.append(b)
    return canonical_partition(tuple(new_blocks))


def state_bit(state: int, idx: int) -> int:
    return (state >> idx) & 1


def row_matches_mech(row_index: int, mech_indices: tuple[int, ...],
                     initial_state: np.ndarray) -> bool:
    for idx in mech_indices:
        if state_bit(row_index, idx) != int(initial_state[idx]):
            return False
    return True


# ═════════════════════════════════════════════════════════════════════════════
#  VariableCodec
# ═════════════════════════════════════════════════════════════════════════════

EMPTY_SYM = "\u2205"


@dataclass(frozen=True)
class VariableCodec:
    labels: tuple[str, ...]
    n_mech: int
    n_purv: int
    partition_space: str = "mech_alc"

    @classmethod
    def from_node_count(cls, n_nodes: int,
                        partition_space: str = "mech_alc") -> "VariableCodec":
        lower = tuple(chr(ord("a") + i) for i in range(n_nodes))
        upper = tuple(chr(ord("A") + i) for i in range(n_nodes))
        if partition_space == "node_pairs":
            labels = tuple(f"{l},{u}" for l, u in zip(lower, upper))
            return cls(labels=labels, n_mech=n_nodes, n_purv=n_nodes,
                       partition_space="node_pairs")
        # mech_alc (default): all lowercase then all uppercase
        return cls(labels=lower + upper, n_mech=n_nodes, n_purv=n_nodes,
                   partition_space="mech_alc")

    def label(self, index: int) -> str:
        return self.labels[index]

    def mask_to_labels(self, mask: int) -> tuple[str, ...]:
        return tuple(self.labels[i] for i in bits_to_indices(mask))

    def mask_to_str(self, mask: int) -> str:
        if mask == 0:
            return EMPTY_SYM
        return "{" + ",".join(self.labels[i] for i in bits_to_indices(mask)) + "}"

    def part_to_str(self, partition: tuple[int, ...]) -> str:
        return " | ".join(self.mask_to_str(b) for b in partition)

    def lower_indices_from_mask(self, mask: int) -> tuple[int, ...]:
        if self.partition_space == "node_pairs":
            return bits_to_indices(mask)
        return tuple(i for i in bits_to_indices(mask) if i < self.n_mech)

    def upper_indices_from_mask(self, mask: int) -> tuple[int, ...]:
        if self.partition_space == "node_pairs":
            return bits_to_indices(mask)
        return tuple(i - self.n_mech for i in bits_to_indices(mask) if i >= self.n_mech)

    @property
    def n_search_vars(self) -> int:
        if self.partition_space == "node_pairs":
            return self.n_mech
        return len(self.labels)


def mask_to_str(mask: int, codec: Optional[VariableCodec] = None) -> str:
    if codec is not None:
        return codec.mask_to_str(mask)
    if mask == 0:
        return EMPTY_SYM
    return "{" + ",".join(str(i) for i in bits_to_indices(mask)) + "}"


def part_to_str(partition: tuple[int, ...],
                codec: Optional[VariableCodec] = None) -> str:
    return " | ".join(mask_to_str(m, codec) for m in partition)


# ═════════════════════════════════════════════════════════════════════════════
#  Set partition enumeration
# ═════════════════════════════════════════════════════════════════════════════

def enumerate_set_partitions(elements: int, k: int) -> Iterator[tuple[int, ...]]:
    if k == 1:
        yield (full_mask(elements),)
        return
    if elements < k:
        return
    if elements == k:
        yield tuple(1 << i for i in range(elements))
        return
    elems = list(range(elements))

    def _rec(remaining, blocks):
        if not remaining:
            if len(blocks) == k:
                masks = tuple(indices_to_mask(tuple(sorted(b))) for b in blocks)
                yield canonical_partition(masks)
            return
        e = remaining[0]
        rest = remaining[1:]
        for i in range(len(blocks)):
            copy = [b[:] for b in blocks]
            copy[i].append(e)
            yield from _rec(rest, copy)
        if len(blocks) < k:
            yield from _rec(rest, blocks + [[e]])

    yield from _rec(elems, [])


def enumerate_node_selection_partitions(n_nodes: int, k: int) -> Iterator[tuple[int, ...]]:
    """Generate all partitions where k-1 nodes are singletons and the rest form the last block.
    Complexity: C(n_nodes, k-1)."""
    from itertools import combinations
    if k > n_nodes:
        return
    if k == 1:
        yield (full_mask(n_nodes),)
        return
    for singletons in combinations(range(n_nodes), k - 1):
        singleton_masks = [1 << i for i in singletons]
        rest_mask = full_mask(n_nodes)
        for m in singleton_masks:
            rest_mask ^= m
        if rest_mask:
            yield canonical_partition(tuple(singleton_masks) + (rest_mask,))


# ═════════════════════════════════════════════════════════════════════════════
#  Split generators
# ═════════════════════════════════════════════════════════════════════════════

_SplitCache: dict[int, list[tuple[int, int, int]]] = {}


def _cached_splits(block_mask: int) -> list[tuple[int, int, int]]:
    if block_mask in _SplitCache:
        return _SplitCache[block_mask]
    if popcount(block_mask) < 2:
        _SplitCache[block_mask] = []
        return []
    fixed = 1 << min_bit_index(block_mask)
    rest = block_mask ^ fixed
    result: list[tuple[int, int, int]] = []
    sub = 0
    while True:
        if sub != rest:
            left = fixed | sub
            right = block_mask ^ left
            result.append((block_mask, left, right))
        if sub == rest:
            break
        sub = (sub - rest) & rest
    _SplitCache[block_mask] = result
    return result


@dataclass
class CandidateSplit:
    block_mask: int
    left_mask: int
    right_mask: int
    source: str = ""


class SplitGenerator(ABC):
    name: str = ""

    @abstractmethod
    def generate(self, partition: tuple[int, ...], block_mask: int,
                 context: "StateNodeTPMContext", top_l: int) -> list[CandidateSplit]:
        ...


class SelectionSplitGenerator(SplitGenerator):
    """Generate node-pair splits: {i} | B-{i} for each i in block."""

    name = "selection"

    def generate(self, partition, block_mask, context, top_l):
        result = []
        indices = bits_to_indices(block_mask)
        for i in indices:
            left = 1 << i
            right = block_mask ^ left
            if left and right:
                result.append(CandidateSplit(block_mask, left, right, "selection"))
        return result


class BruteForceSmallBlockGenerator(SplitGenerator):
    """All canonical splits for small blocks (popcount <= 20)."""

    name = "bruteforce"

    def generate(self, partition, block_mask, context, top_l):
        result = []
        for bm, left, right in _cached_splits(block_mask):
            result.append(CandidateSplit(bm, left, right, "bruteforce"))
        return result


# ═════════════════════════════════════════════════════════════════════════════
#  TPM distribution
# ═════════════════════════════════════════════════════════════════════════════

def validate_state_node_tpm(tpm: np.ndarray) -> None:
    if tpm.ndim != 2:
        raise ValueError(f"TPM must be 2D, got shape {tpm.shape}")
    rows, cols = tpm.shape
    expected = 2 ** cols
    if rows != expected:
        raise ValueError(f"TPM shape mismatch: got ({rows}, {cols}), expected ({expected}, {cols})")
    if np.any(tpm < -1e-12) or np.any(tpm > 1 + 1e-12):
        raise ValueError(f"TPM values must be in [0, 1], got range [{tpm.min():.4f}, {tpm.max():.4f}]")


def state_state_to_state_node_off_probs(tpm_ss: np.ndarray) -> np.ndarray:
    rows, cols = tpm_ss.shape
    if rows != cols:
        raise ValueError(f"State-state TPM must be square, got {tpm_ss.shape}")
    n = int(np.log2(rows))
    if 2 ** n != rows:
        raise ValueError(f"Rows must be a power of 2, got {rows}")
    result = np.zeros((rows, n), dtype=np.float64)
    for row in range(rows):
        for ns in range(cols):
            p = float(tpm_ss[row, ns])
            if p == 0.0:
                continue
            for j in range(n):
                if ((ns >> j) & 1) == 0:
                    result[row, j] += p
    return result


def ensure_state_node_tpm(tpm: np.ndarray) -> np.ndarray:
    if tpm.ndim != 2:
        raise ValueError(f"TPM must be 2D, got shape {tpm.shape}")
    rows, cols = tpm.shape
    n = int(np.log2(rows))
    if 2 ** n != rows:
        raise ValueError(f"Rows must be a power of 2, got {rows}")
    if cols == n:
        validate_state_node_tpm(tpm)
        return tpm.astype(np.float64, copy=False)
    if cols == rows:
        return state_state_to_state_node_off_probs(tpm)
    raise ValueError(f"Cannot interpret TPM shape {tpm.shape}: expected (2^n, n) or (2^n, 2^n)")


def part_distribution(state_node_tpm: np.ndarray,
                      mech_indices: tuple[int, ...],
                      purv_indices: tuple[int, ...],
                      initial_state: np.ndarray) -> np.ndarray:
    n_nodes = state_node_tpm.shape[1]
    if len(mech_indices) == 0:
        row_avg = np.mean(state_node_tpm, axis=0)
    else:
        compatible = [r for r in range(state_node_tpm.shape[0])
                      if row_matches_mech(r, mech_indices, initial_state)]
        row_avg = np.mean(state_node_tpm[compatible], axis=0) if compatible else np.mean(state_node_tpm, axis=0)
    if len(purv_indices) == 0:
        return np.array([1.0])
    dist = np.array([1.0])
    for j in purv_indices:
        p0 = float(row_avg[j])
        dist = np.outer(dist, np.array([p0, 1.0 - p0])).ravel()
    return dist / dist.sum()


def reconstruct_distribution(partition: tuple[int, ...], codec: VariableCodec,
                             state_node_tpm: np.ndarray,
                             initial_state: np.ndarray, n_nodes: int) -> np.ndarray:
    part_dists: list[tuple[np.ndarray, tuple[int, ...]]] = []
    for block in partition:
        mech = codec.lower_indices_from_mask(block)
        purv = codec.upper_indices_from_mask(block)
        d = part_distribution(state_node_tpm, mech, purv, initial_state)
        part_dists.append((d, purv))
    gs = 2 ** n_nodes

    # Vectorized reconstruction: build global_state -> block index mapping
    recon = np.ones(gs, dtype=np.float64)
    for d, purv in part_dists:
        if not purv:
            continue
        # For each global state, compute the index into this block's distribution
        n_p = len(purv)
        # Build integer index for each global state
        idx = np.zeros(gs, dtype=np.int64)
        for k, pidx in enumerate(purv):
            bit = (np.arange(gs) >> pidx) & 1
            idx |= bit << k
        recon *= d[idx]
    return recon / recon.sum()


# ═════════════════════════════════════════════════════════════════════════════
#  StateNodeTPMContext
# ═════════════════════════════════════════════════════════════════════════════

_PhiCache: dict[tuple, float] = {}


def clear_phi_cache():
    _PhiCache.clear()


@dataclass
class StateNodeTPMContext:
    state_node_tpm: np.ndarray
    initial_state: np.ndarray
    codec: VariableCodec
    intact_distribution: np.ndarray
    metric: Callable[[np.ndarray, np.ndarray], float]
    n_nodes: int


def make_context(state_node_tpm: np.ndarray,
                 initial_state: Optional[np.ndarray] = None,
                 metric: str = "emd_effect",
                 partition_space: str = "nodes") -> StateNodeTPMContext:
    validate_state_node_tpm(state_node_tpm)
    n = state_node_tpm.shape[1]
    if initial_state is None:
        initial_state = np.zeros(n, dtype=np.int8)
    initial_state = np.asarray(initial_state, dtype=np.int8)
    if initial_state.shape != (n,):
        raise ValueError(f"initial_state must have shape ({n},), got {initial_state.shape}")
    codec = VariableCodec.from_node_count(n, partition_space)
    if metric == "emd_effect":
        from src.functions.emd import emd_efecto
        mfn = emd_efecto
    else:
        raise ValueError(f"Unknown metric: {metric}")
    intact = part_distribution(state_node_tpm, tuple(range(n)), tuple(range(n)), initial_state)
    return StateNodeTPMContext(state_node_tpm, initial_state, codec, intact, mfn, n)


def phi_partition(partition: tuple[int, ...], ctx: StateNodeTPMContext) -> float:
    if partition in _PhiCache:
        return _PhiCache[partition]
    if len(partition) <= 1:
        _PhiCache[partition] = 0.0
        return 0.0
    recon = reconstruct_distribution(partition, ctx.codec, ctx.state_node_tpm,
                                     ctx.initial_state, ctx.n_nodes)
    phi = float(ctx.metric(ctx.intact_distribution, recon))
    _PhiCache[partition] = phi
    return phi


def count_set_partitions(elements: int, k: int) -> int:
    """Stirling numbers of the second kind S(elements, k)."""
    if elements < k:
        return 0
    if k == 1 or elements == k:
        return 1
    total = 0
    for j in range(k + 1):
        sign = 1 if (k - j) % 2 == 0 else -1
        total += sign * math.comb(k, j) * (j ** elements)
    return abs(total) // math.factorial(k)


# ═════════════════════════════════════════════════════════════════════════════
#  Engine 1: Exact enumeration for final_phi
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class FinalPhiResult:
    partition: tuple[int, ...]
    final_phi: float
    accumulated_loss: float
    path: list[dict]
    path_labels_str: str
    nodes_created: int
    nodes_evaluated: int
    runtime: float
    optimality_certified: bool
    termination_reason: str
    incumbent_source: str
    partial_candidates: int = 0
    complete_partitions: int = 0
    frontier_sizes: Optional[dict[int, int]] = None


def run_exact_final_phi(ctx: StateNodeTPMContext, target_k: int,
                        config: "BnBConfig") -> FinalPhiResult:
    n = ctx.codec.n_search_vars
    best_phi = float("inf")
    best_part = None
    evaluated = 0
    start = time.time()
    codec = ctx.codec

    for part in enumerate_set_partitions(n, target_k):
        evaluated += 1
        phi = phi_partition(part, ctx)
        if phi < best_phi - 1e-12:
            best_phi = phi
            best_part = part

    elapsed = time.time() - start
    labels_str = codec.part_to_str(best_part) if best_part else ""

    return FinalPhiResult(
        partition=best_part or (full_mask(n),),
        final_phi=best_phi if best_phi != float("inf") else -1.0,
        accumulated_loss=0.0,
        path=[],
        path_labels_str="",
        nodes_created=evaluated,
        nodes_evaluated=evaluated,
        runtime=elapsed,
        optimality_certified=True,
        termination_reason="exhausted_all_final_partitions",
        incumbent_source="exact_final_partition_enumeration",
    )


# ═════════════════════════════════════════════════════════════════════════════
#  Engine 2a: Direct selection heuristic (k-1 singletons + rest)
# ═════════════════════════════════════════════════════════════════════════════

def run_selection_direct_final_phi(ctx: StateNodeTPMContext, target_k: int,
                                    config: "BnBConfig") -> FinalPhiResult:
    n = ctx.codec.n_search_vars
    codec = ctx.codec
    start = time.time()
    created = 0
    best_phi = float("inf")
    best_part = None

    for sel_part in enumerate_node_selection_partitions(n, target_k):
        created += 1
        phi = phi_partition(sel_part, ctx)
        if phi < best_phi - 1e-12:
            best_phi = phi
            best_part = sel_part

    elapsed = time.time() - start
    labels_str = codec.part_to_str(best_part) if best_part else ""
    if best_part is not None:
        assert len(best_part) == target_k, (
            f"selection_direct: expected {target_k} blocks, got {len(best_part)}"
        )

    return FinalPhiResult(
        partition=best_part or (full_mask(n),),
        final_phi=best_phi if best_phi != float("inf") else -1.0,
        accumulated_loss=0.0,
        path=[],
        path_labels_str="",
        nodes_created=created,
        nodes_evaluated=created,
        runtime=elapsed,
        optimality_certified=False,
        termination_reason="selection_direct_exhausted",
        incumbent_source="selection_direct",
    )


# ═════════════════════════════════════════════════════════════════════════════
#  Engine 2b: Beam search for final_phi (incremental, level by level)
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class _BeamItem:
    partition: tuple[int, ...]
    phi: float

    def __len__(self):
        return len(self.partition)


def run_heuristic_beam_final_phi(ctx: StateNodeTPMContext, target_k: int,
                                  config: "BnBConfig") -> FinalPhiResult:
    n = ctx.codec.n_search_vars
    codec = ctx.codec
    beam_width = getattr(config, "beam_width", 50)
    generator_names = getattr(config, "generators", ("selection",))
    top_l = getattr(config, "top_l_per_generator", 5)
    max_nodes = config.max_nodes
    timeout = config.timeout_seconds
    start = time.time()
    created = 0
    partial_count = 0
    complete_count = 0
    frontier_sizes: dict[int, int] = {}

    gen_registry = {"selection": SelectionSplitGenerator(), "bruteforce": BruteForceSmallBlockGenerator()}
    generators = [gen_registry[name] for name in generator_names if name in gen_registry]

    # Initial incumbent: use root phi=0, then improve via beam.
    # Skip full selection_direct seeding for large n (too many phi evals).
    best_phi = float("inf")
    best_part = None
    incumbent_source = "none"

    # Limited seeding: evaluate at most 100 selection_direct candidates
    max_sel = 100
    sel_count = 0
    for sel_part in enumerate_node_selection_partitions(n, target_k):
        sel_count += 1
        if sel_count > max_sel:
            break
        created += 1
        complete_count += 1
        phi = phi_partition(sel_part, ctx)
        if phi < best_phi - 1e-12:
            best_phi = phi
            best_part = sel_part
            incumbent_source = "selection_direct"

    # Beam search: level by level from k=1 to k=target_k
    root = (full_mask(n),)
    frontier: list[_BeamItem] = []
    root_phi = phi_partition(root, ctx)
    created += 1
    frontier.append(_BeamItem(root, root_phi))
    frontier_sizes[1] = 1

    for level in range(2, target_k + 1):
        # Phase 1: generate all candidates with cheap proxy scores
        raw_candidates: list[tuple[int, ...]] = []
        for item in frontier:
            for block in item.partition:
                if popcount(block) < 2:
                    continue
                for gen in generators:
                    splits = gen.generate(item.partition, block, ctx, top_l)
                    for sp in splits:
                        child = apply_split(item.partition, sp.block_mask, sp.left_mask, sp.right_mask)
                        raw_candidates.append(child)

        if not raw_candidates:
            raise RuntimeError(
                f"Beam search died at level {level}/{target_k}: "
                f"no valid splits from frontier of size {len(frontier)}"
            )

        # Phase 2: deduplicate by partition identity
        seen_unique: dict[tuple[int, ...], bool] = {}
        for child in raw_candidates:
            seen_unique[child] = True
        unique = list(seen_unique.keys())

        # Phase 3: evaluate exact phi for beam_width candidates (the rest get proxy)
        # Score by block size variance as cheap proxy
        scored: list[tuple[float, tuple[int, ...]]] = []
        for child in unique:
            sizes = [popcount(b) for b in child]
            proxy = float(np.std(sizes)) if len(sizes) > 1 else 0.0
            scored.append((proxy, child))
        scored.sort(key=lambda x: x[0])

        # Only compute exact phi for beam_width candidates
        next_frontier_map: dict[tuple[int, ...], float] = {}
        for _, child in scored[:beam_width]:
            child_phi = phi_partition(child, ctx)
            created += 1

            if len(child) == target_k:
                complete_count += 1
                if child_phi < best_phi - 1e-12:
                    best_phi = child_phi
                    best_part = child
                    incumbent_source = f"beam_level_{level}"
            else:
                partial_count += 1
                next_frontier_map[child] = child_phi

            if max_nodes and created >= max_nodes:
                break

        # For remaining candidates, don't compute phi (save time)
        # They are dropped from the frontier

        frontier = [_BeamItem(p, ph) for p, ph in next_frontier_map.items()]
        frontier.sort(key=lambda x: x.phi)
        frontier = frontier[:beam_width]
        frontier_sizes[level] = len(frontier)

        if max_nodes and created >= max_nodes:
            break
        if timeout and (time.time() - start) >= timeout:
            break

        # If no candidates in frontier and not at target_k, crash
        if not frontier and level < target_k:
            raise RuntimeError(
                f"Beam search died at level {level}/{target_k}: "
                f"no valid splits found."
            )

    elapsed = time.time() - start

    # Validate result
    if best_part is not None:
        assert len(best_part) == target_k, (
            f"BEAM SEARCH BUG: expected k={target_k}, got {len(best_part)}. "
            f"Partition: {codec.part_to_str(best_part)}"
        )
    else:
        raise RuntimeError(
            f"Beam search failed to find any complete k={target_k} partition. "
            f"Selection direct found none either."
        )

    labels_str = codec.part_to_str(best_part) if best_part else ""

    if timeout and (time.time() - start) >= timeout:
        term = "timeout"
    elif max_nodes and created >= max_nodes:
        term = "max_nodes"
    else:
        term = "beam_exhausted"

    return FinalPhiResult(
        partition=best_part or (full_mask(n),),
        final_phi=best_phi if best_phi != float("inf") else -1.0,
        accumulated_loss=0.0,
        path=[],
        path_labels_str="",
        nodes_created=created,
        nodes_evaluated=complete_count,
        runtime=elapsed,
        optimality_certified=False,
        termination_reason=term,
        incumbent_source=incumbent_source,
        partial_candidates=partial_count,
        complete_partitions=complete_count,
        frontier_sizes=frontier_sizes,
    )


# ═════════════════════════════════════════════════════════════════════════════
#  Engine 3: Accumulated path BnB (kept original)
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(slots=True)
class SplitStep:
    step: int
    parent_partition: tuple[int, ...]
    child_partition: tuple[int, ...]
    block_mask: int
    left_mask: int
    right_mask: int
    delta_phi: float
    accumulated_loss_after_split: float


@dataclass(slots=True)
class BBNode:
    partition: tuple[int, ...]
    accumulated_loss: float
    lower_bound: float
    depth: int = 0
    expected_loss: Optional[float] = None
    upper_bound: Optional[float] = None
    parent_id: Optional[int] = None
    node_id: int = -1
    path: tuple[SplitStep, ...] = ()
    split_step: Optional[SplitStep] = None
    status: str = "live"
    prune_reason: Optional[str] = None

    @property
    def current_k(self) -> int:
        return len(self.partition)


_node_id_counter: int = 0


def _reset_nid():
    global _node_id_counter
    _node_id_counter = 0


def _next_nid() -> int:
    global _node_id_counter
    nid = _node_id_counter
    _node_id_counter += 1
    return nid


def make_priority(node: BBNode) -> tuple:
    exp = node.expected_loss if node.expected_loss is not None else float("inf")
    return (node.lower_bound, exp, node.accumulated_loss, -node.current_k, node.node_id)


class MetricsTracker:
    __slots__ = ("created", "expanded", "pruned_by_bound", "pruned_by_dominance", "complete_found")
    def __init__(self):
        self.created = 0
        self.expanded = 0
        self.pruned_by_bound = 0
        self.pruned_by_dominance = 0
        self.complete_found = 0


_ExpectedCache: dict[tuple, tuple] = {}
_UpperCache: dict[tuple, float] = {}
_BestCostSeen: dict[tuple, float] = {}
_DeltaCache: dict[tuple, float] = {}


def clear_caches():
    _SplitCache.clear()
    _DeltaCache.clear()
    _ExpectedCache.clear()
    _UpperCache.clear()
    _BestCostSeen.clear()
    _PhiCache.clear()


def run_accumulated_path_bnb(ctx: StateNodeTPMContext, target_k: int,
                              config: "BnBConfig",
                              verbose: bool = False) -> "SearchReport":
    _reset_nid()
    _clear_acc_caches()

    codec = ctx.codec
    n_vars = codec.n_search_vars
    nodes_by_id: dict[int, BBNode] = {}

    root = BBNode(partition=(full_mask(n_vars),), accumulated_loss=0.0, lower_bound=0.0,
                  node_id=_next_nid())
    all_nodes = [root]
    nodes_by_id[root.node_id] = root
    metrics = MetricsTracker()
    metrics.created += 1

    incumbent_part = root.partition
    incumbent_loss = float("inf")
    incumbent_path: tuple[SplitStep, ...] = ()
    incumbent_source = "none"
    incumbent_updates = 0

    def delta_fn(parent, block, left, right):
        child = apply_split(parent, block, left, right)
        return phi_partition(child, ctx)

    # Greedy incumbent
    if config.use_initial_greedy_incumbent:
        tail = _greedy_tail(root, target_k, delta_fn, config)
        if tail[0] and tail[1] < incumbent_loss:
            incumbent_part = tail[1]
            incumbent_loss = tail[2]
            incumbent_source = "greedy_initial"
            incumbent_updates += 1

    root.expected_loss = incumbent_loss if incumbent_loss != float("inf") else None
    _BestCostSeen[root.partition] = 0.0
    pq = []
    heapq.heappush(pq, make_priority(root))
    start = time.time()

    while pq:
        if config.max_nodes and metrics.created >= config.max_nodes:
            break
        if config.timeout_seconds and (time.time() - start) >= config.timeout_seconds:
            break

        prio = heapq.heappop(pq)
        node = nodes_by_id.get(prio[-1])
        if node is None or node.status != "live":
            continue

        if verbose:
            _log_node(node, codec)

        if config.enable_bound_pruning and node.lower_bound >= incumbent_loss - config.epsilon:
            node.status = "pruned"
            node.prune_reason = "LB >= incumbent"
            metrics.pruned_by_bound += 1
            continue

        if node.current_k == target_k:
            node.status = "complete"
            metrics.complete_found += 1
            if node.accumulated_loss < incumbent_loss - config.epsilon:
                incumbent_part = node.partition
                incumbent_loss = node.accumulated_loss
                incumbent_path = node.path
                incumbent_source = "node_complete"
                incumbent_updates += 1
            continue

        node.status = "expanded"
        children = _expand_acc(node, delta_fn, config)
        metrics.expanded += 1

        for child in children:
            metrics.created += 1
            all_nodes.append(child)
            nodes_by_id[child.node_id] = child

            if child.current_k == target_k:
                child.status = "complete"
                metrics.complete_found += 1
                if child.accumulated_loss < incumbent_loss - config.epsilon:
                    incumbent_part = child.partition
                    incumbent_loss = child.accumulated_loss
                    incumbent_path = child.path
                    incumbent_source = "node_complete"
                    incumbent_updates += 1
                continue

            if config.enable_bound_pruning and child.accumulated_loss >= incumbent_loss - config.epsilon:
                child.status = "pruned"
                child.prune_reason = "LB >= incumbent"
                metrics.pruned_by_bound += 1
                continue

            if config.use_dominance_pruning:
                dk = child.partition
                pv = _BestCostSeen.get(dk)
                if pv is not None and child.accumulated_loss >= pv - config.epsilon:
                    child.status = "pruned"
                    child.prune_reason = "dominance"
                    metrics.pruned_by_dominance += 1
                    continue
                if pv is None or child.accumulated_loss < pv - config.epsilon:
                    _BestCostSeen[dk] = child.accumulated_loss

            tail = _greedy_tail(child, target_k, delta_fn, config)
            if tail[0]:
                child.expected_loss = child.accumulated_loss + tail[1]
                if child.accumulated_loss + tail[1] < incumbent_loss - config.epsilon:
                    incumbent_part = tail[1]
                    incumbent_loss = child.accumulated_loss + tail[1]
                    incumbent_source = "expected_rollout"
                    incumbent_updates += 1
            else:
                child.expected_loss = None

            child.status = "live"
            heapq.heappush(pq, make_priority(child))

    elapsed = time.time() - start
    live_rem = sum(1 for n in all_nodes if n.status == "live")
    if config.max_nodes and metrics.created >= config.max_nodes:
        term = "max_nodes"
    elif config.timeout_seconds and elapsed >= config.timeout_seconds:
        term = "timeout"
    elif live_rem == 0:
        term = "queue_exhausted"
    else:
        term = "partial"

    best_phi = phi_partition(incumbent_part, ctx)

    return SearchReport(
        best_partition=incumbent_part,
        best_accumulated_loss=incumbent_loss,
        best_final_phi=best_phi,
        best_path=[{"step": s.step, "parent": str(s.parent_partition), "child": str(s.child_partition),
                     "delta": s.delta_phi, "C": s.accumulated_loss_after_split} for s in incumbent_path],
        best_partition_str=part_to_str(incumbent_part),
        best_path_str="",
        best_partition_labels_str=codec.part_to_str(incumbent_part),
        best_path_labels_str="",
        target_k=target_k,
        incumbent_source=incumbent_source,
        nodes_created=metrics.created,
        nodes_expanded=metrics.expanded,
        nodes_pruned_by_bound=metrics.pruned_by_bound,
        nodes_pruned_by_dominance=metrics.pruned_by_dominance,
        complete_nodes_found=metrics.complete_found,
        incumbent_updates=incumbent_updates,
        runtime_seconds=elapsed,
        all_nodes=all_nodes,
        M_worst_per_block=config.M_worst_per_block,
        upper_frontier_width=config.upper_frontier_width,
        termination_reason=term,
        optimality_certified=(term == "queue_exhausted"
                              and config.max_nodes is None
                              and config.timeout_seconds is None
                              and config.max_expansion_candidates_per_node == 0),
        live_nodes_remaining=live_rem,
        objective="accumulated_path",
        mode="heuristic",
        partition_space=config.partition_space,
        generators=config.generators,
        n_nodes=ctx.n_nodes,
        n_search_vars=n_vars,
        initial_state_str="".join(str(int(b)) for b in ctx.initial_state),
    )


def _clear_acc_caches():
    _DeltaCache.clear()
    _ExpectedCache.clear()
    _UpperCache.clear()
    _BestCostSeen.clear()


def _greedy_tail(node, target_k, delta_fn, config):
    cur = list(node.partition)
    total = 0.0
    while len(cur) < target_k:
        best = float("inf")
        best_sp = None
        best_nb = None
        for bm in cur:
            if popcount(bm) < 2:
                continue
            for _, left, right in _cached_splits(bm):
                key = (tuple(cur), bm, left, right)
                if key in _DeltaCache:
                    d = _DeltaCache[key]
                else:
                    d = delta_fn(tuple(cur), bm, left, right)
                    _DeltaCache[key] = d
                if d < best:
                    best = d
                    best_sp = (bm, left, right)
                    nb = []
                    for b in cur:
                        if b == bm:
                            nb.append(left)
                            nb.append(right)
                        else:
                            nb.append(b)
                    best_nb = nb
        if best_sp is None:
            return (False, None, None)
        total += best
        cur = best_nb
    return (True, canonical_partition(tuple(cur)), total)


def _expand_acc(node, delta_fn, config):
    children = []
    for bm in node.partition:
        if popcount(bm) < 2:
            continue
        for _, left, right in _cached_splits(bm):
            key = (node.partition, bm, left, right)
            if key in _DeltaCache:
                d = _DeltaCache[key]
            else:
                d = delta_fn(node.partition, bm, left, right)
                _DeltaCache[key] = d
            cpart = canonical_partition(
                tuple([b for b in node.partition if b != bm] + [left, right]))
            cC = node.accumulated_loss + d
            step = SplitStep(len(node.path) + 1, node.partition, cpart, bm, left, right, d, cC)
            child = BBNode(partition=cpart, accumulated_loss=cC, lower_bound=cC,
                           depth=node.depth + 1, parent_id=node.node_id,
                           node_id=_next_nid(), path=node.path + (step,), split_step=step)
            children.append(child)
    # Limit if configured
    lim = config.max_expansion_candidates_per_node
    if lim > 0 and len(children) > lim:
        children.sort(key=lambda c: c.accumulated_loss)
        children = children[:lim]
    return children


def _log_node(node, codec=None):
    e = f"{node.expected_loss:.4f}" if node.expected_loss is not None else "None"
    u = f"{node.upper_bound:.4f}" if node.upper_bound is not None else "None"
    print(f"  Node {node.node_id} k={node.current_k} "
          f"P={part_to_str(node.partition, codec)} "
          f"C={node.accumulated_loss:.4f} LB={node.lower_bound:.4f} "
          f"E={e} U={u} status={node.status}"
          + (f" reason={node.prune_reason}" if node.prune_reason else ""))


def export_search_tree_mermaid(nodes: list[BBNode], max_nodes: int = 100,
                                codec: Optional[VariableCodec] = None) -> str:
    lines = ["graph TD"]
    inc = set()
    for node in nodes[:max_nodes]:
        inc.add(node.node_id)
        parts = [f"k={node.current_k}", f"P={part_to_str(node.partition, codec)[:40]}",
                 f"C={node.accumulated_loss:.4f}", f"LB={node.lower_bound:.4f}"]
        if node.expected_loss is not None:
            parts.append(f"E={node.expected_loss:.4f}")
        if node.upper_bound is not None:
            parts.append(f"U={node.upper_bound:.4f}")
        if node.prune_reason:
            parts.append(f"PODA: {node.prune_reason}")
        lines.append(f'    n{node.node_id}["{"<br/>".join(parts)}"]')
        if node.prune_reason:
            lines.append(f'    style n{node.node_id} fill:#ffcccc,stroke:#cc0000')
        elif node.status == "complete":
            lines.append(f'    style n{node.node_id} fill:#ccffcc,stroke:#00cc00')
    for node in nodes[:max_nodes]:
        if node.parent_id is not None and node.parent_id in inc:
            d = f"delta={node.split_step.delta_phi:.4f}" if node.split_step else ""
            lines.append(f'    n{node.parent_id} -->|"{d}"| n{node.node_id}')
    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════════
#  SearchReport (unified)
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class SearchReport:
    best_partition: tuple[int, ...]
    best_accumulated_loss: float
    best_final_phi: float
    best_path: list[dict]
    best_partition_str: str
    best_path_str: str
    best_partition_labels_str: str
    best_path_labels_str: str
    target_k: int
    incumbent_source: str
    nodes_created: int
    nodes_expanded: int
    nodes_pruned_by_bound: int
    nodes_pruned_by_dominance: int
    complete_nodes_found: int
    incumbent_updates: int
    runtime_seconds: float
    all_nodes: list[BBNode]
    M_worst_per_block: int
    upper_frontier_width: int
    termination_reason: str = "unknown"
    optimality_certified: bool = False
    live_nodes_remaining: int = 0
    objective: str = "final_phi"
    mode: str = "heuristic"
    partition_space: str = "nodes"
    generators: tuple[str, ...] = ()
    dataset_name: str = ""
    csv_path: str = ""
    n_nodes: int = 0
    n_search_vars: int = 0
    initial_state_str: str = ""


# ═════════════════════════════════════════════════════════════════════════════
#  BnBConfig
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class BnBConfig:
    target_k: int = 5
    epsilon: float = 1e-12
    M_worst_per_block: int = 5
    upper_frontier_width: int = 10
    M_best_per_block_expected: int = 5
    use_dominance_pruning: bool = True
    use_initial_greedy_incumbent: bool = True
    enable_bound_pruning: bool = True
    collect_complete_equal_solutions: bool = False
    cache_delta_phi: bool = True
    cache_upper_bound: bool = True
    cache_expected: bool = True
    compute_upper_bound: bool = True
    compute_expected: bool = True
    max_expansion_candidates_per_node: int = 0
    mode: str = "heuristic"
    beam_width: int = 50
    max_nodes: Optional[int] = None
    timeout_seconds: Optional[float] = None
    codec: Optional[VariableCodec] = None
    partition_space: str = "mech_alc"
    objective: str = "final_phi"
    generators: tuple[str, ...] = ("selection",)
    top_l_per_generator: int = 5


# ═════════════════════════════════════════════════════════════════════════════
#  High-level dispatch
# ═════════════════════════════════════════════════════════════════════════════

def branch_and_bound_k_from_state_node_tpm(
    state_node_tpm: np.ndarray,
    target_k: int = 5,
    initial_state: Optional[np.ndarray] = None,
    metric: str = "emd_effect",
    config: Optional[BnBConfig] = None,
    verbose: bool = False,
    dataset_name: str = "",
    csv_path: str = "",
) -> SearchReport:
    if config is None:
        config = BnBConfig(target_k=target_k)
    config.target_k = target_k

    ctx = make_context(state_node_tpm, initial_state, metric, config.partition_space)
    config.codec = ctx.codec
    n_vars = ctx.codec.n_search_vars

    if config.objective == "final_phi":
        if config.mode == "exact" and config.max_expansion_candidates_per_node == 0:
            total = count_set_partitions(n_vars, target_k)
            max_enum = 500_000
            if total <= max_enum:
                res = run_exact_final_phi(ctx, target_k, config)
                return _final_phi_to_report(res, ctx, config, dataset_name, csv_path)

        # Heuristic: always use beam search (which seeds incumbent via selection_direct).
        # selection_direct alone is too slow for large systems (C(30,4)=27405+ phi evals).
        res = run_heuristic_beam_final_phi(ctx, target_k, config)
        return _final_phi_to_report(res, ctx, config, dataset_name, csv_path)

    # accumulated_path
    return run_accumulated_path_bnb(ctx, target_k, config, verbose)


def _final_phi_to_report(res: FinalPhiResult, ctx: StateNodeTPMContext,
                          config: BnBConfig, dataset: str, csv: str) -> SearchReport:
    # Validate: result must have exactly target_k blocks
    if len(res.partition) != config.target_k:
        raise RuntimeError(
            f"BUG: final result has {len(res.partition)} blocks, "
            f"expected {config.target_k}. "
            f"Partition: {ctx.codec.part_to_str(res.partition)}"
        )
    return SearchReport(
        best_partition=res.partition,
        best_accumulated_loss=res.accumulated_loss,
        best_final_phi=res.final_phi,
        best_path=res.path,
        best_partition_str=part_to_str(res.partition),
        best_path_str=res.path_labels_str,
        best_partition_labels_str=ctx.codec.part_to_str(res.partition),
        best_path_labels_str=res.path_labels_str,
        target_k=config.target_k,
        incumbent_source=res.incumbent_source,
        nodes_created=res.nodes_created,
        nodes_expanded=res.partial_candidates,
        nodes_pruned_by_bound=0,
        nodes_pruned_by_dominance=0,
        complete_nodes_found=res.complete_partitions,
        incumbent_updates=1,
        runtime_seconds=res.runtime,
        all_nodes=[],
        M_worst_per_block=config.M_worst_per_block,
        upper_frontier_width=config.upper_frontier_width,
        termination_reason=res.termination_reason,
        optimality_certified=res.optimality_certified,
        live_nodes_remaining=0,
        objective="final_phi",
        mode="exact" if res.optimality_certified else "heuristic",
        partition_space=config.partition_space,
        generators=config.generators,
        dataset_name=dataset,
        csv_path=csv,
        n_nodes=ctx.n_nodes,
        n_search_vars=ctx.codec.n_search_vars,
        initial_state_str="".join(str(int(b)) for b in ctx.initial_state),
    )


# ═════════════════════════════════════════════════════════════════════════════
#  CLI utilities
# ═════════════════════════════════════════════════════════════════════════════

def parse_initial_state(value: str, n_nodes: int) -> np.ndarray:
    if value == "zeros":
        return np.zeros(n_nodes, dtype=np.int8)
    if value == "ones":
        return np.ones(n_nodes, dtype=np.int8)
    if len(value) != n_nodes:
        raise ValueError(f"Initial state must have length {n_nodes}, got '{value}'")
    if any(c not in "01" for c in value):
        raise ValueError(f"Initial state must be binary, got '{value}'")
    return np.array([int(c) for c in value], dtype=np.int8)


def load_tpm_csv(path: str) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"TPM file not found: {p.resolve()}")

    # Peek first line to determine format
    with open(str(p), "r") as f:
        first = f.readline().strip()
    has_comma = "," in first
    if has_comma:
        n_cols = len(first.split(","))
        if n_cols > 1:
            arr = np.loadtxt(str(p), delimiter=",", dtype=np.float64)
            return arr if arr.ndim == 2 else arr.reshape(-1, 1)

    # Single column: decimal integer encoding of binary state
    raw = np.loadtxt(str(p), dtype=np.uint64)
    if raw.ndim == 0:
        raw = np.array([raw.item()])
    n_rows = raw.shape[0]
    n = int(np.log2(n_rows))
    if 2 ** n != n_rows:
        raise ValueError(f"Rows must be a power of 2, got {n_rows}")

    # Efficient bit extraction: allocate result, fill column by column
    # This avoids creating 25 large temporary arrays
    result = np.empty((n_rows, n), dtype=np.float32)
    for j in range(n):
        result[:, j] = 1.0 - ((raw >> j) & 1).astype(np.float32)
    return result

"""
Unlabeled set partitions for k-partition enumeration.

Generates k-partitions of (mechanism, alcance) indices without the
redundancy introduced by labeled-group encoding (which overcounts by k!).
"""

from itertools import combinations, permutations


def set_partitions(elements: list, k: int):
    """Yield all set partitions of `elements` into exactly `k` nonempty blocks.

    Blocks are unlabeled (canonical order based on first occurrence).
    Each block is a list of elements.
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


def _pair_blocks(mech_blocks, alc_blocks, k, p):
    """Pair `p` blocks from each side, yield partitions with exactly `k` groups.

    mech_blocks: list of `a` lists (mech elements per block)
    alc_blocks:  list of `b` lists (alc  elements per block)
    p = a + b - k  (number of shared pairs needed)
    """
    a = len(mech_blocks)
    b = len(alc_blocks)

    for mech_chosen in combinations(range(a), p):
        paired_mech = [mech_blocks[i] for i in mech_chosen]
        solo_mech = [mech_blocks[i] for i in range(a) if i not in mech_chosen]

        for alc_chosen in combinations(range(b), p):
            paired_alc = [alc_blocks[i] for i in alc_chosen]
            solo_alc = [alc_blocks[i] for i in range(b) if i not in alc_chosen]

            for perm in permutations(range(p)):
                groups = [
                    (frozenset(paired_mech[i]), frozenset(paired_alc[perm[i]]))
                    for i in range(p)
                ] + [
                    (frozenset(block), frozenset()) for block in solo_mech
                ] + [
                    (frozenset(), frozenset(block)) for block in solo_alc
                ]
                yield tuple(groups)


def all_k_partitions_unlabeled(mech_indices, alc_indices, k):
    """Yield every k-partition of mech and alc indices, without group-label redundancy.

    For each a ∈ [1, min(m,k)], b ∈ [1, min(n,k)] with a + b ≥ k:
      1. Partition mech into `a` unlabeled nonempty blocks.
      2. Partition alc  into `b` unlabeled nonempty blocks.
      3. Pair p = a + b - k blocks from each side (C(a,p)·C(b,p)·p! ways).
      4. Remaining blocks go solo as (block, ∅) or (∅, block).

    Yields tuples of k (frozenset, frozenset) pairs.
    """
    m = len(mech_indices)
    n = len(alc_indices)

    if k > m + n:
        return

    for a in range(1, min(m, k) + 1):
        for b in range(1, min(n, k) + 1):
            if a + b < k:
                continue
            p = a + b - k

            for m_blocks in set_partitions(list(mech_indices), a):
                for a_blocks in set_partitions(list(alc_indices), b):
                    yield from _pair_blocks(m_blocks, a_blocks, k, p)


def count_k_partitions_unlabeled(m, n, k):
    """Count unlabeled k-partitions (mech=m, alc=n) without enumerating."""
    from math import comb, factorial

    max_n = max(m, n, k)
    s = [[0] * (max_n + 1) for _ in range(max_n + 1)]
    s[0][0] = 1
    for i in range(1, max_n + 1):
        for j in range(1, i + 1):
            s[i][j] = j * s[i - 1][j] + s[i - 1][j - 1]

    total = 0
    for a in range(1, min(m, k) + 1):
        for b in range(1, min(n, k) + 1):
            if a + b < k:
                continue
            p = a + b - k
            total += s[m][a] * s[n][b] * comb(a, p) * comb(b, p) * factorial(p)
    return total


def partition_bitmasks(partition, m):
    """Convert a partition tuple to integer bitmask arrays.

    Returns (mech_masks, alc_masks), each a tuple of k ints where
    bit i is set if index i is in that group.
    """
    k = len(partition)
    mech_masks = [0] * k
    alc_masks = [0] * k
    for g, (mech_block, alc_block) in enumerate(partition):
        for i in mech_block:
            mech_masks[g] |= 1 << int(i)
        for j in alc_block:
            alc_masks[g] |= 1 << int(j)
    return tuple(mech_masks), tuple(alc_masks)

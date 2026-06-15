"""
0/1 Knapsack vía Branch and Bound (maximización de valor).

Items preordenados por ratio value/weight descendente.
Cota superior: relajación fraccional (bound clásico de Dantzig).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from typing import Any

from src.models.bnb_optimizer import BnBOptimizer, Node

# (weight, value)
ITEMS = [(1, 2), (2, 3), (3, 4), (4, 5)]
CAPACITY = 7

# Pre-sort by value/weight descending
ITEMS_SORTED = sorted(ITEMS, key=lambda x: x[1] / x[0], reverse=True)
N = len(ITEMS_SORTED)


def bound_fn(node: Node) -> float:
    idx, cap, val, _ = node.state
    if idx >= N:
        return float(val)

    total_val = val
    remaining_cap = cap

    for i in range(idx, N):
        w, v = ITEMS_SORTED[i]
        if remaining_cap <= 0:
            break
        if w <= remaining_cap:
            total_val += v
            remaining_cap -= w
        else:
            total_val += v * (remaining_cap / w)
            break

    return float(total_val)


def is_complete_fn(node: Node) -> bool:
    idx, _, _, _ = node.state
    return idx == N


def branch_fn(node: Node) -> list[Node]:
    idx, cap, val, mask = node.state
    if idx >= N:
        return []

    w, v = ITEMS_SORTED[idx]

    children = []

    # skip
    children.append(Node(state=(idx + 1, cap, val, mask), bound=0.0))

    # take (if fits)
    if w <= cap:
        children.append(
            Node(state=(idx + 1, cap - w, val + v, mask | (1 << idx)), bound=0.0)
        )

    return children


def estimate_fn(initial_state) -> tuple[Any, float]:
    """Cota inferior inicial: greedy binario (solución factible 0/1)."""
    items_sorted = sorted(ITEMS, key=lambda x: x[1] / x[0], reverse=True)
    val = cap = mask = 0
    for i, (w, v) in enumerate(items_sorted):
        if cap + w <= CAPACITY:
            val += v
            cap += w
            mask |= (1 << i)
    n = len(items_sorted)
    complete_state = (n, CAPACITY - cap, val, mask)
    return complete_state, float(val)


def solve() -> tuple[list[int], int]:
    solver = BnBOptimizer(direction="max")
    result = solver.solve(
        initial_state=(0, CAPACITY, 0, 0),
        branch_fn=branch_fn,
        bound_fn=bound_fn,
        is_complete_fn=is_complete_fn,
        estimate_fn=estimate_fn,
    )

    if result is None:
        raise RuntimeError("No se encontró solución")

    state, valor = result
    _, _, _, mask = state

    # Map sorted indices back to original indices
    sorted_to_original = {i: ITEMS.index(ITEMS_SORTED[i]) for i in range(N)}
    selected = [sorted_to_original[i] for i in range(N) if mask & (1 << i)]

    return selected, int(valor)


if __name__ == "__main__":
    selected, total = solve()
    peso = sum(ITEMS[i][0] for i in selected)
    print(f"Valor máximo: {total}")
    print(f"Peso total: {peso}/{CAPACITY}")
    print(f"Items seleccionados (índices originales): {selected}")
    print(f"  Detalle: {[ITEMS[i] for i in selected]}")

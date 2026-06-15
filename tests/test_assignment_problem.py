"""
Problema de Asignación vía Branch and Bound (minimización).

n trabajadores, n tareas. Cada trabajador i tiene un costo COST[i][j]
para la tarea j. Minimizar costo total asignando cada trabajador a
una tarea distinta.

Estado: (i, tuple_asignaciones, costo_parcial)
  - i: índice del siguiente trabajador a asignar (0..n)
  - tuple_asignaciones[j] = trabajador asignado a tarea j, o -1 si libre
  - costo_parcial: suma acumulada de costos hasta i
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.bnb_optimizer import BnBOptimizer, Node


COST = [
    [9, 6, 8, 5],
    [6, 7, 9, 4],
    [5, 6, 7, 3],
    [7, 6, 9, 4],
]
N = len(COST)


def bound_fn(node: Node) -> float:
    i, asignadas, costo_parcial = node.state

    tareas_libres = [j for j in range(N) if asignadas[j] == -1]
    lower = costo_parcial

    for r in range(i, N):
        lower += min(COST[r][j] for j in tareas_libres)

    return float(lower)


def is_complete_fn(node: Node) -> bool:
    i, _, _ = node.state
    return i == N


def branch_fn(node: Node) -> list[Node]:
    i, asignadas, costo_parcial = node.state
    if i == N:
        return []

    tareas_libres = [j for j in range(N) if asignadas[j] == -1]
    tareas_libres.sort(key=lambda j: COST[i][j])

    children = []
    for j in tareas_libres:
        nueva_lista = list(asignadas)
        nueva_lista[j] = i
        child = Node(
            state=(i + 1, tuple(nueva_lista), costo_parcial + COST[i][j]),
            bound=0.0,
        )
        children.append(child)
    return children


def solve() -> tuple[list[tuple[int, int]], int]:
    solver = BnBOptimizer(direction="min")
    result = solver.solve(
        initial_state=(0, tuple(-1 for _ in range(N)), 0),
        branch_fn=branch_fn,
        bound_fn=bound_fn,
        is_complete_fn=is_complete_fn,
    )

    if result is None:
        raise RuntimeError("No se encontró solución")

    state, valor = result
    _, asignadas, _ = state

    asignaciones = [(asignadas[j], j) for j in range(N)]
    asignaciones.sort()
    return asignaciones, int(valor)


if __name__ == "__main__":
    asignaciones, total = solve()
    print(f"Costo mínimo: {total}")
    for w, j in asignaciones:
        print(f"  Trabajador {w} -> Tarea {j}  (costo {COST[w][j]})")

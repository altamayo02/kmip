import heapq
from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional


@dataclass(slots=True)
class Node:
    state: Any
    bound: float
    depth: int = 0


class BnBOptimizer:
    def __init__(self, direction: Literal["min", "max"] = "min"):
        self.direction = direction
        self.incumbent: Optional[Node] = None
        self._live_nodes: list[tuple[float, int, Node]] = []
        self._node_counter: int = 0

    def _push(self, node: Node):
        if self.direction == "min":
            priority = node.bound
        else:
            priority = -node.bound
        heapq.heappush(self._live_nodes, (priority, -self._node_counter, node))
        self._node_counter += 1

    def _pop(self) -> Node:
        _, _, node = heapq.heappop(self._live_nodes)
        return node

    def solve(
        self,
        initial_state: Any,
        branch_fn: Callable[[Node], list[Node]],
        estimate_fn: Callable[[Any], tuple[Any, float]],
        bound_fn: Callable[[Node], float],
        is_complete_fn: Callable[[Node], bool],
        disable_pruning: bool = False,
    ) -> Optional[tuple[Any, float]]:
        state, bound = estimate_fn(initial_state)
        candidate = Node(state=state, bound=bound)
        if is_complete_fn(candidate):
            self.incumbent = candidate
        self._init_root(initial_state, bound_fn, is_complete_fn)
        while self._live_nodes:
            node = self._pop()
            if not disable_pruning and self._is_pruned(node):
                continue
            self._expand(node, branch_fn, bound_fn, is_complete_fn, disable_pruning)
        return self._result()

    def _init_root(self, initial_state, bound_fn, is_complete_fn):
        root = Node(state=initial_state, bound=0.0)
        root.bound = bound_fn(root)
        if is_complete_fn(root):
            self.incumbent = root
        self._push(root)

    def _is_pruned(self, node: Node) -> bool:
        if self.incumbent is None:
            return False
        if self.direction == "min":
            return node.bound >= self.incumbent.bound
        else:
            return node.bound <= self.incumbent.bound

    def _is_better(self, candidate: Node) -> bool:
        if self.incumbent is None:
            return True
        if self.direction == "min":
            return candidate.bound < self.incumbent.bound
        else:
            return candidate.bound > self.incumbent.bound

    def _expand(self, node, branch_fn, bound_fn, is_complete_fn, disable_pruning=False):
        for child in branch_fn(node):
            child.bound = bound_fn(child)
            if not disable_pruning and self._is_pruned(child):
                continue
            if is_complete_fn(child) and self._is_better(child):
                self.incumbent = child
            else:
                self._push(child)

    def _result(self):
        if self.incumbent:
            return self.incumbent.state, self.incumbent.bound
        return None

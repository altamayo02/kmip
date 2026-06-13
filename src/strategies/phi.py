import time
import random
import collections
from collections.abc import Iterable, Mapping, MutableMapping, Sequence

import numpy as np

if not hasattr(collections, "Iterable"):
    setattr(collections, "Iterable", Iterable)
if not hasattr(collections, "Mapping"):
    setattr(collections, "Mapping", Mapping)
if not hasattr(collections, "MutableMapping"):
    setattr(collections, "MutableMapping", MutableMapping)
if not hasattr(collections, "Sequence"):
    setattr(collections, "Sequence", Sequence)

from pyphi import Network, Subsystem
from pyphi.labels import NodeLabels
from src.config import Config
from src.middlewares.slogger import SafeLogger
from src.middlewares.profile import profiler_manager, profile
from src.functions.labels import ABECEDARY
from src.functions.format import fmt_kparticion
from src.functions.emd import emd_efecto
from src.strategies.base import SIA
from src.solution import Solution


LABEL = "Pyphi"
TAG_STRATEGY = f"{LABEL}_strategy"
TAG_ANALYSIS = f"{LABEL}_analysis"


class Phi(SIA):
    def __init__(self, tpm: np.ndarray, config: Config):
        super().__init__(tpm, config)
        profiler_manager.start_session(
            f"Phi{len(tpm[1])}{config.pagina_muestra}"
        )
        self.logger = SafeLogger(TAG_STRATEGY)

    @profile(context={"type": TAG_ANALYSIS})
    def aplicar_estrategia(
        self,
        estado_inicial: str,
        condiciones: str,
        alcance: str,
        mecanismo: str,
    ):
        self.sia_preparar_subsistema(estado_inicial, condiciones, alcance, mecanismo)

        futuros = self.sia_subsistema.indices_ncubos
        presentes = self.sia_subsistema.dims_ncubos
        m, n = futuros.size, presentes.size

        todas_las_soluciones: list[Solution] = []

        # --- k=2: PyPhi effect_mip (comportamiento original) ---
        self._k2_pyphi(
            todas_las_soluciones, estado_inicial, condiciones, alcance, mecanismo
        )

        # --- k=3,4,5: Pipeline escalable ---
        if m > 0 and n >= 0:
            P_intact = self.sia_dists_marginales

            # Fase 1: Precomputar todas las marginales condicionales
            P_table = self._precompute_all_marginals(
                futuros, presentes, m, n
            )

            # Fase 2: Random restarts + refinement para cada k
            for k in (3, 4, 5):
                if k > m + n:
                    continue

                num_restarts = 30 if n < 10 else 15
                blocks, emd = self._find_best_k_partition(
                    P_intact, P_table, m, n, k, num_restarts=num_restarts
                )
                if blocks is None:
                    continue

                _, alc_bitmap = self._assign_alcance_optimal(
                    blocks, P_intact, P_table, m, k
                )
                part_dist = self._build_partition_distribution(
                    P_table, blocks, alc_bitmap, m, k
                )
                partition_tuples = self._blocks_to_partition(
                    blocks, alc_bitmap, presentes, futuros, m, n, k
                )
                fmt = fmt_kparticion(partition_tuples)
                todas_las_soluciones.append(Solution(
                    estrategia=f"{LABEL}(k={k})",
                    perdida=emd,
                    distribucion_subsistema=self.sia_dists_marginales,
                    distribucion_particion=part_dist,
                    tiempo_ejecucion=time.time() - self.sia_tiempo_inicio,
                    particion=fmt,
                ))

        return todas_las_soluciones

    # ------------------------------------------------------------------
    #  k=2 — PyPhi original
    # ------------------------------------------------------------------
    def _k2_pyphi(self, soluciones, estado_inicial, condiciones, alcance, mecanismo):
        try:
            n_nodes = len(self.tpm[1])
            node_labels = NodeLabels(
                tuple(ABECEDARY[:n_nodes]),
                tuple(range(n_nodes)),
            )
            network = Network(self.tpm, node_labels=node_labels)
            subsystem = Subsystem(
                network=network,
                state=np.array([int(b) for b in estado_inicial]),
                nodes=range(n_nodes),
            )
            efecto_mip = subsystem.effect_mip(
                mechanism=tuple(
                    i for i, b in enumerate(mecanismo) if b == "1"
                ),
                purview=tuple(
                    i for i, b in enumerate(alcance) if b == "1"
                ),
            )
            perdida = float(efecto_mip.phi)
            particion = self._format_mip(efecto_mip)
            soluciones.append(Solution(
                estrategia=f"{LABEL}(k=2)",
                perdida=perdida,
                distribucion_subsistema=np.array([0.0]),
                distribucion_particion=np.array([0.0]),
                tiempo_ejecucion=time.time() - self.sia_tiempo_inicio,
                particion=particion,
                quiere_hablar=True,
            ))
        except Exception as exc:
            self.logger.warn(f"PyPhi k=2 falló: {exc}")

    # ------------------------------------------------------------------
    #  Fase 1: Precomputación de todas las marginales condicionales
    #  via zeta transform (superset sum). O(m · n · 2ⁿ)
    # ------------------------------------------------------------------
    def _precompute_all_marginals(self, futuros, presentes, m, n):
        if n == 0:
            P_table = np.zeros((m, 1), dtype=np.float32)
            for j_idx, cube in enumerate(self.sia_subsistema.ncubos):
                P_table[j_idx, 0] = float(np.mean(cube.data))
            return P_table

        n_masks = 1 << n

        # Initial state as little-endian integer
        t = 0
        for i in range(n):
            t |= int(self.sia_subsistema.estado_inicial[presentes[i]]) << i

        # Precompute match_bits for all 2ⁿ states
        states = np.arange(n_masks, dtype=np.uint32)
        match_bits = (~(states ^ t)) & (n_masks - 1)

        # Precompute popcount for all masks via byte LUT
        byte_popcnt = np.zeros(256, dtype=np.uint32)
        for i in range(256):
            byte_popcnt[i] = i.bit_count()
        popcount = byte_popcnt[
            states.astype(np.uint32).view(np.uint8).reshape(-1, 4)
        ].sum(axis=1)

        P_table = np.zeros((m, n_masks), dtype=np.float32)

        for j_idx, cube in enumerate(self.sia_subsistema.ncubos):
            cube_data = cube.data.ravel().astype(np.float64)
            g = np.zeros(n_masks, dtype=np.float64)

            # Scatter-add: for each state, add to its match_bits bucket
            np.add.at(g, match_bits, cube_data)

            # Superset zeta transform: g[mask] += g[mask | (1<<i)]
            for i in range(n):
                stride = 1 << i
                for offset in range(0, n_masks, 2 * stride):
                    g[offset:offset + stride] += g[offset + stride:offset + 2 * stride]

            # Convert to conditional probabilities
            P_table[j_idx] = (g / (1 << (n - popcount))).astype(np.float32)

        return P_table

    # ------------------------------------------------------------------
    #  Fase 2: Clustering jerárquico (aglomerativo) de nodos mecanismo
    #  Genera particiones anidadas para todos k ∈ [1, n].
    #  Para k > n, añade bloques vacíos extra.
    #  Retorna dict: k -> list[bitmask]
    # ------------------------------------------------------------------
    def _hierarchical_partitions(self, P_intact, P_table, m, n):
        block_masks = [1 << i for i in range(n)]
        partitions: dict[int, list[int]] = {}

        max_k = min(5, m + n)
        if n == 0:
            for k in range(1, max_k + 1):
                partitions[k] = [0] * k
            return partitions

        if n > 0:
            partitions[n] = block_masks[:]
            for k in range(n + 1, max_k + 1):
                partitions[k] = block_masks[:] + [0] * (k - n)

            for _ in range(n - 1):
                k = len(block_masks)
                best_i = best_jj = -1
                best_emd = float("inf")

                for i in range(k):
                    for jj in range(i + 1, k):
                        merged = block_masks[i] | block_masks[jj]

                        emd = 0.0
                        for a_idx in range(m):
                            best = float("inf")
                            for b_idx, bm in enumerate(block_masks):
                                if b_idx == i or b_idx == jj:
                                    val = np.abs(
                                        P_intact[a_idx] - P_table[a_idx, merged]
                                    )
                                else:
                                    val = np.abs(
                                        P_intact[a_idx] - P_table[a_idx, bm]
                                    )
                                if val < best:
                                    best = val
                            emd += best

                        if emd < best_emd:
                            best_emd = emd
                            best_i, best_jj = i, jj

                new_mask = block_masks[best_i] | block_masks[best_jj]
                block_masks.pop(best_jj)
                block_masks.pop(best_i)
                block_masks.append(new_mask)

                k_new = len(block_masks)
                partitions[k_new] = block_masks[:]

        return partitions

    # ------------------------------------------------------------------
    #  Fase 2b: Random restarts + refinement para una k específica
    #  Busca la mejor partición de n nodos en k bloques (cada bloque
    #  con al menos un nodo mecanismo, excepto si k > n).
    # ------------------------------------------------------------------
    def _find_best_k_partition(self, P_intact, P_table, m, n, k, num_restarts=20):
        def _constrained_emd(block_list):
            emd, _ = self._assign_alcance_optimal(
                block_list, P_intact, P_table, m, len(block_list)
            )
            return emd

        best_blocks = None
        best_emd = float("inf")

        candidates = []

        # 1) Hierarchical clustering como candidato inicial
        if n > 0:
            parts = self._hierarchical_partitions(P_intact, P_table, m, n)
            if k in parts:
                candidates.append(list(parts[k]))

        # 2) Random starts (si hay al menos k nodos mecanismo)
        if n >= k:
            for _ in range(num_restarts):
                blocks = [0] * k
                nodes = list(range(n))
                random.shuffle(nodes)
                for b in range(k):
                    blocks[b] |= 1 << nodes.pop()
                for node in nodes:
                    blocks[random.randrange(k)] |= 1 << node
                candidates.append(blocks)
        else:
            # n < k: algunos bloques quedarán vacíos de mecanismo
            for _ in range(num_restarts):
                blocks = [0] * k
                nodes = list(range(n))
                random.shuffle(nodes)
                for b in range(min(k, n)):
                    blocks[b] |= 1 << nodes.pop()
                for node in nodes:
                    blocks[random.randrange(k)] |= 1 << node
                candidates.append(blocks)

        # Refinar y evaluar cada candidato
        for blocks in candidates:
            refined = self._refine_partition(
                blocks, P_intact, P_table, m, n, k
            )
            emd = _constrained_emd(refined)
            if emd < best_emd:
                best_emd = emd
                best_blocks = refined

        return best_blocks, best_emd

    # ------------------------------------------------------------------
    #  Fase 2b: Refinamiento local (hill-climbing) de una partición
    # ------------------------------------------------------------------
    def _refine_partition(self, blocks, P_intact, P_table, m, n, k):
        blocks = list(blocks)
        node_to_block = [-1] * n
        for b_idx, bm in enumerate(blocks):
            for bit in range(n):
                if (bm >> bit) & 1:
                    node_to_block[bit] = b_idx

        def _constrained_emd(block_list):
            emd, _ = self._assign_alcance_optimal(
                block_list, P_intact, P_table, m, len(block_list)
            )
            return emd

        current_emd = _constrained_emd(blocks)

        for _ in range(10):
            improved = False
            for node in range(n):
                old_block = node_to_block[node]
                if old_block < 0:
                    continue
                for new_block in range(k):
                    if new_block == old_block:
                        continue
                    node_bit = 1 << node
                    blocks[old_block] &= ~node_bit
                    blocks[new_block] |= node_bit

                    new_emd = _constrained_emd(blocks)

                    if new_emd < current_emd - 1e-12:
                        current_emd = new_emd
                        node_to_block[node] = new_block
                        improved = True
                        break
                    else:
                        blocks[old_block] |= node_bit
                        blocks[new_block] &= ~node_bit

                if improved:
                    break
            if not improved:
                break

        return blocks[:k]

    # ------------------------------------------------------------------
    #  Fase 3: Asignación óptima de alcance — O(m·k)
    #  Cada nodo de alcance se asigna independientemente al bloque
    #  que minimiza |P_intact[j] - P(j|mask_b)|.
    # ------------------------------------------------------------------
    @staticmethod
    def _assign_alcance_optimal(blocks, P_intact, P_table, m, k):
        empty_mech = [b for b in range(k) if blocks[b] == 0]
        nonempty = [b for b in range(k) if blocks[b] != 0]

        best_block = np.empty(m, dtype=np.int32)
        best_val = np.empty(m, dtype=np.float32)
        for j in range(m):
            bb, bv = 0, float("inf")
            for b in nonempty:
                val = np.abs(P_intact[j] - P_table[j, blocks[b]])
                if val < bv:
                    bv = val
                    bb = b
            best_block[j] = bb
            best_val[j] = bv

        assigned = np.zeros(k, dtype=bool)
        alc_bitmap = 0

        if empty_mech:
            used = set()
            for eb in empty_mech:
                best_j, best_inc = -1, float("inf")
                for j in range(m):
                    if j in used:
                        continue
                    val = np.abs(P_intact[j] - P_table[j, 0])
                    inc = val - best_val[j]
                    if inc < best_inc:
                        best_inc = inc
                        best_j = j
                if best_j >= 0:
                    used.add(best_j)
                    best_block[best_j] = eb
                    best_val[best_j] = np.abs(P_intact[best_j] - P_table[best_j, 0])
                    assigned[eb] = True

            for eb in empty_mech:
                if not assigned[eb]:
                    best_j, best_inc = -1, float("inf")
                    for j in range(m):
                        val = np.abs(P_intact[j] - P_table[j, 0])
                        inc = val - best_val[j]
                        if inc < best_inc:
                            best_inc = inc
                            best_j = j
                    if best_j >= 0:
                        best_block[best_j] = eb
                        best_val[best_j] = np.abs(P_intact[best_j] - P_table[best_j, 0])
                        assigned[eb] = True

        emd = 0.0
        for j in range(m):
            b = int(best_block[j])
            emd += best_val[j]
            alc_bitmap |= 1 << (b * m + j)
        return emd, alc_bitmap

    # ------------------------------------------------------------------
    #  Reconstruir distribución marginal bajo la k-partición
    # ------------------------------------------------------------------
    def _build_partition_distribution(self, P_table, blocks, alc_bitmap, m, k):
        dist = np.zeros(m, dtype=np.float32)
        for j in range(m):
            for b in range(k):
                bit_pos = b * m + j
                if (alc_bitmap >> bit_pos) & 1:
                    dist[j] = P_table[j, blocks[b]]
                    break
        return dist

    # ------------------------------------------------------------------
    #  Convertir blocks + alc_bitmap a tupla de frozensets
    # ------------------------------------------------------------------
    def _blocks_to_partition(self, blocks, alc_bitmap, presentes, futuros, m, n, k):
        partition = []
        for b in range(k):
            mech_nodes = [int(presentes[bit]) for bit in range(n) if (blocks[b] >> bit) & 1]
            alc_nodes = [int(futuros[bit]) for bit in range(m) if (alc_bitmap >> (b * m + bit)) & 1]
            partition.append((frozenset(mech_nodes), frozenset(alc_nodes)))
        return tuple(partition)

    # ------------------------------------------------------------------
    #  Formateo de la MIP de PyPhi (k=2)
    # ------------------------------------------------------------------
    def _format_mip(self, mip):
        try:
            k_partition = tuple(
                (frozenset(part.mechanism), frozenset(part.purview))
                for part in mip.partition.parts
            )
            return fmt_kparticion(k_partition)
        except Exception:
            return "PyPhi partition"

import numpy as np


class GeometricBase:
    """Provides hypercube path computation and transition cost tables.

    Call _compute_geometric_data() after sia_preparar_subsistema()
    to build:
      - _caminos:  dict[nivel -> list[estados]]  (hypercube levels)
      - _tabla_transiciones: dict[(state_a, state_b) -> list[costs]]
    """

    def _compute_geometric_data(self):
        dims = self.sia_subsistema.dims_ncubos
        self._estado_inicial = self.sia_subsistema.estado_inicial[dims]
        self._estado_final = 1 - self._estado_inicial

        self._flat_data = [
            ncubo.data.ravel() for ncubo in self.sia_subsistema.ncubos
        ]
        self._idx_ncubos = list(
            range(len(self.sia_subsistema.indices_ncubos))
        )
        self._caminos = {
            0: [self._estado_inicial.tolist()]
        }
        init_key = (
            tuple(self._caminos[0][0]),
            tuple(self._caminos[0][0]),
        )
        self._tabla_transiciones = {
            init_key: [0.0] * len(self._idx_ncubos)
        }

        for nivel in range(1, len(self._estado_inicial) + 1):
            self._compute_path_level(nivel)

    def _hamming(self, a, b):
        return sum(x != y for x, y in zip(a, b))

    def _compute_transition_cost(self, estado_ini, estado_fin):
        key = (tuple(estado_ini), tuple(estado_fin))
        if key not in self._tabla_transiciones:
            self._tabla_transiciones[key] = [None] * len(self._idx_ncubos)

        dist_hamming = self._hamming(estado_ini, estado_fin)
        factor = 1.0 / (2**dist_hamming)

        ini_int = int("".join(map(str, estado_ini[::-1])), 2)
        fin_int = int("".join(map(str, estado_fin[::-1])), 2)

        diffs = np.abs(
            np.array([flat[ini_int] for flat in self._flat_data])
            - np.array([flat[fin_int] for flat in self._flat_data])
        )
        self._tabla_transiciones[key] = diffs.tolist()

        if dist_hamming > 1:
            for i in range(len(estado_ini)):
                if estado_ini[i] != estado_fin[i]:
                    intermedio = estado_fin.copy()
                    intermedio[i] = estado_ini[i]
                    temp_key = (tuple(estado_ini), tuple(intermedio))
                    for n in self._idx_ncubos:
                        if self._tabla_transiciones[temp_key][n] is not None:
                            self._tabla_transiciones[key][n] = (
                                self._tabla_transiciones[key][n]
                                + self._tabla_transiciones[temp_key][n]
                            )

        result = []
        for v in self._tabla_transiciones[key]:
            result.append(factor * v if v is not None else v)
        self._tabla_transiciones[key] = result

    def _compute_path_level(self, nivel):
        n = len(self._estado_final)
        visitados = set()
        self._caminos[nivel] = []
        for estado_anterior in self._caminos[nivel - 1]:
            estado_actual = np.array(estado_anterior)
            for i in range(n):
                if estado_actual[i] != self._estado_final[i]:
                    nuevo = estado_actual.copy()
                    nuevo[i] = self._estado_final[i]
                    t = tuple(nuevo)
                    if t not in visitados:
                        self._caminos[nivel].append(nuevo.tolist())
                        self._compute_transition_cost(
                            self._caminos[0][0], nuevo.tolist()
                        )
                        visitados.add(t)

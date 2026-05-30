import numpy as np
from numpy.typing import NDArray

from src.models.enums import Notation
from src.models.ncube import NCube


class System:
    def __init__(
        self,
        tpm: np.ndarray,
        estado_inicio: np.ndarray,
    ):
        num_nodos = self._validar_inicial(tpm, estado_inicio)
        self.estado_inicial = estado_inicio
        self.ncubos = tuple(
            NCube(
                indice=idx,
                dims=np.array(range(num_nodos), dtype=np.int8),
                data=tpm[:, idx].reshape((2,) * num_nodos),
            )
            for idx in range(num_nodos)
        )
        self.memo = {}

    def _validar_inicial(self, tpm: np.ndarray, estado_inicio: np.ndarray):
        if estado_inicio.size != (num_nodos := tpm.shape[1]):
            raise ValueError(
                f"Estado inicial debe tener longitud {num_nodos}, "
                f"se obtuvo {estado_inicio.size}"
            )
        return num_nodos

    @property
    def indices_ncubos(self):
        return np.array([cube.indice for cube in self.ncubos], dtype=np.int8)

    @property
    def dims_ncubos(self):
        return (
            self.ncubos[0].dims if len(self.ncubos) > 0 else np.array([])
        )

    def condicionar(self, indices: NDArray[np.int8]) -> "System":
        indices_validos = np.intersect1d(self.indices_ncubos, indices)
        if not indices_validos.size:
            return self
        nuevo_sistema = System.__new__(System)
        nuevo_sistema.estado_inicial = self.estado_inicial
        nuevo_sistema.memo = {}
        nuevo_sistema.ncubos = tuple(
            cube.condicionar(indices_validos, self.estado_inicial)
            for cube in self.ncubos
            if cube.indice not in indices_validos
        )
        return nuevo_sistema

    def substraer(
        self,
        alcance_idx: NDArray[np.int8],
        mecanismo_dims: NDArray[np.int8],
    ) -> "System":
        futuros_validos = np.setdiff1d(self.indices_ncubos, alcance_idx)
        nuevo_sistema = System.__new__(System)
        nuevo_sistema.estado_inicial = self.estado_inicial
        nuevo_sistema.memo = {}
        nuevo_sistema.ncubos = tuple(
            cube.marginalizar(mecanismo_dims)
            for cube in self.ncubos
            if cube.indice in futuros_validos
        )
        return nuevo_sistema

    def bipartir(
        self,
        alcance: NDArray[np.int8],
        mecanismo: NDArray[np.int8],
    ) -> "System":
        nuevo_sistema = System.__new__(System)
        nuevo_sistema.estado_inicial = self.estado_inicial
        nuevo_sistema.memo = self.memo

        clave = tuple(alcance), tuple(mecanismo)
        if clave not in self.memo:
            self.memo[clave] = tuple(
                cube.marginalizar(np.setdiff1d(cube.dims, mecanismo))
                if cube.indice in alcance
                else cube.marginalizar(mecanismo)
                for cube in self.ncubos
            )

        nuevo_sistema.ncubos = self.memo[clave]
        return nuevo_sistema

    def distribucion_marginal(self):
        distribucion = np.empty(self.indices_ncubos.size, dtype=np.float32)
        for i, ncubo in enumerate(self.ncubos):
            probabilidad = ncubo.data
            if ncubo.dims.size:
                inicial = tuple(self.estado_inicial[j] for j in ncubo.dims)
								# HARDCODED - Little-endian encoding
                probabilidad = ncubo.data[inicial[::-1]]
            distribucion[i] = probabilidad
        return distribucion

    def __str__(self) -> str:
        sub_dims = self.dims_ncubos
        cubos_info = [f"{c}" for c in self.ncubos]
        return (
            f"\nSystem(indices={self.indices_ncubos}, dims={sub_dims})"
            f"\nInitial state: {self.estado_inicial}"
            f"\nNCubes:\n" + "\n".join(cubos_info)
        )

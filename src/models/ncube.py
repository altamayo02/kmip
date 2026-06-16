from dataclasses import dataclass, field
from numpy.typing import NDArray
import numpy as np


@dataclass(frozen=True)
class NCube:
    indice: int
    dims: NDArray[np.int8]
    data: np.ndarray
    memo: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.dims.size and self.data.shape != (2,) * self.dims.size:
            raise ValueError(
                f"Forma inválida {self.data.shape} para dimensiones {self.dims}"
            )

    def condicionar(
        self,
        indices_condicionados: NDArray[np.int8],
        estado_inicial: NDArray[np.int8],
    ) -> "NCube":
        numero_dims = self.dims.size
        seleccion = [slice(None)] * numero_dims

        for condicion in indices_condicionados:
            level_arr = numero_dims - (condicion + 1)
            seleccion[level_arr] = estado_inicial[condicion]

        nuevas_dims = np.array(
            [dim for dim in self.dims if dim not in indices_condicionados],
            dtype=np.int8,
        )
        return NCube(
            data=self.data[tuple(seleccion)],
            indice=self.indice,
            dims=nuevas_dims,
        )

    def marginalizar(self, ejes: NDArray[np.int8]) -> "NCube":
        if tuple(ejes) not in self.memo:
            marginable_axis = np.intersect1d(ejes, self.dims)
            if not marginable_axis.size:
                return self
            numero_dims = self.dims.size - 1
            ejes_locales = tuple(
                numero_dims - dim_idx
                for dim_idx, axis in enumerate(self.dims)
                if axis in marginable_axis
            )
            new_dims = np.array(
                [d for d in self.dims if d not in marginable_axis],
                dtype=np.int8,
            )
            self.memo[tuple(ejes)] = (
                np.mean(self.data, axis=ejes_locales, keepdims=False, dtype=np.float32),
                new_dims,
            )
        return NCube(
            data=self.memo[tuple(ejes)][0],
            dims=self.memo[tuple(ejes)][1],
            indice=self.indice,
        )

    def __str__(self) -> str:
        dims_str = f"dims={self.dims}"
        forma_str = f"shape={self.data.shape}"
        datos_str = str(self.data).replace("\n", "\n" + " " * 8)
        return (
            f"NCube(index={self.indice}):\n"
            f"    {dims_str}\n"
            f"    {forma_str}\n"
            f"    data=\n        {datos_str}"
        )

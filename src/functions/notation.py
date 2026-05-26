import numpy as np
from numpy.typing import NDArray

from src.models.enums import Notation


def lil_endian(n: int) -> np.ndarray:
    if n <= 0:
        return np.array([0], dtype=np.uint32)

    size = 1 << n
    result = np.zeros(size, dtype=np.uint32)

    block_bits = max(12, min(16, 28 - int(np.log2(n))))
    block_size = 1 << block_bits

    shifts = np.array([n - i - 1 for i in range(n)], dtype=np.uint32)
    block_result = np.zeros(block_size, dtype=np.uint32)
    bit_group_size = 6 if n > 24 else 4

    for start in range(0, size, block_size):
        end = min(start + block_size, size)
        current_size = end - start
        block_result[:current_size] = 0
        block_indices = np.arange(start, end, dtype=np.uint32)

        for base_bit in range(0, n, bit_group_size):
            bits_remaining = min(bit_group_size, n - base_bit)
            if bits_remaining <= 0:
                break

            group_mask = np.uint32((1 << bits_remaining) - 1)
            group_values = (block_indices >> base_bit) & group_mask

            for j in range(bits_remaining):
                shift = shifts[base_bit + j]
                bit_value = (group_values >> j) & np.uint32(1)
                block_result[:current_size] |= bit_value << shift

        result[start:end] = block_result[:current_size]

    return result


def big_endian(n: int) -> np.ndarray:
    return np.array(range(n), dtype=np.uint32)


def reindexar(n: int, notacion: str = Notation.LIL_ENDIAN.value) -> np.ndarray:
    notaciones = {
        Notation.BIG_ENDIAN.value: big_endian(n),
        Notation.LIL_ENDIAN.value: lil_endian(n),
    }
    return notaciones.get(notacion, lil_endian(n))


def seleccionar_estado(
    subestado: np.ndarray, notacion: str = Notation.LIL_ENDIAN.value
) -> np.ndarray:
    notaciones = {
        Notation.BIG_ENDIAN.value: subestado,
        Notation.LIL_ENDIAN.value: subestado[::-1],
    }
    return notaciones.get(notacion, subestado[::-1])

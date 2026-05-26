from itertools import product

import numpy as np
from numpy.typing import NDArray


def get_labels(n: int) -> tuple[str, ...]:
    def get_excel_column(n: int) -> str:
        if n <= 0:
            return ""
        return get_excel_column((n - 1) // 26) + chr((n - 1) % 26 + ord("A"))

    return tuple(get_excel_column(i) for i in range(1, n + 1))


ABECEDARY = get_labels(40)
LOWER_ABECEDARY = [letter.lower() for letter in ABECEDARY]


def literales(remaining_vars: NDArray[np.int8], lowercase: bool = False) -> str:
    return (
        "".join(
            ABECEDARY[i].lower() if lowercase else ABECEDARY[i]
            for i in remaining_vars
        )
        if remaining_vars.size
        else "\u2205"
    )


def dec2bin(decimal: int, width: int) -> str:
    return format(decimal, f"0{width}b")


def estados_binarios(n: int) -> list[str]:
    return [dec2bin(i, n) for i in range(1 << n)][1:]


def get_restricted_combinations(binary_str: str) -> tuple[list[str], list[str]]:
    ones_count = binary_str.count("1")
    width = len(binary_str)
    one_positions = [i for i, bit in enumerate(binary_str) if bit == "1"]

    def generate_valid_combinations():
        base_combinations = list(product(["0", "1"], repeat=ones_count))
        valid_combinations = []
        for comb in base_combinations:
            result = ["0"] * width
            for pos, bit in zip(one_positions, comb):
                result[pos] = bit
            valid_combinations.append("".join(result))
        return valid_combinations

    B = generate_valid_combinations()
    C = B.copy()
    return B, C


def generate_combinations(A: str) -> list[tuple[str, str, str]]:
    B, C = get_restricted_combinations(A)
    formatted_B = [" ".join(b[i : i + 2] for i in range(0, len(b), 2)) for b in B]
    formatted_C = [" ".join(c[i : i + 2] for i in range(0, len(c), 2)) for c in C]
    formatted_A = " ".join(A[i : i + 2] for i in range(0, len(A), 2))
    return list(product([formatted_A], formatted_B, formatted_C))[1:]

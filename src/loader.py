from pathlib import Path
import time
import os

import numpy as np


SAMPLES_PATH = Path("data/samples")
CSV_EXTENSION = "csv"
COLON_DELIM = ","
ABC_START = "A"


class TpmLoader:
    @staticmethod
    def cargar(
        tamaño: int,
        pagina: str = "A",
        binaria: bool = True
    ) -> np.ndarray:
        filename = f"N{tamaño}{pagina}.{CSV_EXTENSION}"
        filepath = SAMPLES_PATH / filename
        if not filepath.exists():
            alt_paths = [
                Path("src") / ".samples" / filename,
                Path("..") / "data" / "samples" / filename,
                Path(".") / filename,
            ]
            for alt in alt_paths:
                if alt.exists():
                    filepath = alt
                    break
            else:
                raise FileNotFoundError(
                    f"No se encontró la TPM '{filename}'. "
                    f"Buscado en: {SAMPLES_PATH.resolve()}, {alt_paths}"
                )
        if tamaño > 20 and binaria:
            with open(filepath, "r") as fh:
                first = fh.readline().strip()
            is_multi = "," in first
            if is_multi:
                return np.genfromtxt(filepath, delimiter=COLON_DELIM, dtype=np.uint8)
            # Columna única: cargar como enteros sin el paso intermedio de strings Python
            is_hex = any(c in "abcdefABCDEF" for c in first)
            base = 16 if is_hex else 10
            if base == 16:
                raw_str = np.genfromtxt(filepath, delimiter=COLON_DELIM, dtype=str)
                raw = np.array([int(v, 16) for v in raw_str.ravel()], dtype=np.uint64)
            else:
                raw = np.loadtxt(filepath, dtype=np.uint64)
            n = tamaño
            num_states = raw.shape[0]
            result = np.zeros((num_states, n), dtype=np.uint8)
            for col in range(n):
                result[:, col] = (raw >> col) & 1
            return result
        return np.genfromtxt(filepath, delimiter=COLON_DELIM)

    @staticmethod
    def generar(
        n: int,
        binaria: bool = True,
        semilla: int = 42,
    ) -> str:
        np.random.seed(semilla)

        if n < 1:
            raise ValueError("n debe ser positivo")

        num_estados = 1 << n
        total_size_gb = (num_estados * n) / (1024**3)
        estimated_time = total_size_gb * 2

        print(f"Tamaño estimado: {total_size_gb:.6f} GB")
        print(f"Tiempo estimado: {estimated_time:.1f} segundos")

        if total_size_gb > 1 and input(
          "El sistema ocupará más de 1GB. ¿Continuar? (s/n): "
        ).lower() != "s":
            return ''

        base_path = SAMPLES_PATH
        base_path.mkdir(parents=True, exist_ok=True)

        suffix = ABC_START
        filename = f"N{n}{suffix}.{CSV_EXTENSION}"
        filepath = base_path / filename
        if filepath.exists():
            respuesta = input(
                f"Ya existe N{n}{suffix}.{CSV_EXTENSION}. "
                "¿Generar nueva red? (s/n): "
            ).lower()
            if respuesta != 's':
                return filename
        
            while filepath.exists():
                suffix = chr(ord(suffix) + 1)
                filepath = base_path / f"N{n}{suffix}.{CSV_EXTENSION}"
        
        print("Generando estados...")
        start_time = time.time()

        rng = np.random.default_rng(42)
        states: np.ndarray
        if binaria:
            if n <= 20:
                states = rng.integers(2, size=(num_estados, n), dtype=np.bool)
            else:
                states = rng.integers(
                    num_estados,
                    size=num_estados,
                    dtype=np.uint64
                )
        else:
            states = rng.random((num_estados, n), np.float32)

        print(f"Generación completada en {time.time() - start_time:.2f} segundos")

        print(f"Guardando en {filepath}...")
        start_time = time.time()
        format_ = '%f'
        if binaria:
          format_ = '%d' if n <= 20 else '%x'
        
        np.savetxt(
            filepath,
            states,
            delimiter=COLON_DELIM,
            fmt=format_,
        )

        file_size_gb = os.path.getsize(filepath) / (1024**3)
        print(f"Archivo guardado: {file_size_gb:.6f} GB")
        print(f"Tiempo de guardado: {time.time() - start_time:.2f} segundos")

        return filename

#TpmLoader.generar(14)
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
    def cargar(tamano: int, pagina: str = "A") -> np.ndarray:
        filename = f"N{tamano}{pagina}.{CSV_EXTENSION}"
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
            states = rng.integers(2, size=(num_estados, n), dtype=np.bool)
        else:
            states = rng.random((num_estados, n), np.float32)

        print(f"Generación completada en {time.time() - start_time:.2f} segundos")

        print(f"Guardando en {filepath}...")
        start_time = time.time()
        np.savetxt(
            filepath,
            states,
            delimiter=COLON_DELIM,
            fmt="%d" if binaria else "%f",
        )

        file_size_gb = os.path.getsize(filepath) / (1024**3)
        print(f"Archivo guardado: {file_size_gb:.6f} GB")
        print(f"Tiempo de guardado: {time.time() - start_time:.2f} segundos")

        return filename

#TpmLoader.generar(25)
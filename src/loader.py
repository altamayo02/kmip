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
        dimensiones: int,
        semilla: int = 73,
        datos_deterministas: bool = True,
    ) -> str:
        np.random.seed(semilla)

        if dimensiones < 1:
            raise ValueError("Las dimensiones deben ser positivas")

        num_estados = 1 << dimensiones
        total_size_gb = (num_estados * dimensiones) / (1024**3)
        estimated_time = total_size_gb * 2

        print(f"Tamaño estimado: {total_size_gb:.6f} GB")
        print(f"Tiempo estimado: {estimated_time:.1f} segundos")

        if total_size_gb > 1:
            if (
                input("El sistema ocupará más de 1GB. ¿Continuar? (s/n): ").lower()
                != "s"
            ):
                return None

        base_path = SAMPLES_PATH
        base_path.mkdir(parents=True, exist_ok=True)

        suffix = ABC_START
        while (base_path / f"N{dimensiones}{suffix}.{CSV_EXTENSION}").exists():
            if (
                input(
                    f"Ya existe N{dimensiones}{suffix}.{CSV_EXTENSION}. "
                    "¿Generar nueva red? (s/n): "
                ).lower()
                != "s"
            ):
                return f"N{dimensiones}{suffix}.{CSV_EXTENSION}"
            suffix = chr(ord(suffix) + 1)

        filename = f"N{dimensiones}{suffix}.{CSV_EXTENSION}"
        filepath = base_path / filename

        print("Generando estados...")
        start_time = time.time()

        if datos_deterministas:
            states = np.random.randint(
                2, size=(num_estados, dimensiones), dtype=np.int8
            )
        else:
            states = np.random.random(size=(num_estados, dimensiones))

        print(f"Generación completada en {time.time() - start_time:.2f} segundos")

        print(f"Guardando en {filepath}...")
        start_time = time.time()
        np.savetxt(
            filepath,
            states,
            delimiter=COLON_DELIM,
            fmt="%d" if datos_deterministas else "%.6f",
        )

        file_size_gb = os.path.getsize(filepath) / (1024**3)
        print(f"Archivo guardado: {file_size_gb:.6f} GB")
        print(f"Tiempo de guardado: {time.time() - start_time:.2f} segundos")

        return filename

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.loader import TpmLoader, SAMPLES_PATH

SEMILLA = 42


def generar_tpm(N, pagina, deterministico=False):
    filename = f"N{N}{pagina}.csv"
    filepath = SAMPLES_PATH / filename
    if filepath.exists():
        print(f"  {filename} ya existe, omitiendo.")
        return

    num_estados = 1 << N
    size_mb = (num_estados * N * 8) / (1024**2)
    print(f"  Generando {filename} ({num_estados} x {N}, ~{size_mb:.1f} MB)...")

    np.random.seed(SEMILLA)
    if deterministico:
        data = np.random.randint(2, size=(num_estados, N), dtype=np.int8)
    else:
        data = np.random.random(size=(num_estados, N))

    SAMPLES_PATH.mkdir(parents=True, exist_ok=True)
    np.savetxt(filepath, data, delimiter=",", fmt="%.18f" if not deterministico else "%d")
    print(f"  {filename} guardado.")


def main():
    paginas_a_generar = [(16, "A"), (17, "A"), (18, "A")]

    print("Generando TPMs probabilisticos...")
    for N, pagina in paginas_a_generar:
        generar_tpm(N, pagina, deterministico=False)

    print("\nListo.")


if __name__ == "__main__":
    main()

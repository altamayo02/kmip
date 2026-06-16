import os
import sys

os.environ["PYTHONIOENCODING"] = "utf-8"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from _qnodes_base import process_sheet
from src.functions.gpu_backend import HAS_GPU

EXCEL = os.path.join(os.path.dirname(__file__), "..", "data/evaluation/MIKP - Datos Prueba.xlsx")
if __name__ == "__main__":
    print(f"CUDA disponible: {HAS_GPU}")
    if not HAS_GPU:
        print("  [ADVERTENCIA] No se detecto GPU. Instala cupy: pip install cupy-cuda11x")
    process_sheet(EXCEL, "25A-Elementos ", N=25, page="B")

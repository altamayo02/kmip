import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from _phi_base import process_sheet

EXCEL = os.path.join(os.path.dirname(__file__), "..", "data/evaluation/MIKP - Datos Prueba.xlsx")
if __name__ == "__main__":
    process_sheet(EXCEL, "15B-Phi", N=15, page="B", template_sheet_name="15B-Elementos")

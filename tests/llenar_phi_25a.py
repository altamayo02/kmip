import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from _phi_base import process_sheet

EXCEL = os.path.join(os.path.dirname(__file__), "..", "data/evaluation/MIKP - Datos Prueba.xlsx")
if __name__ == "__main__":
    process_sheet(EXCEL, "25A-Phi ", N=25, page="A", template_sheet_name="25A-Elementos ")

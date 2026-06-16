import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from _geometric_base import process_sheet

EXCEL = os.path.join(os.path.dirname(__file__), "..", "data/evaluation/MIKP - Datos Prueba.xlsx")
SHEET = "25A-Elementos "
process_sheet(EXCEL, SHEET, N=25, page="A")

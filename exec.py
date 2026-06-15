import sys
from src.main import iniciar


def main():
		estrategia = sys.argv[1] if len(sys.argv) > 1 else "BruteForce"
		sufijo_csv = sys.argv[2] if len(sys.argv) > 2 else "N3A"
		k = int(sys.argv[3]) if len(sys.argv) > 3 else 2

		if "A" in sufijo_csv[1:]:
			n = int(sufijo_csv[1:sufijo_csv.index("A")])
		elif "B" in sufijo_csv[1:]:
			n = int(sufijo_csv[1:sufijo_csv.index("B")])
		else:
			n = int(sufijo_csv[1:sufijo_csv.index("C")])
		pagina = sufijo_csv[-1]
		estado_inicial = "1" + "0" * (n - 1)

		iniciar(
				estado_inicial=estado_inicial,
				condiciones="1" * n,
				alcance="1" * n,
				mecanismo="1" * n,
				estrategia=estrategia,
				pagina=pagina,
				k=k
		)


if __name__ == "__main__":
		main()

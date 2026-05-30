import sys
from src.main import iniciar


def main():
    estrategia = sys.argv[1] if len(sys.argv) > 1 else "BruteForce"

    iniciar(
        estado_inicial="100",
        condiciones="111",
        alcance="111",
        mecanismo="111",
        estrategia=estrategia,
        pagina="A",
    )


if __name__ == "__main__":
    main()

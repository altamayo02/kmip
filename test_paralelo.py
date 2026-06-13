"""Prueba rapida de BruteForce secuencial vs paralelo con N grande."""
import sys, os, time

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from src.middlewares.profile import profiler_manager
    profiler_manager.enabled = False

    from src.config import Config
    from src.loader import TpmLoader
    from src.strategies.brute_force import BruteForce
    from src.strategies.brute_force_opt import BruteForce_Opt

    N = 10
    PAG = "A"
    tpm = TpmLoader.cargar(N, PAG)
    config = Config(pagina_muestra=f"{N}{PAG}", profiler_habilitado=False)

    estado = "1" + "0" * (N - 1)
    cond = "1" * N

    for label, cls in [("Secuencial", BruteForce), ("Paralelo", BruteForce_Opt)]:
        print(f"\nCorriendo {label} N={N}...")
        inst = cls(tpm, config)
        t0 = time.perf_counter()
        sol = inst.aplicar_estrategia(estado, cond, cond, cond)
        t1 = time.perf_counter()
        print(f"  {label}: {t1-t0:.2f}s  phi={sol[0].perdida:.6f}  soluciones={len(sol)}")

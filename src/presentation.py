from threading import Thread
from typing import Optional

from colorama import init, Fore, Style
import numpy as np
import pyttsx3
from pyttsx3.engine import Engine
from pyttsx3.voice import Voice

from src.solution import Solution
from src.config import Config


init()


def mostrar_solucion(soluciones: list[Solution], config: Config) -> None:
    if not soluciones:
        return

    espaciado = 64
    bilinea = "\u2550" * espaciado
    trilinea = "\u2261" * espaciado

    def formatear_distribucion(
        distribucion: np.ndarray,
        evitar_desbordamiento=True,
    ):
        rango = distribucion.size
        mensaje_desborde = ""
        if evitar_desbordamiento:
            LIMITE = espaciado
            excedente = rango - LIMITE
            if excedente > 0:
                mensaje_desborde = f" {excedente} valores m\u00e1s.."
                rango = LIMITE

        datos = " ".join(
            f"{Fore.WHITE}{distribucion[idx]:.4f}"
            if distribucion[idx] > 0
            else f"{Fore.LIGHTBLACK_EX}0.    "
            for idx in range(rango)
        )
        return f"[ {datos}{mensaje_desborde} {Fore.WHITE}]"

    solucion = soluciones[0]

    if solucion.quiere_hablar:
        voz = Thread(target=_anunciar_solucion, args=(solucion,))
        voz.start()

    es_pyphi = solucion.estrategia == "Pyphi"
    tipo_distribucion = "tensorial" if es_pyphi else "marginal"

    tiempo_hrs = f"{solucion.tiempo_ejecucion / 3600:.2f}"
    tiempo_min = f"{solucion.tiempo_ejecucion / 60:.1f}"
    tiempo_seg = f"{solucion.tiempo_ejecucion:.4f}"

    output = f"""{Fore.CYAN}{bilinea}

{Fore.RED}{solucion.estrategia} fue la estrategia de solucion.

{Fore.BLUE}Distancia métrica utilizada:
{Fore.WHITE}{config.distancia_metrica}
{Fore.BLUE}Notación utilizada en indexación:
{Fore.WHITE}{config.notacion_indexado}

{Fore.YELLOW}Distribucion {tipo_distribucion} del Subsistema:
{Style.RESET_ALL}{formatear_distribucion(solucion.distribucion_subsistema)}
"""

    for i, sol in enumerate(soluciones):
        output += f"""

{'-' * espaciado}
{Fore.YELLOW}Bi-Particion {i + 1} de {len(soluciones)}:
{Style.RESET_ALL}Distribucion {tipo_distribucion} de la Particion:
{formatear_distribucion(sol.distribucion_particion)}

{Fore.MAGENTA}{sol.particion}{Style.RESET_ALL}"""

    output += f"""

{Fore.YELLOW}Perdida minima ( {chr(966)} ) = {solucion.perdida:.4f}{Fore.WHITE}

{Fore.BLUE}Tiempos de ejecucion:
{Fore.WHITE}Horas: {tiempo_hrs} = Minutos: {tiempo_min} = Segundos: {tiempo_seg}

{Fore.CYAN}{trilinea}{Style.RESET_ALL}"""

    try:
        print(output)
    except UnicodeEncodeError:
        safe = output.replace("\u2550", "=")
        safe = safe.replace("\u2261", "=")
        safe = safe.replace("\u03c6", "phi")
        safe = safe.replace("\u239b", "(")
        safe = safe.replace("\u239e", ")")
        safe = safe.replace("\u239d", "(")
        safe = safe.replace("\u23a0", ")")
        safe = safe.replace("\u2205", "0")
        print(safe)


def _anunciar_solucion(solucion: Solution) -> None:
    try:
        motor = pyttsx3.init()

        id_voz = _obtener_voz_espanol(motor)
        if id_voz:
            motor.setProperty("voice", id_voz)

        motor.setProperty("rate", 150)
        motor.setProperty("volume", 0.9)

        mensaje = f"Solución encontrada con {solucion.estrategia}." + (
            f"El valor de fi es de {solucion.perdida:.2f}"
            if solucion.perdida > 0
            else "No hubo pérdida."
        )
        motor.say(mensaje)
        motor.runAndWait()
    except Exception as e:
        print(f"Error al inicializar el motor de voz: {e}")


def _obtener_voz_espanol(motor: Engine) -> Optional[str]:
    voces: list[Voice] = motor.getProperty("voices")

    prioridades = [
        ("sabina", "méxico"),
        ("helena", "españa"),
        ("spanish", None),
        ("español", None),
        ("es-", None),
    ]

    for nombre_buscado, region in prioridades:
        for voz in voces:
            nombre_voz = voz.name.lower()
            id_voz = voz.id.lower()

            if nombre_buscado in nombre_voz or nombre_buscado in id_voz:
                if region is None or region in nombre_voz:
                    return voz.id

    return voces[0].id if voces else None

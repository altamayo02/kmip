from dataclasses import dataclass, field
from src.models.enums import MetricDistance, Notation, TimeEMD


@dataclass
class Config:
    semilla_numpy: int = 73
    pagina_muestra: str = "A"
    distancia_metrica: str = MetricDistance.HAMMING.value
    notacion_indexado: str = Notation.LIL_ENDIAN.value
    tiempo_emd: str = TimeEMD.EMD_EFECTO.value
    profiler_habilitado: bool = True

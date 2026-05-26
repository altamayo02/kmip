from enum import Enum


class MetricDistance(Enum):
    HAMMING = "distancia-hamming"
    MANHATTAN = "distancia-manhattan"
    EUCLIDIANA = "distancia-euclidiana"
    EMD_EFECTO = "emd-effect"
    EMD_CAUSA = "emd-cause"


class Notation(Enum):
    LIL_ENDIAN = "little-endian"
    BIG_ENDIAN = "big-endian"
    GRAY_CODE = "gray-code"
    SIGN_MAGNITUDE = "sign-magnitude"
    TWOS_COMPLEMENT = "two's-complement"


class TimeEMD(Enum):
    EMD_EFECTO = "emd-effect"
    EMD_CAUSA = "emd-cause"
    EMD_INTEGRADA = "emd-cause-effect"

import sys
import logging
from pathlib import Path
from datetime import datetime
from functools import wraps
from typing import Any, Callable

from colorama import init, Fore, Style


LOGS_PATH = Path(".logs")


class ColorFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: Fore.LIGHTBLACK_EX,
        logging.INFO: Fore.BLUE,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.MAGENTA,
        logging.FATAL: Fore.RED + Style.BRIGHT,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        init(autoreset=True)

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelno, "")
        original_levelname = record.levelname
        record.levelname = f"{color}{original_levelname}{Style.RESET_ALL}"
        formatted = super().format(record)
        record.levelname = original_levelname
        return formatted


class SafeLogger:
    def __init__(self, name: str):
        self._logger = self._setup_logger(name)

    def _safe_str(self, obj: Any) -> str:
        try:
            if isinstance(obj, (list, tuple, set, dict)):
                return str(obj)
            return str(obj).encode("utf-8", errors="replace").decode("utf-8")
        except Exception:
            return "[Objeto no representable]"

    def _safe_format(self, *args, **kwargs) -> str:
        args_str = " ".join(self._safe_str(arg) for arg in args)
        if kwargs:
            kwargs_str = " ".join(f"{k}={self._safe_str(v)}" for k, v in kwargs.items())
            return f"{args_str} {kwargs_str}"
        return args_str

    def _setup_logger(self, name: str) -> logging.Logger:
        base_log_dir = LOGS_PATH
        base_log_dir.mkdir(exist_ok=True)

        current_time = datetime.now()
        date_dir = base_log_dir / current_time.strftime("%d_%m_%Y")
        date_dir.mkdir(exist_ok=True)

        hour_dir = date_dir / f"{current_time.strftime('%H')}hrs"
        hour_dir.mkdir(exist_ok=True)

        detailed_log_file = hour_dir / f"{name}.log"
        last_log_file = base_log_dir / f"last_{name}.log"

        logger = logging.getLogger(name)
        logger.setLevel(logging.ERROR)
        logger.propagate = False
        logger.handlers.clear()

        plain_formatter = logging.Formatter(
            "%(asctime)s [%(name)s] %(levelname)s %(processName)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        colored_formatter = ColorFormatter(
            "%(levelname)s (%(asctime)s): %(message)s",
            datefmt="%H:%M:%S",
        )

        detailed_file_handler = logging.FileHandler(
            detailed_log_file, mode="w", encoding="utf-8"
        )
        detailed_file_handler.setLevel(logging.DEBUG)
        detailed_file_handler.setFormatter(plain_formatter)

        last_file_handler = logging.FileHandler(
            last_log_file, mode="w", encoding="utf-8"
        )
        last_file_handler.setLevel(logging.DEBUG)
        last_file_handler.setFormatter(plain_formatter)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(colored_formatter)

        logger.addHandler(detailed_file_handler)
        logger.addHandler(last_file_handler)
        logger.addHandler(console_handler)

        return logger

    def set_log(self, level: int, *args, **kwargs) -> None:
        message = self._safe_format(*args, **kwargs)
        self._logger.log(level, message)

    def debug(self, *args, **kwargs) -> None:
        self.set_log(logging.DEBUG, *args, **kwargs)

    def info(self, *args, **kwargs) -> None:
        self.set_log(logging.INFO, *args, **kwargs)

    def warn(self, *args, **kwargs) -> None:
        self.set_log(logging.WARNING, *args, **kwargs)

    def error(self, *args, **kwargs) -> None:
        self.set_log(logging.ERROR, *args, **kwargs)

    def critic(self, *args, **kwargs) -> None:
        self.set_log(logging.CRITICAL, *args, **kwargs)

    def fatal(self, *args, **kwargs) -> None:
        self.set_log(logging.FATAL, *args, **kwargs)

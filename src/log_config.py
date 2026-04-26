import logging
import sys
from pathlib import Path
from typing import Optional


logger = logging.getLogger(__name__)


def setup_logging(
    console_level: str = "INFO",
    file_level: str = "DEBUG",
    log_file: Optional[Path] = None,
) -> None:
    """Configure console and file logging after CLI options are parsed."""
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.DEBUG)

    detailed_fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(getattr(logging, console_level.upper()))
    ch.setFormatter(detailed_fmt)
    root.addHandler(ch)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setLevel(getattr(logging, file_level.upper()))
        fh.setFormatter(detailed_fmt)
        root.addHandler(fh)


def log_command_args(command: str, **kwargs) -> None:
    """Log parsed click arguments for a command."""
    logger.info("%s arguments: %s", command, kwargs)

"""Logging setup. One function, called once from cli.main, before anything else logs.

Per-run log file is created in log_dir, named with a timestamp. Stdout mirror is
attached so the user sees output live; tqdm bars go to stderr (default) and don't
interleave with our log lines.
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime


_LEVEL_NAMES = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}

# Third-party libraries that flood INFO with model-loading details we don't care
# about by default. Stays at WARNING unless the caller asks for DEBUG explicitly.
_NOISY_LIBS = ("transformers", "accelerate", "PIL", "urllib3", "filelock", "matplotlib")


def parse_level(level: str | int) -> int:
    if isinstance(level, int):
        return level
    name = str(level).strip().upper()
    if name not in _LEVEL_NAMES:
        raise ValueError(f"invalid log level: {level!r}; pick one of {sorted(_LEVEL_NAMES)}")
    return getattr(logging, name)


def setup_logging(log_dir: str = "./logs", level: int | str = logging.INFO) -> str:
    """Wire root logger to a per-run file + stdout. Returns the log file path."""
    level_int = parse_level(level)

    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    log_path = os.path.join(log_dir, f"viral-{ts}.log")

    root = logging.getLogger()
    root.setLevel(level_int)

    # Clear prior handlers — setup_logging is safe to call multiple times.
    for h in list(root.handlers):
        root.removeHandler(h)

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    root.addHandler(stream)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    # Suppress library noise unless the user explicitly asked for DEBUG.
    if level_int > logging.DEBUG:
        for name in _NOISY_LIBS:
            logging.getLogger(name).setLevel(logging.WARNING)

    return log_path

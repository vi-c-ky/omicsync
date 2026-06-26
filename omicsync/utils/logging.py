"""Consistent logging setup for omicsync."""

import logging
from typing import Optional

_logger = logging.getLogger("omicsync")

if not _logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] omicsync: %(message)s",
                          datefmt="%H:%M:%S")
    )
    _logger.addHandler(_handler)
    _logger.setLevel(logging.WARNING)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return the omicsync logger or a child logger.

    Parameters
    ----------
    name:
        Optional child name, e.g. ``"loaders.csv"``.

    Returns
    -------
    logging.Logger
    """
    if name:
        return logging.getLogger(f"omicsync.{name}")
    return _logger


def set_verbose(verbose: bool) -> None:
    """Enable or disable verbose (INFO-level) logging.

    Parameters
    ----------
    verbose:
        ``True`` to enable INFO logging, ``False`` to restore WARNING level.
    """
    level = logging.INFO if verbose else logging.WARNING
    _logger.setLevel(level)

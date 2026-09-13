from __future__ import annotations

import logging


def setup_logging(level=logging.INFO):
    """
    Configure logging for the application.
    """

    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")


def get_logger(name=None):
    """
    Get a logger instance with the specified name.
    """
    return logging.getLogger(name)

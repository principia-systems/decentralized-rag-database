"""
Logging utilities for src.

This module provides consistent logging configuration and utility functions
for the src package.
"""

import logging
import os
import sys
from pathlib import Path


def setup_logger(name=None, level=None, log_file=None):
    """
    Configure and return a logger with consistent formatting.

    Args:
        name: Logger name (defaults to root logger if None)
        level: Logging level (defaults to INFO if None or if env var not set)
        log_file: Optional file path to write logs to

    Returns:
        Configured logger instance
    """
    # Get logger level from environment or use default
    if level is None:
        level_name = os.environ.get("SRC_LOG_LEVEL", "INFO")
        level = getattr(logging, level_name.upper(), logging.INFO)

    # Get or create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers to avoid duplicates when called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create formatters
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        # Ensure directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name=None):
    """
    Get an existing logger or create a new one with default settings.

    Args:
        name: Logger name (optional)

    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)

    # If logger doesn't have handlers, set it up
    if not logger.hasHandlers():
        # Check if log directory is set in environment
        log_dir = os.environ.get("SRC_LOG_DIR")
        log_file = None

        if log_dir:
            # Create path based on logger name
            log_filename = f"{name or 'src'}.log"
            log_file = Path(log_dir) / log_filename

        logger = setup_logger(name, log_file=log_file)

    return logger

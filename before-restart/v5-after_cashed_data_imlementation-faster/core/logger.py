"""
core/logger.py
──────────────
Structured logging via loguru.
Import `log` everywhere – never use print() for operational messages.
"""
from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

from config.settings import settings


def setup_logger() -> None:
    logger.remove()  # Remove default handler

    # Console – human-readable
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> – {message}",
        level="INFO",
        colorize=True,
    )

    # File – JSON structured for audit trail
    log_dir = settings.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.add(
        log_dir / "app_{time:YYYY-MM-DD}.log",
        rotation="00:00",        # Rotate at midnight
        retention="30 days",
        compression="zip",
        format="{time:YYYY-MM-DDTHH:mm:ss.SSSZ} | {level} | {name}:{line} | {message}",
        level="DEBUG",
        serialize=False,
        enqueue=True,            # Thread-safe
    )

    # Separate trade audit log – never rotated away without archive
    logger.add(
        log_dir / "trades.log",
        rotation="100 MB",
        retention=None,          # Keep forever
        compression="zip",
        filter=lambda r: "TRADE" in r["message"] or r["extra"].get("trade_audit"),
        format="{time:YYYY-MM-DDTHH:mm:ss.SSSZ} | {level} | {message}",
        level="INFO",
        enqueue=True,
    )


setup_logger()
log = logger

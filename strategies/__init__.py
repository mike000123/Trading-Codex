"""
Import all strategies here so they self-register via @register_strategy.
To add a new strategy: create the file, add the import below.
"""
from .base import BaseStrategy, register_strategy, get_strategy, list_strategies

# Auto-register all built-in strategies
from . import rsi_strategy       # noqa: F401
from . import ma_crossover       # noqa: F401
from . import macd_strategy      # noqa: F401

__all__ = ["BaseStrategy", "register_strategy", "get_strategy", "list_strategies"]

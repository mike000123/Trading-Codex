"""
Import all strategies here so they self-register via @register_strategy.
To add a new strategy: create the file, add the import below.
"""
from .base import BaseStrategy, register_strategy, get_strategy, list_strategies

# Auto-register all built-in strategies
from . import rsi_strategy           # noqa: F401
from . import ma_crossover           # noqa: F401
from . import macd_strategy          # noqa: F401
from . import fixed_level_strategy   # noqa: F401
from . import vwap_rsi_strategy      # noqa: F401  — GC=F intraday
from . import bollinger_rsi_strategy # noqa: F401  — UVXY mean reversion
from . import atr_rsi_strategy       # noqa: F401  — adaptive, both instruments
from . import ema_trend_rsi_strategy # noqa: F401  — GC=F trend-following
from . import earnings_event_hybrid_strategy  # noqa: F401  — unified research-only earnings event hybrid
from . import earnings_positive_continuation_strategy  # noqa: F401  — standalone research-only positive earnings continuation
from . import earnings_overshoot_hybrid_strategy  # noqa: F401  — research-only earnings wave / failed-reclaim hybrid
from . import earnings_negative_rebound_strategy  # noqa: F401  — research-only negative earnings rebound long
from . import earnings_negative_hybrid_strategy  # noqa: F401  — research-only negative earnings rebound / failure selector

__all__ = ["BaseStrategy", "register_strategy", "get_strategy", "list_strategies"]

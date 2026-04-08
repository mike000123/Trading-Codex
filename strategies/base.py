"""
strategies/base.py
──────────────────
Abstract base for all strategies.
Every strategy must subclass BaseStrategy and implement `generate_signal()`.

Registration:
    Strategies self-register via the @register_strategy decorator so the UI
    can discover them dynamically without any manual wiring.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Optional, Type

import pandas as pd

from core.models import Signal, SignalAction


# ─── Registry ────────────────────────────────────────────────────────────────

_REGISTRY: dict[str, Type["BaseStrategy"]] = {}


def register_strategy(cls: Type["BaseStrategy"]) -> Type["BaseStrategy"]:
    """Class decorator that adds the strategy to the global registry."""
    _REGISTRY[cls.strategy_id] = cls
    return cls


def get_strategy(strategy_id: str) -> Type["BaseStrategy"]:
    if strategy_id not in _REGISTRY:
        raise KeyError(f"Unknown strategy '{strategy_id}'. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[strategy_id]


def list_strategies() -> list[dict[str, str]]:
    return [
        {"id": sid, "name": cls.name, "description": cls.description}
        for sid, cls in _REGISTRY.items()
    ]


# ─── Base class ──────────────────────────────────────────────────────────────

class BaseStrategy(ABC):
    """
    Interface every strategy must implement.

    Class attributes (define in subclass):
        strategy_id  – machine-readable unique key  e.g. "rsi_threshold"
        name         – human-readable display name
        description  – short explanation shown in UI

    Instance:
        params       – dict of user-configurable parameters passed on __init__
    """

    strategy_id: ClassVar[str] = ""
    name: ClassVar[str] = "Unnamed Strategy"
    description: ClassVar[str] = ""

    def __init__(self, params: Optional[dict[str, Any]] = None) -> None:
        self.params: dict[str, Any] = params or {}

    @abstractmethod
    def generate_signal(self, data: pd.DataFrame, symbol: str) -> Signal:
        """
        Analyse `data` (OHLCV DataFrame) and return a Signal.

        Args:
            data    – standard OHLCV frame (columns: date, open, high, low, close, volume)
            symbol  – ticker being analysed

        Returns:
            Signal dataclass with action BUY / SELL / HOLD
        """

    def generate_signals_bulk(
        self,
        data:   "pd.DataFrame",
        symbol: str,
    ) -> tuple[list, list]:
        """
        Pre-compute signals for ALL bars in one pass — used by BacktestEngine
        for a 50-100× speedup on large datasets.

        Returns (actions_list, meta_list) where each list is parallel to data rows.
        meta_list[i] = {"suggested_tp": float|None, "suggested_sl": float|None,
                         "metadata": dict}

        Default: raises NotImplementedError → engine falls back to bar-by-bar.
        Override in strategies that vectorise their indicators.
        """
        raise NotImplementedError

    def default_params(self) -> dict[str, Any]:
        """Return {param_name: default_value} for UI form generation."""
        return {}

    def symbol_param_overrides(
        self,
        symbol: str,
        source: Optional[str] = None,
        interval: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Optional symbol-specific default overrides.

        Use this for instrument families that benefit from the same strategy
        architecture but need different baseline thresholds.
        """
        return {}

    def effective_default_params(
        self,
        symbol: Optional[str] = None,
        source: Optional[str] = None,
        interval: Optional[str] = None,
    ) -> dict[str, Any]:
        params = dict(self.default_params())
        if symbol:
            params.update(self.symbol_param_overrides(symbol, source=source, interval=interval))
        return params

    def resolve_params(
        self,
        symbol: Optional[str] = None,
        source: Optional[str] = None,
        interval: Optional[str] = None,
    ) -> dict[str, Any]:
        params = self.effective_default_params(symbol=symbol, source=source, interval=interval)
        params.update(self.params)
        return params

    def validate_params(self) -> list[str]:
        """Return list of validation error strings (empty = valid)."""
        return []

    def companion_symbols(
        self,
        symbol: str,
        source: Optional[str] = None,
        interval: Optional[str] = None,
    ) -> list[str]:
        """
        Optional companion symbols required by the strategy.

        The data pipeline can use this to fetch/algn extra market context
        before the strategy runs in backtests, forward tests, or paper trading.
        """
        return []

    def companion_contexts(
        self,
        symbol: str,
        source: Optional[str] = None,
        interval: Optional[str] = None,
    ) -> list[str]:
        """
        Generic context dependencies requested by the strategy.

        Example:
            ["equity_benchmark"]

        The data pipeline resolves these context types to real symbols using the
        primary ticker's symbol profile.
        """
        return []

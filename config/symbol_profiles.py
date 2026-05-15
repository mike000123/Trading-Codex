"""
config/symbol_profiles.py
Strategy-agnostic symbol context mappings.

Strategies can request generic context types such as "equity_benchmark".
This module resolves those context needs to actual companion tickers per symbol.
"""
from __future__ import annotations

from typing import Any, Optional


_SYMBOL_CONTEXTS: dict[str, dict[str, str]] = {
    "APLD": {
        "equity_benchmark": "QQQ",
        "crypto_benchmark": "BTC-USD",
    },
    "GLD": {
        "dollar_benchmark": "UUP",
        "rates_benchmark": "IEF",
        "precious_metal_peer": "SLV",
        "miners_proxy": "GDX",
        "riskoff_proxy": "VIXY",
    },
    "SLV": {
        "dollar_benchmark": "UUP",
        "rates_benchmark": "IEF",
        "precious_metal_peer": "GLD",
        "miners_proxy": "GDX",
        "riskoff_proxy": "VIXY",
    },
    "UVXY": {
        "equity_benchmark": "SPY",
    },
    "VXZ": {
        "equity_benchmark": "SPY",
    },
    "VXX": {
        "equity_benchmark": "SPY",
    },
    "VIXY": {
        "equity_benchmark": "SPY",
    },
    "VIXM": {
        "equity_benchmark": "SPY",
    },
    "UVIX": {
        "equity_benchmark": "SPY",
    },
    "SVXY": {
        "equity_benchmark": "SPY",
    },
    "SVIX": {
        "equity_benchmark": "SPY",
    },
}

_SYMBOL_METADATA: dict[str, dict[str, Any]] = {
    "APLD": {
        "execution_defaults": {
            "monday_open_delay": False,
        },
    },
}

_CONTEXT_PREFIX: dict[str, str] = {
    "equity_benchmark": "benchmark",
    "crypto_benchmark": "crypto",
    "dollar_benchmark": "dollar",
    "rates_benchmark": "rates",
    "long_rates_benchmark": "long_rates",
    "precious_metal_peer": "metal_peer",
    "miners_proxy": "miners",
    "riskoff_proxy": "riskoff",
}

_CONTEXT_LABEL: dict[str, str] = {
    "equity_benchmark": "Equity benchmark",
    "crypto_benchmark": "Crypto benchmark",
    "dollar_benchmark": "Dollar benchmark",
    "rates_benchmark": "Rates benchmark",
    "long_rates_benchmark": "Long-rates benchmark",
    "precious_metal_peer": "Precious-metals peer",
    "miners_proxy": "Gold miners proxy",
    "riskoff_proxy": "Risk-off proxy",
}


def resolve_context_symbol(primary_symbol: str, context_key: str) -> Optional[str]:
    symbol = primary_symbol.strip().upper()
    return _SYMBOL_CONTEXTS.get(symbol, {}).get(context_key)


def resolve_symbol_metadata(primary_symbol: str, metadata_key: str, default: Any = None) -> Any:
    symbol = primary_symbol.strip().upper()
    return _SYMBOL_METADATA.get(symbol, {}).get(metadata_key, default)


def resolve_execution_default(primary_symbol: str, setting_key: str, default: Any = None) -> Any:
    symbol = primary_symbol.strip().upper()
    settings_map = _SYMBOL_METADATA.get(symbol, {}).get("execution_defaults", {})
    if not isinstance(settings_map, dict):
        return default
    return settings_map.get(setting_key, default)


def context_prefix(context_key: str) -> str:
    return _CONTEXT_PREFIX.get(context_key, context_key)


def context_label(context_key: str) -> str:
    return _CONTEXT_LABEL.get(context_key, context_key.replace("_", " ").title())

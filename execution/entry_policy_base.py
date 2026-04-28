"""
execution/entry_policy_base.py
──────────────────────────────
Shared types for paper-trading entry policies.

An "entry policy" is the set of pre-fill gates that decide whether a signal
gets routed to the OrderRouter, and whether it needs any capital adjustment
before execution. We keep the policy behind a small uniform API so the
paper-trading loop doesn't need to know which flavour is active — the user
can swap them at runtime via a dropdown.

Concrete policies live in their own modules so additional variants (e.g. a
broker-specific preset) can be added without touching anything else:

  - execution/entry_policy_classic.py  → ClassicEntryPolicy  (no gates)
  - execution/entry_policy_alpaca.py   → AlpacaEntryPolicy   (Tier-1 gates)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Protocol

import pandas as pd

from core.models import Direction


@dataclass
class EntryContext:
    """Everything a policy needs to decide on a single entry attempt."""
    symbol: str
    direction: Direction
    entry_price: float
    bar: Any                       # latest bar (pd.Series or dict)
    bar_time: Any                  # timestamp of the bar
    prices: pd.DataFrame           # recent history incl. current bar (date, close, ...)
    requested_capital: float
    leverage: float
    portfolio_equity: float
    portfolio_trades: Iterable     # all paper trades (for PDT rolling count)


@dataclass
class EntryDecision:
    """Uniform outcome returned by every policy."""
    allowed: bool
    skip_reason: str = ""
    adjusted_capital: Optional[float] = None  # None → keep requested_capital
    notes_prefix: str = ""                    # appended to trade.notes before OrderRouter call
    post_fill_note: str = ""                  # appended to trade.notes after fill

    @classmethod
    def allow(cls, **kw) -> "EntryDecision":
        return cls(allowed=True, **kw)

    @classmethod
    def reject(cls, reason: str) -> "EntryDecision":
        return cls(allowed=False, skip_reason=reason)


class EntryPolicy(Protocol):
    """Uniform protocol every policy must satisfy."""
    name: str
    label: str

    def evaluate(self, ctx: EntryContext) -> EntryDecision: ...


# ── Factory ─────────────────────────────────────────────────────────────────

_POLICIES: dict[str, type] = {}


def register_policy(name: str, cls: type) -> None:
    """Register a policy class under `name` (case-insensitive)."""
    _POLICIES[name.lower()] = cls


def get_policy(name: str, **kwargs) -> EntryPolicy:
    """
    Build a policy by registered name. Unknown names fall back to 'classic'
    so a config typo never breaks the paper loop.
    """
    key = (name or "classic").lower()
    cls = _POLICIES.get(key) or _POLICIES["classic"]
    return cls(**kwargs)


def available_policies() -> list[tuple[str, str]]:
    """Return [(name, human label), ...] for UI dropdowns."""
    out = []
    for name, cls in _POLICIES.items():
        label = getattr(cls, "label", name.title())
        out.append((name, label))
    # Stable order: classic first, then by name
    out.sort(key=lambda t: (0 if t[0] == "classic" else 1, t[0]))
    return out

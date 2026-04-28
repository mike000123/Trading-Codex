"""
execution/alpaca_constraints.py
───────────────────────────────
Alpaca-realistic execution gates shared by backtest and paper simulation.

Goal: when the sim allows a trade, Alpaca would allow it too. When Alpaca
would reject or restrict it, the sim reflects that by skipping the trade
with a clear reason — so simulated P&L is not inflated by trades that
couldn't happen live.

Constraints implemented:
  1. Regular Trading Hours + NYSE holiday calendar
  2. Pattern Day Trader (PDT) rule for margin accounts under $25k
  3. Short-Sale Restriction (SSR) heuristic — prior day close to current < -10%
  4. Shortability check — Alpaca's asset.shortable / easy_to_borrow flags
  5. Fractional-share rules — Alpaca requires integer qty for shorts
  6. Fill-timing diagnostic — records bar context for later cost calibration

Usage (paper / backtest):
    from execution.alpaca_constraints import (
        is_regular_trading_hour, is_trading_day,
        pdt_guard, ssr_guard, shortable_guard,
        normalize_qty_for_direction, fill_timing_note,
    )
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import timedelta
from typing import Callable, Iterable, Optional, Union

import pandas as pd

try:
    import pandas_market_calendars as mcal
    _NYSE = mcal.get_calendar("NYSE")
except Exception:  # pragma: no cover
    _NYSE = None

from core.models import Direction


# ── Market hours ─────────────────────────────────────────────────────────────

def _coerce_utc(ts) -> Optional[pd.Timestamp]:
    try:
        t = pd.Timestamp(ts)
    except Exception:
        return None
    if t is pd.NaT:
        return None
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return t


def is_trading_day(ts) -> bool:
    """True if `ts`'s NYSE calendar date is a full trading day."""
    t = _coerce_utc(ts)
    if t is None or _NYSE is None:
        return True  # fail-open so a misconfigured calendar doesn't block everything
    date_str = t.tz_convert("America/New_York").strftime("%Y-%m-%d")
    try:
        sched = _NYSE.schedule(start_date=date_str, end_date=date_str)
    except Exception:
        return True
    return not sched.empty


def is_regular_trading_hour(ts, *, extended_hours: bool = False) -> bool:
    """
    True if `ts` is inside NYSE regular trading hours (09:30-16:00 ET).
    If extended_hours=True, also accepts premarket (04:00-09:30 ET) and
    after-hours (16:00-20:00 ET). Always False on non-trading days.
    """
    t = _coerce_utc(ts)
    if t is None:
        return False
    if _NYSE is None:
        # Fall-open if calendar isn't installed (keep behaviour predictable)
        return True

    et = t.tz_convert("America/New_York")
    date_str = et.strftime("%Y-%m-%d")
    try:
        sched = _NYSE.schedule(start_date=date_str, end_date=date_str)
    except Exception:
        return True
    if sched.empty:
        return False  # holiday / weekend

    open_ts = sched.iloc[0]["market_open"]
    close_ts = sched.iloc[0]["market_close"]
    if open_ts <= t <= close_ts:
        return True

    if extended_hours:
        pre_start = et.normalize() + pd.Timedelta(hours=4)    # 04:00 ET
        pre_end = et.normalize() + pd.Timedelta(hours=9, minutes=30)
        post_start = et.normalize() + pd.Timedelta(hours=16)
        post_end = et.normalize() + pd.Timedelta(hours=20)
        return (pre_start <= et < pre_end) or (post_start < et <= post_end)

    return False


def monday_open_delay_guard(
    ts,
    *,
    enforce: bool = True,
    delay_minutes: int = 30,
) -> tuple[bool, str]:
    """
    Optional user-defined execution gate:
    block entries on Mondays for `delay_minutes` after the regular NYSE open.

    This is independent from Alpaca's own rules; we expose it as a practical
    "cool-off" gate that can be applied consistently across backtest, forward
    test, and paper trading.
    """
    if not enforce:
        return True, ""

    t = _coerce_utc(ts)
    if t is None:
        return True, ""

    et = t.tz_convert("America/New_York")
    if et.weekday() != 0:  # Monday
        return True, ""

    if _NYSE is None:
        open_et = et.normalize() + pd.Timedelta(hours=9, minutes=30)
        blocked_until = open_et + pd.Timedelta(minutes=int(delay_minutes))
        if open_et <= et < blocked_until:
            return False, (
                f"Monday open-delay: entries blocked until "
                f"{blocked_until.strftime('%H:%M ET')}."
            )
        return True, ""

    date_str = et.strftime("%Y-%m-%d")
    try:
        sched = _NYSE.schedule(start_date=date_str, end_date=date_str)
    except Exception:
        return True, ""
    if sched.empty:
        return True, ""

    open_ts = sched.iloc[0]["market_open"]
    blocked_until = open_ts + pd.Timedelta(minutes=int(delay_minutes))
    if open_ts <= t < blocked_until:
        return False, (
            f"Monday open-delay: entries blocked until "
            f"{blocked_until.tz_convert('America/New_York').strftime('%H:%M ET')}."
        )
    return True, ""


# ── Pattern Day Trader (PDT) ─────────────────────────────────────────────────

def _ts_attr(obj, key):
    """Read `key` from dict or dataclass-like object; return pd.Timestamp or None."""
    v = obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)
    if v in (None, ""):
        return None
    try:
        t = pd.Timestamp(v)
    except Exception:
        return None
    if t is pd.NaT:
        return None
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return t


def count_day_trades(trades: Iterable, *, window_days: int = 5, as_of=None) -> int:
    """
    Count closed trades where entry_time and exit_time land on the same NYSE
    trading day AND the exit was within the trailing `window_days` business days
    relative to `as_of` (defaults to now). Accepts a list of dicts or
    TradeRecord-like objects. For backtests, pass `as_of=current_bar_time`.
    """
    if as_of is None:
        ref = pd.Timestamp.utcnow()
    else:
        ref = _coerce_utc(as_of) or pd.Timestamp.utcnow()
    if ref.tzinfo is None:
        ref = ref.tz_localize("UTC")
    # 5 business days spans up to ~7-9 calendar days with weekends/holidays; be generous
    cutoff = ref - pd.Timedelta(days=window_days * 2)
    count = 0
    for t in trades:
        e = _ts_attr(t, "entry_time")
        x = _ts_attr(t, "exit_time")
        if e is None or x is None:
            continue
        if x < cutoff or x > ref:
            continue
        if e.tz_convert("America/New_York").date() == x.tz_convert("America/New_York").date():
            count += 1
    return count


def pdt_guard(trades: Iterable, equity: float, *,
              enforce: bool = True,
              threshold: float = 25_000.0,
              as_of=None) -> tuple[bool, str]:
    """
    (allowed, reason). For margin accounts with equity < `threshold`, block
    the 4th day-trade inside a rolling 5-day window (Alpaca would reject it).
    `as_of` pins the rolling window reference point for backtest use.
    """
    if not enforce:
        return True, ""
    if float(equity) >= threshold:
        return True, ""
    dt = count_day_trades(trades, window_days=5, as_of=as_of)
    if dt >= 3:
        return False, (
            f"PDT: {dt} day-trades in last 5 days, equity ${equity:,.2f} < "
            f"${threshold:,.0f} — Alpaca would block the next day-trade."
        )
    if dt == 2:
        return True, f"PDT warn: {dt}/3 day-trades in last 5 days."
    return True, ""


# ── Short-Sale Restriction (SSR) ─────────────────────────────────────────────

def ssr_guard(prices: pd.DataFrame, direction) -> tuple[bool, str]:
    """
    SSR heuristic: if today's price is ≥10% below prior trading day's close,
    Alpaca restricts shorts to upticks. For a market order this effectively
    rejects the entry. Only applies when direction is SHORT.

    `prices` must be a DataFrame with `date` and `close` columns sorted
    ascending. Returns (allowed, reason).
    """
    if direction != Direction.SHORT:
        return True, ""
    if prices is None or len(prices) < 2 or "close" not in prices.columns:
        return True, ""
    try:
        p = prices.copy()
        p["_date"] = pd.to_datetime(p["date"]).dt.tz_localize(None).dt.date
        daily_close = p.groupby("_date")["close"].last()
        if len(daily_close) < 2:
            return True, ""
        prev_close = float(daily_close.iloc[-2])
        current = float(p["close"].iloc[-1])
        if prev_close <= 0:
            return True, ""
        drop_pct = (current - prev_close) / prev_close * 100
        if drop_pct <= -10.0:
            return False, (
                f"SSR: down {drop_pct:.2f}% from prior close — "
                f"Alpaca restricts shorts (uptick rule)."
            )
    except Exception:
        return True, ""
    return True, ""


# ── Shortability (asset.shortable + easy_to_borrow) ─────────────────────────

# Module-level TTL cache so we hit Alpaca at most once per (symbol, paper) per
# `_SHORTABLE_TTL_SEC`. The cache is cleared on process restart, which is fine
# — shortability changes daily at most.
_SHORTABLE_TTL_SEC = 600  # 10 minutes
_SHORTABLE_LOCK = threading.Lock()
_SHORTABLE_CACHE: dict[tuple[str, bool], tuple[float, bool, bool, str]] = {}


def _read_shortable_flags(client, symbol: str) -> tuple[bool, bool, str]:
    """
    Hit Alpaca's get_asset(symbol) and return (shortable, easy_to_borrow, note).
    `note` summarizes the asset state for trade metadata. On any failure we
    return (True, True, "") so a transient API issue doesn't lock all shorts.
    """
    try:
        asset = client.get_asset(symbol)
    except Exception as exc:  # pragma: no cover - network path
        return True, True, f"shortable_lookup_failed:{exc.__class__.__name__}"
    shortable = bool(getattr(asset, "shortable", False))
    etb = bool(getattr(asset, "easy_to_borrow", False))
    tradable = bool(getattr(asset, "tradable", True))
    note = f"shortable={shortable} etb={etb} tradable={tradable}"
    if not tradable:
        return False, False, note + " (not tradable)"
    return shortable, etb, note


def _cache_get(symbol: str, paper: bool) -> Optional[tuple[bool, bool, str]]:
    key = (symbol.upper(), paper)
    with _SHORTABLE_LOCK:
        entry = _SHORTABLE_CACHE.get(key)
    if entry is None:
        return None
    expires_at, shortable, etb, note = entry
    if time.time() >= expires_at:
        return None
    return shortable, etb, note


def _cache_put(symbol: str, paper: bool, shortable: bool, etb: bool, note: str) -> None:
    key = (symbol.upper(), paper)
    with _SHORTABLE_LOCK:
        _SHORTABLE_CACHE[key] = (time.time() + _SHORTABLE_TTL_SEC, shortable, etb, note)


def shortable_guard(
    symbol: str,
    direction,
    *,
    client_factory: Optional[Callable[[bool], object]] = None,
    paper: bool = True,
    require_easy_to_borrow: bool = False,
    enforce: bool = True,
) -> tuple[bool, str]:
    """
    Verify Alpaca will accept a SHORT on `symbol` before submitting.

    Long entries always pass. Short entries hit `client_factory(paper)` once
    per `_SHORTABLE_TTL_SEC` to read `asset.shortable` (and optionally
    `asset.easy_to_borrow`). Returns (allowed, reason).

    `client_factory` is a callable that takes `paper: bool` and returns a
    cached TradingClient. When the factory is missing or returns None we
    fail-open so backtest paths (no API key configured) keep working.
    """
    if not enforce:
        return True, ""
    if direction != Direction.SHORT:
        return True, ""
    sym = (symbol or "").strip().upper()
    if not sym:
        return True, ""

    cached = _cache_get(sym, paper)
    if cached is None:
        if client_factory is None:
            return True, ""  # backtest path with no client wired
        try:
            client = client_factory(paper)
        except Exception as exc:  # pragma: no cover
            return True, f"shortable lookup unavailable ({exc.__class__.__name__}); fail-open"
        if client is None:
            return True, ""
        shortable, etb, note = _read_shortable_flags(client, sym)
        _cache_put(sym, paper, shortable, etb, note)
    else:
        shortable, etb, note = cached

    if not shortable:
        return False, (
            f"Shortability: {sym} is not currently shortable on Alpaca "
            f"(paper={paper}). Live order would be rejected."
        )
    if require_easy_to_borrow and not etb:
        return False, (
            f"Shortability: {sym} is shortable but not on the easy-to-borrow "
            "list — locate fees apply. require_easy_to_borrow=True blocked entry."
        )
    return True, note  # note becomes a notes_prefix so the trade row records ETB state


def invalidate_shortable_cache(symbol: Optional[str] = None) -> None:
    """Manual cache buster for the UI. Pass None to clear all symbols."""
    with _SHORTABLE_LOCK:
        if symbol is None:
            _SHORTABLE_CACHE.clear()
            return
        sym = symbol.upper()
        for key in list(_SHORTABLE_CACHE.keys()):
            if key[0] == sym:
                del _SHORTABLE_CACHE[key]


# ── Fractional-share routing ─────────────────────────────────────────────────

def normalize_qty_for_direction(qty: float, direction) -> tuple[float, str]:
    """
    Alpaca allows fractional qty for LONG market orders; SHORT requires
    integer qty >= 1. Returns (normalized_qty, reason_if_changed).
    """
    q = float(qty)
    if direction == Direction.SHORT:
        floored = float(int(q))  # drop fractional part
        if floored < 1.0:
            return 0.0, (
                f"Fractional short rejected (qty={q:.6f} → 0). "
                f"Alpaca requires integer qty ≥ 1 for short orders."
            )
        if floored != q:
            return floored, f"Short qty floored {q:.6f} → {floored:.0f} (Alpaca integer rule)."
        return floored, ""
    return q, ""


# ── Fill-timing diagnostic ───────────────────────────────────────────────────

@dataclass
class FillTimingNote:
    """Diagnostic snapshot for later cost calibration against real Alpaca fills."""
    symbol: str
    entry_time: str
    signal_close: float
    bar_high: float
    bar_low: float
    bar_range_pct: float

    def as_note_str(self) -> str:
        return (
            f"fill_diag: close={self.signal_close:.4f} "
            f"H={self.bar_high:.4f} L={self.bar_low:.4f} "
            f"range={self.bar_range_pct:.3f}%"
        )


def fill_timing_note(symbol: str, bar) -> FillTimingNote:
    """
    Capture the entry bar's close/high/low/range — so when you later
    compare to Alpaca's actual filled_avg_price you can back-calibrate
    spread_pct + slippage_pct.
    """
    close = float(bar["close"])
    high = float(bar["high"])
    low = float(bar["low"])
    rng_pct = ((high - low) / close * 100.0) if close > 0 else 0.0
    return FillTimingNote(
        symbol=symbol,
        entry_time=str(bar.get("date") if isinstance(bar, dict) else bar["date"]),
        signal_close=close,
        bar_high=high,
        bar_low=low,
        bar_range_pct=rng_pct,
    )

"""
execution/entry_policy_alpaca.py
────────────────────────────────
Alpaca-realistic paper-trading entry policy.

Applies the Tier-1 gates that Alpaca would enforce on a live order, in the
same order the paper loop used to run them inline:

  1. Regular Trading Hours + NYSE holiday calendar
  2. Short-Sale Restriction (SSR) heuristic — prior day close to current < -10%
  3. Shortability — asset.shortable + (optionally) easy_to_borrow
  4. Pattern Day Trader (PDT) rolling 5-day count vs. equity
  5. Fractional-share routing — shorts require integer qty ≥ 1
  6. Fill-timing diagnostic — bar high/low/range appended to trade notes

Each gate is individually configurable so a user can disable one without
falling all the way back to the "classic" no-gate policy. Any gate can also
be globally bypassed by picking the ClassicEntryPolicy instead.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from core.models import Direction
from execution.alpaca_constraints import (
    is_regular_trading_hour,
    is_trading_day,
    monday_open_delay_guard,
    pdt_guard,
    ssr_guard,
    shortable_guard,
    normalize_qty_for_direction,
    fill_timing_note,
)
from execution.entry_policy_base import (
    EntryContext,
    EntryDecision,
    register_policy,
)


# Module-level cached TradingClients so we don't construct one per signal.
_ALPACA_CLIENT_CACHE: dict[bool, object] = {}


def _default_alpaca_client_factory(paper: bool):
    """Lazy-construct a cached TradingClient for shortability lookups.

    Returns None if Alpaca creds aren't configured so backtests stay fail-open.
    """
    paper = bool(paper)
    if paper in _ALPACA_CLIENT_CACHE:
        return _ALPACA_CLIENT_CACHE[paper]
    try:
        from config.settings import settings
        from alpaca.trading.client import TradingClient
        if paper and not settings.alpaca.has_paper_credentials():
            return None
        if (not paper) and not (
            settings.alpaca.live_api_key and settings.alpaca.live_secret_key
        ):
            return None
        if paper:
            client = TradingClient(
                api_key=settings.alpaca.paper_api_key,
                secret_key=settings.alpaca.paper_secret_key,
                paper=True,
            )
        else:
            client = TradingClient(
                api_key=settings.alpaca.live_api_key,
                secret_key=settings.alpaca.live_secret_key,
                paper=False,
            )
        _ALPACA_CLIENT_CACHE[paper] = client
        return client
    except Exception:
        return None


@dataclass
class AlpacaEntryPolicy:
    name: str = "alpaca"
    label: str = "Alpaca-realistic (RTH / SSR / shortable / PDT / fractional)"

    enforce_rth: bool = True
    extended_hours: bool = False
    enforce_ssr: bool = True
    enforce_pdt: bool = True
    enforce_fractional: bool = True
    enforce_shortable: bool = True
    require_easy_to_borrow: bool = False
    fill_diagnostic: bool = True
    enforce_monday_open_delay: bool = False
    # Used so backtests (no live account) and tests can swap in a stub.
    client_factory: Optional[Callable[[bool], object]] = None
    use_paper_endpoint: bool = True

    def evaluate(self, ctx: EntryContext) -> EntryDecision:
        allowed, reason = monday_open_delay_guard(
            ctx.bar_time,
            enforce=self.enforce_monday_open_delay,
        )
        if not allowed:
            return EntryDecision.reject(reason)

        # 1. Regular Trading Hours + NYSE holiday calendar
        if self.enforce_rth:
            if not is_trading_day(ctx.bar_time) or not is_regular_trading_hour(
                ctx.bar_time, extended_hours=self.extended_hours
            ):
                return EntryDecision.reject(
                    f"Outside trading hours ({ctx.bar_time}). Alpaca would reject. "
                    "Enable 'Allow extended hours' to permit 04:00–20:00 ET."
                )

        # 2. Short-Sale Restriction (shorts only)
        if self.enforce_ssr:
            allowed, reason = ssr_guard(ctx.prices, ctx.direction)
            if not allowed:
                return EntryDecision.reject(reason)

        # 3. Shortability — asset.shortable + optional easy_to_borrow check
        notes_prefix_short = ""
        if self.enforce_shortable and ctx.direction == Direction.SHORT:
            client_fn = self.client_factory or _default_alpaca_client_factory
            allowed, reason = shortable_guard(
                ctx.symbol,
                ctx.direction,
                client_factory=client_fn,
                paper=self.use_paper_endpoint,
                require_easy_to_borrow=self.require_easy_to_borrow,
                enforce=True,
            )
            if not allowed:
                return EntryDecision.reject(reason)
            # `reason` is repurposed as a status note when the trade IS allowed.
            if reason:
                notes_prefix_short = reason

        # 4. Pattern Day Trader rolling 5-day count
        if self.enforce_pdt:
            allowed, reason = pdt_guard(ctx.portfolio_trades, ctx.portfolio_equity)
            if not allowed:
                return EntryDecision.reject(reason)

        adjusted_capital: float | None = None
        notes_prefix = notes_prefix_short

        # 5. Fractional-share routing (shorts need integer qty ≥ 1)
        if self.enforce_fractional:
            est_qty = (
                (ctx.requested_capital * float(ctx.leverage)) / ctx.entry_price
                if ctx.entry_price > 0 else 0.0
            )
            norm_qty, norm_reason = normalize_qty_for_direction(est_qty, ctx.direction)
            if norm_qty <= 0:
                return EntryDecision.reject(norm_reason or "Qty resolved to 0.")
            if norm_reason and ctx.direction == Direction.SHORT and est_qty > 0:
                scale = norm_qty / est_qty
                adjusted_capital = ctx.requested_capital * scale
                notes_prefix = (notes_prefix + "; " + norm_reason).strip("; ")

        # 6. Fill-timing diagnostic — computed here so the bar context at
        #    decision time is the one recorded, not whatever comes later.
        post_fill_note = ""
        if self.fill_diagnostic:
            try:
                post_fill_note = fill_timing_note(ctx.symbol, ctx.bar).as_note_str()
            except Exception:
                post_fill_note = ""

        return EntryDecision.allow(
            adjusted_capital=adjusted_capital,
            notes_prefix=notes_prefix,
            post_fill_note=post_fill_note,
        )


register_policy("alpaca", AlpacaEntryPolicy)

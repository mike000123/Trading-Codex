"""
execution/entry_policy_alpaca.py
────────────────────────────────
Alpaca-realistic paper-trading entry policy.

Applies the Tier-1 gates that Alpaca would enforce on a live order, in the
same order the paper loop used to run them inline:

  1. Regular Trading Hours + NYSE holiday calendar
  2. Short-Sale Restriction (SSR) heuristic — prior day close to current < -10%
  3. Pattern Day Trader (PDT) rolling 5-day count vs. equity
  4. Fractional-share routing — shorts require integer qty ≥ 1
  5. Fill-timing diagnostic — bar high/low/range appended to trade notes

Each gate is individually configurable so a user can disable one without
falling all the way back to the "classic" no-gate policy. Any gate can also
be globally bypassed by picking the ClassicEntryPolicy instead.
"""
from __future__ import annotations

from dataclasses import dataclass

from core.models import Direction
from execution.alpaca_constraints import (
    is_regular_trading_hour,
    is_trading_day,
    pdt_guard,
    ssr_guard,
    normalize_qty_for_direction,
    fill_timing_note,
)
from execution.entry_policy_base import (
    EntryContext,
    EntryDecision,
    register_policy,
)


@dataclass
class AlpacaEntryPolicy:
    name: str = "alpaca"
    label: str = "Alpaca-realistic (RTH / PDT / SSR / fractional)"

    enforce_rth: bool = True
    extended_hours: bool = False
    enforce_ssr: bool = True
    enforce_pdt: bool = True
    enforce_fractional: bool = True
    fill_diagnostic: bool = True

    def evaluate(self, ctx: EntryContext) -> EntryDecision:
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

        # 3. Pattern Day Trader rolling 5-day count
        if self.enforce_pdt:
            allowed, reason = pdt_guard(ctx.portfolio_trades, ctx.portfolio_equity)
            if not allowed:
                return EntryDecision.reject(reason)

        adjusted_capital: float | None = None
        notes_prefix = ""

        # 4. Fractional-share routing (shorts need integer qty ≥ 1)
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
                notes_prefix = norm_reason

        # 5. Fill-timing diagnostic — computed here so the bar context at
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

"""
execution/entry_policy_classic.py
─────────────────────────────────
"Classic" paper-trading entry policy — the logic we used *before* adding
Alpaca-realistic gates.

Behaviour:
  - No RTH / NYSE holiday gate
  - No PDT rolling-window guard
  - No Short-Sale Restriction (SSR) check
  - No fractional-share routing rule
  - No fill-timing diagnostic

Any entry request that has a valid direction + entry price is allowed to pass
through. This exists so a user can A/B compare Alpaca-gated results against
the unconstrained baseline in the forward-testing (paper) page.
"""
from __future__ import annotations

from dataclasses import dataclass

from execution.alpaca_constraints import monday_open_delay_guard
from execution.entry_policy_base import (
    EntryContext,
    EntryDecision,
    register_policy,
)


@dataclass
class ClassicEntryPolicy:
    name: str = "classic"
    label: str = "Classic (pre-Alpaca gates)"
    enforce_monday_open_delay: bool = False

    def evaluate(self, ctx: EntryContext) -> EntryDecision:  # noqa: D401
        allowed, reason = monday_open_delay_guard(
            ctx.bar_time,
            enforce=self.enforce_monday_open_delay,
        )
        if not allowed:
            return EntryDecision.reject(reason)
        # No-op policy: accept every entry; request no capital adjustment;
        # emit no notes.
        return EntryDecision.allow()


register_policy("classic", ClassicEntryPolicy)

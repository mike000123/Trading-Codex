"""
risk/manager.py
───────────────
Centralised risk gate.
Every order passes through RiskManager.check() before execution.
If the check fails, the order is BLOCKED with a reason – nothing is sent to the broker.

Controls enforced:
  1. Daily loss limit    – halt if today's realised P&L < -max_daily_loss_pct%
  2. Max open positions  – reject new entries when limit reached
  3. Capital per trade   – cap size to max_capital_per_trade_pct% of portfolio
  4. TP/SL validation    – reject orders with no SL or inverted levels
  5. SL capping          – clamp SL so loss ≤ max_loss_pct_of_capital
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from config.settings import RiskConfig
from core.models import Direction
from core.logger import log


@dataclass
class RiskCheckResult:
    approved: bool
    reason: str = ""
    adjusted_sl: Optional[float] = None
    adjusted_size: Optional[float] = None   # capital amount after sizing


class RiskManager:
    """
    Stateless + stateful risk checks.

    State that changes per session is injected via update_portfolio_state().
    """

    def __init__(self, config: RiskConfig) -> None:
        self.config = config
        self._daily_pnl: float = 0.0
        self._open_positions: int = 0
        self._total_equity: float = 0.0

    # ─── State update (call after each fill) ────────────────────────────────

    def update_portfolio_state(
        self,
        daily_pnl: float,
        open_positions: int,
        total_equity: float,
    ) -> None:
        self._daily_pnl = daily_pnl
        self._open_positions = open_positions
        self._total_equity = total_equity

    # ─── Main gate ──────────────────────────────────────────────────────────

    def check(
        self,
        *,
        direction: Direction,
        entry_price: float,
        take_profit: Optional[float],
        stop_loss: Optional[float],
        leverage: float,
        capital_requested: float,
    ) -> RiskCheckResult:
        """
        Run all risk checks in priority order.
        Returns RiskCheckResult with approved=True and possibly adjusted values.
        """

        # 1. Daily loss circuit-breaker
        if self._total_equity > 0:
            daily_loss_pct = (-self._daily_pnl / self._total_equity) * 100
            if daily_loss_pct >= self.config.max_daily_loss_pct:
                msg = (
                    f"Daily loss limit reached: {daily_loss_pct:.1f}% >= "
                    f"{self.config.max_daily_loss_pct:.1f}%. Trading halted for today."
                )
                log.warning(f"RISK BLOCK: {msg}")
                return RiskCheckResult(approved=False, reason=msg)

        # 2. Open positions cap
        if self._open_positions >= self.config.max_open_positions:
            msg = f"Max open positions ({self.config.max_open_positions}) reached."
            log.warning(f"RISK BLOCK: {msg}")
            return RiskCheckResult(approved=False, reason=msg)

        # 3. SL required
        if stop_loss is None:
            msg = "Order rejected: stop-loss is required."
            log.warning(f"RISK BLOCK: {msg}")
            return RiskCheckResult(approved=False, reason=msg)

        # 4. TP/SL direction validation
        if take_profit is not None:
            if direction == Direction.LONG and take_profit <= entry_price:
                return RiskCheckResult(
                    approved=False,
                    reason="Long TP must be above entry price.",
                )
            if direction == Direction.SHORT and take_profit >= entry_price:
                return RiskCheckResult(
                    approved=False,
                    reason="Short TP must be below entry price.",
                )
            if direction == Direction.LONG and stop_loss >= entry_price:
                return RiskCheckResult(
                    approved=False,
                    reason="Long SL must be below entry price.",
                )
            if direction == Direction.SHORT and stop_loss <= entry_price:
                return RiskCheckResult(
                    approved=False,
                    reason="Short SL must be above entry price.",
                )

        # 5. SL capping to max loss per trade
        adjusted_sl, was_capped = self._cap_stop_loss(
            entry_price, stop_loss, leverage, direction,
            self.config.default_max_loss_pct_of_capital,
        )
        if was_capped:
            log.info(f"RISK: SL adjusted {stop_loss:.4f} → {adjusted_sl:.4f} (max-loss cap)")

        # 6. Position sizing cap
        max_capital = (
            self._total_equity * self.config.max_capital_per_trade_pct / 100
            if self._total_equity > 0
            else capital_requested
        )
        adjusted_size = min(capital_requested, max_capital) if max_capital > 0 else capital_requested

        return RiskCheckResult(
            approved=True,
            reason="All checks passed.",
            adjusted_sl=adjusted_sl,
            adjusted_size=adjusted_size,
        )

    # ─── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _cap_stop_loss(
        entry: float,
        sl: float,
        leverage: float,
        direction: Direction,
        max_loss_pct: float,
    ) -> Tuple[float, bool]:
        loss_move = max_loss_pct / 100 / leverage
        if direction == Direction.LONG:
            min_sl = entry * (1 - loss_move)
            adjusted = max(sl, min_sl)
        else:
            max_sl = entry * (1 + loss_move)
            adjusted = min(sl, max_sl)
        was_adjusted = abs(adjusted - sl) > 1e-10
        return adjusted, was_adjusted

    @staticmethod
    def implied_stop_floor(
        entry: float,
        leverage: float,
        direction: Direction,
        max_loss_pct: float,
    ) -> float:
        """Return the closest allowed SL price given max loss cap."""
        result, _ = RiskManager._cap_stop_loss(entry, 0.0 if direction == Direction.LONG else 1e12, leverage, direction, max_loss_pct)
        return result

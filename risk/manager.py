"""
risk/manager.py
───────────────
Centralised risk gate. Every order passes through RiskManager.check() before execution.

FIX: SL direction validation and SL cap now run unconditionally,
     regardless of whether TP is set. Previously both were inside
     `if take_profit is not None` which meant TP-disabled trades
     skipped the SL floor entirely.
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
    adjusted_size: Optional[float] = None


class RiskManager:
    def __init__(self, config: RiskConfig) -> None:
        self.config = config
        self._daily_pnl: float = 0.0
        self._open_positions: int = 0
        self._total_equity: float = 0.0

    def update_portfolio_state(self, daily_pnl: float, open_positions: int, total_equity: float) -> None:
        self._daily_pnl      = daily_pnl
        self._open_positions = open_positions
        self._total_equity   = total_equity

    def check(
        self, *,
        direction: Direction,
        entry_price: float,
        take_profit: Optional[float],
        stop_loss: Optional[float],
        leverage: float,
        capital_requested: float,
    ) -> RiskCheckResult:

        # 1. Daily loss circuit-breaker
        if self._total_equity > 0:
            daily_loss_pct = (-self._daily_pnl / self._total_equity) * 100
            if daily_loss_pct >= self.config.max_daily_loss_pct:
                msg = (f"Daily loss limit reached: {daily_loss_pct:.1f}% >= "
                       f"{self.config.max_daily_loss_pct:.1f}%. Trading halted.")
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

        # 4. SL direction validation — runs ALWAYS, not only when TP is set
        if direction == Direction.LONG and stop_loss >= entry_price:
            return RiskCheckResult(approved=False, reason=f"Long SL ({stop_loss:.4f}) must be below entry ({entry_price:.4f}).")
        if direction == Direction.SHORT and stop_loss <= entry_price:
            return RiskCheckResult(approved=False, reason=f"Short SL ({stop_loss:.4f}) must be above entry ({entry_price:.4f}).")

        # 5. TP direction validation — only when TP is provided
        if take_profit is not None:
            if direction == Direction.LONG and take_profit <= entry_price:
                return RiskCheckResult(approved=False, reason="Long TP must be above entry price.")
            if direction == Direction.SHORT and take_profit >= entry_price:
                return RiskCheckResult(approved=False, reason="Short TP must be below entry price.")

        # 6. SL cap — runs ALWAYS
        adjusted_sl, was_capped = self._cap_stop_loss(
            entry_price, stop_loss, leverage, direction,
            self.config.default_max_loss_pct_of_capital,
        )
        if was_capped:
            log.info(f"RISK: SL adjusted {stop_loss:.4f} → {adjusted_sl:.4f} "
                     f"(max {self.config.default_max_loss_pct_of_capital:.0f}% capital loss cap, "
                     f"lev={leverage}x → max price move = "
                     f"{self.config.default_max_loss_pct_of_capital/leverage:.2f}%)")

        # 7. Position sizing cap
        max_capital = (self._total_equity * self.config.max_capital_per_trade_pct / 100
                       if self._total_equity > 0 else capital_requested)
        adjusted_size = min(capital_requested, max_capital) if max_capital > 0 else capital_requested

        return RiskCheckResult(approved=True, reason="All checks passed.",
                               adjusted_sl=adjusted_sl, adjusted_size=adjusted_size)

    @staticmethod
    def _cap_stop_loss(entry: float, sl: float, leverage: float,
                       direction: Direction, max_loss_pct: float) -> Tuple[float, bool]:
        """
        Clamp SL so that the leveraged loss cannot exceed max_loss_pct% of capital.
        max_loss_pct=20, leverage=5  →  max price move = 20/5 = 4%
          Long:  SL floor = entry * (1 - 0.04) = entry * 0.96
          Short: SL ceil  = entry * (1 + 0.04) = entry * 1.04
        """
        loss_move = max_loss_pct / 100.0 / leverage
        if direction == Direction.LONG:
            floor    = entry * (1 - loss_move)
            adjusted = max(sl, floor)
        else:
            ceiling  = entry * (1 + loss_move)
            adjusted = min(sl, ceiling)
        was_adjusted = abs(adjusted - sl) > 1e-10
        return adjusted, was_adjusted

    @staticmethod
    def implied_stop_floor(entry: float, leverage: float,
                           direction: Direction, max_loss_pct: float) -> float:
        result, _ = RiskManager._cap_stop_loss(
            entry,
            0.0 if direction == Direction.LONG else 1e12,
            leverage, direction, max_loss_pct
        )
        return result

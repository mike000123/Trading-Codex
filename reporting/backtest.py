"""
reporting/backtest.py
──────────────────────
Walk-forward backtesting engine.

Exit priority per bar:
  1. SL hit (price low/high crosses stop-loss)
  2. TP hit (price high/low crosses take-profit)
  3. Counter-signal (opposing strategy signal, if counter_signal_exit=True)

Outcome labels are descriptive:
  "TP hit"               – price reached take-profit level
  "SL hit"               – price reached stop-loss level
  "RSI overbought exit"  – RSI crossed OB threshold while Long → close
  "RSI oversold exit"    – RSI crossed OS threshold while Short → close
  "Counter-signal exit"  – non-RSI strategy fired opposing signal
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import pandas as pd

from core.models import Direction, Signal, SignalAction, TradeOutcome, TradeRecord
from core.logger import log
from risk.manager import RiskManager
from strategies.base import BaseStrategy
import uuid


@dataclass
class BacktestResult:
    trades: list[TradeRecord]
    equity_curve: pd.DataFrame
    total_return_pct: float
    win_rate_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win_pct: float
    avg_loss_pct: float

    def summary(self) -> dict:
        return {
            "Total Trades": self.total_trades,
            "Win Rate":     f"{self.win_rate_pct:.1f}%",
            "Total Return": f"{self.total_return_pct:.2f}%",
            "Max Drawdown": f"{self.max_drawdown_pct:.2f}%",
            "Sharpe Ratio": f"{self.sharpe_ratio:.3f}",
            "Avg Win":      f"{self.avg_win_pct:.2f}%",
            "Avg Loss":     f"{self.avg_loss_pct:.2f}%",
        }


class BacktestEngine:
    def __init__(
        self,
        strategy: BaseStrategy,
        risk_manager: Optional[RiskManager] = None,
        direction_filter: Optional[Direction] = None,
        counter_signal_exit: bool = True,
    ) -> None:
        self.strategy            = strategy
        self.risk                = risk_manager
        self.direction_filter    = direction_filter
        self.counter_signal_exit = counter_signal_exit

    def run(
        self,
        data: pd.DataFrame,
        symbol: str,
        leverage: float = 1.0,
        capital_per_trade: float = 1000.0,
        starting_equity: float = 10_000.0,
    ) -> BacktestResult:
        log.info(f"Backtest START: {symbol} | {self.strategy.strategy_id} | "
                 f"{len(data)} bars | lev={leverage}x | capital/trade={capital_per_trade}")

        trades: list[TradeRecord] = []
        equity = starting_equity
        equity_curve: list[dict] = []
        open_trade: Optional[TradeRecord] = None

        for i in range(1, len(data)):
            bar          = data.iloc[i]
            current_date = bar["date"]

            # Always generate signal (needed for counter-signal exit check)
            window       = data.iloc[:i + 1].copy()
            signal       = self.strategy.generate_signal(window, symbol)
            new_direction= self._signal_to_direction(signal)

            # ── 1. Check SL / TP hit ──────────────────────────────────────────
            if open_trade is not None:
                open_trade = self._check_exit(open_trade, bar)
                if open_trade.outcome not in (TradeOutcome.OPEN, TradeOutcome.NO_DATA):
                    if open_trade.leveraged_return_pct is not None:
                        trade_pnl       = open_trade.capital_allocated * open_trade.leveraged_return_pct / 100
                        equity         += trade_pnl
                        open_trade.pnl  = trade_pnl
                    trades.append(open_trade)
                    open_trade = None

            # ── 2. Counter-signal exit ────────────────────────────────────────
            if (self.counter_signal_exit
                    and open_trade is not None
                    and new_direction is not None):
                current_dir = open_trade.direction
                is_reversal = (
                    (current_dir == Direction.LONG  and new_direction == Direction.SHORT) or
                    (current_dir == Direction.SHORT and new_direction == Direction.LONG)
                )
                if is_reversal:
                    exit_px = float(bar["close"])
                    pct     = (exit_px - open_trade.entry_price) / open_trade.entry_price
                    if current_dir == Direction.SHORT:
                        pct = -pct
                    open_trade.leveraged_return_pct = pct * open_trade.leverage * 100
                    open_trade.exit_price = exit_px
                    open_trade.exit_time  = (current_date if isinstance(current_date, datetime)
                                             else current_date.to_pydatetime())
                    # Descriptive outcome: which threshold triggered it?
                    open_trade.outcome = self._counter_signal_outcome(
                        open_trade.direction, signal
                    )
                    open_trade.notes += f" | {open_trade.outcome.value}"
                    trade_pnl         = open_trade.capital_allocated * open_trade.leveraged_return_pct / 100
                    equity           += trade_pnl
                    open_trade.pnl    = trade_pnl
                    trades.append(open_trade)
                    open_trade = None
                    log.debug(f"COUNTER-SIGNAL EXIT: {symbol} @ {exit_px:.4f} "
                              f"outcome={open_trade.outcome if open_trade else 'appended'}")

            # ── 3. Open new trade if flat and signal fired ────────────────────
            if open_trade is None and new_direction is not None and signal.suggested_sl is not None:
                if self.risk:
                    check = self.risk.check(
                        direction=new_direction,
                        entry_price=float(bar["close"]),
                        take_profit=signal.suggested_tp,
                        stop_loss=signal.suggested_sl,
                        leverage=leverage,
                        capital_requested=capital_per_trade,
                    )
                    if not check.approved:
                        equity_curve.append({"date": current_date, "equity": equity})
                        continue
                    effective_sl      = check.adjusted_sl or signal.suggested_sl
                    effective_capital = check.adjusted_size or capital_per_trade
                else:
                    effective_sl      = signal.suggested_sl
                    effective_capital = capital_per_trade

                entry_px = float(bar["close"])
                raw_sl   = signal.suggested_sl
                log.debug(f"TRADE OPEN: {symbol} {new_direction.value} @ {entry_px:.4f} | "
                          f"SL={effective_sl:.4f} (strategy={raw_sl:.4f}) | "
                          f"TP={signal.suggested_tp} | lev={leverage}x")

                # Build descriptive open notes
                meta      = signal.metadata or {}
                triggered = meta.get("triggered_level")
                tp_str    = f"{signal.suggested_tp:.4f}" if signal.suggested_tp else "none"
                notes = (f"Entry: {new_direction.value} @ {entry_px:.4f} | "
                         f"SL={effective_sl:.4f} | TP={tp_str} | "
                         f"Triggered by RSI {triggered}" if triggered
                         else f"Entry: {new_direction.value} @ {entry_px:.4f} | "
                              f"SL={effective_sl:.4f} | TP={tp_str}")

                open_trade = TradeRecord(
                    id=str(uuid.uuid4()),
                    symbol=symbol,
                    direction=new_direction,
                    entry_price=entry_px,
                    take_profit=signal.suggested_tp,
                    stop_loss=effective_sl,
                    leverage=leverage,
                    capital_allocated=effective_capital,
                    entry_time=(current_date if isinstance(current_date, datetime)
                                else current_date.to_pydatetime()),
                    mode="backtest",
                    strategy_id=self.strategy.strategy_id,
                    outcome=TradeOutcome.OPEN,
                    notes=notes,
                )

            equity_curve.append({"date": current_date, "equity": equity})

        # Close any still-open trade at last bar
        if open_trade is not None:
            last = data.iloc[-1]
            open_trade.exit_price = float(last["close"])
            open_trade.exit_time  = last["date"]
            open_trade.outcome    = TradeOutcome.OPEN
            trades.append(open_trade)

        return self._compute_result(trades, equity_curve, starting_equity)

    # ─── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _counter_signal_outcome(
        open_direction: Direction, closing_signal: Signal
    ) -> TradeOutcome:
        """
        Return a descriptive outcome label based on which RSI level triggered the exit.
        Long closed by sell signal → RSI overbought exit
        Short closed by buy signal → RSI oversold exit
        """
        meta          = closing_signal.metadata or {}
        strategy_id   = closing_signal.strategy_id
        triggered_lvl = meta.get("triggered_level")

        if "rsi" in strategy_id:
            if open_direction == Direction.LONG:
                lvl_str = f" @ RSI {triggered_lvl:.0f}" if triggered_lvl else ""
                return TradeOutcome.SIGNAL_RSI_OB   # RSI crossed OB → close Long
            else:
                return TradeOutcome.SIGNAL_RSI_OS   # RSI crossed OS → close Short
        return TradeOutcome.SIGNAL_EXIT

    @staticmethod
    def _check_exit(trade: TradeRecord, bar: pd.Series) -> TradeRecord:
        high  = float(bar["high"])
        low   = float(bar["low"])
        tp    = trade.take_profit
        sl    = trade.stop_loss
        entry = trade.entry_price

        hit_tp = hit_sl = False
        if trade.direction == Direction.LONG:
            hit_sl = sl is not None and low  <= sl
            hit_tp = tp is not None and high >= tp
        else:
            hit_sl = sl is not None and high >= sl
            hit_tp = tp is not None and low  <= tp

        # SL takes priority over TP if both hit same bar
        if hit_sl and hit_tp:
            trade.outcome   = TradeOutcome.AMBIGUOUS
            trade.exit_time = bar["date"]
        elif hit_sl:
            trade.outcome   = TradeOutcome.STOP_LOSS
            trade.exit_price= sl
            trade.exit_time = bar["date"]
            pct = (sl - entry) / entry
            if trade.direction == Direction.SHORT: pct = -pct
            trade.leveraged_return_pct = pct * trade.leverage * 100
        elif hit_tp:
            trade.outcome   = TradeOutcome.TAKE_PROFIT
            trade.exit_price= tp
            trade.exit_time = bar["date"]
            pct = (tp - entry) / entry
            if trade.direction == Direction.SHORT: pct = -pct
            trade.leveraged_return_pct = pct * trade.leverage * 100

        return trade

    @staticmethod
    def _signal_to_direction(signal: Signal) -> Optional[Direction]:
        if signal.action == SignalAction.BUY:  return Direction.LONG
        if signal.action == SignalAction.SELL: return Direction.SHORT
        return None

    @staticmethod
    def _compute_result(
        trades: list[TradeRecord],
        equity_curve: list[dict],
        starting_equity: float,
    ) -> BacktestResult:
        closed = [t for t in trades if t.leveraged_return_pct is not None]
        wins   = [t for t in closed if (t.leveraged_return_pct or 0) > 0]
        losses = [t for t in closed if (t.leveraged_return_pct or 0) <= 0]

        eq_df     = pd.DataFrame(equity_curve)
        final_eq  = float(eq_df["equity"].iloc[-1]) if not eq_df.empty else starting_equity
        total_ret = (final_eq - starting_equity) / starting_equity * 100 if starting_equity else 0

        if not eq_df.empty:
            roll_max = eq_df["equity"].cummax()
            drawdown = (eq_df["equity"] - roll_max) / roll_max * 100
            max_dd   = float(drawdown.min())
        else:
            max_dd = 0.0

        if len(eq_df) > 1:
            daily_ret = eq_df["equity"].pct_change().dropna()
            sharpe    = float(daily_ret.mean() / (daily_ret.std() + 1e-9) * (252 ** 0.5))
        else:
            sharpe = 0.0

        win_rets  = [t.leveraged_return_pct for t in wins]
        loss_rets = [t.leveraged_return_pct for t in losses]

        return BacktestResult(
            trades           = trades,
            equity_curve     = eq_df,
            total_return_pct = round(total_ret, 3),
            win_rate_pct     = round(len(wins) / len(closed) * 100 if closed else 0, 1),
            max_drawdown_pct = round(max_dd, 3),
            sharpe_ratio     = round(sharpe, 4),
            total_trades     = len(closed),
            winning_trades   = len(wins),
            losing_trades    = len(losses),
            avg_win_pct      = round(sum(win_rets)  / len(win_rets)  if win_rets  else 0, 3),
            avg_loss_pct     = round(sum(loss_rets) / len(loss_rets) if loss_rets else 0, 3),
        )

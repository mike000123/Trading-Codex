"""
reporting/backtest.py
──────────────────────
Walk-forward backtesting engine.

Usage:
    engine = BacktestEngine(strategy, risk_manager)
    result = engine.run(data, symbol, leverage, capital_per_trade)
    print(result.summary())
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
    equity_curve: pd.DataFrame        # columns: date, equity
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
            "Win Rate": f"{self.win_rate_pct:.1f}%",
            "Total Return": f"{self.total_return_pct:.2f}%",
            "Max Drawdown": f"{self.max_drawdown_pct:.2f}%",
            "Sharpe Ratio": f"{self.sharpe_ratio:.3f}",
            "Avg Win": f"{self.avg_win_pct:.2f}%",
            "Avg Loss": f"{self.avg_loss_pct:.2f}%",
        }


class BacktestEngine:
    """
    Walk-forward backtester.

    For each bar in the data window:
      1. Feed data[0..i] to strategy → Signal
      2. If BUY/SELL signal and no open trade, open a position
      3. On each subsequent bar, check if TP or SL was hit
      4. Record outcome and move to next signal
    """

    def __init__(
        self,
        strategy: BaseStrategy,
        risk_manager: Optional[RiskManager] = None,
        direction_filter: Optional[Direction] = None,
    ) -> None:
        self.strategy = strategy
        self.risk = risk_manager
        self.direction_filter = direction_filter  # None = follow signal, Long/Short = force

    def run(
        self,
        data: pd.DataFrame,
        symbol: str,
        leverage: float = 1.0,
        capital_per_trade: float = 1000.0,
        starting_equity: float = 10_000.0,
    ) -> BacktestResult:
        log.info(
            f"Backtest START: {symbol} | {self.strategy.strategy_id} | "
            f"{len(data)} bars | lev={leverage}x | capital/trade={capital_per_trade}"
        )

        trades: list[TradeRecord] = []
        equity = starting_equity
        equity_curve: list[dict] = []
        open_trade: Optional[TradeRecord] = None

        for i in range(1, len(data)):
            bar = data.iloc[i]
            current_date = bar["date"]

            # ── Check if open trade hit TP or SL ───────────────────────────
            if open_trade is not None:
                open_trade = self._check_exit(open_trade, bar)
                if open_trade.outcome != TradeOutcome.OPEN:
                    if open_trade.leveraged_return_pct is not None:
                        trade_pnl = open_trade.capital_allocated * open_trade.leveraged_return_pct / 100
                        equity += trade_pnl
                        open_trade.pnl = trade_pnl
                    trades.append(open_trade)
                    open_trade = None

            # ── Generate signal on data up to current bar ──────────────────
            if open_trade is None:
                window = data.iloc[: i + 1].copy()
                signal: Signal = self.strategy.generate_signal(window, symbol)

                direction = self._signal_to_direction(signal)
                if direction is not None and signal.suggested_sl is not None:

                    if self.risk:
                        check = self.risk.check(
                            direction=direction,
                            entry_price=float(bar["close"]),
                            take_profit=signal.suggested_tp,
                            stop_loss=signal.suggested_sl,
                            leverage=leverage,
                            capital_requested=capital_per_trade,
                        )
                        if not check.approved:
                            continue
                        effective_sl = check.adjusted_sl or signal.suggested_sl
                        effective_capital = check.adjusted_size or capital_per_trade
                    else:
                        effective_sl = signal.suggested_sl
                        effective_capital = capital_per_trade

                    open_trade = TradeRecord(
                        id=str(uuid.uuid4()),
                        symbol=symbol,
                        direction=direction,
                        entry_price=float(bar["close"]),
                        take_profit=signal.suggested_tp,
                        stop_loss=effective_sl,
                        leverage=leverage,
                        capital_allocated=effective_capital,
                        entry_time=current_date if isinstance(current_date, datetime) else current_date.to_pydatetime(),
                        mode="backtest",
                        strategy_id=self.strategy.strategy_id,
                        outcome=TradeOutcome.OPEN,
                    )

            equity_curve.append({"date": current_date, "equity": equity})

        # Close any open trade at last bar
        if open_trade is not None:
            last = data.iloc[-1]
            open_trade.exit_price = float(last["close"])
            open_trade.exit_time = last["date"]
            open_trade.outcome = TradeOutcome.OPEN
            trades.append(open_trade)

        return self._compute_result(trades, equity_curve, starting_equity)

    # ─── Private helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _check_exit(trade: TradeRecord, bar: pd.Series) -> TradeRecord:
        high = float(bar["high"])
        low = float(bar["low"])
        tp = trade.take_profit
        sl = trade.stop_loss

        hit_tp = hit_sl = False
        if trade.direction == Direction.LONG:
            hit_tp = tp is not None and high >= tp
            hit_sl = sl is not None and low <= sl
        else:
            hit_tp = tp is not None and low <= tp
            hit_sl = sl is not None and high >= sl

        if hit_tp and hit_sl:
            trade.outcome = TradeOutcome.AMBIGUOUS
            trade.exit_time = bar["date"]
        elif hit_tp:
            trade.outcome = TradeOutcome.TAKE_PROFIT
            trade.exit_price = tp
            trade.exit_time = bar["date"]
            pct = (tp - trade.entry_price) / trade.entry_price
            if trade.direction == Direction.SHORT:
                pct = -pct
            trade.leveraged_return_pct = pct * trade.leverage * 100
        elif hit_sl:
            trade.outcome = TradeOutcome.STOP_LOSS
            trade.exit_price = sl
            trade.exit_time = bar["date"]
            pct = (sl - trade.entry_price) / trade.entry_price
            if trade.direction == Direction.SHORT:
                pct = -pct
            trade.leveraged_return_pct = pct * trade.leverage * 100

        return trade

    @staticmethod
    def _signal_to_direction(signal: Signal) -> Optional[Direction]:
        if signal.action == SignalAction.BUY:
            return Direction.LONG
        if signal.action == SignalAction.SELL:
            return Direction.SHORT
        return None

    @staticmethod
    def _compute_result(
        trades: list[TradeRecord],
        equity_curve: list[dict],
        starting_equity: float,
    ) -> BacktestResult:
        closed = [t for t in trades if t.leveraged_return_pct is not None]
        wins = [t for t in closed if (t.leveraged_return_pct or 0) > 0]
        losses = [t for t in closed if (t.leveraged_return_pct or 0) <= 0]

        eq_df = pd.DataFrame(equity_curve)
        final_eq = eq_df["equity"].iloc[-1] if not eq_df.empty else starting_equity
        total_return = (final_eq - starting_equity) / starting_equity * 100 if starting_equity else 0

        # Drawdown
        if not eq_df.empty:
            roll_max = eq_df["equity"].cummax()
            drawdown = (eq_df["equity"] - roll_max) / roll_max * 100
            max_dd = float(drawdown.min())
        else:
            max_dd = 0.0

        # Sharpe (simplified daily)
        if len(eq_df) > 1:
            daily_ret = eq_df["equity"].pct_change().dropna()
            sharpe = float(daily_ret.mean() / (daily_ret.std() + 1e-9) * (252 ** 0.5))
        else:
            sharpe = 0.0

        win_rates = [t.leveraged_return_pct for t in wins]
        loss_rates = [t.leveraged_return_pct for t in losses]

        return BacktestResult(
            trades=trades,
            equity_curve=eq_df,
            total_return_pct=round(total_return, 3),
            win_rate_pct=round(len(wins) / len(closed) * 100 if closed else 0, 1),
            max_drawdown_pct=round(max_dd, 3),
            sharpe_ratio=round(sharpe, 4),
            total_trades=len(closed),
            winning_trades=len(wins),
            losing_trades=len(losses),
            avg_win_pct=round(sum(win_rates) / len(win_rates) if win_rates else 0, 3),
            avg_loss_pct=round(sum(loss_rates) / len(loss_rates) if loss_rates else 0, 3),
        )

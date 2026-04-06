"""
reporting/backtest.py
─────────────────────
Walk-forward backtesting engine — optimised for large datasets.

Key optimisation vs original:
  BEFORE: data.iloc[:i+1].copy() on every bar → O(n²) memory copies
  AFTER:  strategy pre-computes all indicator series once → O(n) total

Speedup on 100k bars (UVXY 1-min / 1 year): ~50-100×

ATR Trailing Stop (added for Decay / Spike regimes):
  Signals can set  metadata["trailing_atr_mult"] = <float>  to request a
  trailing stop instead of (or in addition to) a fixed TP.

  Mechanics for SHORT:
    trail_best  = running low since entry (moves down as price falls)
    trailing_sl = trail_best + mult × ATR[i]
    stop_loss is tightened each bar (never widened)

  Mechanics for LONG:
    trail_best  = running high since entry
    trailing_sl = trail_best - mult × ATR[i]
    stop_loss tightened each bar.

  A fixed SL from the signal acts as a hard-cap floor/ceiling —
  whichever is *tighter* at each bar wins.

  IMPORTANT: trail_best starts at entry_price on bar 0 and is updated live
  each bar. Do NOT initialise trail_peak = entry_px and then never update it —
  that caused bad interactions with spike longs in a previous session.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from core.models import Direction, Signal, SignalAction, TradeOutcome, TradeRecord
from core.logger import log
from risk.manager import RiskManager
from strategies.base import BaseStrategy
import uuid


@dataclass
class BacktestResult:
    trades:            list[TradeRecord]
    equity_curve:      pd.DataFrame
    total_return_pct:  float
    win_rate_pct:      float
    max_drawdown_pct:  float
    sharpe_ratio:      float
    total_trades:      int
    winning_trades:    int
    losing_trades:     int
    avg_win_pct:       float
    avg_loss_pct:      float

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


def _calc_atr_series(data: pd.DataFrame, period: int = 14) -> np.ndarray:
    hi   = data["high"].to_numpy(dtype=float)
    lo   = data["low"].to_numpy(dtype=float)
    cl   = data["close"].to_numpy(dtype=float)
    n    = len(cl)
    tr   = np.empty(n)
    tr[0] = hi[0] - lo[0]
    for i in range(1, n):
        tr[i] = max(hi[i] - lo[i], abs(hi[i] - cl[i - 1]), abs(lo[i] - cl[i - 1]))
    alpha  = 1.0 / period
    atr    = np.empty(n)
    atr[0] = tr[0]
    for i in range(1, n):
        atr[i] = alpha * tr[i] + (1.0 - alpha) * atr[i - 1]
    return atr


class BacktestEngine:
    def __init__(
        self,
        strategy: BaseStrategy,
        risk_manager: Optional[RiskManager] = None,
        direction_filter: Optional[Direction] = None,
        counter_signal_exit: bool = True,
        spread_pct: float = 0.0,
        slippage_pct: float = 0.0,
        commission_per_trade: float = 0.0,
    ) -> None:
        self.strategy = strategy
        self.risk = risk_manager
        self.direction_filter = direction_filter
        self.counter_signal_exit = counter_signal_exit
        self.spread_pct = spread_pct
        self.slippage_pct = slippage_pct
        self.commission_per_trade = commission_per_trade

    def run(
        self,
        data: pd.DataFrame,
        symbol: str,
        leverage: float = 1.0,
        capital_per_trade: float = 1000.0,
        starting_equity: float = 10_000.0,
    ) -> BacktestResult:

        n = len(data)
        log.info(
            f"Backtest START: {symbol} | {self.strategy.strategy_id} | "
            f"{n:,} bars | lev={leverage}x | capital/trade={capital_per_trade}"
        )

        try:
            actions_s, meta_s = self.strategy.generate_signals_bulk(data, symbol)
            log.info(f"Bulk signal generation complete: {n:,} bars")
        except NotImplementedError:
            log.warning(
                f"{self.strategy.strategy_id} has no bulk mode — "
                f"falling back to bar-by-bar (slow on large datasets)"
            )
            actions_s = []
            meta_s = []
            for i in range(n):
                sig = self.strategy.generate_signal(data.iloc[: i + 1].copy(), symbol)
                actions_s.append(sig.action)
                meta_s.append(
                    {
                        "suggested_tp": sig.suggested_tp,
                        "suggested_sl": sig.suggested_sl,
                        "metadata": sig.metadata,
                    }
                )

        atr_arr = _calc_atr_series(data, period=14)
        high_arr = data["high"].to_numpy(dtype=float)
        low_arr = data["low"].to_numpy(dtype=float)
        dates = data["date"].to_numpy()

        trades: list[TradeRecord] = []
        equity = starting_equity
        equity_curve: list[dict] = []
        open_trade: Optional[TradeRecord] = None

        trail_best: Optional[float] = None
        trail_mult: Optional[float] = None
        trail_pct: Optional[float] = None
        trail_giveback_frac: Optional[float] = None
        trail_giveback_min_pct: float = 0.0
        trail_hard_sl: Optional[float] = None
        trail_grace: int = 0
        trail_bars: int = 0

        def _reset_trail():
            nonlocal trail_best, trail_mult, trail_pct, trail_giveback_frac, trail_giveback_min_pct, trail_hard_sl, trail_grace, trail_bars
            trail_best = None
            trail_mult = None
            trail_pct = None
            trail_giveback_frac = None
            trail_giveback_min_pct = 0.0
            trail_hard_sl = None
            trail_grace = 0
            trail_bars = 0

        for i in range(1, n):
            bar = data.iloc[i]
            current_date = dates[i]
            action = actions_s[i]
            meta = meta_s[i]
            new_direction = self._signal_to_direction(action)

            if open_trade is not None and (trail_mult is not None or trail_pct is not None or trail_giveback_frac is not None):
                trail_bars += 1
                if trail_bars > trail_grace:
                    if open_trade.direction == Direction.SHORT:
                        trail_best = min(trail_best, low_arr[i])
                        if trail_mult is not None:
                            candidate_sl = trail_best + trail_mult * atr_arr[i]
                        elif trail_pct is not None:
                            candidate_sl = trail_best * (1 + trail_pct / 100)
                        else:
                            profit_move = max(open_trade.entry_price - trail_best, 0.0)
                            profit_move_pct = (profit_move / open_trade.entry_price) * 100 if open_trade.entry_price > 0 else 0.0
                            if profit_move_pct >= trail_giveback_min_pct:
                                candidate_sl = trail_best + trail_giveback_frac * profit_move
                            else:
                                candidate_sl = trail_hard_sl if trail_hard_sl is not None else open_trade.stop_loss
                        open_trade.stop_loss = min(candidate_sl, trail_hard_sl) if trail_hard_sl is not None else candidate_sl
                    else:
                        trail_best = max(trail_best, high_arr[i])
                        if trail_mult is not None:
                            candidate_sl = trail_best - trail_mult * atr_arr[i]
                        elif trail_pct is not None:
                            candidate_sl = trail_best * (1 - trail_pct / 100)
                        else:
                            profit_move = max(trail_best - open_trade.entry_price, 0.0)
                            profit_move_pct = (profit_move / open_trade.entry_price) * 100 if open_trade.entry_price > 0 else 0.0
                            if profit_move_pct >= trail_giveback_min_pct:
                                candidate_sl = trail_best - trail_giveback_frac * profit_move
                            else:
                                candidate_sl = trail_hard_sl if trail_hard_sl is not None else open_trade.stop_loss
                        open_trade.stop_loss = max(candidate_sl, trail_hard_sl) if trail_hard_sl is not None else candidate_sl

            if open_trade is not None:
                open_trade = self._check_exit(open_trade, bar)
                if open_trade.outcome not in (TradeOutcome.OPEN, TradeOutcome.NO_DATA):
                    if open_trade.leveraged_return_pct is not None:
                        trade_pnl = open_trade.capital_allocated * open_trade.leveraged_return_pct / 100
                        trade_pnl -= self._trade_cost(open_trade.capital_allocated)
                        equity += trade_pnl
                        open_trade.pnl = trade_pnl
                    trades.append(open_trade)
                    open_trade = None
                    _reset_trail()

            _spike_trade = open_trade is not None and any(
                r in (open_trade.notes or "")
                for r in ("regime=spike_long", "regime=spike_momentum_long", "regime=post_spike_short", "regime=event_target_short")
            )
            if self.counter_signal_exit and open_trade is not None and new_direction is not None and not _spike_trade:
                current_dir = open_trade.direction
                is_reversal = (
                    (current_dir == Direction.LONG and new_direction == Direction.SHORT)
                    or (current_dir == Direction.SHORT and new_direction == Direction.LONG)
                )
                if is_reversal:
                    exit_px = float(bar["close"])
                    pct = (exit_px - open_trade.entry_price) / open_trade.entry_price
                    if current_dir == Direction.SHORT:
                        pct = -pct
                    open_trade.leveraged_return_pct = pct * open_trade.leverage * 100
                    open_trade.exit_price = exit_px
                    open_trade.exit_time = current_date if isinstance(current_date, datetime) else pd.Timestamp(current_date).to_pydatetime()
                    open_trade.outcome = self._counter_signal_outcome(
                        open_trade.direction, action, self.strategy.strategy_id, meta.get("metadata", {})
                    )
                    trade_pnl = open_trade.capital_allocated * open_trade.leveraged_return_pct / 100
                    trade_pnl -= self._trade_cost(open_trade.capital_allocated)
                    equity += trade_pnl
                    open_trade.pnl = trade_pnl
                    trades.append(open_trade)
                    open_trade = None
                    _reset_trail()

            if i > 1:
                prev_action = actions_s[i - 1]
                prev_meta = meta_s[i - 1]
                prev_dir = self._signal_to_direction(prev_action)
                prev_sl = prev_meta.get("suggested_sl")
                prev_tp = prev_meta.get("suggested_tp")

                if open_trade is None and prev_dir is not None and prev_sl is not None:
                    entry_px = float(bar["open"])
                    signal_close = float(data.iloc[i - 1]["close"])
                    adj_sl = self._reanchor_level(
                        direction=prev_dir,
                        level=prev_sl,
                        signal_close=signal_close,
                        entry_price=entry_px,
                        is_stop=True,
                    )
                    adj_tp = self._reanchor_level(
                        direction=prev_dir,
                        level=prev_tp,
                        signal_close=signal_close,
                        entry_price=entry_px,
                        is_stop=False,
                    )

                    if self.risk:
                        check = self.risk.check(
                            direction=prev_dir,
                            entry_price=entry_px,
                            take_profit=adj_tp,
                            stop_loss=adj_sl,
                            leverage=leverage,
                            capital_requested=capital_per_trade,
                        )
                        if not check.approved:
                            equity_curve.append({"date": current_date, "equity": equity})
                            continue
                        effective_sl = check.adjusted_sl or adj_sl
                        effective_capital = check.adjusted_size or capital_per_trade
                    else:
                        effective_sl = adj_sl
                        effective_capital = capital_per_trade

                    prev_sig_meta = prev_meta.get("metadata", {})
                    prev_regime = prev_sig_meta.get("regime", "normal")

                    open_trade = TradeRecord(
                        id=str(uuid.uuid4()),
                        symbol=symbol,
                        direction=prev_dir,
                        entry_price=entry_px,
                        take_profit=adj_tp,
                        stop_loss=effective_sl,
                        leverage=leverage,
                        capital_allocated=effective_capital,
                        entry_time=current_date if isinstance(current_date, datetime) else pd.Timestamp(current_date).to_pydatetime(),
                        mode="backtest",
                        strategy_id=self.strategy.strategy_id,
                        outcome=TradeOutcome.OPEN,
                        notes=(
                            f"Entry: {prev_dir.value} @ {entry_px:.4f} (next-bar open) | "
                            f"regime={prev_regime} | "
                            f"SL={effective_sl:.4f} | "
                            + (f"TP={adj_tp:.4f}" if adj_tp is not None else "TP=none")
                        ),
                    )

                    req_atr = prev_sig_meta.get("trailing_atr_mult")
                    req_pct = prev_sig_meta.get("pct_trail")
                    req_giveback = prev_sig_meta.get("profit_giveback_frac")
                    req_giveback_min_pct = prev_sig_meta.get("profit_giveback_min_pct", 0.0)
                    if req_atr is not None:
                        trail_mult = float(req_atr)
                        trail_pct = None
                        trail_giveback_frac = None
                        trail_giveback_min_pct = 0.0
                        trail_best = entry_px
                        trail_hard_sl = effective_sl
                        trail_grace = 0
                        trail_bars = 0
                        open_trade.notes += f" | trail=atr:{trail_mult:.2f}"
                    elif req_pct is not None:
                        trail_mult = None
                        trail_pct = float(req_pct)
                        trail_giveback_frac = None
                        trail_giveback_min_pct = 0.0
                        trail_best = entry_px
                        trail_hard_sl = effective_sl
                        trail_grace = 1
                        trail_bars = 0
                        open_trade.notes += f" | trail=pct:{trail_pct:.2f}"
                    elif req_giveback is not None:
                        trail_mult = None
                        trail_pct = None
                        trail_giveback_frac = float(req_giveback)
                        trail_giveback_min_pct = float(req_giveback_min_pct or 0.0)
                        trail_best = entry_px
                        trail_hard_sl = effective_sl
                        trail_grace = 1
                        trail_bars = 0
                        open_trade.notes += f" | trail=giveback:{trail_giveback_frac:.2f},min:{trail_giveback_min_pct:.2f}"
                    else:
                        _reset_trail()

            equity_curve.append({"date": current_date, "equity": equity})

        if open_trade is not None:
            last = data.iloc[-1]
            open_trade.exit_price = float(last["close"])
            open_trade.exit_time = last["date"]
            open_trade.outcome = TradeOutcome.OPEN
            trades.append(open_trade)

        return self._compute_result(trades, equity_curve, starting_equity)

    def _trade_cost(self, capital: float) -> float:
        """Round-trip cost: spread + slippage (% of capital) + flat commission."""
        return capital * (self.spread_pct + self.slippage_pct) / 100.0 + self.commission_per_trade

    @staticmethod
    def _reanchor_level(
        *,
        direction: Direction,
        level: Optional[float],
        signal_close: float,
        entry_price: float,
        is_stop: bool,
    ) -> Optional[float]:
        if level is None or signal_close <= 0 or entry_price <= 0:
            return level
        if direction == Direction.LONG:
            move = ((signal_close - level) / signal_close) if is_stop else ((level - signal_close) / signal_close)
            return entry_price * (1 - move) if is_stop else entry_price * (1 + move)
        move = ((level - signal_close) / signal_close) if is_stop else ((signal_close - level) / signal_close)
        return entry_price * (1 + move) if is_stop else entry_price * (1 - move)

    @staticmethod
    def _counter_signal_outcome(
        open_direction: Direction,
        action: SignalAction,
        strategy_id: str,
        metadata: dict,
    ) -> TradeOutcome:
        if "rsi" in strategy_id:
            return TradeOutcome.SIGNAL_RSI_OB if open_direction == Direction.LONG else TradeOutcome.SIGNAL_RSI_OS
        return TradeOutcome.SIGNAL_EXIT

    @staticmethod
    def _check_exit(trade: TradeRecord, bar: pd.Series) -> TradeRecord:
        high = float(bar["high"])
        low = float(bar["low"])
        tp = trade.take_profit
        sl = trade.stop_loss
        entry = trade.entry_price

        hit_tp = hit_sl = False
        if trade.direction == Direction.LONG:
            hit_sl = sl is not None and low <= sl
            hit_tp = tp is not None and high >= tp
        else:
            hit_sl = sl is not None and high >= sl
            hit_tp = tp is not None and low <= tp

        if hit_sl and hit_tp:
            trade.outcome = TradeOutcome.AMBIGUOUS
            trade.exit_time = bar["date"]
        elif hit_sl:
            trade.outcome = TradeOutcome.TRAIL_STOP if "trail=" in (trade.notes or "") else TradeOutcome.STOP_LOSS
            trade.exit_price = sl
            trade.exit_time = bar["date"]
            pct = (sl - entry) / entry
            if trade.direction == Direction.SHORT:
                pct = -pct
            trade.leveraged_return_pct = pct * trade.leverage * 100
        elif hit_tp:
            trade.outcome = TradeOutcome.TAKE_PROFIT
            trade.exit_price = tp
            trade.exit_time = bar["date"]
            pct = (tp - entry) / entry
            if trade.direction == Direction.SHORT:
                pct = -pct
            trade.leveraged_return_pct = pct * trade.leverage * 100

        return trade

    @staticmethod
    def _signal_to_direction(action: SignalAction) -> Optional[Direction]:
        if action == SignalAction.BUY:
            return Direction.LONG
        if action == SignalAction.SELL:
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
        if len(eq_df) > 2000:
            step = len(eq_df) // 2000
            eq_df = (
                pd.concat([eq_df.iloc[::step], eq_df.iloc[[-1]]])
                .drop_duplicates("date")
                .sort_values("date")
                .reset_index(drop=True)
            )

        final_eq = float(eq_df["equity"].iloc[-1]) if not eq_df.empty else starting_equity
        total_ret = ((final_eq - starting_equity) / starting_equity * 100 if starting_equity else 0)

        if not eq_df.empty:
            roll_max = eq_df["equity"].cummax()
            drawdown = (eq_df["equity"] - roll_max) / roll_max * 100
            max_dd = float(drawdown.min())
        else:
            max_dd = 0.0

        if len(eq_df) > 1:
            daily_ret = eq_df["equity"].pct_change().dropna()
            sharpe = float(daily_ret.mean() / (daily_ret.std() + 1e-9) * (252 ** 0.5))
        else:
            sharpe = 0.0

        win_rets = [t.leveraged_return_pct for t in wins]
        loss_rets = [t.leveraged_return_pct for t in losses]

        return BacktestResult(
            trades=trades,
            equity_curve=eq_df,
            total_return_pct=round(total_ret, 3),
            win_rate_pct=round(len(wins) / len(closed) * 100 if closed else 0, 1),
            max_drawdown_pct=round(max_dd, 3),
            sharpe_ratio=round(sharpe, 4),
            total_trades=len(closed),
            winning_trades=len(wins),
            losing_trades=len(losses),
            avg_win_pct=round(sum(win_rets) / len(win_rets) if win_rets else 0, 3),
            avg_loss_pct=round(sum(loss_rets) / len(loss_rets) if loss_rets else 0, 3),
        )

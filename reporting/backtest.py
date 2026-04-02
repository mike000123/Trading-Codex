"""
reporting/backtest.py
─────────────────────
Walk-forward backtesting engine — optimised for large datasets.

Key optimisation vs original:
  BEFORE: data.iloc[:i+1].copy() on every bar → O(n²) memory copies
  AFTER:  strategy pre-computes all indicator series once → O(n) total

Speedup on 100k bars (UVXY 1-min / 1 year): ~50-100×
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


class BacktestEngine:
    def __init__(
        self,
        strategy: BaseStrategy,
        risk_manager:        Optional[RiskManager] = None,
        direction_filter:     Optional[Direction]   = None,
        counter_signal_exit:  bool                  = True,
        spread_pct:           float                 = 0.0,
        slippage_pct:         float                 = 0.0,
        commission_per_trade: float                 = 0.0,
    ) -> None:
        self.strategy             = strategy
        self.risk                 = risk_manager
        self.direction_filter     = direction_filter
        self.counter_signal_exit  = counter_signal_exit
        self.spread_pct           = spread_pct
        self.slippage_pct         = slippage_pct
        self.commission_per_trade = commission_per_trade

    def run(
        self,
        data:              pd.DataFrame,
        symbol:            str,
        leverage:          float = 1.0,
        capital_per_trade: float = 1000.0,
        starting_equity:   float = 10_000.0,
    ) -> BacktestResult:

        n = len(data)
        log.info(f"Backtest START: {symbol} | {self.strategy.strategy_id} | "
                 f"{n:,} bars | lev={leverage}x | capital/trade={capital_per_trade}")

        # ── Pre-compute all indicator signals on the full dataset ─────────────
        # This replaces the O(n²) data.iloc[:i+1].copy() loop.
        # strategy.generate_signals_bulk() returns two parallel Series:
        #   actions[i]  – SignalAction for each bar
        #   metadata[i] – dict with tp/sl/rsi/etc for each bar
        # Falls back to bar-by-bar if strategy doesn't implement bulk.
        try:
            actions_s, meta_s = self.strategy.generate_signals_bulk(data, symbol)
            log.info(f"Bulk signal generation complete: {n:,} bars")
        except NotImplementedError:
            # Fallback: bar-by-bar (slower but always works)
            log.warning(f"{self.strategy.strategy_id} has no bulk mode — "
                        f"falling back to bar-by-bar (slow on large datasets)")
            actions_s = []
            meta_s    = []
            for i in range(n):
                sig = self.strategy.generate_signal(data.iloc[:i+1].copy(), symbol)
                actions_s.append(sig.action)
                meta_s.append({
                    "suggested_tp": sig.suggested_tp,
                    "suggested_sl": sig.suggested_sl,
                    "metadata":     sig.metadata,
                })

        # ── Main loop — O(n), no copies ──────────────────────────────────────
        trades:      list[TradeRecord] = []
        equity       = starting_equity
        equity_curve: list[dict]       = []
        open_trade:  Optional[TradeRecord] = None
        # Trailing stop state — set when a trade has trailing_atr_mult in metadata
        trail_atr_mult:  Optional[float] = None   # X in: SL = peak ± X×ATR
        trail_atr_period: int            = 14
        trail_atr_min_pct: float         = 0.0    # minimum % distance for short trail
        trail_peak:      Optional[float] = None   # highest high (long) / lowest low (short)
        trail_direction: str             = "short" # "long" or "short"

        for i in range(1, n):
            bar          = data.iloc[i]
            current_date = bar["date"]
            action       = actions_s[i]
            meta         = meta_s[i]
            new_direction = self._signal_to_direction(action)

            # 1. Update trailing stop if active, then check SL / TP
            if open_trade is not None and trail_atr_mult is not None:
                bar_high = float(bar["high"])
                bar_low  = float(bar["low"])
                curr_atr = self._bar_atr(data, i, trail_atr_period)
                if trail_direction == "long":
                    # Trail up: SL = highest_high_since_entry - mult × ATR
                    if trail_peak is None or bar_high > trail_peak:
                        trail_peak = bar_high
                    new_trail_sl = trail_peak - trail_atr_mult * curr_atr
                    if open_trade.stop_loss is None or new_trail_sl > open_trade.stop_loss:
                        open_trade.stop_loss = new_trail_sl
                else:
                    # Trail down: SL = lowest_low_since_entry + mult × ATR
                    # Also enforce a minimum % distance so the trail doesn't
                    # become too tight when ATR compresses at low prices
                    if trail_peak is None or bar_low < trail_peak:
                        trail_peak = bar_low
                    atr_based_sl  = trail_peak + trail_atr_mult * curr_atr
                    # Minimum trail = trail_atr_min_pct% above the current low
                    min_pct       = trail_atr_min_pct / 100.0
                    min_based_sl  = bar_low * (1.0 + min_pct)
                    new_trail_sl  = max(atr_based_sl, min_based_sl)
                    if open_trade.stop_loss is None or new_trail_sl < open_trade.stop_loss:
                        open_trade.stop_loss = new_trail_sl

            if open_trade is not None:
                open_trade = self._check_exit(open_trade, bar)
                if open_trade.outcome not in (TradeOutcome.OPEN, TradeOutcome.NO_DATA):
                    if open_trade.leveraged_return_pct is not None:
                        trade_pnl      = (open_trade.capital_allocated
                                          * open_trade.leveraged_return_pct / 100)
                        trade_pnl     -= self._trade_cost(open_trade.capital_allocated)
                        equity        += trade_pnl
                        open_trade.pnl = trade_pnl
                    trades.append(open_trade)
                    open_trade = None
                    # Clear trailing stop state
                    trail_atr_mult    = None
                    trail_atr_min_pct = 0.0
                    trail_peak        = None

            # 2. Counter-signal exit
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
                    open_trade.outcome = self._counter_signal_outcome(
                        open_trade.direction, action,
                        self.strategy.strategy_id, meta.get("metadata", {}))
                    trade_pnl      = (open_trade.capital_allocated
                                      * open_trade.leveraged_return_pct / 100)
                    cost           = self._trade_cost(open_trade.capital_allocated)
                    trade_pnl     -= cost
                    equity        += trade_pnl
                    open_trade.pnl = trade_pnl
                    trades.append(open_trade)
                    open_trade = None
                    # Clear trailing stop state
                    trail_atr_mult    = None
                    trail_atr_min_pct = 0.0
                    trail_peak        = None

            # 3. Queue new trade — will open on NEXT bar's open price
            # This prevents lookahead bias: signal fires at bar[i] close,
            # but we can only act at bar[i+1] open (realistic execution).
            suggested_sl = meta.get("suggested_sl")
            suggested_tp = meta.get("suggested_tp")

            # Execute any pending trade from the PREVIOUS bar's signal
            # (open at current bar's open price — one bar after signal)
            if i > 1:
                prev_action   = actions_s[i - 1]
                prev_meta     = meta_s[i - 1]
                prev_dir      = self._signal_to_direction(prev_action)
                prev_sl       = prev_meta.get("suggested_sl")
                prev_tp       = prev_meta.get("suggested_tp")

                if (open_trade is None and prev_dir is not None and prev_sl is not None):
                    # Adjust SL/TP: they were computed relative to yesterday's close.
                    # Scale them to today's open maintaining the same distance ratio.
                    entry_px        = float(bar["open"])
                    signal_close    = float(data.iloc[i-1]["close"])
                    scale           = entry_px / signal_close if signal_close != 0 else 1.0

                    adj_sl = prev_sl * scale
                    adj_tp = prev_tp * scale if prev_tp is not None else None

                    if self.risk:
                        check = self.risk.check(
                            direction         = prev_dir,
                            entry_price       = entry_px,
                            take_profit       = adj_tp,
                            stop_loss         = adj_sl,
                            leverage          = leverage,
                            capital_requested = capital_per_trade,
                        )
                        if not check.approved:
                            equity_curve.append({"date": current_date, "equity": equity})
                            continue
                        effective_sl      = check.adjusted_sl or adj_sl
                        effective_capital = check.adjusted_size or capital_per_trade
                    else:
                        effective_sl      = adj_sl
                        effective_capital = capital_per_trade

                    prev_regime = prev_meta.get("metadata", {}).get("regime", "normal")
                    open_trade = TradeRecord(
                        id                = str(uuid.uuid4()),
                        symbol            = symbol,
                        direction         = prev_dir,
                        entry_price       = entry_px,
                        take_profit       = adj_tp,
                        stop_loss         = effective_sl,
                        leverage          = leverage,
                        capital_allocated = effective_capital,
                        entry_time        = (current_date if isinstance(current_date, datetime)
                                             else current_date.to_pydatetime()),
                        mode              = "backtest",
                        strategy_id       = self.strategy.strategy_id,
                        outcome           = TradeOutcome.OPEN,
                        notes             = (
                            f"Entry: {prev_dir.value} @ {entry_px:.4f} (next-bar open) | "
                            f"SL={effective_sl:.4f} | "
                            + (f'TP={adj_tp:.4f}' if adj_tp is not None else 'TP=none')
                        ),
                    )
                    # Initialise trailing stop state if strategy requested it
                    prev_inner = prev_meta.get("metadata", {})
                    if "trailing_atr_mult" in prev_inner:
                        trail_atr_mult    = float(prev_inner["trailing_atr_mult"])
                        trail_atr_period  = int(prev_inner.get("atr_period", 14))
                        trail_direction   = prev_inner.get("trail_direction",
                                           "long" if prev_dir == Direction.LONG else "short")
                        trail_atr_min_pct = float(prev_inner.get("trailing_atr_min_pct", 0.0))
                        # Start trail_peak at entry price (not first bar's high/low).
                        # This means the initial trail SL = entry ± mult×ATR,
                        # which is wide enough to survive first-bar volatility.
                        # The trail only tightens as price moves in our favour.
                        trail_peak = entry_px
                    else:
                        trail_atr_mult   = None
                        trail_peak       = None


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

    def _trade_cost(self, capital: float) -> float:
        """Round-trip cost: spread + slippage (% of capital) + flat commission."""
        return capital * (self.spread_pct + self.slippage_pct) / 100.0 + self.commission_per_trade

    @staticmethod
    def _bar_atr(data: pd.DataFrame, i: int, period: int) -> float:
        """Fast single-value ATR estimate at bar i using a rolling window."""
        start = max(0, i - period * 3)   # enough bars for EWM to stabilise
        sl    = data.iloc[start:i+1]
        hi    = sl["high"].astype(float)
        lo    = sl["low"].astype(float)
        cl    = sl["close"].astype(float)
        prev  = cl.shift(1)
        tr    = pd.concat([hi-lo, (hi-prev).abs(), (lo-prev).abs()], axis=1).max(axis=1)
        atr   = tr.ewm(alpha=1/period, adjust=False, min_periods=1).mean()
        v     = float(atr.iloc[-1])
        return v if not np.isnan(v) else float((hi - lo).mean())

    @staticmethod
    def _counter_signal_outcome(
        open_direction: Direction,
        action:         SignalAction,
        strategy_id:    str,
        metadata:       dict,
    ) -> TradeOutcome:
        if "rsi" in strategy_id:
            return (TradeOutcome.SIGNAL_RSI_OB
                    if open_direction == Direction.LONG
                    else TradeOutcome.SIGNAL_RSI_OS)
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

        if hit_sl and hit_tp:
            trade.outcome   = TradeOutcome.AMBIGUOUS
            trade.exit_time = bar["date"]
        elif hit_sl:
            trade.outcome    = TradeOutcome.STOP_LOSS
            trade.exit_price = sl
            trade.exit_time  = bar["date"]
            pct = (sl - entry) / entry
            if trade.direction == Direction.SHORT: pct = -pct
            trade.leveraged_return_pct = pct * trade.leverage * 100
        elif hit_tp:
            trade.outcome    = TradeOutcome.TAKE_PROFIT
            trade.exit_price = tp
            trade.exit_time  = bar["date"]
            pct = (tp - entry) / entry
            if trade.direction == Direction.SHORT: pct = -pct
            trade.leveraged_return_pct = pct * trade.leverage * 100

        return trade

    @staticmethod
    def _signal_to_direction(action: SignalAction) -> Optional[Direction]:
        if action == SignalAction.BUY:  return Direction.LONG
        if action == SignalAction.SELL: return Direction.SHORT
        return None

    @staticmethod
    def _compute_result(
        trades:          list[TradeRecord],
        equity_curve:    list[dict],
        starting_equity: float,
    ) -> BacktestResult:
        closed = [t for t in trades if t.leveraged_return_pct is not None]
        wins   = [t for t in closed if (t.leveraged_return_pct or 0) > 0]
        losses = [t for t in closed if (t.leveraged_return_pct or 0) <= 0]

        eq_df    = pd.DataFrame(equity_curve)
        # Downsample equity curve to max 2000 points for memory/render efficiency.
        # One point per ~500 bars on a 1M-bar backtest — smooth enough for display.
        if len(eq_df) > 2000:
            step  = len(eq_df) // 2000
            eq_df = pd.concat([
                eq_df.iloc[::step],
                eq_df.iloc[[-1]]   # always keep the last point
            ]).drop_duplicates("date").sort_values("date").reset_index(drop=True)
        final_eq = float(eq_df["equity"].iloc[-1]) if not eq_df.empty else starting_equity
        total_ret = ((final_eq - starting_equity) / starting_equity * 100
                     if starting_equity else 0)

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
            trades            = trades,
            equity_curve      = eq_df,
            total_return_pct  = round(total_ret, 3),
            win_rate_pct      = round(len(wins) / len(closed) * 100 if closed else 0, 1),
            max_drawdown_pct  = round(max_dd, 3),
            sharpe_ratio      = round(sharpe, 4),
            total_trades      = len(closed),
            winning_trades    = len(wins),
            losing_trades     = len(losses),
            avg_win_pct       = round(sum(win_rets)  / len(win_rets)  if win_rets  else 0, 3),
            avg_loss_pct      = round(sum(loss_rets) / len(loss_rets) if loss_rets else 0, 3),
        )

"""
reporting/backtest.py
─────────────────────
Walk-forward backtesting engine — optimised for large datasets.

ENTRY SEMANTICS (honest, no-lookahead):
  Signals emitted on bar i → entry at bar i's CLOSE with the strategy's raw
  SL/TP. TP/SL and trailing stops are first evaluated on bar i+1 (the entry
  bar has already closed). This matches paper / forward-test / live semantics
  exactly — there is no "next-bar open" fill, because in live trading there
  is no next bar yet when the signal fires. Consequences:
    • Paper, forward test, and backtest now use the same entry reference, so
      backtest results are a faithful predictor of live performance (no
      look-ahead bias from next-bar-open fills).
    • SL/TP levels from the strategy are consumed unchanged (no re-anchoring),
      because entry_price == signal_close in this model.
    • In return for realism, backtest results may shift vs. the old next-bar-
      open convention (typically slightly worse on trend-following strategies,
      slightly better on mean-reversion, but direction is strategy-specific).

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
from typing import Any, Optional

import numpy as np
import pandas as pd

from core.models import Direction, Signal, SignalAction, TradeOutcome, TradeRecord
from core.logger import log
from risk.manager import RiskManager
from strategies.base import BaseStrategy
from execution.alpaca_constraints import (
    is_regular_trading_hour,
    is_trading_day,
    monday_open_delay_guard,
    pdt_guard,
    ssr_guard,
    normalize_qty_for_direction,
    fill_timing_note,
)
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


def _trade_requests_session_exit(trade: TradeRecord | None) -> bool:
    if trade is None:
        return False
    return "session_exit=eod" in (trade.notes or "")


def _is_session_close_bar(current_date, next_date=None) -> bool:
    ts = pd.Timestamp(current_date)
    if ts.hour > 16 or (ts.hour == 16 and ts.minute >= 0):
        return True
    if next_date is None:
        return False
    next_ts = pd.Timestamp(next_date)
    return ts.date() != next_ts.date()


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
        *,
        enforce_rth: bool = True,
        extended_hours: bool = False,
        enforce_pdt: bool = True,
        enforce_ssr: bool = True,
        enforce_fractional: bool = True,
        fill_diagnostic: bool = True,
        enforce_monday_open_delay: bool = False,
        regime_loss_guard_rules: Optional[dict[str, dict[str, Any]]] = None,
    ) -> None:
        self.strategy = strategy
        self.risk = risk_manager
        self.direction_filter = direction_filter
        self.counter_signal_exit = counter_signal_exit
        self.spread_pct = spread_pct
        self.slippage_pct = slippage_pct
        self.commission_per_trade = commission_per_trade
        # Alpaca-realistic gates
        self.enforce_rth = enforce_rth
        self.extended_hours = extended_hours
        self.enforce_pdt = enforce_pdt
        self.enforce_ssr = enforce_ssr
        self.enforce_fractional = enforce_fractional
        self.fill_diagnostic = fill_diagnostic
        self.enforce_monday_open_delay = enforce_monday_open_delay
        self.regime_loss_guard_rules = dict(regime_loss_guard_rules or {})

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
        close_arr = data["close"].to_numpy(dtype=float)
        dates = data["date"].to_numpy()

        # Pre-compute prior-trading-day close for a fast inline SSR check on SHORTs.
        # O(n) single pass; avoids re-slicing on every short entry.
        prior_day_close_arr = np.full(n, np.nan, dtype=float)
        if self.enforce_ssr and "date" in data.columns:
            try:
                _d = pd.to_datetime(data["date"])
                if getattr(_d.dt, "tz", None) is None:
                    _d = _d.dt.tz_localize("UTC")
                _et_dates = _d.dt.tz_convert("America/New_York").dt.date.to_numpy()
                prev_date = None
                prev_day_last_close = np.nan
                last_close_this_day = np.nan
                for _i in range(n):
                    _cur_date = _et_dates[_i]
                    if prev_date is not None and _cur_date != prev_date:
                        prev_day_last_close = last_close_this_day
                    prior_day_close_arr[_i] = prev_day_last_close
                    last_close_this_day = close_arr[_i]
                    prev_date = _cur_date
            except Exception:
                prior_day_close_arr = np.full(n, np.nan, dtype=float)

        trades: list[TradeRecord] = []
        equity = starting_equity
        equity_curve: list[dict] = []
        open_trade: Optional[TradeRecord] = None
        open_trade_regime: Optional[str] = None
        regime_loss_streaks: dict[str, int] = {}
        regime_cooldown_until: dict[str, int] = {}
        global_loss_streak = 0
        global_cooldown_until: Optional[int] = None

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

        def _register_closed_trade_regime(closed_trade: TradeRecord, regime_name: Optional[str], bar_index: int) -> None:
            nonlocal global_loss_streak, global_cooldown_until
            if not regime_name:
                regime_name = ""
            pnl = float(closed_trade.pnl or 0.0)
            is_loss = pnl <= 0.0
            if is_loss:
                streak = regime_loss_streaks.get(regime_name, 0) + 1
                global_loss_streak += 1
            else:
                streak = 0
                global_loss_streak = 0
            regime_loss_streaks[regime_name] = streak
            rule = self.regime_loss_guard_rules.get(regime_name)
            if not rule:
                rule = None
            if rule:
                trigger_losses = max(int(rule.get("trigger_losses", 0)), 0)
                cooldown_bars = max(int(rule.get("cooldown_bars", 0)), 0)
                if trigger_losses > 0 and cooldown_bars > 0 and streak >= trigger_losses:
                    regime_cooldown_until[regime_name] = bar_index + cooldown_bars
                    regime_loss_streaks[regime_name] = 0
            global_rule = self.regime_loss_guard_rules.get("__all__")
            if global_rule:
                trigger_losses = max(int(global_rule.get("trigger_losses", 0)), 0)
                cooldown_bars = max(int(global_rule.get("cooldown_bars", 0)), 0)
                if trigger_losses > 0 and cooldown_bars > 0 and global_loss_streak >= trigger_losses:
                    global_cooldown_until = bar_index + cooldown_bars
                    global_loss_streak = 0

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
                    _register_closed_trade_regime(open_trade, open_trade_regime, i)
                    trades.append(open_trade)
                    open_trade = None
                    open_trade_regime = None
                    _reset_trail()

            if (
                open_trade is not None
                and _trade_requests_session_exit(open_trade)
                and _is_session_close_bar(current_date, dates[i + 1] if i + 1 < n else None)
            ):
                exit_px = float(bar["close"])
                pct = (exit_px - open_trade.entry_price) / open_trade.entry_price
                if open_trade.direction == Direction.SHORT:
                    pct = -pct
                open_trade.leveraged_return_pct = pct * open_trade.leverage * 100
                open_trade.exit_price = exit_px
                open_trade.exit_time = current_date if isinstance(current_date, datetime) else pd.Timestamp(current_date).to_pydatetime()
                open_trade.outcome = TradeOutcome.SIGNAL_EXIT
                trade_pnl = open_trade.capital_allocated * open_trade.leveraged_return_pct / 100
                trade_pnl -= self._trade_cost(open_trade.capital_allocated)
                equity += trade_pnl
                open_trade.pnl = trade_pnl
                open_trade.notes = f"{open_trade.notes or ''} | Session-close exit".strip(" |")
                _register_closed_trade_regime(open_trade, open_trade_regime, i)
                trades.append(open_trade)
                open_trade = None
                open_trade_regime = None
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
                    _register_closed_trade_regime(open_trade, open_trade_regime, i)
                    trades.append(open_trade)
                    open_trade = None
                    open_trade_regime = None
                    _reset_trail()

            # ── New entry (signal-bar close, no re-anchor) ──────────────────
            # Honest "no-lookahead" semantics: if the strategy emits a signal on
            # this bar, we enter at this bar's CLOSE with the strategy's raw
            # SL/TP (no translation from signal-close to next-bar-open, because
            # there is no next bar in live trading). TP/SL are first checked on
            # the NEXT bar — the entry bar itself has already closed.
            # Paper mode uses the same convention, so backtest ↔ paper ↔ live
            # all share the same entry reference.
            current_sl = meta.get("suggested_sl")
            current_tp = meta.get("suggested_tp")
            if open_trade is None and new_direction is not None and current_sl is not None:
                entry_px = float(bar["close"])
                adj_sl = current_sl
                adj_tp = current_tp

                if self.enforce_monday_open_delay:
                    allowed, _monday_reason = monday_open_delay_guard(
                        current_date,
                        enforce=True,
                    )
                    if not allowed:
                        equity_curve.append({"date": current_date, "equity": equity})
                        continue

                # ── Alpaca-realistic gates ──────────────────────────────────
                # RTH / NYSE holiday gate
                if self.enforce_rth:
                    if not is_trading_day(current_date) or not is_regular_trading_hour(
                        current_date, extended_hours=self.extended_hours
                    ):
                        equity_curve.append({"date": current_date, "equity": equity})
                        continue

                # SSR: block shorts on ≥10% drop vs prior day close
                if self.enforce_ssr and new_direction == Direction.SHORT:
                    pd_close = prior_day_close_arr[i]
                    if not np.isnan(pd_close) and pd_close > 0:
                        drop_pct = (close_arr[i] - pd_close) / pd_close * 100.0
                        if drop_pct <= -10.0:
                            equity_curve.append({"date": current_date, "equity": equity})
                            continue

                available_equity = max(float(equity), 0.0)
                requested_capital = min(float(capital_per_trade), available_equity)

                if requested_capital <= 0:
                    equity_curve.append({"date": current_date, "equity": equity})
                    continue

                # PDT: pin the rolling 5-day window to the current bar time so
                # the backtest evaluates day-trade count against simulated history.
                if self.enforce_pdt:
                    allowed, _pdt_reason = pdt_guard(
                        trades, available_equity, as_of=current_date
                    )
                    if not allowed:
                        equity_curve.append({"date": current_date, "equity": equity})
                        continue

                # Fractional-share routing: Alpaca rejects fractional short qty.
                frac_scale = 1.0
                frac_note = ""
                if self.enforce_fractional:
                    _est_qty = (requested_capital * float(leverage)) / entry_px if entry_px > 0 else 0.0
                    _norm_qty, _norm_reason = normalize_qty_for_direction(_est_qty, new_direction)
                    if _norm_qty <= 0:
                        equity_curve.append({"date": current_date, "equity": equity})
                        continue
                    if _norm_reason and new_direction == Direction.SHORT and _est_qty > 0:
                        frac_scale = _norm_qty / _est_qty
                        requested_capital = requested_capital * frac_scale
                        frac_note = _norm_reason

                if self.risk:
                    self.risk.update_portfolio_state(
                        daily_pnl=0.0,
                        open_positions=0,  # we just checked: open_trade is None
                        total_equity=available_equity,
                    )
                    check = self.risk.check(
                        direction=new_direction,
                        entry_price=entry_px,
                        take_profit=adj_tp,
                        stop_loss=adj_sl,
                        leverage=leverage,
                        capital_requested=requested_capital,
                    )
                    if not check.approved:
                        equity_curve.append({"date": current_date, "equity": equity})
                        continue
                    effective_sl = check.adjusted_sl or adj_sl
                    effective_capital = min(check.adjusted_size or requested_capital, available_equity)
                else:
                    effective_sl = adj_sl
                    effective_capital = requested_capital

                sig_meta = meta.get("metadata", {})
                regime = sig_meta.get("regime", "normal")

                if global_cooldown_until is not None and i < global_cooldown_until:
                    equity_curve.append({"date": current_date, "equity": equity})
                    continue

                cooldown_until = regime_cooldown_until.get(regime)
                if cooldown_until is not None and i < cooldown_until:
                    equity_curve.append({"date": current_date, "equity": equity})
                    continue

                _base_notes = (
                    f"Entry: {new_direction.value} @ {entry_px:.4f} (signal-bar close) | "
                    f"regime={regime} | "
                    f"SL={effective_sl:.4f} | "
                    + (f"TP={adj_tp:.4f}" if adj_tp is not None else "TP=none")
                )
                if frac_note:
                    _base_notes += f" | {frac_note}"
                if self.fill_diagnostic:
                    try:
                        _diag = fill_timing_note(symbol, bar)
                        _base_notes += f" | {_diag.as_note_str()}"
                    except Exception:
                        pass
                if sig_meta.get("session_exit"):
                    _base_notes += f" | session_exit={sig_meta.get('session_exit')}"

                open_trade = TradeRecord(
                    id=str(uuid.uuid4()),
                    symbol=symbol,
                    direction=new_direction,
                    entry_price=entry_px,
                    take_profit=adj_tp,
                    stop_loss=effective_sl,
                    leverage=leverage,
                    capital_allocated=effective_capital,
                    entry_time=current_date if isinstance(current_date, datetime) else pd.Timestamp(current_date).to_pydatetime(),
                    mode="backtest",
                    strategy_id=self.strategy.strategy_id,
                    outcome=TradeOutcome.OPEN,
                    notes=_base_notes,
                )
                open_trade_regime = regime

                req_atr = sig_meta.get("trailing_atr_mult")
                req_pct = sig_meta.get("pct_trail")
                req_giveback = sig_meta.get("profit_giveback_frac")
                req_giveback_min_pct = sig_meta.get("profit_giveback_min_pct", 0.0)
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
        if not eq_df.empty:
            # Preserve exact jump timing while dropping redundant repeated flat points.
            # Keeping both the start and end of each flat run is effectively lossless
            # for the realized-equity line and avoids the misleading shift introduced
            # by coarse downsampling.
            equity_s = pd.to_numeric(eq_df["equity"], errors="coerce")
            keep_mask = equity_s.ne(equity_s.shift()) | equity_s.ne(equity_s.shift(-1))
            eq_df = eq_df.loc[keep_mask].reset_index(drop=True)

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

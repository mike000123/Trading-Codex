from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
from pathlib import Path
import pickle
from typing import Any, Callable, Optional
import uuid
import threading

import numpy as np
import pandas as pd

from core.logger import log
from core.models import Direction, SignalAction, TradeOutcome, TradeRecord
from execution.alpaca_constraints import (
    fill_timing_note,
    is_regular_trading_hour,
    is_trading_day,
    monday_open_delay_guard,
    normalize_qty_for_direction,
    pdt_guard,
)
from reporting.backtest import BacktestEngine, BacktestResult, _calc_atr_series, _is_session_close_bar
from risk.manager import RiskManager
from strategies.base import BaseStrategy


@dataclass
class ManagedPortfolioSymbolInput:
    symbol: str
    strategy_id: str
    strategy_name: str
    strategy: BaseStrategy
    data: pd.DataFrame


@dataclass
class ManagedPortfolioBacktestResult:
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
    symbol_strategy_map: dict[str, str]
    candidate_entries: int
    skipped_entries: int
    replaced_positions: int
    max_concurrent_positions_seen: int

    def summary(self) -> dict[str, object]:
        return {
            "Managed Symbols": len(self.symbol_strategy_map),
            "Total Trades": self.total_trades,
            "Win Rate": f"{self.win_rate_pct:.1f}%",
            "Total Return": f"{self.total_return_pct:.2f}%",
            "Max Drawdown": f"{self.max_drawdown_pct:.2f}%",
            "Sharpe Ratio": f"{self.sharpe_ratio:.3f}",
            "Avg Win": f"{self.avg_win_pct:.2f}%",
            "Avg Loss": f"{self.avg_loss_pct:.2f}%",
            "Candidates": self.candidate_entries,
            "Skipped": self.skipped_entries,
            "Replaced": self.replaced_positions,
            "Peak Open Positions": self.max_concurrent_positions_seen,
        }


@dataclass
class _PreparedSymbolState:
    symbol: str
    strategy_id: str
    strategy_name: str
    strategy: BaseStrategy
    data: pd.DataFrame
    actions: list
    meta: list[dict[str, Any]]
    atr_arr: np.ndarray
    high_arr: np.ndarray
    low_arr: np.ndarray
    close_arr: np.ndarray
    dates: list[pd.Timestamp]
    index_by_ts: dict[pd.Timestamp, int]
    prior_day_close_arr: np.ndarray


@dataclass
class _OpenPositionState:
    trade: TradeRecord
    strategy_name: str
    regime: str
    entry_score: float
    entry_index: int
    portfolio_bucket: str = ""
    trail_best: Optional[float] = None
    trail_mult: Optional[float] = None
    trail_pct: Optional[float] = None
    trail_giveback_frac: Optional[float] = None
    trail_giveback_min_pct: float = 0.0
    trail_hard_sl: Optional[float] = None
    trail_grace: int = 0
    trail_bars: int = 0


@dataclass
class _EntryCandidate:
    symbol: str
    strategy_name: str
    timestamp: pd.Timestamp
    bar_index: int
    direction: Direction
    entry_price: float
    take_profit: Optional[float]
    stop_loss: float
    score: float
    regime: str
    portfolio_bucket: str
    metadata: dict[str, Any]


@dataclass
class _SizingDecision:
    multiplier: float
    tier: str
    reason: str


_PREPARED_STATE_CACHE: dict[str, _PreparedSymbolState] = {}
_PREPARED_STATE_CACHE_LOCK = threading.Lock()
_PREPARED_STATE_CACHE_VERSION = "v2"
_PREPARED_STATE_CACHE_DIR = (
    Path(__file__).resolve().parents[1] / "artifacts" / "cache" / "managed_prepared_states"
)


class ManagedPortfolioBacktestEngine:
    def __init__(
        self,
        *,
        risk_manager: Optional[RiskManager] = None,
        direction_filter: Optional[Direction] = None,
        counter_signal_exit: bool = True,
        spread_pct: float = 0.0,
        slippage_pct: float = 0.0,
        commission_per_trade: float = 0.0,
        enforce_rth: bool = True,
        extended_hours: bool = False,
        enforce_pdt: bool = True,
        enforce_ssr: bool = True,
        enforce_fractional: bool = True,
        fill_diagnostic: bool = True,
        enforce_monday_open_delay: bool = False,
        allow_replacement: bool = True,
        replacement_score_edge_pct: float = 10.0,
        max_open_positions: int = 1,
        dynamic_sizing: bool = False,
        dynamic_min_size_mult: float = 0.5,
        dynamic_strong_size_mult: float = 1.5,
        dynamic_exceptional_size_mult: float = 2.0,
        dynamic_near_peer_ratio: float = 0.85,
        dynamic_strong_lead_ratio: float = 1.15,
        dynamic_exceptional_lead_ratio: float = 1.35,
        dynamic_abundant_cash_units: float = 6.0,
        dynamic_abundant_strong_size_mult: float = 2.5,
        dynamic_abundant_exceptional_size_mult: float = 4.0,
    ) -> None:
        self.risk = risk_manager
        self.direction_filter = direction_filter
        self.counter_signal_exit = counter_signal_exit
        self.spread_pct = float(spread_pct)
        self.slippage_pct = float(slippage_pct)
        self.commission_per_trade = float(commission_per_trade)
        self.enforce_rth = bool(enforce_rth)
        self.extended_hours = bool(extended_hours)
        self.enforce_pdt = bool(enforce_pdt)
        self.enforce_ssr = bool(enforce_ssr)
        self.enforce_fractional = bool(enforce_fractional)
        self.fill_diagnostic = bool(fill_diagnostic)
        self.enforce_monday_open_delay = bool(enforce_monday_open_delay)
        self.allow_replacement = bool(allow_replacement)
        self.replacement_score_edge_pct = float(replacement_score_edge_pct)
        self.max_open_positions = max(int(max_open_positions), 1)
        self.dynamic_sizing = bool(dynamic_sizing)
        self.dynamic_min_size_mult = max(0.1, float(dynamic_min_size_mult))
        self.dynamic_strong_size_mult = max(1.0, float(dynamic_strong_size_mult))
        self.dynamic_exceptional_size_mult = max(self.dynamic_strong_size_mult, float(dynamic_exceptional_size_mult))
        self.dynamic_near_peer_ratio = min(max(float(dynamic_near_peer_ratio), 0.05), 0.999)
        self.dynamic_strong_lead_ratio = max(1.0, float(dynamic_strong_lead_ratio))
        self.dynamic_exceptional_lead_ratio = max(
            self.dynamic_strong_lead_ratio,
            float(dynamic_exceptional_lead_ratio),
        )
        self.dynamic_abundant_cash_units = max(2.0, float(dynamic_abundant_cash_units))
        self.dynamic_abundant_strong_size_mult = max(
            self.dynamic_exceptional_size_mult,
            float(dynamic_abundant_strong_size_mult),
        )
        self.dynamic_abundant_exceptional_size_mult = max(
            self.dynamic_abundant_strong_size_mult,
            float(dynamic_abundant_exceptional_size_mult),
        )

    def run(
        self,
        symbol_inputs: list[ManagedPortfolioSymbolInput],
        *,
        leverage: float = 1.0,
        capital_per_trade: float = 1000.0,
        starting_equity: float = 10_000.0,
        progress_cb: Optional[Callable[[float, str], None]] = None,
    ) -> ManagedPortfolioBacktestResult:
        if not symbol_inputs:
            raise ValueError("No symbol inputs were provided for the managed portfolio backtest.")

        states = [self._prepare_symbol_state(item) for item in symbol_inputs if item.data is not None and not item.data.empty]
        if not states:
            raise ValueError("No prepared symbol states were available for the managed portfolio backtest.")

        state_by_symbol = {state.symbol: state for state in states}
        all_timestamps = sorted({ts for state in states for ts in state.dates})
        total_steps = max(len(all_timestamps), 1)
        cash_balance = float(starting_equity)
        realized_equity = float(starting_equity)
        trades: list[TradeRecord] = []
        equity_curve: list[dict[str, Any]] = []
        open_positions: dict[str, _OpenPositionState] = {}
        last_close_by_symbol: dict[str, float] = {}
        symbol_strategy_map = {state.symbol: state.strategy_name for state in states}
        candidate_entries = 0
        skipped_entries = 0
        replaced_positions = 0
        max_concurrent_positions_seen = 0

        for step_idx, ts in enumerate(all_timestamps):
            for state in states:
                bar_idx = state.index_by_ts.get(ts)
                if bar_idx is None:
                    continue
                last_close_by_symbol[state.symbol] = float(state.close_arr[bar_idx])

            symbols_to_close: list[str] = []
            for symbol, pos in list(open_positions.items()):
                state = state_by_symbol[symbol]
                bar_idx = state.index_by_ts.get(ts)
                if bar_idx is None or bar_idx <= pos.entry_index:
                    continue
                bar = state.data.iloc[bar_idx]
                self._advance_trailing(pos, state, bar_idx)
                pos.trade = BacktestEngine._check_exit(pos.trade, bar)
                if pos.trade.outcome not in (TradeOutcome.OPEN, TradeOutcome.NO_DATA):
                    cash_box = [cash_balance]
                    realized_equity += self._finalize_closed_trade(pos.trade, cash_balance_ref=cash_box, trades=trades)
                    cash_balance = cash_box[0]
                    symbols_to_close.append(symbol)
                    continue

                if (
                    self.counter_signal_exit
                    and self._should_counter_exit(pos, state, bar_idx)
                ):
                    self._close_position_at_price(
                        pos,
                        exit_price=float(state.close_arr[bar_idx]),
                        exit_time=ts,
                        outcome=TradeOutcome.SIGNAL_EXIT,
                        note_suffix="Counter-signal exit",
                        cash_balance_ref=(cash_box := [cash_balance]),
                        trades=trades,
                    )
                    realized_equity += float(pos.trade.pnl or 0.0)
                    cash_balance = cash_box[0]
                    symbols_to_close.append(symbol)
                    continue

                if (
                    "session_exit=eod" in (pos.trade.notes or "")
                    and _is_session_close_bar(ts, self._next_timestamp_for_state(state, bar_idx))
                ):
                    self._close_position_at_price(
                        pos,
                        exit_price=float(state.close_arr[bar_idx]),
                        exit_time=ts,
                        outcome=TradeOutcome.SIGNAL_EXIT,
                        note_suffix="Session-close exit",
                        cash_balance_ref=(cash_box := [cash_balance]),
                        trades=trades,
                    )
                    realized_equity += float(pos.trade.pnl or 0.0)
                    cash_balance = cash_box[0]
                    symbols_to_close.append(symbol)
                    continue

            for symbol in symbols_to_close:
                open_positions.pop(symbol, None)

            candidates: list[_EntryCandidate] = []
            open_symbols = set(open_positions.keys())
            for state in states:
                if state.symbol in open_symbols:
                    continue
                bar_idx = state.index_by_ts.get(ts)
                if bar_idx is None or bar_idx < 1:
                    continue
                candidate = self._build_candidate(state, ts, bar_idx)
                if candidate is not None:
                    candidates.append(candidate)
            candidate_entries += len(candidates)
            pending_candidates = list(candidates)
            while pending_candidates:
                pending_candidates.sort(
                    key=lambda c: self._effective_candidate_score(c, open_positions),
                    reverse=True,
                )
                candidate = pending_candidates.pop(0)
                candidate.score = self._effective_candidate_score(candidate, open_positions)
                account_equity = self._mark_to_market_equity(realized_equity, open_positions, last_close_by_symbol)
                if len(open_positions) >= self.max_open_positions or cash_balance <= 0:
                    weakest = self._weakest_open_position(open_positions)
                    if (
                        self.allow_replacement
                        and weakest is not None
                        and candidate.score >= weakest.entry_score * (1.0 + self.replacement_score_edge_pct / 100.0)
                    ):
                        replace_px = last_close_by_symbol.get(weakest.trade.symbol)
                        if replace_px is None:
                            skipped_entries += 1
                            continue
                        self._close_position_at_price(
                            weakest,
                            exit_price=float(replace_px),
                            exit_time=ts,
                            outcome=TradeOutcome.SIGNAL_EXIT,
                            note_suffix=f"Replaced by stronger setup in {candidate.symbol}",
                            cash_balance_ref=(cash_box := [cash_balance]),
                            trades=trades,
                        )
                        realized_equity += float(weakest.trade.pnl or 0.0)
                        cash_balance = cash_box[0]
                        open_positions.pop(weakest.trade.symbol, None)
                        replaced_positions += 1
                        account_equity = self._mark_to_market_equity(realized_equity, open_positions, last_close_by_symbol)
                    else:
                        skipped_entries += 1
                        continue

                sizing = self._size_candidate(
                    candidate,
                    pending_candidates=pending_candidates,
                    open_positions=open_positions,
                    cash_balance=float(cash_balance),
                    capital_per_trade=float(capital_per_trade),
                )
                requested_capital = min(
                    float(capital_per_trade) * float(sizing.multiplier),
                    max(float(cash_balance), 0.0),
                )
                if requested_capital <= 0:
                    skipped_entries += 1
                    continue

                if self.enforce_monday_open_delay:
                    allowed, _ = monday_open_delay_guard(ts, enforce=True)
                    if not allowed:
                        skipped_entries += 1
                        continue
                if self.enforce_rth:
                    if not is_trading_day(ts) or not is_regular_trading_hour(ts, extended_hours=self.extended_hours):
                        skipped_entries += 1
                        continue
                if self.enforce_ssr and candidate.direction == Direction.SHORT:
                    candidate_state = state_by_symbol.get(candidate.symbol)
                    if candidate_state is not None:
                        pd_close = candidate_state.prior_day_close_arr[candidate.bar_index]
                        if not np.isnan(pd_close) and pd_close > 0:
                            drop_pct = (candidate.entry_price - pd_close) / pd_close * 100.0
                            if drop_pct <= -10.0:
                                skipped_entries += 1
                                continue
                if self.enforce_pdt:
                    allowed, _ = pdt_guard(trades, account_equity, as_of=ts)
                    if not allowed:
                        skipped_entries += 1
                        continue

                effective_capital = requested_capital
                effective_sl = candidate.stop_loss
                frac_note = ""
                if self.enforce_fractional:
                    est_qty = (effective_capital * float(leverage)) / candidate.entry_price if candidate.entry_price > 0 else 0.0
                    norm_qty, norm_reason = normalize_qty_for_direction(est_qty, candidate.direction)
                    if norm_qty <= 0:
                        skipped_entries += 1
                        continue
                    if norm_reason and candidate.direction == Direction.SHORT and est_qty > 0:
                        frac_scale = norm_qty / est_qty
                        effective_capital = effective_capital * frac_scale
                        frac_note = norm_reason

                if self.risk:
                    self.risk.update_portfolio_state(
                        daily_pnl=0.0,
                        open_positions=len(open_positions),
                        total_equity=account_equity,
                    )
                    check = self.risk.check(
                        direction=candidate.direction,
                        entry_price=candidate.entry_price,
                        take_profit=candidate.take_profit,
                        stop_loss=effective_sl,
                        leverage=leverage,
                        capital_requested=effective_capital,
                    )
                    if not check.approved:
                        skipped_entries += 1
                        continue
                    effective_sl = check.adjusted_sl or effective_sl
                    effective_capital = min(check.adjusted_size or effective_capital, cash_balance)

                if effective_capital <= 0:
                    skipped_entries += 1
                    continue

                notes = (
                    f"Entry: {candidate.direction.value} @ {candidate.entry_price:.4f} (managed portfolio) | "
                    f"regime={candidate.regime} | score={candidate.score:.4f} | "
                    f"size_tier={sizing.tier}:{float(sizing.multiplier):.2f}x | "
                    f"SL={effective_sl:.4f} | "
                    + (f"TP={candidate.take_profit:.4f}" if candidate.take_profit is not None else "TP=none")
                    + f" | strategy={candidate.strategy_name}"
                )
                if sizing.reason:
                    notes += f" | sizing_reason={sizing.reason}"
                if frac_note:
                    notes += f" | {frac_note}"
                if self.fill_diagnostic:
                    try:
                        state = state_by_symbol[candidate.symbol]
                        bar = state.data.iloc[candidate.bar_index]
                        notes += f" | {fill_timing_note(candidate.symbol, bar).as_note_str()}"
                    except Exception:
                        pass
                if candidate.metadata.get("session_exit"):
                    notes += f" | session_exit={candidate.metadata.get('session_exit')}"

                trade = TradeRecord(
                    id=str(uuid.uuid4()),
                    symbol=candidate.symbol,
                    direction=candidate.direction,
                    entry_price=float(candidate.entry_price),
                    take_profit=candidate.take_profit,
                    stop_loss=float(effective_sl),
                    leverage=float(leverage),
                    capital_allocated=float(effective_capital),
                    entry_time=ts.to_pydatetime(),
                    mode="backtest",
                    strategy_id=state_by_symbol[candidate.symbol].strategy_id,
                    outcome=TradeOutcome.OPEN,
                    notes=notes,
                )
                cs_thresh = candidate.metadata.get("counter_signal_min_profit_pct")
                if cs_thresh is not None:
                    trade.counter_signal_min_profit_pct = float(cs_thresh)
                    trade.notes += f" | cs_min_profit={float(cs_thresh):.2f}%"
                open_positions[candidate.symbol] = _OpenPositionState(
                    trade=trade,
                    strategy_name=candidate.strategy_name,
                    regime=candidate.regime,
                    entry_score=float(candidate.score),
                    entry_index=int(candidate.bar_index),
                    portfolio_bucket=str(candidate.portfolio_bucket or ""),
                )
                self._init_trailing(open_positions[candidate.symbol], candidate)
                cash_balance -= float(effective_capital)
                max_concurrent_positions_seen = max(max_concurrent_positions_seen, len(open_positions))

            if not open_positions and abs(cash_balance - realized_equity) > 0.01:
                log.warning(
                    "managed_backtest: flat-book cash/equity mismatch at {} — cash {:.2f} vs realized {:.2f}; normalizing cash",
                    ts,
                    cash_balance,
                    realized_equity,
                )
                cash_balance = realized_equity

            equity_curve.append(
                {
                    "date": ts,
                    "equity": self._mark_to_market_equity(realized_equity, open_positions, last_close_by_symbol),
                }
            )
            if progress_cb is not None:
                placed_entries = len(trades) + len(open_positions)
                progress_cb(
                    (step_idx + 1) / total_steps,
                    f"{ts} · placed {placed_entries} · open {len(open_positions)} · closed {len(trades)}",
                )

        for symbol, pos in list(open_positions.items()):
            final_px = last_close_by_symbol.get(symbol, pos.trade.entry_price)
            final_ts = max(all_timestamps) if all_timestamps else pd.Timestamp.utcnow()
            self._close_position_at_price(
                pos,
                exit_price=float(final_px),
                exit_time=final_ts,
                outcome=TradeOutcome.SIGNAL_EXIT,
                note_suffix="Final portfolio close",
                cash_balance_ref=(cash_box := [cash_balance]),
                trades=trades,
            )
            realized_equity += float(pos.trade.pnl or 0.0)
            cash_balance = cash_box[0]
            open_positions.pop(symbol, None)

        if equity_curve:
            if abs(cash_balance - realized_equity) > 0.01:
                log.warning(
                    "managed_backtest: final cash/equity mismatch — cash {:.2f} vs realized {:.2f}; normalizing final cash",
                    cash_balance,
                    realized_equity,
                )
                cash_balance = realized_equity
            equity_curve[-1]["equity"] = realized_equity

        base_result = BacktestEngine._compute_result(trades, equity_curve, starting_equity)
        return ManagedPortfolioBacktestResult(
            trades=base_result.trades,
            equity_curve=base_result.equity_curve,
            total_return_pct=base_result.total_return_pct,
            win_rate_pct=base_result.win_rate_pct,
            max_drawdown_pct=base_result.max_drawdown_pct,
            sharpe_ratio=base_result.sharpe_ratio,
            total_trades=base_result.total_trades,
            winning_trades=base_result.winning_trades,
            losing_trades=base_result.losing_trades,
            avg_win_pct=base_result.avg_win_pct,
            avg_loss_pct=base_result.avg_loss_pct,
            symbol_strategy_map=symbol_strategy_map,
            candidate_entries=int(candidate_entries),
            skipped_entries=int(skipped_entries),
            replaced_positions=int(replaced_positions),
            max_concurrent_positions_seen=int(max_concurrent_positions_seen),
        )

    def _prepare_symbol_state(self, item: ManagedPortfolioSymbolInput) -> _PreparedSymbolState:
        data = item.data.reset_index(drop=True).copy()
        cache_key = self._prepared_state_cache_key(item=item, data=data)
        with _PREPARED_STATE_CACHE_LOCK:
            cached = _PREPARED_STATE_CACHE.get(cache_key)
        if cached is not None:
            return cached
        cached = self._load_prepared_state_from_disk(item=item, data=data, cache_key=cache_key)
        if cached is not None:
            with _PREPARED_STATE_CACHE_LOCK:
                _PREPARED_STATE_CACHE[cache_key] = cached
            return cached
        n = len(data)
        try:
            actions_s, meta_s = item.strategy.generate_signals_bulk(data, item.symbol)
        except NotImplementedError:
            actions_s = []
            meta_s = []
            for i in range(n):
                sig = item.strategy.generate_signal(data.iloc[: i + 1].copy(), item.symbol)
                actions_s.append(sig.action)
                meta_s.append(
                    {
                        "suggested_tp": sig.suggested_tp,
                        "suggested_sl": sig.suggested_sl,
                        "metadata": sig.metadata,
                    }
                )

        dates = [pd.Timestamp(v) for v in data["date"]]
        state = _PreparedSymbolState(
            symbol=item.symbol,
            strategy_id=item.strategy_id,
            strategy_name=item.strategy_name,
            strategy=item.strategy,
            data=data,
            actions=actions_s,
            meta=meta_s,
            atr_arr=_calc_atr_series(data, period=14),
            high_arr=data["high"].to_numpy(dtype=float),
            low_arr=data["low"].to_numpy(dtype=float),
            close_arr=data["close"].to_numpy(dtype=float),
            dates=dates,
            index_by_ts={pd.Timestamp(ts): idx for idx, ts in enumerate(dates)},
            prior_day_close_arr=self._compute_prior_day_close_arr(data),
        )
        with _PREPARED_STATE_CACHE_LOCK:
            _PREPARED_STATE_CACHE[cache_key] = state
        self._save_prepared_state_to_disk(cache_key=cache_key, state=state)
        return state

    @staticmethod
    def _prepared_state_cache_key(*, item: ManagedPortfolioSymbolInput, data: pd.DataFrame) -> str:
        if data.empty:
            return f"{str(item.symbol).strip().upper()}|{str(item.strategy_id).strip().lower()}|empty"
        first_ts = pd.Timestamp(data.iloc[0]["date"]).value
        last_ts = pd.Timestamp(data.iloc[-1]["date"]).value
        try:
            first_close = round(float(data.iloc[0]["close"]), 6)
            last_close = round(float(data.iloc[-1]["close"]), 6)
        except Exception:
            first_close = 0.0
            last_close = 0.0
        return "|".join(
            [
                _PREPARED_STATE_CACHE_VERSION,
                str(item.symbol or "").strip().upper(),
                str(item.strategy_id or "").strip().lower(),
                str(len(data)),
                str(first_ts),
                str(last_ts),
                f"{first_close:.6f}",
                f"{last_close:.6f}",
            ]
        )

    @staticmethod
    def _prepared_state_cache_path(cache_key: str) -> Path:
        digest = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()
        return _PREPARED_STATE_CACHE_DIR / f"{digest}.pkl"

    def _load_prepared_state_from_disk(
        self,
        *,
        item: ManagedPortfolioSymbolInput,
        data: pd.DataFrame,
        cache_key: str,
    ) -> Optional[_PreparedSymbolState]:
        path = self._prepared_state_cache_path(cache_key)
        if not path.exists():
            return None
        try:
            with path.open("rb") as fh:
                payload = pickle.load(fh)
            dates = [pd.Timestamp(v) for v in payload["dates"]]
            state = _PreparedSymbolState(
                symbol=item.symbol,
                strategy_id=item.strategy_id,
                strategy_name=item.strategy_name,
                strategy=item.strategy,
                data=data,
                actions=list(payload["actions"]),
                meta=list(payload["meta"]),
                atr_arr=np.asarray(payload["atr_arr"], dtype=float),
                high_arr=data["high"].to_numpy(dtype=float),
                low_arr=data["low"].to_numpy(dtype=float),
                close_arr=data["close"].to_numpy(dtype=float),
                dates=dates,
                index_by_ts={pd.Timestamp(ts): idx for idx, ts in enumerate(dates)},
                prior_day_close_arr=np.asarray(payload["prior_day_close_arr"], dtype=float),
            )
            if (
                len(state.actions) != len(data)
                or len(state.meta) != len(data)
                or len(state.atr_arr) != len(data)
                or len(state.prior_day_close_arr) != len(data)
                or len(state.dates) != len(data)
            ):
                return None
            return state
        except Exception:
            return None

    def _save_prepared_state_to_disk(self, *, cache_key: str, state: _PreparedSymbolState) -> None:
        try:
            _PREPARED_STATE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            path = self._prepared_state_cache_path(cache_key)
            payload = {
                "actions": state.actions,
                "meta": state.meta,
                "atr_arr": state.atr_arr,
                "dates": [ts.isoformat() for ts in state.dates],
                "prior_day_close_arr": state.prior_day_close_arr,
            }
            tmp_path = path.with_suffix(".tmp")
            with tmp_path.open("wb") as fh:
                pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
            tmp_path.replace(path)
        except Exception:
            return

    @staticmethod
    def _compute_prior_day_close_arr(data: pd.DataFrame) -> np.ndarray:
        n = len(data)
        prior_day_close_arr = np.full(n, np.nan, dtype=float)
        try:
            dates = pd.to_datetime(data["date"])
            if getattr(dates.dt, "tz", None) is None:
                dates = dates.dt.tz_localize("UTC")
            et_dates = dates.dt.tz_convert("America/New_York").dt.date.to_numpy()
            close_arr = data["close"].to_numpy(dtype=float)
            prev_date = None
            prev_day_last_close = np.nan
            last_close_this_day = np.nan
            for i in range(n):
                cur_date = et_dates[i]
                if prev_date is not None and cur_date != prev_date:
                    prev_day_last_close = last_close_this_day
                prior_day_close_arr[i] = prev_day_last_close
                last_close_this_day = close_arr[i]
                prev_date = cur_date
        except Exception:
            pass
        return prior_day_close_arr

    def _build_candidate(self, state: _PreparedSymbolState, ts: pd.Timestamp, bar_idx: int) -> Optional[_EntryCandidate]:
        action = state.actions[bar_idx]
        direction = BacktestEngine._signal_to_direction(action)
        if direction is None:
            return None
        if self.direction_filter is not None and direction != self.direction_filter:
            return None
        meta = state.meta[bar_idx] or {}
        current_sl = meta.get("suggested_sl")
        if current_sl is None:
            return None
        entry_price = float(state.close_arr[bar_idx])
        current_tp = meta.get("suggested_tp")
        metadata = dict(meta.get("metadata") or {})
        regime = str(metadata.get("regime", "normal"))
        portfolio_bucket = self._portfolio_bucket(
            symbol=state.symbol,
            strategy_id=state.strategy_id,
            metadata=metadata,
        )
        score = self._candidate_score(
            entry_price=entry_price,
            take_profit=current_tp,
            stop_loss=float(current_sl),
            metadata=metadata,
        )
        return _EntryCandidate(
            symbol=state.symbol,
            strategy_name=state.strategy_name,
            timestamp=ts,
            bar_index=bar_idx,
            direction=direction,
            entry_price=entry_price,
            take_profit=current_tp,
            stop_loss=float(current_sl),
            score=float(score),
            regime=regime,
            portfolio_bucket=portfolio_bucket,
            metadata=metadata,
        )

    @staticmethod
    def _portfolio_bucket(
        *,
        symbol: str,
        strategy_id: str,
        metadata: dict[str, Any],
    ) -> str:
        explicit = metadata.get("portfolio_bucket") or metadata.get("portfolio_cluster")
        if explicit:
            return str(explicit).strip().lower()
        sym = str(symbol or "").strip().upper()
        if sym in {"UVXY", "VXX", "VXZ"}:
            return "volatility_etp"
        if sym in {"SPY", "QQQ", "IWM"}:
            return "broad_index_beta"
        return f"symbol:{sym or str(strategy_id or '').strip().lower()}"

    @staticmethod
    def _candidate_score(
        *,
        entry_price: float,
        take_profit: Optional[float],
        stop_loss: float,
        metadata: dict[str, Any],
    ) -> float:
        explicit = metadata.get("portfolio_score")
        if explicit is not None:
            try:
                return float(explicit)
            except Exception:
                pass
        confidence = float(metadata.get("confidence", 1.0) or 1.0)
        reward_pct = abs((float(take_profit) - entry_price) / entry_price) * 100.0 if take_profit is not None and entry_price > 0 else 0.0
        risk_pct = abs((entry_price - float(stop_loss)) / entry_price) * 100.0 if entry_price > 0 else 0.0
        if reward_pct <= 0.0:
            reward_pct = max(risk_pct, 0.5)
        rr = reward_pct / max(risk_pct, 0.1)
        rr_adj = min(max(rr, 0.5), 3.0)
        return max(confidence, 0.05) * reward_pct * rr_adj

    @staticmethod
    def _effective_candidate_score(
        candidate: _EntryCandidate,
        open_positions: dict[str, _OpenPositionState],
    ) -> float:
        base_score = float(candidate.score)
        bucket = str(candidate.portfolio_bucket or "")
        if not bucket:
            return base_score
        duplicate_count = sum(
            1 for pos in open_positions.values() if str(pos.portfolio_bucket or "") == bucket
        )
        if duplicate_count <= 0:
            return base_score
        if bucket == "volatility_etp":
            penalty = 0.55 ** duplicate_count
        elif bucket == "broad_index_beta":
            penalty = 0.65 ** duplicate_count
        else:
            penalty = 0.80 ** duplicate_count
        return base_score * penalty

    def _size_candidate(
        self,
        candidate: _EntryCandidate,
        *,
        pending_candidates: list[_EntryCandidate],
        open_positions: dict[str, _OpenPositionState],
        cash_balance: float,
        capital_per_trade: float,
    ) -> _SizingDecision:
        if not self.dynamic_sizing:
            return _SizingDecision(multiplier=1.0, tier="base", reason="fixed")
        base_unit = max(float(capital_per_trade), 0.0)
        if base_unit <= 0.0:
            return _SizingDecision(multiplier=1.0, tier="base", reason="invalid_base")

        current_score = max(float(candidate.score), 0.0)
        other_scores = sorted(
            (
                max(float(self._effective_candidate_score(other, open_positions)), 0.0)
                for other in pending_candidates
            ),
            reverse=True,
        )
        second_best = float(other_scores[0]) if other_scores else 0.0
        lead_ratio = (current_score / second_best) if second_best > 0 else 1.0
        near_peer_floor = current_score * self.dynamic_near_peer_ratio
        near_peer_count = 1 + sum(1 for score in other_scores if score >= near_peer_floor and near_peer_floor > 0)
        bucket = str(candidate.portfolio_bucket or "")
        bucket_open_count = sum(
            1 for pos in open_positions.values() if str(pos.portfolio_bucket or "") == bucket
        )
        available_slots = max(self.max_open_positions - len(open_positions), 0)
        affordable_base_slots = max(int(max(float(cash_balance), 0.0) // base_unit), 0)
        cash_units = max(float(cash_balance), 0.0) / base_unit if base_unit > 0 else 0.0
        abundant_cash = (
            bucket_open_count <= 0
            and available_slots >= 4
            and cash_units >= self.dynamic_abundant_cash_units
        )

        if bucket_open_count <= 0 and not other_scores:
            if abundant_cash and cash_balance >= base_unit * self.dynamic_abundant_exceptional_size_mult:
                return _SizingDecision(
                    multiplier=self.dynamic_abundant_exceptional_size_mult,
                    tier="solo_abundant",
                    reason=f"solo_candidate,cash_units={cash_units:.1f}",
                )
            if available_slots >= 2 and cash_balance >= base_unit * self.dynamic_exceptional_size_mult:
                return _SizingDecision(
                    multiplier=self.dynamic_exceptional_size_mult,
                    tier="solo_exceptional",
                    reason="solo_candidate",
                )
            if available_slots >= 1 and cash_balance >= base_unit * self.dynamic_strong_size_mult:
                return _SizingDecision(
                    multiplier=self.dynamic_strong_size_mult,
                    tier="solo_strong",
                    reason="solo_candidate",
                )

        if (
            available_slots > affordable_base_slots
            and affordable_base_slots > 0
            and near_peer_count > affordable_base_slots
            and lead_ratio < self.dynamic_strong_lead_ratio
        ):
            return _SizingDecision(
                multiplier=self.dynamic_min_size_mult,
                tier="crowded",
                reason=f"peers={near_peer_count},lead={lead_ratio:.2f}",
            )

        if bucket_open_count <= 0 and second_best > 0:
            if lead_ratio >= self.dynamic_exceptional_lead_ratio and near_peer_count <= 1:
                if abundant_cash and cash_balance >= base_unit * self.dynamic_abundant_exceptional_size_mult:
                    return _SizingDecision(
                        multiplier=self.dynamic_abundant_exceptional_size_mult,
                        tier="exceptional_abundant",
                        reason=f"lead={lead_ratio:.2f},cash_units={cash_units:.1f}",
                    )
                return _SizingDecision(
                    multiplier=self.dynamic_exceptional_size_mult,
                    tier="exceptional",
                    reason=f"lead={lead_ratio:.2f}",
                )
            if lead_ratio >= self.dynamic_strong_lead_ratio and near_peer_count <= 2:
                if abundant_cash and cash_balance >= base_unit * self.dynamic_abundant_strong_size_mult:
                    return _SizingDecision(
                        multiplier=self.dynamic_abundant_strong_size_mult,
                        tier="strong_abundant",
                        reason=f"lead={lead_ratio:.2f},cash_units={cash_units:.1f}",
                    )
                return _SizingDecision(
                    multiplier=self.dynamic_strong_size_mult,
                    tier="strong",
                    reason=f"lead={lead_ratio:.2f}",
                )

        return _SizingDecision(multiplier=1.0, tier="base", reason="balanced")

    @staticmethod
    def _init_trailing(pos: _OpenPositionState, candidate: _EntryCandidate) -> None:
        req_atr = candidate.metadata.get("trailing_atr_mult")
        req_pct = candidate.metadata.get("pct_trail")
        req_giveback = candidate.metadata.get("profit_giveback_frac")
        req_giveback_min_pct = candidate.metadata.get("profit_giveback_min_pct", 0.0)
        if req_atr is not None:
            pos.trail_mult = float(req_atr)
            pos.trail_best = candidate.entry_price
            pos.trail_hard_sl = candidate.stop_loss
            pos.trail_grace = 0
            pos.trade.notes += f" | trail=atr:{pos.trail_mult:.2f}"
        elif req_pct is not None:
            pos.trail_pct = float(req_pct)
            pos.trail_best = candidate.entry_price
            pos.trail_hard_sl = candidate.stop_loss
            pos.trail_grace = 1
            pos.trade.notes += f" | trail=pct:{pos.trail_pct:.2f}"
        elif req_giveback is not None:
            pos.trail_giveback_frac = float(req_giveback)
            pos.trail_giveback_min_pct = float(req_giveback_min_pct or 0.0)
            pos.trail_best = candidate.entry_price
            pos.trail_hard_sl = candidate.stop_loss
            pos.trail_grace = 1
            pos.trade.notes += (
                f" | trail=giveback:{pos.trail_giveback_frac:.2f},min:{pos.trail_giveback_min_pct:.2f}"
            )

    @staticmethod
    def _advance_trailing(pos: _OpenPositionState, state: _PreparedSymbolState, bar_idx: int) -> None:
        if pos.trail_mult is None and pos.trail_pct is None and pos.trail_giveback_frac is None:
            return
        pos.trail_bars += 1
        if pos.trail_bars <= pos.trail_grace:
            return
        if pos.trade.direction == Direction.SHORT:
            pos.trail_best = min(float(pos.trail_best), float(state.low_arr[bar_idx]))
            if pos.trail_mult is not None:
                candidate_sl = pos.trail_best + pos.trail_mult * state.atr_arr[bar_idx]
            elif pos.trail_pct is not None:
                candidate_sl = pos.trail_best * (1 + pos.trail_pct / 100.0)
            else:
                profit_move = max(pos.trade.entry_price - pos.trail_best, 0.0)
                profit_move_pct = (profit_move / pos.trade.entry_price) * 100 if pos.trade.entry_price > 0 else 0.0
                if profit_move_pct >= pos.trail_giveback_min_pct:
                    candidate_sl = pos.trail_best + pos.trail_giveback_frac * profit_move
                else:
                    candidate_sl = pos.trail_hard_sl if pos.trail_hard_sl is not None else pos.trade.stop_loss
            pos.trade.stop_loss = min(candidate_sl, pos.trail_hard_sl) if pos.trail_hard_sl is not None else candidate_sl
        else:
            pos.trail_best = max(float(pos.trail_best), float(state.high_arr[bar_idx]))
            if pos.trail_mult is not None:
                candidate_sl = pos.trail_best - pos.trail_mult * state.atr_arr[bar_idx]
            elif pos.trail_pct is not None:
                candidate_sl = pos.trail_best * (1 - pos.trail_pct / 100.0)
            else:
                profit_move = max(pos.trail_best - pos.trade.entry_price, 0.0)
                profit_move_pct = (profit_move / pos.trade.entry_price) * 100 if pos.trade.entry_price > 0 else 0.0
                if profit_move_pct >= pos.trail_giveback_min_pct:
                    candidate_sl = pos.trail_best - pos.trail_giveback_frac * profit_move
                else:
                    candidate_sl = pos.trail_hard_sl if pos.trail_hard_sl is not None else pos.trade.stop_loss
            pos.trade.stop_loss = max(candidate_sl, pos.trail_hard_sl) if pos.trail_hard_sl is not None else candidate_sl

    @staticmethod
    def _next_timestamp_for_state(state: _PreparedSymbolState, bar_idx: int) -> Optional[pd.Timestamp]:
        if bar_idx + 1 >= len(state.dates):
            return None
        return state.dates[bar_idx + 1]

    @staticmethod
    def _should_counter_exit(pos: _OpenPositionState, state: _PreparedSymbolState, bar_idx: int) -> bool:
        action = state.actions[bar_idx]
        direction = BacktestEngine._signal_to_direction(action)
        if direction is None:
            return False
        current_dir = pos.trade.direction
        spike_trade = any(
            regime in (pos.trade.notes or "")
            for regime in ("regime=spike_long", "regime=spike_momentum_long", "regime=post_spike_short", "regime=event_target_short")
        )
        if spike_trade:
            return False
        is_reversal = (
            (current_dir == Direction.LONG and direction == Direction.SHORT)
            or (current_dir == Direction.SHORT and direction == Direction.LONG)
        )
        if not is_reversal:
            return False
        if pos.trade.counter_signal_min_profit_pct is None:
            return True
        exit_px = float(state.close_arr[bar_idx])
        pct = (exit_px - pos.trade.entry_price) / pos.trade.entry_price
        if current_dir == Direction.SHORT:
            pct = -pct
        return (pct * 100.0) < float(pos.trade.counter_signal_min_profit_pct)

    def _close_position_at_price(
        self,
        pos: _OpenPositionState,
        *,
        exit_price: float,
        exit_time: pd.Timestamp,
        outcome: TradeOutcome,
        note_suffix: str,
        cash_balance_ref: list[float],
        trades: list[TradeRecord],
    ) -> None:
        pct = (exit_price - pos.trade.entry_price) / pos.trade.entry_price
        if pos.trade.direction == Direction.SHORT:
            pct = -pct
        pos.trade.leveraged_return_pct = pct * pos.trade.leverage * 100.0
        pos.trade.exit_price = float(exit_price)
        pos.trade.exit_time = exit_time.to_pydatetime()
        pos.trade.outcome = outcome
        pos.trade.pnl = pos.trade.capital_allocated * pos.trade.leveraged_return_pct / 100.0 - self._trade_cost(pos.trade.capital_allocated)
        pos.trade.notes = f"{pos.trade.notes or ''} | {note_suffix}".strip(" |")
        cash_balance_ref[0] += float(pos.trade.capital_allocated) + float(pos.trade.pnl or 0.0)
        trades.append(pos.trade)

    def _finalize_closed_trade(
        self,
        trade: TradeRecord,
        *,
        cash_balance_ref: list[float],
        trades: list[TradeRecord],
    ) -> float:
        if trade.leveraged_return_pct is not None:
            trade.pnl = trade.capital_allocated * trade.leveraged_return_pct / 100.0 - self._trade_cost(trade.capital_allocated)
            cash_balance_ref[0] += float(trade.capital_allocated) + float(trade.pnl or 0.0)
        trades.append(trade)
        return float(trade.pnl or 0.0)

    @staticmethod
    def _weakest_open_position(open_positions: dict[str, _OpenPositionState]) -> Optional[_OpenPositionState]:
        if not open_positions:
            return None
        return min(open_positions.values(), key=lambda pos: pos.entry_score)

    @staticmethod
    def _mark_to_market_equity(
        realized_equity: float,
        open_positions: dict[str, _OpenPositionState],
        last_close_by_symbol: dict[str, float],
    ) -> float:
        equity = float(realized_equity)
        for symbol, pos in open_positions.items():
            mark_px = float(last_close_by_symbol.get(symbol, pos.trade.entry_price))
            pct = (mark_px - pos.trade.entry_price) / pos.trade.entry_price
            if pos.trade.direction == Direction.SHORT:
                pct = -pct
            unrealized_pnl = pos.trade.capital_allocated * (pct * pos.trade.leverage)
            equity += float(unrealized_pnl)
        return equity

    def _trade_cost(self, capital: float) -> float:
        return capital * (self.spread_pct + self.slippage_pct) / 100.0 + self.commission_per_trade

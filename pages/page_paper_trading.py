"""
pages/page_paper_trading.py
────────────────────────────
Paper (simulated) live trading — multi-ticker, strategy-driven, fully automatic.

Behaviour matches Forward Test, but orders are routed through the real
OrderRouter and persisted in the DB with mode="paper":

  • On each tick (manual Refresh or auto-refresh), for every active symbol:
      1. Fetch fresh OHLCV (forward-blended).
      2. For any open paper trade on this symbol:
           - Apply trailing-stop logic (pct / atr / giveback) from the strategy.
           - Persist the updated SL back to the DB.
           - If TP / SL are hit on the latest bar, auto-close (DB + P&L).
      3. If no open position remains, run the strategy and, when it fires a
         BUY/SELL signal, automatically place a paper order through
         OrderRouter (which runs risk checks, then calls _execute_paper → DB).
  • There is no "Submit" button. HOLD simply means "no entry this bar" — the
    strategy is re-evaluated on every tick.
  • Strategy defaults (Bollinger+RSI spike-aware rules, trailing stops, etc.)
    are auto-loaded via render_strategy_params → effective_default_params,
    so picking a strategy from the dropdown activates all of its behaviour.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

import altair as alt
import pandas as pd
import streamlit as st

from config.settings import settings, TradingMode
from core import kill_switch as ks
from core.logger import log
from core.models import Direction, SignalAction, TradeOutcome
from data.ingestion import load_forward_blended_data, prepare_strategy_data
from db.database import Database
from execution.router import OrderRouter, ROUTE_ALPACA_PAPER, ROUTE_SIM
from execution import trading_stream as ts
from execution.entry_policy_base import (
    EntryContext,
    available_policies,
    get_policy,
)
# Importing these registers the two policies with the factory.
import execution.entry_policy_classic  # noqa: F401
import execution.entry_policy_alpaca   # noqa: F401
from risk.manager import RiskManager
from strategies import list_strategies, get_strategy
from ui.autorefresh import render_autorefresh_timer
from ui.components import render_mode_banner, render_strategy_params
from ui.charts import rsi_chart


# ── Styling (matches forward test) ───────────────────────────────────────────
_GREEN = "#26a69a"
_RED   = "#ef5350"
_BLUE  = "#4a9eff"
_GOLD  = "#ffd54f"
_GREY  = "#9e9eb8"
_AXIS  = dict(gridColor="#2a2d3e", labelColor="#d0d4f0", titleColor="#d0d4f0",
              labelFontSize=12, titleFontSize=13)
_TITLE = dict(color="#e8eaf6", fontSize=14, fontWeight="bold")


# ── Session-state keys ───────────────────────────────────────────────────────
_RUNS    = "pt_active_runs"     # dict[symbol → run_config_dict]
_CACHE   = "pt_prices_cache"    # dict[symbol → pd.DataFrame]
_SIGNALS = "pt_all_signals"     # list[signal_row_dict]
_TRAIL   = "pt_trail_state"     # dict[trade_id → trailing metadata dict]
_SYNC    = "pt_last_alpaca_sync"    # dict — most recent _sync_alpaca_paper_orders() result
_ACCT    = "pt_alpaca_account"      # dict — Alpaca account snapshot
_POS     = "pt_alpaca_positions"    # dict — {positions: [...], fetched_at}
_RECON   = "pt_last_reconcile"      # dict — last position-reconciliation summary
_CLOCK   = "pt_alpaca_clock"        # dict — Alpaca market-clock snapshot
_KS_CONF = "pt_kill_switch_confirm" # bool — confirmation-step flag for the kill button
_RESTORE = "pt_restored_from_config"
_RUNS_CFG_KEY = "paper_trading_runs_v1"
_SIGNALS_CFG_KEY = "paper_trading_signals_v1"
_MAX_PERSISTED_SIGNALS = 500


# ── Helpers ──────────────────────────────────────────────────────────────────

def _db() -> Database:
    return Database(settings.db_path)


def _fmt_price(value) -> str:
    return f"{float(value):.4f}" if value is not None else "—"


def _init_state() -> None:
    for k, v in [(_RUNS, {}), (_CACHE, {}), (_SIGNALS, []), (_TRAIL, {}),
                  (_SYNC, {}), (_ACCT, {}), (_POS, {}), (_RECON, {}),
                  (_CLOCK, {}), (_KS_CONF, False), (_RESTORE, False)]:
        if k not in st.session_state:
            st.session_state[k] = v


def _persist_runs_config() -> None:
    runs = st.session_state.get(_RUNS) or {}
    payload = {
        "runs": runs,
        "saved_at": datetime.utcnow().isoformat(timespec="seconds"),
    }
    try:
        _db().save_config(_RUNS_CFG_KEY, payload)
    except Exception as exc:
        log.warning(f"Paper runs config save failed: {exc}")


def _json_safe(value):
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def _normalize_signal_row(row: dict) -> dict:
    normalized = dict(row)
    date_val = normalized.get("date")
    if date_val is not None:
        try:
            normalized["date"] = pd.Timestamp(date_val).isoformat()
        except Exception:
            normalized["date"] = str(date_val)
    return normalized


def _persist_signals_config() -> None:
    signals = st.session_state.get(_SIGNALS) or []
    payload = {
        "signals": [_normalize_signal_row(_json_safe(s)) for s in signals[-_MAX_PERSISTED_SIGNALS:]],
        "saved_at": datetime.utcnow().isoformat(timespec="seconds"),
    }
    try:
        _db().save_config(_SIGNALS_CFG_KEY, payload)
    except Exception as exc:
        log.warning(f"Paper signals config save failed: {exc}")


def _restore_runs_config() -> bool:
    if st.session_state.get(_RESTORE):
        return False
    st.session_state[_RESTORE] = True
    if st.session_state.get(_RUNS):
        return False
    try:
        payload = _db().load_config(_RUNS_CFG_KEY) or {}
    except Exception as exc:
        log.warning(f"Paper runs config load failed: {exc}")
        return False

    runs = payload.get("runs") or {}
    if not isinstance(runs, dict) or not runs:
        return False

    restored: dict[str, dict] = {}
    for symbol, cfg in runs.items():
        if not isinstance(cfg, dict):
            continue
        sym = str(cfg.get("symbol") or symbol or "").upper().strip()
        if not sym:
            continue
        clean_cfg = dict(cfg)
        clean_cfg["symbol"] = sym
        clean_cfg.setdefault("active", True)
        clean_cfg.setdefault("started_at", datetime.utcnow().isoformat())
        restored[sym] = clean_cfg

    if not restored:
        return False

    st.session_state[_RUNS] = restored
    return True


def _restore_signals_config() -> bool:
    if st.session_state.get(_SIGNALS):
        return False
    try:
        payload = _db().load_config(_SIGNALS_CFG_KEY) or {}
    except Exception as exc:
        log.warning(f"Paper signals config load failed: {exc}")
        return False
    signals = payload.get("signals") or []
    if not isinstance(signals, list) or not signals:
        return False
    st.session_state[_SIGNALS] = [_normalize_signal_row(s) for s in signals[-_MAX_PERSISTED_SIGNALS:] if isinstance(s, dict)]
    return True


def _interval_td(interval: str) -> timedelta:
    return {"1m": timedelta(minutes=1), "2m": timedelta(minutes=2),
            "5m": timedelta(minutes=5), "15m": timedelta(minutes=15),
            "30m": timedelta(minutes=30), "1h": timedelta(hours=1),
            "1d": timedelta(days=1)}.get(interval, timedelta(minutes=5))


def _fetch(symbol: str, interval: str, lookback: int) -> pd.DataFrame:
    delta = _interval_td(interval)
    end   = pd.Timestamp.now()
    start = end - delta * max(lookback * 3, 500)
    return load_forward_blended_data(symbol, interval, start, end, lookback=lookback)


def _local_tz():
    """Return the user's local timezone (tzinfo)."""
    try:
        return datetime.now().astimezone().tzinfo
    except Exception:
        return None


def _to_local(value):
    """Convert a single timestamp (naive UTC or tz-aware) to user's local tz."""
    if value is None:
        return None
    try:
        ts = pd.Timestamp(value)
    except Exception:
        return value
    if ts is pd.NaT:
        return value
    tz = _local_tz()
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert(tz) if tz is not None else ts


def _to_local_series(series: pd.Series) -> pd.Series:
    """Convert a datetime Series to local tz (treat naive as UTC)."""
    s = pd.to_datetime(series, errors="coerce")
    try:
        if s.dt.tz is None:
            s = s.dt.tz_localize("UTC")
        tz = _local_tz()
        if tz is not None:
            s = s.dt.tz_convert(tz)
    except Exception:
        pass
    return s


def _latest_atr(prices: pd.DataFrame, period: int = 14) -> Optional[float]:
    if len(prices) < 2:
        return None
    high  = prices["high"].astype(float)
    low   = prices["low"].astype(float)
    close = prices["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(period, min_periods=1).mean().iloc[-1]
    return float(atr) if pd.notna(atr) else None


def _leveraged_ret(entry: float, exit_p: float, leverage: float, direction: str) -> float:
    raw = (exit_p - entry) / entry
    if direction == "Short":
        raw = -raw
    return raw * leverage * 100


def _trade_cost(cap: float, spread_pct: float, slippage_pct: float, commission: float) -> float:
    return float(cap) * (float(spread_pct) + float(slippage_pct)) / 100.0 + float(commission)


def _open_paper_trades(symbol: Optional[str] = None) -> list[dict]:
    try:
        trades = _db().get_trades(mode="paper", symbol=symbol)
    except Exception:
        return []
    return [t for t in trades if t.get("outcome") == "Open"]


def _open_alpaca_paper_trades(symbol: Optional[str] = None) -> list[dict]:
    """Shadow-mode Alpaca-paper trades still considered 'live' at the broker."""
    try:
        trades = _db().get_trades(mode="alpaca_paper", symbol=symbol)
    except Exception:
        return []
    # We consider a trade "live at the broker" if its outcome is still Open
    # AND the broker_status is not a terminal failure state.
    terminal_fail = {"canceled", "expired", "rejected"}
    live = []
    for t in trades:
        if t.get("outcome") != "Open":
            continue
        bs = (t.get("broker_status") or "").lower()
        if bs in terminal_fail:
            continue
        if t.get("broker_order_id"):
            live.append(t)
    return live


# ── Alpaca-paper order sync (Step 4) ─────────────────────────────────────────

def _sync_alpaca_paper_orders() -> dict:
    """
    Poll Alpaca for every open shadow-mode trade's latest order state and
    persist any updates (broker_status, filled_qty, filled_avg_price,
    filled_at, last_synced_at, and notes/outcome when terminal).

    Returns a small summary dict for UI display:
      {"polled": N, "updated": M, "errors": K,
       "filled": [symbols...], "rejected": [...],
       "at": datetime}

    Safe to call repeatedly — it only contacts Alpaca when there's actually
    a tracked open shadow order with a broker_order_id, and swallows any
    per-order error so one bad row doesn't stop the loop.
    """
    open_shadow = _open_alpaca_paper_trades()
    summary = {
        "polled": 0, "updated": 0, "errors": 0,
        "filled": [], "rejected": [],
        "at": datetime.utcnow(),
    }
    if not open_shadow:
        return summary
    if not settings.alpaca.has_paper_credentials():
        return summary

    # Re-use one router instance across this pass
    router = OrderRouter(risk_manager=RiskManager(settings.risk))
    db = _db()
    terminal = OrderRouter.ALPACA_TERMINAL_STATES

    for t in open_shadow:
        oid = t.get("broker_order_id")
        if not oid:
            continue
        summary["polled"] += 1
        poll = router.fetch_alpaca_order(oid, paper=True)
        if not poll.get("ok"):
            summary["errors"] += 1
            continue

        new_status  = (poll.get("status") or "").lower() or None
        filled_qty  = poll.get("filled_qty")
        filled_px   = poll.get("filled_avg_price")
        filled_at   = poll.get("filled_at")

        # Nothing changed? skip writing.
        _old_status = (t.get("broker_status") or "").lower() or None
        _old_qty    = t.get("filled_qty")
        _old_px     = t.get("filled_avg_price")
        _changed = (
            new_status != _old_status
            or (filled_qty is not None and filled_qty != _old_qty)
            or (filled_px  is not None and filled_px  != _old_px)
        )
        if not _changed:
            # Still refresh last_synced_at so the UI shows activity
            _persist_alpaca_paper_update(t, last_synced_at=datetime.utcnow())
            continue

        # Track summary categories
        if new_status == "filled":
            summary["filled"].append(t.get("symbol", "?"))
        elif new_status in ("rejected", "canceled", "expired"):
            summary["rejected"].append(f"{t.get('symbol','?')} ({new_status})")

        # Terminal-fail → mark the shadow trade closed with no-data outcome
        outcome_override: Optional[TradeOutcome] = None
        notes_suffix = ""
        if new_status in ("rejected", "canceled", "expired"):
            outcome_override = TradeOutcome.NO_DATA
            notes_suffix = f"Alpaca {new_status} at {datetime.utcnow().isoformat(timespec='seconds')}"

        _persist_alpaca_paper_update(
            t,
            broker_status=new_status,
            filled_qty=filled_qty,
            filled_avg_price=filled_px,
            filled_at=filled_at,
            last_synced_at=datetime.utcnow(),
            outcome_override=outcome_override,
            notes_suffix=notes_suffix,
        )
        summary["updated"] += 1

    return summary


def _persist_alpaca_paper_update(
    trade_row: dict,
    *,
    broker_status: Optional[str] = None,
    filled_qty: Optional[float] = None,
    filled_avg_price: Optional[float] = None,
    filled_at: Optional[datetime] = None,
    last_synced_at: Optional[datetime] = None,
    outcome_override: Optional[TradeOutcome] = None,
    notes_suffix: str = "",
) -> None:
    """
    Upsert an alpaca_paper trade row preserving all non-broker fields and
    merging new broker state. Mirrors _update_trade_in_db but stays in the
    'alpaca_paper' mode namespace.
    """
    entry_time = trade_row.get("entry_time")
    if isinstance(entry_time, str) and entry_time:
        entry_time = datetime.fromisoformat(entry_time)
    elif entry_time is None:
        entry_time = datetime.utcnow()

    current_outcome_raw = trade_row.get("outcome") or "Open"
    try:
        current_outcome = TradeOutcome(current_outcome_raw)
    except Exception:
        current_outcome = TradeOutcome.OPEN

    def _preserve_dt(v):
        if isinstance(v, datetime):
            return v
        if isinstance(v, str) and v:
            try:
                return datetime.fromisoformat(v)
            except Exception:
                return None
        return None

    merged_notes = trade_row.get("notes") or ""
    if notes_suffix:
        merged_notes = f"{merged_notes} | {notes_suffix}".strip(" |")

    row = {
        "id":                trade_row["id"],
        "symbol":            trade_row["symbol"],
        "direction":         Direction(trade_row["direction"]),
        "entry_price":       float(trade_row["entry_price"]),
        "take_profit":       trade_row.get("take_profit"),
        "stop_loss":         trade_row.get("stop_loss"),
        "leverage":          float(trade_row["leverage"]),
        "capital_allocated": float(trade_row["capital_allocated"]),
        "entry_time":        entry_time,
        "mode":              "alpaca_paper",
        "strategy_id":       trade_row.get("strategy_id"),
        "exit_price":        trade_row.get("exit_price"),
        "exit_time":         _preserve_dt(trade_row.get("exit_time")),
        "outcome":           outcome_override if outcome_override is not None else current_outcome,
        "leveraged_return_pct": trade_row.get("leveraged_return_pct"),
        "pnl":               trade_row.get("pnl"),
        "notes":             merged_notes,
        # Broker-lifecycle fields — merge new values, preserve old ones
        "broker_order_id":     trade_row.get("broker_order_id"),
        "broker_status":       broker_status if broker_status is not None else trade_row.get("broker_status"),
        "broker_submitted_at": _preserve_dt(trade_row.get("broker_submitted_at")),
        "filled_qty":          filled_qty if filled_qty is not None else trade_row.get("filled_qty"),
        "filled_avg_price":    filled_avg_price if filled_avg_price is not None else trade_row.get("filled_avg_price"),
        "filled_at":           filled_at if filled_at is not None else _preserve_dt(trade_row.get("filled_at")),
        "last_synced_at":      last_synced_at if last_synced_at is not None else _preserve_dt(trade_row.get("last_synced_at")),
    }
    _db().save_trade(_TR(row))


# ── Alpaca account + positions sync (Step 5) ────────────────────────────────

def _sync_alpaca_account_and_positions() -> tuple[dict, dict]:
    """
    Fetch account snapshot + open positions from Alpaca paper. Stores each
    into session state (_ACCT / _POS) and returns them. Cheap no-op when
    paper credentials are missing.
    """
    if not settings.alpaca.has_paper_credentials():
        return ({}, {})
    router = OrderRouter(risk_manager=RiskManager(settings.risk))
    acct = router.fetch_alpaca_account(paper=True)
    pos  = router.fetch_alpaca_positions(paper=True)
    st.session_state[_ACCT] = acct or {}
    st.session_state[_POS]  = pos  or {}
    return acct, pos


def _reconcile_positions_against_db(pos_snapshot: dict) -> dict:
    """
    Compare Alpaca's current positions vs our DB's open alpaca_paper trades.

    Three buckets are produced:
      - closed_by_broker: trades the DB thinks are open but Alpaca no longer
        holds a position in → close them (bracket TP/SL filled, or manually
        flattened via the Alpaca UI).
      - orphan_positions: symbols Alpaca holds that we have no DB entry for
        (pre-existing positions, or manual trades outside the app).
      - tracked_open: symbols present on both sides → healthy live positions.

    For each closed_by_broker trade we attempt to query the actual closing
    order via router.find_closing_order() so the DB row gets the REAL fill
    price + time instead of an approximation. If that lookup fails we fall
    back to the latest cached bar close.

    Returns a summary dict for UI display.
    """
    summary = {
        "closed_by_broker": [],   # list of {symbol, exit_price, exit_time, source}
        "orphan_positions": [],   # list of symbols Alpaca holds that DB doesn't
        "tracked_open":     [],   # list of symbols live on both sides
        "errors":           [],
        "at":               datetime.utcnow(),
    }

    if not pos_snapshot or not pos_snapshot.get("ok"):
        summary["errors"].append(pos_snapshot.get("error") if pos_snapshot else "no snapshot")
        return summary

    alpaca_positions = pos_snapshot.get("positions") or []
    alpaca_symbols   = {p["symbol"]: p for p in alpaca_positions if p.get("symbol")}

    open_shadow = _open_alpaca_paper_trades()
    open_by_sym = {}
    for t in open_shadow:
        open_by_sym.setdefault(t.get("symbol"), []).append(t)

    router = OrderRouter(risk_manager=RiskManager(settings.risk))

    # ── Bucket 1: trades we think are open but Alpaca has no position ────
    for sym, trades in open_by_sym.items():
        if sym in alpaca_symbols:
            summary["tracked_open"].append(sym)
            continue
        for t in trades:
            # Only reconcile trades whose broker_status indicates the entry
            # actually got filled. If still pending (accepted/new), skip —
            # the order-status poll will handle it.
            bs = (t.get("broker_status") or "").lower()
            if bs != "filled":
                continue

            # Opposite side for the closing order
            dir_str = t.get("direction")
            opp_side = "sell" if dir_str == "Long" else "buy"

            # Try to find the real closing order first
            closing = None
            try:
                _after = None
                _sub = t.get("broker_submitted_at") or t.get("entry_time")
                if isinstance(_sub, str) and _sub:
                    try:
                        _after = datetime.fromisoformat(_sub)
                    except Exception:
                        _after = None
                elif isinstance(_sub, datetime):
                    _after = _sub
                closing = router.find_closing_order(
                    sym, after=_after, opposite_side=opp_side, paper=True,
                )
            except Exception as _exc:
                summary["errors"].append(f"{sym}: find_closing_order — {_exc}")

            if closing and closing.get("filled_avg_price") is not None:
                exit_px   = float(closing["filled_avg_price"])
                exit_time = closing.get("filled_at") or datetime.utcnow()
                source    = f"alpaca order {closing['order_id'][:8]}"
            else:
                # Fallback: use last cached bar close if we have it
                _cache = st.session_state.get(_CACHE, {}).get(sym)
                if _cache is not None and not _cache.empty:
                    exit_px   = float(_cache.iloc[-1]["close"])
                    exit_time = pd.Timestamp(_cache.iloc[-1]["date"]).to_pydatetime()
                    source    = "last-bar-close (approx)"
                else:
                    exit_px   = float(t.get("entry_price") or 0)
                    exit_time = datetime.utcnow()
                    source    = "entry-price (approx)"

            # Close the trade in DB
            try:
                run_cfg = st.session_state[_RUNS].get(sym, {})
                # We don't recompute PnL using sim costs here because the
                # Alpaca side's P&L is the broker's to report. We store the
                # exit details + a SIGNAL_EXIT outcome so the row looks closed.
                lev = float(t["leverage"])
                ep  = float(t["entry_price"])
                ret = (exit_px - ep) / ep * (1 if dir_str == "Long" else -1) * lev * 100
                cap = float(t["capital_allocated"])
                # No sim costs applied on the Alpaca side — the broker's fill
                # price already embeds slippage + spread.
                pnl = cap * ret / 100.0

                _persist_alpaca_paper_update(
                    t,
                    broker_status=t.get("broker_status"),
                    last_synced_at=datetime.utcnow(),
                    outcome_override=TradeOutcome.SIGNAL_EXIT,
                    notes_suffix=(
                        f"Reconciled closed @ {exit_px:.4f} "
                        f"({source}) on {pd.Timestamp(exit_time).strftime('%Y-%m-%d %H:%M:%S')}"
                    ),
                )
                # The outcome override is persisted, but _persist_alpaca_paper_update
                # doesn't know about exit_price/exit_time/pnl — set those directly.
                _update_alpaca_paper_exit(
                    t, exit_price=exit_px, exit_time=exit_time,
                    leveraged_return_pct=ret, pnl=pnl,
                )
                summary["closed_by_broker"].append({
                    "symbol":     sym,
                    "exit_price": exit_px,
                    "exit_time":  exit_time,
                    "source":     source,
                    "pnl":        pnl,
                })
            except Exception as _exc:
                summary["errors"].append(f"{sym}: close — {_exc}")

    # ── Bucket 2: Alpaca positions we don't track in DB ──────────────────
    db_open_symbols = set(open_by_sym.keys())
    for sym in alpaca_symbols:
        if sym not in db_open_symbols:
            summary["orphan_positions"].append(sym)

    st.session_state[_RECON] = summary
    return summary


# ── Alpaca market-clock sync (Step 6) ───────────────────────────────────────

def _sync_alpaca_clock() -> dict:
    """
    Pull Alpaca's market-clock (is_open / next_open / next_close) into session
    state so the pre-flight banner can show it without re-hitting the network.
    Cheap no-op when paper credentials are missing.
    """
    if not settings.alpaca.has_paper_credentials():
        return {}
    try:
        router = OrderRouter(risk_manager=RiskManager(settings.risk))
        clk = router.fetch_alpaca_clock(paper=True) or {}
    except Exception as exc:
        clk = {"ok": False, "error": str(exc)}
    st.session_state[_CLOCK] = clk
    return clk


# ── Flatten-all (kill switch companion) ─────────────────────────────────────

def _flatten_all_positions() -> dict:
    """
    Emergency close of every open position — local sim AND Alpaca paper.

    Local sim  → each open `paper` trade is closed at the last cached bar
                 close (net of spread/slippage/commission from its run config).
    Alpaca    → close_position_alpaca_paper(symbol) is called for every
                symbol Alpaca currently reports a position in. This lets us
                flatten even positions that were opened before shadow mode
                was on (orphans) because we target Alpaca's view, not ours.

    Returns a summary dict: {sim_closed: [...], alpaca_closed: [...],
                              errors: [...], at: datetime}.
    """
    summary = {
        "sim_closed":    [],
        "alpaca_closed": [],
        "errors":        [],
        "at":            datetime.utcnow(),
    }

    # 1) Close every open local-sim trade at its last cached bar close
    try:
        open_sim = _open_paper_trades()
    except Exception as exc:
        summary["errors"].append(f"sim lookup: {exc}")
        open_sim = []

    runs = st.session_state.get(_RUNS) or {}
    for trade in open_sim:
        sym = trade.get("symbol")
        if not sym:
            continue
        prices = st.session_state[_CACHE].get(sym)
        if prices is None or prices.empty:
            summary["errors"].append(f"{sym}: no cached bar to close sim trade")
            continue
        last_bar = prices.iloc[-1]
        run      = runs.get(sym) or {}
        try:
            _close_trade(
                trade,
                exit_price=float(last_bar["close"]),
                exit_time=pd.Timestamp(last_bar.name).to_pydatetime(),
                outcome=TradeOutcome.SIGNAL_EXIT,
                spread_pct=run.get("spread_pct", 0.0),
                slippage_pct=run.get("slippage_pct", 0.0),
                commission=run.get("commission", 0.0),
                notes=f"Flatten-all: closed at {float(last_bar['close']):.4f}",
            )
            summary["sim_closed"].append(sym)
        except Exception as exc:
            summary["errors"].append(f"{sym} sim close: {exc}")

    # 2) Flatten Alpaca paper — iterate whatever Alpaca itself says is open.
    if settings.alpaca.has_paper_credentials():
        try:
            router = OrderRouter(risk_manager=RiskManager(settings.risk))
            pos = router.fetch_alpaca_positions(paper=True) or {}
            for p in (pos.get("positions") or []):
                sym = p.get("symbol")
                if not sym:
                    continue
                res = router.close_position_alpaca_paper(sym)
                if res.get("ok"):
                    summary["alpaca_closed"].append(sym)
                else:
                    summary["errors"].append(
                        f"{sym} alpaca close: {res.get('error', 'unknown')}"
                    )
        except Exception as exc:
            summary["errors"].append(f"alpaca flatten: {exc}")

    return summary


def _update_alpaca_paper_exit(
    trade_row: dict,
    *,
    exit_price: float,
    exit_time: datetime,
    leveraged_return_pct: float,
    pnl: float,
) -> None:
    """
    Narrow-purpose updater: stamp exit_price/exit_time/pnl on an alpaca_paper
    row without touching broker-lifecycle fields. Used by reconciliation
    after _persist_alpaca_paper_update has set outcome + notes.
    """
    # Re-read the row to pick up the notes/outcome the prior call just wrote.
    try:
        fresh = [
            r for r in _db().get_trades(mode="alpaca_paper", symbol=trade_row["symbol"])
            if r["id"] == trade_row["id"]
        ]
        base = fresh[0] if fresh else trade_row
    except Exception:
        base = trade_row

    entry_time = base.get("entry_time")
    if isinstance(entry_time, str) and entry_time:
        entry_time = datetime.fromisoformat(entry_time)
    elif entry_time is None:
        entry_time = datetime.utcnow()

    def _preserve_dt(v):
        if isinstance(v, datetime):
            return v
        if isinstance(v, str) and v:
            try:
                return datetime.fromisoformat(v)
            except Exception:
                return None
        return None

    try:
        outcome = TradeOutcome(base.get("outcome") or "SignalExit")
    except Exception:
        outcome = TradeOutcome.SIGNAL_EXIT

    row = {
        "id":                base["id"],
        "symbol":            base["symbol"],
        "direction":         Direction(base["direction"]),
        "entry_price":       float(base["entry_price"]),
        "take_profit":       base.get("take_profit"),
        "stop_loss":         base.get("stop_loss"),
        "leverage":          float(base["leverage"]),
        "capital_allocated": float(base["capital_allocated"]),
        "entry_time":        entry_time,
        "mode":              "alpaca_paper",
        "strategy_id":       base.get("strategy_id"),
        "exit_price":        float(exit_price),
        "exit_time":         exit_time,
        "outcome":           outcome,
        "leveraged_return_pct": float(leveraged_return_pct),
        "pnl":               float(pnl),
        "notes":             base.get("notes") or "",
        "broker_order_id":     base.get("broker_order_id"),
        "broker_status":       base.get("broker_status"),
        "broker_submitted_at": _preserve_dt(base.get("broker_submitted_at")),
        "filled_qty":          base.get("filled_qty"),
        "filled_avg_price":    base.get("filled_avg_price"),
        "filled_at":           _preserve_dt(base.get("filled_at")),
        "last_synced_at":      datetime.utcnow(),
    }
    _db().save_trade(_TR(row))


# ── Persist helpers (save to DB via duck-typed TradeRecord) ──────────────────

class _TR:
    """Lightweight TradeRecord stand-in accepted by Database.save_trade."""
    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, v)


def _update_trade_in_db(trade_row: dict, *, stop_loss: float | None = None,
                        take_profit: float | None = None,
                        outcome: TradeOutcome | None = None,
                        exit_price: float | None = None,
                        exit_time: datetime | None = None,
                        leveraged_return_pct: float | None = None,
                        pnl: float | None = None,
                        notes: str | None = None) -> None:
    """Upsert a trade row, preserving fields we don't explicitly change."""
    entry_time = trade_row.get("entry_time")
    if isinstance(entry_time, str) and entry_time:
        entry_time = datetime.fromisoformat(entry_time)
    elif entry_time is None:
        entry_time = datetime.utcnow()

    current_outcome_raw = trade_row.get("outcome") or "Open"
    try:
        current_outcome = TradeOutcome(current_outcome_raw)
    except Exception:
        current_outcome = TradeOutcome.OPEN

    row = {
        "id":                trade_row["id"],
        "symbol":            trade_row["symbol"],
        "direction":         Direction(trade_row["direction"]),
        "entry_price":       float(trade_row["entry_price"]),
        "take_profit":       take_profit if take_profit is not None else trade_row.get("take_profit"),
        "stop_loss":         stop_loss   if stop_loss   is not None else trade_row.get("stop_loss"),
        "leverage":          float(trade_row["leverage"]),
        "capital_allocated": float(trade_row["capital_allocated"]),
        "entry_time":        entry_time,
        "mode":              "paper",
        "strategy_id":       trade_row.get("strategy_id"),
        "exit_price":        exit_price if exit_price is not None else trade_row.get("exit_price"),
        "exit_time":         exit_time  if exit_time  is not None else (
                                datetime.fromisoformat(trade_row["exit_time"])
                                if isinstance(trade_row.get("exit_time"), str) and trade_row["exit_time"]
                                else trade_row.get("exit_time")),
        "outcome":           outcome if outcome is not None else current_outcome,
        "leveraged_return_pct": (leveraged_return_pct if leveraged_return_pct is not None
                                  else trade_row.get("leveraged_return_pct")),
        "pnl":               pnl if pnl is not None else trade_row.get("pnl"),
        "notes":             notes if notes is not None else trade_row.get("notes"),
    }
    _db().save_trade(_TR(row))


# ── Trailing + exit logic (ported from forward test) ────────────────────────

def _apply_trailing(trade_row: dict, trail: dict, bar: pd.Series,
                    atr_value: Optional[float]) -> float:
    """
    Update trailing stop based on latest bar. Returns the (possibly new) SL.
    Mutates trail dict (best, bars) in place.
    """
    hi, lo = float(bar["high"]), float(bar["low"])
    ep     = float(trade_row["entry_price"])
    d      = trade_row["direction"]
    sl     = float(trade_row.get("stop_loss"))

    trail["trail_bars"] = int(trail.get("trail_bars", 0)) + 1
    if trail["trail_bars"] <= int(trail.get("trail_grace", 0)):
        return sl

    hard_sl = float(trail.get("hard_stop_loss", sl))
    kind    = trail.get("trail_kind")
    val     = float(trail.get("trail_value", 0) or 0)

    if d == "Short":
        trail["trail_best"] = min(float(trail.get("trail_best", ep)), lo)
        if kind == "pct":
            candidate = trail["trail_best"] * (1 + val / 100)
        elif kind == "atr" and atr_value is not None:
            candidate = trail["trail_best"] + val * atr_value
        elif kind == "giveback":
            profit_move = max(ep - trail["trail_best"], 0.0)
            profit_pct  = profit_move / ep * 100 if ep > 0 else 0.0
            if profit_pct >= float(trail.get("trail_min_profit_pct", 0.0)):
                candidate = trail["trail_best"] + val * profit_move
            else:
                candidate = hard_sl
        else:
            candidate = hard_sl
        return min(candidate, hard_sl)

    # Long
    trail["trail_best"] = max(float(trail.get("trail_best", ep)), hi)
    if kind == "pct":
        candidate = trail["trail_best"] * (1 - val / 100)
    elif kind == "atr" and atr_value is not None:
        candidate = trail["trail_best"] - val * atr_value
    elif kind == "giveback":
        profit_move = max(trail["trail_best"] - ep, 0.0)
        profit_pct  = profit_move / ep * 100 if ep > 0 else 0.0
        if profit_pct >= float(trail.get("trail_min_profit_pct", 0.0)):
            candidate = trail["trail_best"] - val * profit_move
        else:
            candidate = hard_sl
    else:
        candidate = hard_sl
    return max(candidate, hard_sl)


def _check_tp_sl(trade_row: dict, bar: pd.Series, sl: float) -> Optional[TradeOutcome]:
    hi = float(bar["high"]); lo = float(bar["low"])
    tp = trade_row.get("take_profit")
    d  = trade_row["direction"]
    hit_sl = (hi >= sl) if d == "Short" else (lo <= sl)
    hit_tp = ((lo <= tp) if d == "Short" else (hi >= tp)) if tp else False
    if hit_sl and hit_tp:
        return TradeOutcome.AMBIGUOUS
    if hit_sl:
        return TradeOutcome.STOP_LOSS
    if hit_tp:
        return TradeOutcome.TAKE_PROFIT
    return None


# Regime prefixes that indicate a spike trade (mirrors backtest.py).
_SPIKE_REGIMES = (
    "regime=spike_long",
    "regime=spike_momentum_long",
    "regime=post_spike_short",
    "regime=event_target_short",
)


def _is_spike_trade(trade_row: dict) -> bool:
    notes = trade_row.get("notes") or ""
    return any(r in notes for r in _SPIKE_REGIMES)


def _counter_signal_outcome(strategy_id: str, open_direction: str) -> TradeOutcome:
    """
    Map a counter-signal exit to the right TradeOutcome tag, mirroring
    reporting/backtest._counter_signal_outcome.
    """
    sid = (strategy_id or "").lower()
    if "rsi" in sid:
        return (TradeOutcome.SIGNAL_RSI_OB if open_direction == "Long"
                else TradeOutcome.SIGNAL_RSI_OS)
    return TradeOutcome.SIGNAL_EXIT


def _close_trade(trade_row: dict, *, exit_price: float, exit_time: datetime,
                 outcome: TradeOutcome, spread_pct: float, slippage_pct: float,
                 commission: float, notes: str) -> tuple[float, float]:
    ep   = float(trade_row["entry_price"])
    lev  = float(trade_row["leverage"])
    cap  = float(trade_row["capital_allocated"])
    d    = trade_row["direction"]
    ret  = _leveraged_ret(ep, exit_price, lev, d)
    gross = cap * ret / 100
    pnl   = gross - _trade_cost(cap, spread_pct, slippage_pct, commission)
    _update_trade_in_db(
        trade_row,
        outcome=outcome,
        exit_price=exit_price,
        exit_time=exit_time,
        leveraged_return_pct=ret,
        pnl=pnl,
        notes=notes,
    )
    # Clear trail state for this id
    st.session_state[_TRAIL].pop(trade_row.get("id"), None)
    return ret, pnl


# ── Core per-tick logic: exits first, then maybe entries ────────────────────

def _run_tick(symbol: str, run: dict) -> None:
    try:
        prices = _fetch(symbol, run["interval"], run["lookback"])
    except Exception as e:
        st.warning(f"⚠️ {symbol}: data fetch failed — {e}")
        return
    if prices is None or prices.empty:
        st.warning(f"⚠️ {symbol}: no data returned.")
        return

    st.session_state[_CACHE][symbol] = prices
    atr_val = _latest_atr(prices)
    latest  = prices.iloc[-1]
    latest_ts = latest["date"]

    # ── 1) Manage any open paper trades on this symbol (trailing + exits) ────
    for open_t in _open_paper_trades(symbol):
        trail = st.session_state[_TRAIL].get(open_t["id"])
        current_sl = float(open_t["stop_loss"]) if open_t.get("stop_loss") is not None else None

        new_sl = current_sl
        if trail and current_sl is not None:
            new_sl = _apply_trailing(open_t, trail, latest, atr_val)
            if new_sl is not None and new_sl != current_sl:
                _update_trade_in_db(open_t, stop_loss=new_sl,
                                     notes=f"Trailing stop updated to {new_sl:.4f} at {latest_ts}")
                open_t["stop_loss"] = new_sl  # reflect in local copy for TP/SL check

        if new_sl is None:
            continue
        hit = _check_tp_sl(open_t, latest, new_sl)
        if hit is None:
            continue

        tp_val = float(open_t["take_profit"]) if open_t.get("take_profit") is not None else None
        if hit == TradeOutcome.AMBIGUOUS:
            exit_price = float(latest["close"])
        elif hit == TradeOutcome.TAKE_PROFIT and tp_val is not None:
            exit_price = tp_val
        else:
            exit_price = new_sl

        _close_trade(
            open_t,
            exit_price=exit_price,
            exit_time=pd.Timestamp(latest_ts).to_pydatetime(),
            outcome=hit,
            spread_pct=run.get("spread_pct", 0.0),
            slippage_pct=run.get("slippage_pct", 0.0),
            commission=run.get("commission", 0.0),
            notes=("Trail stop" if trail and hit == TradeOutcome.STOP_LOSS
                   else hit.value if hasattr(hit, "value") else str(hit)),
        )

    # ── 2) Generate signal on prepared data ──────────────────────────────────
    cls      = get_strategy(run["strategy_id"])
    strategy = cls(params=run["params"])
    prepared = prepare_strategy_data(
        prices, strategy,
        primary_symbol=symbol, source="forward_blend",
        interval=run["interval"],
        start=prices["date"].min(), end=prices["date"].max(),
    )
    st.session_state[_CACHE][symbol] = prepared
    latest = prepared.iloc[-1]
    latest_ts = latest["date"]
    signal = strategy.generate_signal(prepared, symbol)

    sig_meta = signal.metadata or {}
    st.session_state[_SIGNALS].append({
        "date":     pd.Timestamp(_to_local(latest_ts)).isoformat(),
        "symbol":   symbol,
        "action":   signal.action.value,
        "close":    float(latest["close"]),
        "strategy": cls.name,
        "confidence": signal.confidence,
        "tp":       signal.suggested_tp,
        "sl":       signal.suggested_sl,
        "rsi":      sig_meta.get("rsi"),
        "regime":   sig_meta.get("regime"),
        "verdict_reason": sig_meta.get("verdict_reason"),
    })

    run["_last_signal"] = {
        "action":     signal.action.value,
        "confidence": signal.confidence,
        "tp":         signal.suggested_tp,
        "sl":         signal.suggested_sl,
        "rsi":        sig_meta.get("rsi"),
        "regime":     sig_meta.get("regime"),
        "verdict_reason": sig_meta.get("verdict_reason"),
        "entry_price": float(latest["close"]),
        "timestamp":   str(_to_local(latest_ts)),
    }
    _persist_runs_config()
    _persist_signals_config()

    # ── 3a) Counter-signal exit (mirror backtest) ────────────────────────────
    # If an open non-spike trade exists and the new signal is the opposite
    # direction, close it at this bar's close price.
    if run.get("counter_signal_exit", True) and signal.action != SignalAction.HOLD:
        new_dir = Direction.LONG if signal.action == SignalAction.BUY else Direction.SHORT
        for open_t in _open_paper_trades(symbol):
            if _is_spike_trade(open_t):
                continue
            current_dir = open_t.get("direction")
            is_reversal = (
                (current_dir == "Long"  and new_dir == Direction.SHORT) or
                (current_dir == "Short" and new_dir == Direction.LONG)
            )
            if not is_reversal:
                continue
            exit_px = float(latest["close"])
            outcome = _counter_signal_outcome(run.get("strategy_id", ""), current_dir)
            _close_trade(
                open_t,
                exit_price=exit_px,
                exit_time=pd.Timestamp(latest_ts).to_pydatetime(),
                outcome=outcome,
                spread_pct=run.get("spread_pct", 0.0),
                slippage_pct=run.get("slippage_pct", 0.0),
                commission=run.get("commission", 0.0),
                notes=f"Counter-signal exit at {exit_px:.4f} ({outcome.value})",
            )
            # Shadow mode: flatten any Alpaca-paper position in this symbol
            # so both populations stay in sync. Safe no-op if none open.
            if run.get("shadow_alpaca") and settings.alpaca.has_paper_credentials():
                try:
                    _shadow_router = OrderRouter(
                        risk_manager=RiskManager(settings.risk)
                    )
                    _shadow_router.close_position_alpaca_paper(symbol)
                except Exception as _exc:
                    st.warning(
                        f"⚠️ {symbol}: shadow-mode Alpaca close failed — {_exc}"
                    )

    # ── 3b) Auto-enter if signal fires AND no open position remains ──────────
    if _open_paper_trades(symbol):
        return  # already in a trade — no pyramiding
    if signal.action == SignalAction.HOLD:
        return
    if signal.suggested_sl is None:
        return  # strategy didn't give us a stop — skip

    # ── KILL-SWITCH GUARD ────────────────────────────────────────────────────
    # Blocks every NEW entry (local sim + shadow-mode Alpaca) while the switch
    # is tripped. Exits, trailing-stop updates, and reconciliation above this
    # point still run so open positions can be closed & squared safely.
    try:
        if ks.is_tripped(_db()):
            run["_last_signal"]["skipped_reason"] = (
                "🛑 Kill switch tripped — entry blocked."
            )
            _persist_runs_config()
            return
    except Exception:
        # Fail-open on DB hiccups: we don't want a transient read error to
        # silently swallow trades. The banner will still show the state.
        pass

    direction = Direction.LONG if signal.action == SignalAction.BUY else Direction.SHORT
    entry_price = float(latest["close"])

    # ── 3c) Equity-aware sizing (mirror backtest) ────────────────────────────
    # Available equity = starting capital + realised P&L on this symbol.
    # Request min(capital_per_trade, available_equity); skip if non-positive.
    requested_capital = float(run["capital"])
    if run.get("equity_aware_sizing", True):
        try:
            sym_closed_all = [
                t for t in _db().get_trades(mode="paper", symbol=symbol)
                if t.get("outcome") != "Open"
            ]
        except Exception:
            sym_closed_all = []
        sym_realised = float(sum(float(t.get("pnl") or 0) for t in sym_closed_all))
        available_equity = max(float(run["capital"]) + sym_realised, 0.0)
        requested_capital = min(float(run["capital"]), available_equity)
        if requested_capital <= 0:
            run["_last_signal"]["skipped_reason"] = "Equity depleted — entry skipped."
            _persist_runs_config()
            return

    # ── 3d) RiskManager state: real portfolio equity + open positions ────────
    # Mirrors backtest: the risk manager needs live equity to apply any
    # equity-based caps, and the real open-position count for concurrency
    # rules. Equity = Σ per-symbol starting capital + Σ realised P&L across
    # all closed paper trades (portfolio-wide view).
    try:
        _all_paper_trades = _db().get_trades(mode="paper")
    except Exception:
        _all_paper_trades = []
    _all_realised = float(sum(
        float(t.get("pnl") or 0)
        for t in _all_paper_trades if t.get("outcome") != "Open"
    ))
    _portfolio_start = float(sum(
        float(r.get("capital", 0) or 0)
        for r in st.session_state[_RUNS].values()
    ))
    _portfolio_equity = max(_portfolio_start + _all_realised, 0.0)
    _open_count = sum(1 for t in _all_paper_trades if t.get("outcome") == "Open")

    # ── Entry policy: Classic vs Alpaca-gated (pluggable) ────────────────────
    # The policy encapsulates every pre-fill gate (RTH / SSR / PDT / fractional
    # / fill-diagnostic). Swapping policies via the dropdown is the ONLY thing
    # needed to compare unconstrained paper results against Alpaca-realistic
    # ones — no logic lives inline here anymore.
    policy = get_policy(
        run.get("execution_logic", "alpaca"),
        **{
            k: run[k] for k in (
                "enforce_rth", "extended_hours", "enforce_pdt",
                "enforce_ssr", "enforce_fractional", "fill_diagnostic",
            ) if k in run
        },
    )
    _ctx = EntryContext(
        symbol=symbol,
        direction=direction,
        entry_price=entry_price,
        bar=latest,
        bar_time=latest_ts,
        prices=prepared,
        requested_capital=requested_capital,
        leverage=float(run["leverage"]),
        portfolio_equity=_portfolio_equity,
        portfolio_trades=_all_paper_trades,
    )
    decision = policy.evaluate(_ctx)
    if not decision.allowed:
        run["_last_signal"]["skipped_reason"] = decision.skip_reason
        _persist_runs_config()
        return
    if decision.adjusted_capital is not None:
        requested_capital = float(decision.adjusted_capital)
    if decision.notes_prefix:
        run["_last_signal"]["notes"] = decision.notes_prefix
        _persist_runs_config()

    risk = RiskManager(settings.risk)
    risk.update_portfolio_state(
        daily_pnl=0.0,
        open_positions=_open_count,
        total_equity=_portfolio_equity,
    )
    router = OrderRouter(risk_manager=risk)
    trade = router.execute(
        symbol=symbol, direction=direction,
        entry_price=entry_price,
        take_profit=signal.suggested_tp,
        stop_loss=signal.suggested_sl,
        leverage=float(run["leverage"]),
        capital=requested_capital,
        strategy_id=run["strategy_id"],
        confirm_live=False,
        route=ROUTE_SIM,   # local-sim side of shadow mode (or sole order otherwise)
    )

    # Policy-provided post-fill note (e.g. Alpaca fill-timing diagnostic).
    # Classic policy returns empty string → no-op.
    if decision.post_fill_note and "REJECTED" not in (trade.notes or ""):
        trade.notes = f"{trade.notes or ''} | {decision.post_fill_note}".strip(" |")

    # Persist + stash trailing state if the strategy emitted one
    _db().save_signal(signal)
    _db().save_trade(trade)

    if "REJECTED" in (trade.notes or ""):
        _persist_runs_config()
        return

    # ── 3e) Shadow mode: mirror the entry to Alpaca's paper endpoint ────────
    # When enabled, every approved sim entry is also submitted to Alpaca
    # paper. The resulting TradeRecord is persisted with mode="alpaca_paper"
    # so the two populations can be compared without touching the sim
    # side's open-position tracking (_open_paper_trades filters by mode).
    if run.get("shadow_alpaca") and settings.alpaca.has_paper_credentials():
        try:
            shadow_trade = router.execute(
                symbol=symbol, direction=direction,
                entry_price=entry_price,
                take_profit=signal.suggested_tp,
                stop_loss=signal.suggested_sl,
                leverage=float(run["leverage"]),
                capital=requested_capital,
                strategy_id=run["strategy_id"],
                confirm_live=False,
                route=ROUTE_ALPACA_PAPER,
            )
            # Tag the shadow record so the UI can pair it with its sim twin.
            pair_tag = f"shadow_of={trade.id}"
            shadow_trade.notes = (
                f"{shadow_trade.notes or ''} | {pair_tag}".strip(" |")
            )
            _db().save_trade(shadow_trade)
            run["_last_signal"]["shadow_status"] = (
                "REJECTED" if "REJECTED" in (shadow_trade.notes or "") else "submitted"
            )
        except Exception as exc:
            run["_last_signal"]["shadow_status"] = f"error: {exc}"
    _persist_runs_config()

    eff_sl = trade.stop_loss
    trail_state = None
    if sig_meta.get("trailing_atr_mult") is not None:
        trail_state = {
            "trail_kind":  "atr",
            "trail_value": float(sig_meta["trailing_atr_mult"]),
            "trail_best":  entry_price,
            "hard_stop_loss": eff_sl,
            "trail_grace": 0,
            "trail_bars":  0,
        }
    elif sig_meta.get("pct_trail") is not None:
        trail_state = {
            "trail_kind":  "pct",
            "trail_value": float(sig_meta["pct_trail"]),
            "trail_best":  entry_price,
            "hard_stop_loss": eff_sl,
            "trail_grace": 1,
            "trail_bars":  0,
        }
    elif sig_meta.get("profit_giveback_frac") is not None:
        trail_state = {
            "trail_kind":  "giveback",
            "trail_value": float(sig_meta["profit_giveback_frac"]),
            "trail_min_profit_pct": float(sig_meta.get("profit_giveback_min_pct", 0.0) or 0.0),
            "trail_best":  entry_price,
            "hard_stop_loss": eff_sl,
            "trail_grace": 1,
            "trail_bars":  0,
        }
    if trail_state:
        st.session_state[_TRAIL][trade.id] = trail_state


# ── Charts ───────────────────────────────────────────────────────────────────

def _equity_curve_chart(closed_trades: list[dict], starting_capital: float,
                        title: str = "Portfolio Equity",
                        anchor_time: Optional[pd.Timestamp] = None) -> alt.LayerChart:
    """
    Equity curve from closed paper trades. Starts at starting_capital at
    `anchor_time` (defaults to now if not supplied), then steps up/down by
    each closed trade's net P&L at its exit_time.

    Even with zero closed trades, a flat visible line at starting_capital
    is drawn from anchor_time to now so the chart is not blank.
    """
    closed = [t for t in closed_trades if t.get("pnl") is not None and t.get("exit_time")]
    now_local = pd.Timestamp(_to_local(pd.Timestamp.utcnow()))
    anchor_local = (pd.Timestamp(_to_local(anchor_time)) if anchor_time is not None
                    else now_local)

    if not closed:
        # Flat line: anchor → now, both at starting_capital
        plot_df = pd.DataFrame({
            "time":   [anchor_local, now_local],
            "equity": [starting_capital, starting_capital],
            "pnl":    [0.0, 0.0],
        })
        df = pd.DataFrame({"time": [], "equity": [], "pnl": []})
    else:
        df = pd.DataFrame(closed).copy()
        df["time"]   = _to_local_series(df["exit_time"])
        df = df.sort_values("time").reset_index(drop=True)
        df["equity"] = starting_capital + df["pnl"].astype(float).cumsum()
        df["pnl"]    = df["pnl"].astype(float)

        # Anchor at start and extend to "now" so the latest equity is visible
        first_time  = min(anchor_local, df["time"].iloc[0])
        last_equity = float(df["equity"].iloc[-1])
        anchor = pd.DataFrame({"time": [first_time],
                                "equity": [starting_capital],
                                "pnl": [0.0]})
        tail = pd.DataFrame({"time":   [now_local],
                              "equity": [last_equity],
                              "pnl":    [0.0]})
        plot_df = pd.concat([anchor, df[["time", "equity", "pnl"]], tail],
                             ignore_index=True)

    base_line = (alt.Chart(pd.DataFrame({"y": [starting_capital]}))
                 .mark_rule(color=_GREY, strokeDash=[4,4], strokeWidth=1).encode(y="y:Q"))
    area = (alt.Chart(plot_df)
            .mark_area(line={"color": _BLUE, "strokeWidth": 2},
                       color=alt.Gradient(gradient="linear",
                           stops=[alt.GradientStop(color="rgba(74,158,255,0.25)", offset=0),
                                  alt.GradientStop(color="rgba(74,158,255,0.0)",  offset=1)],
                           x1=1, x2=1, y1=1, y2=0))
            .encode(x=alt.X("time:T", title="Time", axis=alt.Axis(**_AXIS)),
                    y=alt.Y("equity:Q", title="Equity ($)", scale=alt.Scale(zero=False),
                            axis=alt.Axis(**_AXIS)),
                    tooltip=["time:T",
                             alt.Tooltip("equity:Q", format="$,.2f"),
                             alt.Tooltip("pnl:Q",    format="+$,.2f", title="Trade P&L")]))
    dots = (alt.Chart(df).mark_point(size=60, filled=True)
            .encode(x="time:T", y="equity:Q",
                    color=alt.condition(alt.datum.pnl > 0,
                                         alt.value(_GREEN), alt.value(_RED)),
                    tooltip=["time:T",
                             alt.Tooltip("equity:Q", format="$,.2f"),
                             alt.Tooltip("pnl:Q",    format="+$,.2f", title="Trade P&L")]))
    return (alt.layer(base_line, area, dots)
            .properties(title=alt.TitleParams(title, **_TITLE), height=260)
            .configure_view(strokeOpacity=0)
            .configure_axis(**_AXIS)
            .configure_title(**_TITLE))


def _paper_price_chart(prices: pd.DataFrame, symbol: str,
                       signals: list[dict], open_trades: list[dict],
                       closed_trades: list[dict]) -> alt.LayerChart:
    # Display copy with dates converted to the user's local timezone
    prices = prices.copy()
    prices["date"] = _to_local_series(prices["date"])
    base = (alt.Chart(prices).mark_line(color=_BLUE, strokeWidth=1.4)
            .encode(x=alt.X("date:T", title="Date / Time", axis=alt.Axis(**_AXIS)),
                    y=alt.Y("close:Q", title="Price", scale=alt.Scale(zero=False),
                            axis=alt.Axis(**_AXIS)),
                    tooltip=["date:T", alt.Tooltip("close:Q", format=".4f")]))
    layers = [base]

    sig_rows = [s for s in signals if s.get("symbol") == symbol]
    if sig_rows:
        sdf = pd.DataFrame(sig_rows)
        buys  = sdf[sdf["action"] == "BUY"].copy()
        sells = sdf[sdf["action"] == "SELL"].copy()
        tt = ["date:T", "action:N", alt.Tooltip("close:Q", format=".4f")]
        if not buys.empty:
            buys["y"] = buys["close"] * 0.997
            layers.append(alt.Chart(buys)
                .mark_point(shape="triangle-up", size=110, filled=True, color=_GREEN)
                .encode(x="date:T", y="y:Q", tooltip=tt))
        if not sells.empty:
            sells["y"] = sells["close"] * 1.003
            layers.append(alt.Chart(sells)
                .mark_point(shape="triangle-down", size=110, filled=True, color=_RED)
                .encode(x="date:T", y="y:Q", tooltip=tt))

    # Closed paper trade exits for this symbol
    sym_closed = [t for t in closed_trades
                  if t.get("symbol") == symbol and t.get("exit_price") and t.get("exit_time")]
    if sym_closed:
        ex_df = pd.DataFrame(sym_closed).rename(
            columns={"exit_time": "date", "exit_price": "price"})
        ex_df["date"] = _to_local_series(ex_df["date"])
        ret_col = "leveraged_return_pct"
        win  = ex_df[ex_df.get(ret_col, pd.Series(dtype=float)).fillna(0) > 0] \
               if ret_col in ex_df.columns else pd.DataFrame()
        loss = ex_df[ex_df.get(ret_col, pd.Series(dtype=float)).fillna(0) <= 0] \
               if ret_col in ex_df.columns else ex_df
        tt_x = ["date:T", "outcome:N",
                alt.Tooltip("price:Q", format=".4f"),
                alt.Tooltip(f"{ret_col}:Q", format=".2f", title="Return %")]
        for sub, col in [(win, _GREEN), (loss, _RED)]:
            if not sub.empty:
                layers.append(alt.Chart(sub)
                    .mark_point(shape="cross", size=100, strokeWidth=2.5, color=col)
                    .encode(x="date:T", y="price:Q", tooltip=tt_x))

    # Open paper trade TP/SL lines
    for ot in open_trades:
        sl = ot.get("stop_loss"); tp = ot.get("take_profit")
        if sl:
            sl_df = pd.DataFrame({"y":[sl], "label":[f"SL {float(sl):.4f}"]})
            layers += [
                alt.Chart(sl_df).mark_rule(color=_RED, strokeDash=[4,4], strokeWidth=1.3).encode(y="y:Q"),
                alt.Chart(sl_df).mark_text(color=_RED, align="left", dx=4, dy=-6, fontSize=11)
                    .encode(y="y:Q", x=alt.value(4), text="label:N"),
            ]
        if tp:
            tp_df = pd.DataFrame({"y":[tp], "label":[f"TP {float(tp):.4f}"]})
            layers += [
                alt.Chart(tp_df).mark_rule(color=_GREEN, strokeDash=[4,4], strokeWidth=1.3).encode(y="y:Q"),
                alt.Chart(tp_df).mark_text(color=_GREEN, align="left", dx=4, dy=-6, fontSize=11)
                    .encode(y="y:Q", x=alt.value(4), text="label:N"),
            ]

    return (alt.layer(*layers)
            .properties(title=alt.TitleParams(
                f"{symbol} – Paper Trading  ▲ BUY  ▼ SELL  ✕ Exit", **_TITLE), height=300)
            .configure_view(strokeOpacity=0)
            .configure_axis(**_AXIS).configure_title(**_TITLE))


# ── Pre-flight banner + kill switch (Step 6) ────────────────────────────────

def _render_preflight_and_kill_switch() -> None:
    """
    Trading safety header shown at the very top of the page:

      • RED full-width warning if the kill switch is tripped (blocks all new
        entries). "Re-arm" button to clear it.
      • Otherwise: a compact status strip with market-open, account status,
        credentials, and shadow-run count.
      • Always: a "🛑 Kill switch" button that lets the user arm the switch
        (with a confirmation step) and a "Flatten all" emergency button that
        appears when there's at least one open sim or Alpaca paper position.

    Does NOT early-return the page — the user can still view history, charts,
    and configure runs while the switch is tripped. Only NEW entries are
    blocked (enforced inside _run_tick).
    """
    db = _db()
    try:
        ks_state = ks.load(db)
    except Exception:
        ks_state = {"tripped": False}
    tripped = bool(ks_state.get("tripped"))

    # ── Gather quick signals for the status strip ───────────────────────────
    runs       = st.session_state.get(_RUNS) or {}
    shadow_n   = sum(1 for r in runs.values() if r.get("shadow_alpaca"))
    creds_ok   = settings.alpaca.has_paper_credentials()
    acct       = st.session_state.get(_ACCT) or {}
    clk        = st.session_state.get(_CLOCK) or {}

    # Refresh clock on page render (cheap, once per render, only if creds set).
    if creds_ok and not clk.get("ok"):
        clk = _sync_alpaca_clock()

    # ── Tripped — big red warning + re-arm ──────────────────────────────────
    if tripped:
        reason     = ks_state.get("reason") or "no reason given"
        tripped_at = ks_state.get("tripped_at")
        actor      = ks_state.get("actor") or "user"
        when_str   = ""
        if tripped_at:
            try:
                when_str = f" at {pd.Timestamp(tripped_at).strftime('%Y-%m-%d %H:%M UTC')}"
            except Exception:
                when_str = f" at {tripped_at}"
        st.error(
            f"🛑 **KILL SWITCH TRIPPED** — new entries are blocked until re-armed.  \n"
            f"_Reason:_ {reason}  \n"
            f"_Tripped by {actor}{when_str}_"
        )
        col_re, col_flat, _ = st.columns([1, 1, 3])
        with col_re:
            if st.button("✅ Clear kill switch", key="pt_ks_untrip", type="primary"):
                ks.untrip(db, actor="user")
                st.session_state[_KS_CONF] = False
                st.rerun()
        with col_flat:
            if _any_open_positions() and st.button(
                "🧯 Flatten all positions",
                key="pt_flat_all_tripped",
                help="Close every open sim trade AND every Alpaca paper position.",
            ):
                _flat = _flatten_all_positions()
                _msg  = (
                    f"Flattened — sim: {len(_flat['sim_closed'])}, "
                    f"alpaca: {len(_flat['alpaca_closed'])}"
                )
                if _flat["errors"]:
                    st.warning(_msg + "  \n" + " · ".join(_flat["errors"]))
                else:
                    st.success(_msg)
        st.divider()
        return

    # ── Armed (safe) — compact status strip ─────────────────────────────────
    bits: list[str] = []

    # Market open/closed
    if creds_ok and clk.get("ok"):
        if clk.get("is_open"):
            nxt = clk.get("next_close")
            nxt_str = pd.Timestamp(nxt).strftime("%H:%M") if nxt else "?"
            bits.append(f"🟢 Market **open** (closes {nxt_str})")
        else:
            nxt = clk.get("next_open")
            nxt_str = pd.Timestamp(nxt).strftime("%a %H:%M") if nxt else "?"
            bits.append(f"🌙 Market **closed** (opens {nxt_str})")
    elif creds_ok:
        bits.append("⚠️ Market clock unavailable")
    else:
        bits.append("⚪ No Alpaca credentials")

    # Account status (only surfaced if shadow mode is on somewhere)
    if creds_ok and acct.get("ok"):
        status = (acct.get("status") or "unknown").lower()
        if acct.get("trading_blocked") or acct.get("account_blocked"):
            bits.append("🚫 Account blocked")
        elif status == "active":
            eq = acct.get("equity") or 0
            bits.append(f"🏦 Alpaca active · equity ${eq:,.0f}")
        else:
            bits.append(f"🏦 Alpaca status: {status}")
        if acct.get("pattern_day_trader"):
            bits.append(f"🏷 PDT · {acct.get('daytrade_count', 0)} day-trades (5d)")
    elif creds_ok and shadow_n > 0:
        bits.append("🏦 Alpaca account: not yet synced")

    # Shadow summary
    if shadow_n > 0:
        bits.append(f"🔁 Shadow: {shadow_n} run(s)")
    else:
        bits.append("🔁 Shadow: off")

    # Kill-switch state
    bits.append("🟢 Kill switch armed (trading enabled)")

    st.info(" · ".join(bits))

    # ── Action row: kill-switch button (+ flatten if anything open) ─────────
    action_cols = st.columns([1.2, 1.2, 3])
    confirm = bool(st.session_state.get(_KS_CONF))

    with action_cols[0]:
        if not confirm:
            if st.button("🛑 Kill switch", key="pt_ks_arm", type="secondary",
                         help="Halt all new entries. Exits, trailing stops, "
                              "and reconciliation continue to run."):
                st.session_state[_KS_CONF] = True
                st.rerun()
        else:
            if st.button("🛑 Confirm — trip kill switch", key="pt_ks_confirm",
                         type="primary"):
                ks.trip(db, reason="manual from paper-trading page", actor="user")
                st.session_state[_KS_CONF] = False
                st.rerun()

    with action_cols[1]:
        if confirm:
            if st.button("Cancel", key="pt_ks_cancel"):
                st.session_state[_KS_CONF] = False
                st.rerun()
        elif _any_open_positions():
            if st.button(
                "🧯 Flatten all positions",
                key="pt_flat_all",
                help="Close every open sim trade AND every Alpaca paper position.",
            ):
                _flat = _flatten_all_positions()
                _msg  = (
                    f"Flattened — sim: {len(_flat['sim_closed'])}, "
                    f"alpaca: {len(_flat['alpaca_closed'])}"
                )
                if _flat["errors"]:
                    st.warning(_msg + "  \n" + " · ".join(_flat["errors"]))
                else:
                    st.success(_msg)

    if confirm:
        st.warning(
            "⚠️ You are about to halt all new trade entries. Existing positions "
            "will continue to be managed (TP/SL, trailing stops, reconciliation). "
            "Press **Confirm** to trip, or **Cancel** to abort."
        )


def _any_open_positions() -> bool:
    """Cheap check used by the Flatten button — true if we have anything to flatten."""
    try:
        if _open_paper_trades():
            return True
    except Exception:
        pass
    pos = st.session_state.get(_POS) or {}
    if (pos.get("positions") or []):
        return True
    return False


# ── Page ─────────────────────────────────────────────────────────────────────

def render() -> None:
    _init_state()
    _restored_runs = _restore_runs_config()
    _restored_signals = _restore_signals_config()
    render_mode_banner()

    # Pre-flight banner + kill switch sit above everything else so the user
    # sees safety state before configuring runs or reading history.
    _render_preflight_and_kill_switch()

    mode = settings.trading_mode
    if mode == TradingMode.LIVE:
        st.warning("🔴 **Live mode detected.** Switch to Paper mode in Settings to use this page safely.")

    st.title("📝 Paper Trading")
    st.caption(
        "Strategies run **automatically** on live market data — BUY/SELL signals are placed as "
        "paper orders via the risk router; TP/SL (and trailing stops) are enforced on every tick. "
        "Multi-ticker, with live price + RSI charts."
    )
    st.info(
        "**Flow:** Backtester → Forward Test → **Paper Trading** ← you are here → Live  \n"
        "Paper Trading = real-time data + strategy auto-execution + DB persistence (mode=`paper`).  \n"
        "HOLD simply means _no entry this bar_ — the strategy is re-evaluated on every refresh."
    )
    if _restored_runs:
        st.caption("Restored active paper-trading symbols from saved server state after reconnect/refresh.")
    elif _restored_signals:
        st.caption("Restored recent paper-trading signals from saved server state after reconnect/refresh.")
    st.divider()

    # ── Add new ticker run ───────────────────────────────────────────────────
    with st.expander("➕ Add Symbol to Paper Trading", expanded=len(st.session_state[_RUNS]) == 0):
        strategies  = list_strategies()
        strat_names = {s["name"]: s["id"] for s in strategies}

        col1, col2, col3 = st.columns(3)
        with col1:
            new_symbol   = st.text_input("Symbol", value="UVXY", key="pt_new_sym").upper()
            new_interval = st.selectbox("Interval",
                                        ["1m","2m","5m","15m","30m","1h","1d"],
                                        index=0, key="pt_new_interval")
        with col2:
            new_lookback = st.number_input("Warm-up bars", 50, 5000, 2000, 50, key="pt_new_lb")
            new_capital  = st.number_input("Capital / trade ($)", 10.0, value=500.0, key="pt_new_cap")
        with col3:
            new_leverage = st.number_input("Leverage", 1.0, 100.0, 1.0, 0.5, key="pt_new_lev")
            new_max_loss = st.slider("Max capital loss %", 5, 100, 50, key="pt_new_ml")

        c_cost1, c_cost2, c_cost3 = st.columns(3)
        with c_cost1:
            new_spread = st.number_input("Spread % (round-trip)", 0.0, value=0.06,
                                          step=0.01, format="%.2f", key="pt_spread")
        with c_cost2:
            new_slippage = st.number_input("Slippage % (round-trip)", 0.0, value=0.02,
                                            step=0.01, format="%.2f", key="pt_slippage")
        with c_cost3:
            new_commission = st.number_input("Commission / trade ($)", 0.0, value=0.0,
                                              step=0.01, format="%.2f", key="pt_commission")

        new_counter_exit = st.checkbox(
            "Counter-signal exit (close open non-spike trade on opposing signal)",
            value=True,
            key="pt_counter_exit",
            help=(
                "Mirrors the Backtester behaviour: when an open trade is non-spike "
                "and the strategy emits the opposite direction signal, close at that "
                "bar's close (net P&L after costs)."
            ),
        )
        new_equity_aware = st.checkbox(
            "Equity-aware sizing (cap new-trade size at available equity)",
            value=True,
            key="pt_equity_aware",
            help=(
                "Mirrors the Backtester: available equity = capital + realised P&L; "
                "each new entry uses min(capital_per_trade, available_equity). "
                "Trades are skipped if equity is ≤ 0."
            ),
        )

        # ── Shadow mode: mirror every entry to Alpaca paper ────────────────
        _has_paper_creds = settings.alpaca.has_paper_credentials()
        new_shadow_alpaca = st.checkbox(
            "Send to Alpaca paper in parallel (shadow mode)",
            value=False,
            key="pt_shadow_alpaca",
            disabled=not _has_paper_creds,
            help=(
                "When on, every approved local-sim entry is ALSO submitted as a "
                "real order to Alpaca's paper endpoint (no real money). Both "
                "trades are persisted — the sim one with mode='paper', the "
                "Alpaca one with mode='alpaca_paper' — so you can compare "
                "simulated vs. broker-routed fills, fees, and timing. "
                "Counter-signal / manual exits also flatten the Alpaca side."
            ),
        )
        if not _has_paper_creds:
            st.caption(
                "_(Shadow mode disabled — fill in `ALPACA_PAPER_API_KEY` and "
                "`ALPACA_PAPER_SECRET_KEY` in your `.env` to enable.)_"
            )

        # ── Execution logic selector ────────────────────────────────────────
        _policy_opts = available_policies()          # [(name, label), ...]
        _policy_labels = [lbl for _, lbl in _policy_opts]
        _policy_names = [nm for nm, _ in _policy_opts]
        # Default to Alpaca-realistic so paper results track live Alpaca behaviour.
        _default_idx = _policy_names.index("alpaca") if "alpaca" in _policy_names else 0
        st.markdown("**Execution logic**")
        _chosen_label = st.selectbox(
            "Which entry-gate policy to use for this run?",
            _policy_labels,
            index=_default_idx,
            key="pt_exec_logic",
            help=(
                "Classic = the unconstrained logic we had before Alpaca gates "
                "were added. Alpaca-realistic = RTH / PDT / SSR / fractional / "
                "fill-diagnostic applied at entry so paper tracks live trading. "
                "Runs are independent, so you can start two symbols under "
                "different policies and compare side-by-side."
            ),
        )
        new_execution_logic = _policy_names[_policy_labels.index(_chosen_label)]

        _is_alpaca = new_execution_logic == "alpaca"
        if _is_alpaca:
            st.markdown(
                "**Alpaca execution rules** (fine-tune which gates to apply):"
            )
            ac1, ac2, ac3 = st.columns(3)
            with ac1:
                new_enforce_rth = st.checkbox(
                    "Regular Trading Hours only",
                    value=True, key="pt_rth",
                    help="Skip entries outside NYSE RTH (09:30-16:00 ET) and on holidays.",
                )
                new_extended_hrs = st.checkbox(
                    "Allow extended hours (04:00-20:00 ET)",
                    value=False, key="pt_ext_hrs",
                    help=(
                        "If checked AND RTH-only is on, premarket/afterhours entries are "
                        "accepted (Alpaca requires extended_hours=true on the order)."
                    ),
                )
            with ac2:
                new_enforce_pdt = st.checkbox(
                    "PDT rule (margin < $25k)",
                    value=True, key="pt_pdt",
                    help="Block the 4th day-trade in 5 days when account equity < $25k.",
                )
                new_enforce_ssr = st.checkbox(
                    "Short-Sale Restriction",
                    value=True, key="pt_ssr",
                    help="Skip short entries when price is ≥10% below prior trading day's close.",
                )
            with ac3:
                new_enforce_fractional = st.checkbox(
                    "Fractional-share routing",
                    value=True, key="pt_frac",
                    help="Short orders require integer qty ≥ 1 (Alpaca rule).",
                )
                new_fill_diag = st.checkbox(
                    "Fill-timing diagnostic",
                    value=True, key="pt_fill_diag",
                    help=(
                        "Attach bar close/high/low/range to each entry's notes — used later "
                        "to calibrate spread_pct + slippage_pct against real Alpaca fills."
                    ),
                )
        else:
            st.caption(
                "Classic mode — no Alpaca gates are applied (no RTH / PDT / SSR / "
                "fractional / fill-diagnostic). Useful as an unconstrained baseline."
            )
            # Defaults are kept so run_cfg still has these keys; they're ignored
            # by the Classic policy but preserved so the same run can be switched
            # back to Alpaca without losing its settings.
            new_enforce_rth = True
            new_extended_hrs = False
            new_enforce_pdt = True
            new_enforce_ssr = True
            new_enforce_fractional = True
            new_fill_diag = True

        default_strat_name = "Bollinger + RSI (Spike-Aware)"
        default_idx = (list(strat_names.keys()).index(default_strat_name)
                       if default_strat_name in strat_names else 0)
        new_strat_name = st.selectbox("Strategy", list(strat_names.keys()),
                                      index=default_idx, key="pt_new_strat")
        new_strat_id   = strat_names[new_strat_name]
        new_params     = render_strategy_params(
            new_strat_id,
            leverage=new_leverage,
            max_capital_loss_pct=float(new_max_loss),
            symbol=new_symbol,
            source="forward_blend",
            interval=new_interval,
        )

        if st.button("➕ Add & Start", type="primary", key="pt_add"):
            run_cfg = {
                "symbol":       new_symbol,
                "interval":     new_interval,
                "lookback":     new_lookback,
                "capital":      new_capital,
                "leverage":     new_leverage,
                "max_loss":     new_max_loss,
                "spread_pct":   new_spread,
                "slippage_pct": new_slippage,
                "commission":   new_commission,
                "strategy_id":  new_strat_id,
                "params":       dict(new_params),
                "started_at":   datetime.now().isoformat(),
                "active":       True,
                "counter_signal_exit": bool(new_counter_exit),
                "equity_aware_sizing": bool(new_equity_aware),
                "shadow_alpaca":       bool(new_shadow_alpaca),
                # Entry-policy selector (see execution/entry_policy_*.py)
                "execution_logic":    str(new_execution_logic),
                # Alpaca execution rules (consumed only when execution_logic=='alpaca')
                "enforce_rth":        bool(new_enforce_rth),
                "extended_hours":     bool(new_extended_hrs),
                "enforce_pdt":        bool(new_enforce_pdt),
                "enforce_ssr":        bool(new_enforce_ssr),
                "enforce_fractional": bool(new_enforce_fractional),
                "fill_diagnostic":    bool(new_fill_diag),
            }
            st.session_state[_RUNS][new_symbol] = run_cfg
            _persist_runs_config()
            _run_tick(new_symbol, run_cfg)   # prime cache + possibly first entry
            st.rerun()

    runs = st.session_state[_RUNS]
    if not runs:
        st.info("No active paper trading runs. Add a symbol above to begin.")
        return

    # ── Global controls ──────────────────────────────────────────────────────
    gcol1, gcol2, gcol3 = st.columns(3)
    refresh_all = gcol1.button("🔄 Refresh All Symbols", type="primary", key="pt_refresh_all")
    auto        = gcol2.checkbox("Auto-refresh (60s)", value=True, key="pt_auto")
    if gcol3.button("🗑️ Clear All", key="pt_clear_all"):
        for k in [_RUNS, _CACHE, _SIGNALS, _TRAIL]:
            st.session_state[k] = {} if isinstance(st.session_state[k], dict) else []
        _persist_runs_config()
        _persist_signals_config()
        st.rerun()

    if refresh_all or auto:
        # ── Alpaca-paper status poll (Step 4) ────────────────────────────────
        # Runs BEFORE per-symbol ticks so any terminal-state transitions
        # (filled, rejected, canceled, expired) are already reflected in the
        # DB when the UI renders the summary + history tables.
        # No-op when no shadow-mode trades are tracked or creds missing.
        # Refresh the market clock on every tick so the pre-flight banner
        # stays current across auto-refreshes (cheap — one HTTP call).
        if settings.alpaca.has_paper_credentials():
            try:
                _sync_alpaca_clock()
            except Exception:
                pass

        _has_shadow = any(r.get("shadow_alpaca") for r in runs.values())

        # TradingStream auto-start (Step 7): when shadow mode is active on
        # any run we start the WS listener so fills land in the DB within
        # ~100ms instead of waiting for the next REST poll. Idempotent —
        # safe to call every tick.
        if _has_shadow and settings.alpaca.has_paper_credentials():
            try:
                ts.start_trading_stream(paper=True)
            except Exception as _exc:
                log.warning(f"TradingStream start skipped: {_exc}")

        if _has_shadow and settings.alpaca.has_paper_credentials():
            try:
                st.session_state[_SYNC] = _sync_alpaca_paper_orders()
            except Exception as _exc:
                st.session_state[_SYNC] = {
                    "polled": 0, "updated": 0, "errors": 1,
                    "filled": [], "rejected": [],
                    "at": datetime.utcnow(),
                    "error": str(_exc),
                }

            # ── Alpaca account + positions snapshot (Step 5) ─────────────────
            # Also runs before per-symbol ticks so the reconciliation can
            # detect bracket TP/SL closes that happened since last refresh.
            try:
                _acct, _pos = _sync_alpaca_account_and_positions()
                if _pos and _pos.get("ok"):
                    _reconcile_positions_against_db(_pos)
            except Exception as _exc:
                st.session_state[_RECON] = {
                    "closed_by_broker": [], "orphan_positions": [],
                    "tracked_open": [], "errors": [str(_exc)],
                    "at": datetime.utcnow(),
                }

        for symbol, run in runs.items():
            if run.get("active"):
                _run_tick(symbol, run)
        if not auto:
            st.rerun()

    # ── Alpaca sync indicator ────────────────────────────────────────────────
    _sync_info = st.session_state.get(_SYNC) or {}
    if _sync_info.get("at"):
        _sync_at_local = _to_local(_sync_info["at"])
        _sync_line = (
            f"📡 Alpaca sync: polled **{_sync_info.get('polled',0)}** shadow order(s) · "
            f"updated **{_sync_info.get('updated',0)}** · "
            f"errors **{_sync_info.get('errors',0)}**"
        )
        if _sync_info.get("filled"):
            _sync_line += f" · filled: {', '.join(_sync_info['filled'])}"
        if _sync_info.get("rejected"):
            _sync_line += f" · rejected: {', '.join(_sync_info['rejected'])}"
        _sync_line += f" · at {pd.Timestamp(_sync_at_local).strftime('%H:%M:%S')}"
        if _sync_info.get("error"):
            st.warning(_sync_line + f"  \n⚠️ {_sync_info['error']}")
        else:
            st.caption(_sync_line)

    # ── TradingStream (WS) indicator (Step 7) ───────────────────────────────
    # Shown only when the stream has been started at least once. The stream
    # runs in a background daemon thread, so its status lives outside
    # session_state — we pull it fresh each render.
    try:
        _stream_status = ts.get_stream_status()
    except Exception:
        _stream_status = {"running": False}
    if _stream_status.get("running") or _stream_status.get("events_received"):
        dot = "🟢" if _stream_status.get("connected") else "🔴"
        _ws_line = (
            f"{dot} TradingStream WS: "
            f"{'connected' if _stream_status.get('connected') else 'disconnected'} · "
            f"events **{_stream_status.get('events_received',0)}**"
        )
        _last_at = _stream_status.get("last_event_at")
        if _last_at:
            _ws_line += (
                f" · last: {_stream_status.get('last_event_summary','?')} "
                f"@ {pd.Timestamp(_to_local(_last_at)).strftime('%H:%M:%S')}"
            )
        if _stream_status.get("last_error"):
            st.warning(
                _ws_line + f"  \n⚠️ {_stream_status['last_error']}"
            )
        else:
            st.caption(_ws_line)

    # ── Reconciliation banner (positions closed by the broker) ──────────────
    _recon = st.session_state.get(_RECON) or {}
    if _recon.get("closed_by_broker"):
        _closed_lines = [
            f"**{c['symbol']}** @ {c['exit_price']:.4f} "
            f"({c['source']}, P&L {c.get('pnl', 0):+.2f})"
            for c in _recon["closed_by_broker"]
        ]
        st.success(
            "🔄 Alpaca reconciliation closed: " + " · ".join(_closed_lines)
        )
    if _recon.get("orphan_positions"):
        st.warning(
            "⚠️ Alpaca holds positions with no matching DB trade: "
            f"{', '.join(_recon['orphan_positions'])}. These were likely "
            "opened manually in Alpaca's UI or before shadow mode was enabled. "
            "They are NOT being managed by this app."
        )
    if _recon.get("errors"):
        st.caption(
            f"_Reconciliation notes: {' · '.join(_recon['errors'])}_"
        )

    # ── Alpaca paper account snapshot ───────────────────────────────────────
    _acct_snap = st.session_state.get(_ACCT) or {}
    _pos_snap  = st.session_state.get(_POS)  or {}
    if _acct_snap.get("ok"):
        with st.expander("🏦 Alpaca Paper Account (live)", expanded=False):
            ac_cols = st.columns(4)
            ac_cols[0].metric(
                "Equity",
                f"${(_acct_snap.get('equity') or 0):,.2f}",
                delta=(
                    f"{((_acct_snap.get('equity') or 0) - (_acct_snap.get('last_equity') or 0)):+,.2f}"
                    if _acct_snap.get("last_equity") else None
                ),
            )
            ac_cols[1].metric("Cash",         f"${(_acct_snap.get('cash') or 0):,.2f}")
            ac_cols[2].metric("Buying Power", f"${(_acct_snap.get('buying_power') or 0):,.2f}")
            ac_cols[3].metric("Day-trades (5d)", f"{_acct_snap.get('daytrade_count', 0)}")

            status_bits = []
            status_bits.append(f"status: **{_acct_snap.get('status','?')}**")
            if _acct_snap.get("pattern_day_trader"):
                status_bits.append("🏷 PDT flag ON")
            if _acct_snap.get("trading_blocked"):
                status_bits.append("🚫 trading blocked")
            if _acct_snap.get("account_blocked"):
                status_bits.append("🚫 account blocked")
            status_bits.append(f"id: `{(_acct_snap.get('id') or '')[:8]}…`")
            _fetched = _acct_snap.get("fetched_at")
            if _fetched:
                status_bits.append(
                    f"synced {pd.Timestamp(_to_local(_fetched)).strftime('%H:%M:%S')}"
                )
            st.caption(" · ".join(status_bits))

            # Alpaca positions table
            positions = _pos_snap.get("positions") or []
            if positions:
                pos_df = pd.DataFrame(positions)
                # Order columns for readability
                _col_order = [
                    "symbol", "side", "qty", "avg_entry_price", "current_price",
                    "market_value", "unrealized_pl", "unrealized_plpc", "cost_basis",
                ]
                pos_df = pos_df[[c for c in _col_order if c in pos_df.columns]]
                st.markdown("**Open positions at Alpaca:**")
                st.dataframe(pos_df, width='stretch')
            else:
                st.caption("_No open positions at Alpaca paper._")
    elif _acct_snap.get("error"):
        st.warning(
            f"⚠️ Alpaca account sync failed: {_acct_snap['error']}"
        )

    st.divider()

    # Pull all paper trades once (used by summary, equity curve, tabs) ───────
    try:
        all_paper = _db().get_trades(mode="paper")
    except Exception:
        all_paper = []
    closed_paper = [t for t in all_paper if t.get("outcome") != "Open"]

    # ── Summary table ────────────────────────────────────────────────────────
    st.subheader("📊 Active Symbols")
    summary_rows = []
    for sym, run in runs.items():
        prices  = st.session_state[_CACHE].get(sym)
        last_px = float(prices.iloc[-1]["close"]) if prices is not None and not prices.empty else None
        open_ts = _open_paper_trades(sym)
        open_str = (f"{open_ts[0]['direction']} @ {float(open_ts[0]['entry_price']):.4f}"
                    if open_ts else "None")
        last_sig = run.get("_last_signal") or {}
        rsi_val  = last_sig.get("rsi")
        summary_rows.append({
            "Symbol":    sym,
            "Strategy":  run["strategy_id"],
            "Interval":  run["interval"],
            "Logic":     "Alpaca" if run.get("execution_logic", "alpaca") == "alpaca" else "Classic",
            "Shadow":    "🟢 Alpaca-paper" if run.get("shadow_alpaca") else "—",
            "Last Price":  f"{last_px:.4f}" if last_px else "—",
            "Last RSI":    f"{rsi_val:.1f}" if rsi_val is not None else "—",
            "Last Signal": last_sig.get("action", "—"),
            "Open Position": open_str,
            "Costs":   f"{float(run.get('spread_pct',0)):.2f}% + {float(run.get('slippage_pct',0)):.2f}%",
            "Active":    "✅" if run.get("active") else "⏸",
        })
    st.dataframe(pd.DataFrame(summary_rows), width='stretch')

    # ── Portfolio equity curve (all symbols) ─────────────────────────────────
    _starting_capital = float(sum(float(r.get("capital", 0) or 0) for r in runs.values()))
    _total_pnl = float(sum(float(t.get("pnl") or 0) for t in closed_paper))
    _current_equity = _starting_capital + _total_pnl
    st.subheader("📈 Portfolio Equity")
    eq_cols = st.columns(3)
    eq_cols[0].metric("Starting Capital",  f"${_starting_capital:,.2f}")
    eq_cols[1].metric("Realised P&L",      f"${_total_pnl:+,.2f}")
    eq_cols[2].metric("Current Equity",    f"${_current_equity:,.2f}",
                       delta=f"{(_total_pnl/_starting_capital*100 if _starting_capital else 0):+.2f}%")
    _earliest_start = None
    for r in runs.values():
        sa = r.get("started_at")
        if sa:
            try:
                ts = pd.Timestamp(sa)
                if _earliest_start is None or ts < _earliest_start:
                    _earliest_start = ts
            except Exception:
                pass
    st.altair_chart(
        _equity_curve_chart(closed_paper, _starting_capital,
                             title="Portfolio Equity (all symbols)",
                             anchor_time=_earliest_start),
        width='stretch',
    )
    st.divider()

    # ── Per-symbol tabs ──────────────────────────────────────────────────────
    sym_list = list(runs.keys())
    tabs = st.tabs([f"{'🟢' if runs[s].get('active') else '⏸'} {s}" for s in sym_list])

    for tab, symbol in zip(tabs, sym_list):
        with tab:
            run    = runs[symbol]
            prices = st.session_state[_CACHE].get(symbol)

            # Per-symbol controls
            c1, c2, c3, c4 = st.columns(4)
            if c1.button(f"🔄 Refresh {symbol}", key=f"pt_ref_{symbol}"):
                _run_tick(symbol, run)
                st.rerun()
            if c2.button(f"{'⏸ Pause' if run.get('active') else '▶ Resume'}",
                          key=f"pt_toggle_{symbol}"):
                st.session_state[_RUNS][symbol]["active"] = not run.get("active", True)
                _persist_runs_config()
                st.rerun()
            # Manual close of any open position
            open_t_list = _open_paper_trades(symbol)
            if c3.button(f"❌ Close Open Trade", key=f"pt_close_{symbol}",
                          disabled=not open_t_list):
                if prices is not None and not prices.empty and open_t_list:
                    target = open_t_list[0]
                    xp     = float(prices.iloc[-1]["close"])
                    _close_trade(
                        target,
                        exit_price=xp,
                        exit_time=pd.Timestamp(prices.iloc[-1]["date"]).to_pydatetime(),
                        outcome=TradeOutcome.SIGNAL_EXIT,
                        spread_pct=run.get("spread_pct", 0.0),
                        slippage_pct=run.get("slippage_pct", 0.0),
                        commission=run.get("commission", 0.0),
                        notes="Manually closed.",
                    )
                    # Shadow mode: flatten Alpaca-paper position too
                    if run.get("shadow_alpaca") and settings.alpaca.has_paper_credentials():
                        try:
                            _shadow_router = OrderRouter(
                                risk_manager=RiskManager(settings.risk)
                            )
                            _shadow_router.close_position_alpaca_paper(symbol)
                        except Exception as _exc:
                            st.warning(
                                f"⚠️ {symbol}: shadow-mode Alpaca close failed — {_exc}"
                            )
                    st.rerun()
            if c4.button(f"🗑️ Remove {symbol}", key=f"pt_remove_{symbol}"):
                del st.session_state[_RUNS][symbol]
                st.session_state[_CACHE].pop(symbol, None)
                st.session_state[_SIGNALS] = [
                    s for s in st.session_state[_SIGNALS]
                    if s.get("symbol") != symbol
                ]
                _persist_runs_config()
                _persist_signals_config()
                st.rerun()

            if prices is None or prices.empty:
                st.info(f"Click **🔄 Refresh {symbol}** to fetch first bars.")
                continue

            last_close = float(prices["close"].iloc[-1])
            st.success(f"**{symbol}** — {len(prices)} bars · last close: **{last_close:.4f}**")

            # ── Current signal + open position status ────────────────────────
            last_sig = run.get("_last_signal") or {}
            if last_sig:
                rsi_val = last_sig.get("rsi")
                conf_val = last_sig.get("confidence") or 0
                line = (f"Latest signal: **{last_sig.get('action','—')}** · "
                        f"conf {float(conf_val):.0%}")
                if rsi_val is not None:
                    line += f" · RSI {rsi_val:.1f}"
                if last_sig.get("regime"):
                    line += f" · regime `{last_sig['regime']}`"
                st.caption(line)
                reason = last_sig.get("verdict_reason")
                if reason:
                    st.caption(f"_why:_ {reason}")

            if open_t_list:
                ot = open_t_list[0]
                curr = last_close
                ep   = float(ot["entry_price"]); lev = float(ot["leverage"])
                d    = ot["direction"]
                unreal = (curr - ep) / ep * (1 if d == "Long" else -1) * lev * 100
                col = "green" if unreal >= 0 else "red"
                tp_str = f"{float(ot['take_profit']):.4f}" if ot.get("take_profit") else "—"
                sl_str = f"{float(ot['stop_loss']):.4f}"   if ot.get("stop_loss")   else "—"
                st.markdown(
                    f'<div style="border:1px solid #2a2d3e;border-radius:8px;'
                    f'padding:8px 14px;margin:6px 0;">'
                    f'<b>Open paper position:</b> {d} @ <code>{ep:.4f}</code> · '
                    f'SL <code>{sl_str}</code> · TP <code>{tp_str}</code> · '
                    f'Unrealised: <span style="color:{col};font-weight:bold">'
                    f'{unreal:+.2f}%</span></div>',
                    unsafe_allow_html=True,
                )

            # Local-tz display copy used by both price and RSI charts
            prices_local = prices.copy()
            prices_local["date"] = _to_local_series(prices_local["date"])

            st.markdown(f"#### Live Charts · {symbol}")
            st.caption("These charts belong to Paper Trading for the active symbol and are independent from the Backtester.")

            # ── Live price chart ─────────────────────────────────────────────
            st.altair_chart(
                _paper_price_chart(prices_local, symbol,
                                    st.session_state[_SIGNALS],
                                    open_t_list, closed_paper),
                width='stretch',
            )

            # ── Live RSI chart ───────────────────────────────────────────────
            if run["strategy_id"] in ("rsi_threshold", "atr_rsi", "vwap_rsi",
                                        "bollinger_rsi", "ema_trend_rsi"):
                p_cfg = run.get("params", {})
                rsi_p = int(p_cfg.get("rsi_period", 9))
                try:
                    buy_lvls = [float(x) for x in
                                str(p_cfg.get("buy_levels", "30")).replace(";", ",").split(",")
                                if x.strip()] or [30]
                    sell_lvls = [float(x) for x in
                                  str(p_cfg.get("sell_levels", "70")).replace(";", ",").split(",")
                                  if x.strip()] or [70]
                except Exception:
                    buy_lvls, sell_lvls = [30], [70]
                st.altair_chart(
                    rsi_chart(prices_local, rsi_p, buy_lvls, sell_lvls)
                        .properties(title=alt.TitleParams(f"{symbol} – Paper Trading RSI ({rsi_p})", **_TITLE))
                        .configure_view(strokeOpacity=0)
                        .configure_axis(**_AXIS)
                        .configure_title(**_TITLE),
                    width='stretch',
                )

            # ── Per-symbol equity curve (always visible) ─────────────────────
            sym_closed = [t for t in all_paper if t.get("symbol") == symbol and t.get("outcome") != "Open"]
            sym_pnl = float(sum(float(t.get("pnl") or 0) for t in sym_closed))
            m_cols = st.columns(3)
            m_cols[0].metric("Start",      f"${float(run['capital']):,.2f}")
            m_cols[1].metric("Realised",   f"${sym_pnl:+,.2f}")
            m_cols[2].metric("Equity",     f"${float(run['capital']) + sym_pnl:,.2f}")
            try:
                _sym_anchor = pd.Timestamp(run.get("started_at")) if run.get("started_at") else None
            except Exception:
                _sym_anchor = None
            st.altair_chart(
                _equity_curve_chart(sym_closed, float(run["capital"]),
                                     title=f"{symbol} – Paper Trading Equity",
                                     anchor_time=_sym_anchor),
                width='stretch',
            )

            # ── Recent signals / trades expanders ────────────────────────────
            with st.expander("📡 Recent Signals", expanded=False):
                sym_sigs = [s for s in st.session_state[_SIGNALS]
                            if s.get("symbol") == symbol]
                if sym_sigs:
                    sig_df = pd.DataFrame(sym_sigs).copy()
                    if "date" in sig_df.columns:
                        sig_df["date"] = pd.to_datetime(sig_df["date"], errors="coerce")
                        sig_df = sig_df.sort_values("date", ascending=False)
                    st.dataframe(sig_df, width='stretch')
                else:
                    st.info("No signals yet — press Refresh.")

            with st.expander("📋 Trades for this symbol", expanded=False):
                sym_trades = [t for t in all_paper if t.get("symbol") == symbol]
                if sym_trades:
                    st.dataframe(pd.DataFrame(sym_trades), width='stretch')
                else:
                    st.info("No paper trades for this symbol yet.")

    # ── Closed paper history (all symbols) ───────────────────────────────────
    st.divider()
    with st.expander("📋 Paper Trade History (all symbols)", expanded=False):
        if closed_paper:
            hist_df = pd.DataFrame(closed_paper).copy()
            if "entry_time" in hist_df.columns:
                hist_df["entry_time"] = _to_local_series(hist_df["entry_time"])
            if "exit_time" in hist_df.columns:
                hist_df["exit_time"] = _to_local_series(hist_df["exit_time"])
            st.dataframe(hist_df, width='stretch')
        else:
            st.info("No closed paper trades yet.")

    # ── Auto-refresh loop ────────────────────────────────────────────────────
    auto_enabled = auto and any(r.get("active") for r in runs.values())
    min_interval = 60
    if auto_enabled:
        min_interval = min(
            int(_interval_td(r["interval"]).total_seconds())
            for r in runs.values() if r.get("active")
        )
    render_autorefresh_timer(
        auto_enabled,
        max(min_interval, 60),
        key="paper_trading_refresh",
    )

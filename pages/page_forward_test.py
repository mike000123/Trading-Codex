"""
pages/page_forward_test.py
───────────────────────────
Forward Test — live data, simulated orders, no broker.
Supports multiple tickers simultaneously.
All trades persisted to DB with mode="forward_test".
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from typing import Optional

import altair as alt
import pandas as pd
import streamlit as st

from config.settings import settings
from core.models import Direction, SignalAction
from data.ingestion import load_forward_blended_data, prepare_strategy_data
from db.database import Database
from execution.entry_policy_base import (
    EntryContext,
    available_policies,
    get_policy,
)
# Importing these registers both policies with the factory.
import execution.entry_policy_classic  # noqa: F401
import execution.entry_policy_alpaca   # noqa: F401
from risk.manager import RiskManager
from strategies import list_strategies, get_strategy
from ui.components import render_mode_banner, render_strategy_params, render_metrics_row
from ui.charts import rsi_chart

_GREEN  = "#26a69a"
_RED    = "#ef5350"
_BLUE   = "#4a9eff"
_GOLD   = "#ffd54f"
_GREY   = "#9e9eb8"
_AXIS   = dict(gridColor="#2a2d3e", labelColor="#d0d4f0", titleColor="#d0d4f0",
               labelFontSize=12, titleFontSize=13)
_TITLE  = dict(color="#e8eaf6", fontSize=14, fontWeight="bold")

# ── Shared session state (read by page_portfolio) ─────────────────────────────
# ft_active_runs : dict[symbol → run_config_dict]
# ft_open_trades : dict[symbol → open_trade_dict | None]
# ft_all_signals : list[signal_row_dict]
# ft_prices_cache: dict[symbol → pd.DataFrame]
_RUNS     = "ft_active_runs"
_OPEN     = "ft_open_trades"
_SIGNALS  = "ft_all_signals"
_CACHE    = "ft_prices_cache"
_EQUITY   = "ft_equity"   # dict[symbol → list[{time, equity, pnl}]]


def _db() -> Database:
    return Database(settings.db_path)


def _init_state() -> None:
    for k, v in [(_RUNS, {}), (_OPEN, {}), (_SIGNALS, []),
                  (_CACHE, {}), (_EQUITY, {})]:
        if k not in st.session_state:
            st.session_state[k] = v


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


def _leveraged_ret(entry: float, exit_p: float,
                   leverage: float, direction: str) -> float:
    raw = (exit_p - entry) / entry
    if direction == "Short":
        raw = -raw
    return raw * leverage * 100


def _latest_atr(prices: pd.DataFrame, period: int = 14) -> Optional[float]:
    if len(prices) < 2:
        return None
    high = prices["high"].astype(float)
    low = prices["low"].astype(float)
    close = prices["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(period, min_periods=1).mean().iloc[-1]
    return float(atr) if pd.notna(atr) else None


def _check_exit(trade: dict, bar: pd.Series, atr_value: Optional[float] = None) -> dict:
    hi, lo = float(bar["high"]), float(bar["low"])
    tp, sl = trade.get("take_profit"), trade["stop_loss"]
    ep, lev, d = trade["entry_price"], trade["leverage"], trade["direction"]

    if trade.get("trail_kind"):
        trade["trail_bars"] = int(trade.get("trail_bars", 0)) + 1
        if trade["trail_bars"] > int(trade.get("trail_grace", 0)):
            hard_sl = float(trade.get("hard_stop_loss", sl))
            if d == "Short":
                trade["trail_best"] = min(float(trade.get("trail_best", ep)), lo)
                if trade.get("trail_kind") == "pct":
                    candidate_sl = trade["trail_best"] * (1 + float(trade["trail_value"]) / 100)
                elif trade.get("trail_kind") == "atr" and atr_value is not None:
                    candidate_sl = trade["trail_best"] + float(trade["trail_value"]) * atr_value
                elif trade.get("trail_kind") == "giveback":
                    profit_move = max(ep - trade["trail_best"], 0.0)
                    profit_move_pct = profit_move / ep * 100 if ep > 0 else 0.0
                    if profit_move_pct >= float(trade.get("trail_min_profit_pct", 0.0)):
                        candidate_sl = trade["trail_best"] + float(trade["trail_value"]) * profit_move
                    else:
                        candidate_sl = hard_sl
                else:
                    candidate_sl = hard_sl
                trade["stop_loss"] = min(candidate_sl, hard_sl)
            else:
                trade["trail_best"] = max(float(trade.get("trail_best", ep)), hi)
                if trade.get("trail_kind") == "pct":
                    candidate_sl = trade["trail_best"] * (1 - float(trade["trail_value"]) / 100)
                elif trade.get("trail_kind") == "atr" and atr_value is not None:
                    candidate_sl = trade["trail_best"] - float(trade["trail_value"]) * atr_value
                elif trade.get("trail_kind") == "giveback":
                    profit_move = max(trade["trail_best"] - ep, 0.0)
                    profit_move_pct = profit_move / ep * 100 if ep > 0 else 0.0
                    if profit_move_pct >= float(trade.get("trail_min_profit_pct", 0.0)):
                        candidate_sl = trade["trail_best"] - float(trade["trail_value"]) * profit_move
                    else:
                        candidate_sl = hard_sl
                else:
                    candidate_sl = hard_sl
                trade["stop_loss"] = max(candidate_sl, hard_sl)
            sl = trade["stop_loss"]

    hit_sl = hi >= sl if d == "Short" else lo <= sl
    hit_tp = (lo <= tp if d == "Short" else hi >= tp) if tp else False
    if hit_sl and hit_tp:
        trade.update({"outcome": "Ambiguous candle", "exit_time": bar["date"]})
    elif hit_sl:
        ret = _leveraged_ret(ep, sl, lev, d)
        outcome = "Trail stop" if trade.get("trail_kind") else "SL hit"
        gross_pnl = trade["capital"] * ret / 100
        net_pnl = gross_pnl - _trade_cost(trade)
        trade.update({"outcome": outcome, "exit_price": sl,
                      "exit_time": bar["date"], "leveraged_return_%": ret,
                      "gross_pnl": round(gross_pnl, 2),
                      "cost": round(_trade_cost(trade), 2),
                      "pnl": round(net_pnl, 2)})
    elif hit_tp:
        ret = _leveraged_ret(ep, tp, lev, d)
        gross_pnl = trade["capital"] * ret / 100
        net_pnl = gross_pnl - _trade_cost(trade)
        trade.update({"outcome": "TP hit", "exit_price": tp,
                      "exit_time": bar["date"], "leveraged_return_%": ret,
                      "gross_pnl": round(gross_pnl, 2),
                      "cost": round(_trade_cost(trade), 2),
                      "pnl": round(net_pnl, 2)})
    return trade


def _trade_cost(trade: dict) -> float:
    return (
        float(trade.get("capital", 0.0) or 0.0)
        * (float(trade.get("spread_pct", 0.0) or 0.0) + float(trade.get("slippage_pct", 0.0) or 0.0))
        / 100.0
        + float(trade.get("commission", 0.0) or 0.0)
    )


def _save_trade_to_db(trade: dict, symbol: str, strategy_id: str) -> None:
    """Persist a closed forward-test trade to the database."""
    from core.models import TradeRecord, Direction as Dir, TradeOutcome
    outcome_map = {
        "SL hit": TradeOutcome.STOP_LOSS,
        "TP hit": TradeOutcome.TAKE_PROFIT,
        "Trail stop": TradeOutcome.TRAIL_STOP,
        "Manual close": TradeOutcome.SIGNAL_EXIT,
        "Ambiguous candle": TradeOutcome.AMBIGUOUS,
    }
    try:
        rec = TradeRecord(
            id               = trade.get("id", str(uuid.uuid4())[:8]),
            symbol           = symbol,
            direction        = Dir(trade["direction"]),
            entry_price      = trade["entry_price"],
            take_profit      = trade.get("take_profit"),
            stop_loss        = trade["stop_loss"],
            leverage         = trade["leverage"],
            capital_allocated= trade["capital"],
            entry_time       = pd.Timestamp(trade["entry_time"]).to_pydatetime(),
            mode             = "forward_test",
            strategy_id      = strategy_id,
            exit_price       = trade.get("exit_price"),
            exit_time        = (pd.Timestamp(trade["exit_time"]).to_pydatetime()
                                if trade.get("exit_time") else None),
            outcome          = outcome_map.get(trade.get("outcome", ""), TradeOutcome.OPEN),
            leveraged_return_pct = trade.get("leveraged_return_%"),
            pnl              = trade.get("pnl"),
            notes            = f"Forward test | strategy={strategy_id}",
        )
        _db().save_trade(rec)
    except Exception as e:
        pass  # don't crash the UI on DB errors


# ── Charts ────────────────────────────────────────────────────────────────────

def _price_chart(prices: pd.DataFrame, closed_trades: list,
                 open_trade: Optional[dict], signals: list,
                 symbol: str) -> alt.LayerChart:
    base = (alt.Chart(prices).mark_line(color=_BLUE, strokeWidth=1.4)
            .encode(x=alt.X("date:T", title="Date / Time", axis=alt.Axis(**_AXIS)),
                    y=alt.Y("close:Q", title="Price", scale=alt.Scale(zero=False),
                            axis=alt.Axis(**_AXIS)),
                    tooltip=["date:T", alt.Tooltip("close:Q", format=".4f")]))
    layers = [base]

    # Signal triangles
    sig_rows = [s for s in signals if s.get("symbol") == symbol]
    if sig_rows:
        sdf   = pd.DataFrame(sig_rows)
        buys  = sdf[sdf["action"] == "BUY"].copy()
        sells = sdf[sdf["action"] == "SELL"].copy()
        tt    = ["date:T", "action:N", alt.Tooltip("close:Q", format=".4f")]
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

    # Closed trade exits
    sym_closed = [t for t in closed_trades if t.get("symbol") == symbol
                  and t.get("exit_price")]
    if sym_closed:
        ex_df = pd.DataFrame(sym_closed).rename(
            columns={"exit_time": "date", "exit_price": "price"})
        ex_df["date"] = pd.to_datetime(ex_df["date"])
        win  = ex_df[ex_df.get("leveraged_return_%", pd.Series(dtype=float)).fillna(0) > 0] \
               if "leveraged_return_%" in ex_df.columns else pd.DataFrame()
        loss = ex_df[ex_df.get("leveraged_return_%", pd.Series(dtype=float)).fillna(0) <= 0] \
               if "leveraged_return_%" in ex_df.columns else ex_df
        tt_x = ["date:T", "outcome:N",
                alt.Tooltip("price:Q", format=".4f"),
                alt.Tooltip("leveraged_return_%:Q", format=".2f", title="Return %")]
        for sub, col in [(win, _GREEN), (loss, _RED)]:
            if not sub.empty:
                layers.append(alt.Chart(sub)
                    .mark_point(shape="cross", size=100, strokeWidth=2.5, color=col)
                    .encode(x="date:T", y="price:Q", tooltip=tt_x))

    # Open trade SL/TP lines
    if open_trade:
        sl = open_trade.get("stop_loss")
        tp = open_trade.get("take_profit")
        if sl:
            sl_df = pd.DataFrame({"y": [sl], "label": [f"SL {sl:.4f}"]})
            layers += [
                alt.Chart(sl_df).mark_rule(color=_RED, strokeDash=[4,4], strokeWidth=1.3).encode(y="y:Q"),
                alt.Chart(sl_df).mark_text(color=_RED, align="left", dx=4, dy=-6, fontSize=11)
                    .encode(y="y:Q", x=alt.value(4), text="label:N"),
            ]
        if tp:
            tp_df = pd.DataFrame({"y": [tp], "label": [f"TP {tp:.4f}"]})
            layers += [
                alt.Chart(tp_df).mark_rule(color=_GREEN, strokeDash=[4,4], strokeWidth=1.3).encode(y="y:Q"),
                alt.Chart(tp_df).mark_text(color=_GREEN, align="left", dx=4, dy=-6, fontSize=11)
                    .encode(y="y:Q", x=alt.value(4), text="label:N"),
            ]

    return (alt.layer(*layers)
            .properties(title=alt.TitleParams(
                f"{symbol} – Forward Test  ▲ BUY  ▼ SELL  ✕ Exit", **_TITLE), height=300)
            .configure_view(strokeOpacity=0)
            .configure_axis(**_AXIS).configure_title(**_TITLE))


def _equity_chart(eq_history: list, starting_capital: float,
                  symbol: str) -> alt.LayerChart:
    if len(eq_history) < 2:
        return alt.Chart(pd.DataFrame()).mark_line()
    df   = pd.DataFrame(eq_history)
    base = (alt.Chart(pd.DataFrame({"y": [starting_capital]}))
            .mark_rule(color=_GREY, strokeDash=[4,4], strokeWidth=1).encode(y="y:Q"))
    area = (alt.Chart(df)
            .mark_area(line={"color": _BLUE, "strokeWidth": 2},
                       color=alt.Gradient(gradient="linear",
                           stops=[alt.GradientStop(color="rgba(74,158,255,0.2)", offset=0),
                                  alt.GradientStop(color="rgba(74,158,255,0.0)", offset=1)],
                           x1=1, x2=1, y1=1, y2=0))
            .encode(x=alt.X("time:T", title="Time", axis=alt.Axis(**_AXIS)),
                    y=alt.Y("equity:Q", title="Equity ($)", scale=alt.Scale(zero=False),
                            axis=alt.Axis(**_AXIS)),
                    tooltip=["time:T",
                             alt.Tooltip("equity:Q", format="$,.2f"),
                             alt.Tooltip("pnl:Q",    format="+$,.2f", title="P&L")]))
    dots = (alt.Chart(df).mark_point(size=50, filled=True)
            .encode(x="time:T", y="equity:Q",
                    color=alt.condition(alt.datum.pnl > 0,
                                        alt.value(_GREEN), alt.value(_RED)),
                    tooltip=["time:T",
                             alt.Tooltip("equity:Q", format="$,.2f"),
                             alt.Tooltip("pnl:Q",    format="+$,.2f", title="P&L")]))
    return (alt.layer(base, area, dots)
            .properties(title=alt.TitleParams(f"{symbol} – Equity", **_TITLE), height=220)
            .configure_view(strokeOpacity=0).configure_axis(**_AXIS).configure_title(**_TITLE))


# ── Core fetch + evaluate logic (called per symbol per refresh) ───────────────

def _run_tick(symbol: str, run: dict, closed_trades_acc: list) -> None:
    """Fetch latest bar, check exits, check entries, store results."""
    try:
        prices = _fetch(symbol, run["interval"], run["lookback"])
        st.session_state[_CACHE][symbol] = prices
    except Exception as e:
        st.warning(f"⚠️ {symbol}: data fetch failed — {e}")
        return

    latest = prices.iloc[-1]
    latest_ts = latest["date"]

    # Check open trade exit
    open_trade = st.session_state[_OPEN].get(symbol)
    if open_trade:
        open_trade = _check_exit(open_trade, latest, atr_value=_latest_atr(prices))
        if open_trade.get("outcome") not in (None, "Open"):
            closed_trades_acc.append(open_trade)
            _save_trade_to_db(open_trade, symbol, run["strategy_id"])
            # Update equity
            pnl = open_trade.get("pnl", 0)
            prev_eq = (st.session_state[_EQUITY].get(symbol, [{"equity": run["capital"]}])[-1]["equity"])
            st.session_state[_EQUITY].setdefault(symbol, []).append(
                {"time": latest_ts, "equity": round(prev_eq + pnl, 2), "pnl": round(pnl, 2)})
            st.session_state[_OPEN][symbol] = None
            open_trade = None

    # Generate signal
    if open_trade is None:
        cls      = get_strategy(run["strategy_id"])
        strategy = cls(params=run["params"])
        prepared_prices = prepare_strategy_data(
            prices,
            strategy,
            primary_symbol=symbol,
            source="forward_blend",
            interval=run["interval"],
            start=prices["date"].min() if not prices.empty else None,
            end=prices["date"].max() if not prices.empty else None,
        )
        st.session_state[_CACHE][symbol] = prepared_prices
        latest = prepared_prices.iloc[-1]
        latest_ts = latest["date"]
        signal   = strategy.generate_signal(prepared_prices, symbol)

        signal_meta = signal.metadata or {}
        gate_values = signal_meta.get("gate_values") or {}
        sig_row = {
            "date": latest_ts, "symbol": symbol,
            "action": signal.action.value,
            "close": float(latest["close"]),
            "strategy": cls.name,
            "rsi": signal_meta.get("rsi"),
            "atr_%": gate_values.get("atr_pct"),
            "atr_ok": gate_values.get("atr_ok"),
            "regime": signal_meta.get("regime") or ("no_trade" if signal.action == SignalAction.HOLD else None),
            "episode_phase": gate_values.get("episode_phase"),
            "spike_type": signal_meta.get("spike_type") or gate_values.get("episode_type"),
            "verdict": signal_meta.get("verdict", signal.action.value),
            "verdict_reason": signal_meta.get("verdict_reason"),
            "gates": signal_meta.get("gate_summary"),
        }
        st.session_state[_SIGNALS].append(sig_row)

        if signal.action != SignalAction.HOLD and signal.suggested_sl is not None:
            risk = RiskManager(settings.risk)
            direction = (Direction.LONG if signal.action == SignalAction.BUY
                         else Direction.SHORT)
            entry_price_ft = float(latest["close"])

            # ── Entry policy: Classic vs Alpaca-gated (pluggable) ──────────
            # The policy encapsulates every pre-fill gate (RTH / SSR / PDT /
            # fractional / fill-diagnostic). Policy is picked per-run via the
            # dropdown, so the same signal can be compared under both logics
            # by starting two runs side by side.
            try:
                _ft_all_trades = _db().get_trades(mode="forward_test")
            except Exception:
                _ft_all_trades = []
            _ft_realised = float(sum(
                float(t.get("pnl") or 0)
                for t in _ft_all_trades if t.get("outcome") != "Open"
            ))
            _ft_portfolio_start = float(sum(
                float(r.get("capital", 0) or 0)
                for r in st.session_state[_RUNS].values()
            ))
            _ft_portfolio_equity = max(_ft_portfolio_start + _ft_realised, 0.0)

            policy = get_policy(
                run.get("execution_logic", "alpaca"),
                **{
                    k: run[k] for k in (
                        "enforce_rth", "extended_hours", "enforce_pdt",
                        "enforce_ssr", "enforce_fractional", "fill_diagnostic",
                    ) if k in run
                },
            )
            _ft_ctx = EntryContext(
                symbol=symbol,
                direction=direction,
                entry_price=entry_price_ft,
                bar=latest,
                bar_time=latest_ts,
                prices=prepared_prices,
                requested_capital=float(run["capital"]),
                leverage=float(run["leverage"]),
                portfolio_equity=_ft_portfolio_equity,
                portfolio_trades=_ft_all_trades,
            )
            decision = policy.evaluate(_ft_ctx)
            if not decision.allowed:
                # Record the skip reason on the signal row so the UI can show it,
                # then bail out without creating a trade.
                if st.session_state[_SIGNALS]:
                    st.session_state[_SIGNALS][-1]["skipped_reason"] = decision.skip_reason
                return

            # Policy may have scaled capital (e.g. fractional-short floor).
            capital_ft = float(
                decision.adjusted_capital
                if decision.adjusted_capital is not None
                else run["capital"]
            )

            check = risk.check(
                direction=direction,
                entry_price=entry_price_ft,
                take_profit=signal.suggested_tp,
                stop_loss=signal.suggested_sl,
                leverage=run["leverage"],
                capital_requested=capital_ft,
            )
            if check.approved:
                eff_sl = check.adjusted_sl or signal.suggested_sl
                signal_meta = signal.metadata or {}
                _extra_notes = " | ".join(
                    s for s in (decision.notes_prefix, decision.post_fill_note) if s
                )
                new_trade = {
                    "id":          str(uuid.uuid4())[:8],
                    "symbol":      symbol,
                    "direction":   direction.value,
                    "entry_price": entry_price_ft,
                    "take_profit": signal.suggested_tp,
                    "stop_loss":   eff_sl,
                    "leverage":    run["leverage"],
                    "capital":     capital_ft,
                    "entry_time":  latest_ts,
                    "strategy":    cls.name,
                    "outcome":     "Open",
                    "regime":      signal_meta.get("regime"),
                    "spread_pct":   float(run.get("spread_pct", 0.0) or 0.0),
                    "slippage_pct": float(run.get("slippage_pct", 0.0) or 0.0),
                    "commission":   float(run.get("commission", 0.0) or 0.0),
                    "policy_notes": _extra_notes,
                }
                if signal_meta.get("trailing_atr_mult") is not None:
                    new_trade.update({
                        "trail_kind": "atr",
                        "trail_value": float(signal_meta["trailing_atr_mult"]),
                        "trail_best": float(latest["close"]),
                        "hard_stop_loss": eff_sl,
                        "trail_grace": 0,
                        "trail_bars": 0,
                    })
                elif signal_meta.get("pct_trail") is not None:
                    new_trade.update({
                        "trail_kind": "pct",
                        "trail_value": float(signal_meta["pct_trail"]),
                        "trail_best": float(latest["close"]),
                        "hard_stop_loss": eff_sl,
                        "trail_grace": 1,
                        "trail_bars": 0,
                    })
                elif signal_meta.get("profit_giveback_frac") is not None:
                    new_trade.update({
                        "trail_kind": "giveback",
                        "trail_value": float(signal_meta["profit_giveback_frac"]),
                        "trail_min_profit_pct": float(signal_meta.get("profit_giveback_min_pct", 0.0) or 0.0),
                        "trail_best": float(latest["close"]),
                        "hard_stop_loss": eff_sl,
                        "trail_grace": 1,
                        "trail_bars": 0,
                    })
                st.session_state[_OPEN][symbol] = new_trade


# ── Page ─────────────────────────────────────────────────────────────────────

def render() -> None:
    _init_state()
    render_mode_banner()

    st.title("🔭 Forward Test")
    st.caption(
        "Run strategies on **live market data** with **simulated orders** — "
        "no broker needed. Supports multiple tickers simultaneously. "
        "All trades saved to the Portfolio."
    )
    st.info(
        "**Flow:** Backtester → **Forward Test** ← you are here → Paper Trading → Live  \n"
        "Forward Test = real-time data + local simulation. No orders sent anywhere."
    )
    st.divider()

    # ── Add new ticker run ────────────────────────────────────────────────────
    with st.expander("➕ Add Symbol to Forward Test", expanded=len(st.session_state[_RUNS]) == 0):
        strategies  = list_strategies()
        strat_names = {s["name"]: s["id"] for s in strategies}

        col1, col2, col3 = st.columns(3)
        with col1:
            new_symbol   = st.text_input("Symbol", value="UVXY", key="ft_new_sym").upper()
            new_interval = st.selectbox("Interval",
                                        ["1m","2m","5m","15m","30m","1h","1d"],
                                        index=0, key="ft_new_interval")
        with col2:
            new_lookback = st.number_input("Warm-up bars", 50, 5000, 2000, 50, key="ft_new_lb")
            new_capital  = st.number_input("Capital / trade ($)", 10.0, value=1000.0, key="ft_new_cap")
        with col3:
            new_leverage = st.number_input("Leverage", 1.0, 100.0, 1.0, 0.5, key="ft_new_lev")
            new_max_loss = st.slider("Max capital loss %", 5, 100, 50, key="ft_new_ml")
        c_cost1, c_cost2, c_cost3 = st.columns(3)
        with c_cost1:
            new_spread = st.number_input("Spread % (round-trip)", 0.0, value=0.06, step=0.01, format="%.2f", key="ft_spread")
        with c_cost2:
            new_slippage = st.number_input("Slippage % (round-trip)", 0.0, value=0.02, step=0.01, format="%.2f", key="ft_slippage")
        with c_cost3:
            new_commission = st.number_input("Commission / trade ($)", 0.0, value=0.0, step=0.01, format="%.2f", key="ft_commission")

        # ── Execution logic selector ────────────────────────────────────────
        _ft_policy_opts = available_policies()          # [(name, label), ...]
        _ft_policy_labels = [lbl for _, lbl in _ft_policy_opts]
        _ft_policy_names = [nm for nm, _ in _ft_policy_opts]
        _ft_default_idx = _ft_policy_names.index("alpaca") if "alpaca" in _ft_policy_names else 0
        st.markdown("**Execution logic**")
        _ft_chosen_label = st.selectbox(
            "Which entry-gate policy to use for this run?",
            _ft_policy_labels,
            index=_ft_default_idx,
            key="ft_exec_logic",
            help=(
                "Classic = the unconstrained logic we had before Alpaca gates "
                "were added. Alpaca-realistic = RTH / PDT / SSR / fractional / "
                "fill-diagnostic applied at entry. Runs are independent, so you "
                "can start two symbols under different policies and compare."
            ),
        )
        new_execution_logic = _ft_policy_names[_ft_policy_labels.index(_ft_chosen_label)]

        _ft_is_alpaca = new_execution_logic == "alpaca"
        if _ft_is_alpaca:
            st.markdown("**Alpaca execution rules** (fine-tune which gates to apply):")
            fac1, fac2, fac3 = st.columns(3)
            with fac1:
                ft_enforce_rth = st.checkbox(
                    "RTH only", value=True, key="ft_rth",
                    help="Skip entries outside NYSE RTH (09:30-16:00 ET) and on holidays.",
                )
                ft_extended_hrs = st.checkbox(
                    "Extended hours", value=False, key="ft_ext_hrs",
                    help="Allow 04:00-20:00 ET entries on trading days.",
                )
            with fac2:
                ft_enforce_pdt = st.checkbox(
                    "PDT (<$25k)", value=True, key="ft_pdt",
                    help="Block 4th day-trade in 5 days when equity < $25k.",
                )
                ft_enforce_ssr = st.checkbox(
                    "SSR (shorts)", value=True, key="ft_ssr",
                    help="Skip shorts on ≥10% gap-down vs prior close.",
                )
            with fac3:
                ft_enforce_frac = st.checkbox(
                    "Fractional rule", value=True, key="ft_frac",
                    help="Shorts need integer qty ≥ 1 (Alpaca rule).",
                )
                ft_fill_diag = st.checkbox(
                    "Fill-timing diag", value=True, key="ft_fill_diag",
                    help="Attach bar H/L/range to each entry's notes.",
                )
        else:
            st.caption(
                "Classic mode — no Alpaca gates applied (no RTH / PDT / SSR / "
                "fractional / fill-diagnostic). Useful as an unconstrained baseline."
            )
            ft_enforce_rth = True
            ft_extended_hrs = False
            ft_enforce_pdt = True
            ft_enforce_ssr = True
            ft_enforce_frac = True
            ft_fill_diag = True

        new_strat_name = st.selectbox("Strategy", list(strat_names.keys()), key="ft_new_strat")
        new_strat_id   = strat_names[new_strat_name]
        new_params     = render_strategy_params(new_strat_id,
                                                leverage=new_leverage,
                                                max_capital_loss_pct=float(new_max_loss),
                                                symbol=new_symbol,
                                                source="yfinance",
                                                interval=new_interval)

        if st.button("➕ Add & Start", type="primary", key="ft_add"):
            run_cfg = {
                "symbol":      new_symbol,
                "interval":    new_interval,
                "lookback":    new_lookback,
                "capital":     new_capital,
                "leverage":    new_leverage,
                "max_loss":    new_max_loss,
                "spread_pct":  new_spread,
                "slippage_pct": new_slippage,
                "commission":  new_commission,
                "strategy_id": new_strat_id,
                "params":      dict(new_params),
                "started_at":  datetime.now().isoformat(),
                "active":      True,
                # Entry-policy selector (see execution/entry_policy_*.py)
                "execution_logic":    str(new_execution_logic),
                # Alpaca execution rules (consumed only when execution_logic=='alpaca')
                "enforce_rth":        bool(ft_enforce_rth),
                "extended_hours":     bool(ft_extended_hrs),
                "enforce_pdt":        bool(ft_enforce_pdt),
                "enforce_ssr":        bool(ft_enforce_ssr),
                "enforce_fractional": bool(ft_enforce_frac),
                "fill_diagnostic":    bool(ft_fill_diag),
            }
            st.session_state[_RUNS][new_symbol]  = run_cfg
            st.session_state[_OPEN][new_symbol]  = None
            st.session_state[_EQUITY][new_symbol] = []
            st.rerun()

    # ── Active runs ───────────────────────────────────────────────────────────
    runs = st.session_state[_RUNS]
    if not runs:
        st.info("No active forward tests. Add a symbol above.")
        return

    # Global controls
    gcol1, gcol2, gcol3 = st.columns(3)
    refresh_all = gcol1.button("🔄 Refresh All Symbols", type="primary", key="ft_refresh_all")
    auto        = gcol2.checkbox("Auto-refresh (60s)", value=False, key="ft_auto")
    if gcol3.button("🗑️ Clear All", key="ft_clear"):
        for k in [_RUNS, _OPEN, _SIGNALS, _CACHE, _EQUITY]:
            st.session_state[k] = {} if isinstance(st.session_state[k], dict) else []
        st.rerun()

    # ── Collect all closed trades from DB for this session ────────────────────
    try:
        db_trades = _db().get_trades(mode="forward_test")
        closed_this_session = [t for t in db_trades
                                if any(t.get("symbol") == sym for sym in runs)]
    except Exception:
        closed_this_session = []

    # ── Per-symbol refresh ────────────────────────────────────────────────────
    if refresh_all or (auto):
        for symbol, run in runs.items():
            if run.get("active"):
                _run_tick(symbol, run, closed_this_session)
        st.rerun() if not auto else None

    st.divider()

    # ── Summary table across all symbols ─────────────────────────────────────
    st.subheader("📊 Active Symbols")
    summary_rows = []
    for sym, run in runs.items():
        open_t   = st.session_state[_OPEN].get(sym)
        prices   = st.session_state[_CACHE].get(sym)
        last_px  = float(prices.iloc[-1]["close"]) if prices is not None else None
        sym_closed = [t for t in closed_this_session if t.get("symbol") == sym]
        total_pnl  = sum(t.get("pnl", 0) for t in sym_closed if t.get("pnl"))
        trades_cnt = len([t for t in sym_closed if t.get("pnl") is not None])
        wins       = len([t for t in sym_closed if (t.get("pnl") or 0) > 0])
        summary_rows.append({
            "Symbol":    sym,
            "Strategy":  run["strategy_id"],
            "Interval":  run["interval"],
            "Logic":     "Alpaca" if run.get("execution_logic", "alpaca") == "alpaca" else "Classic",
            "Last Price": f"{last_px:.4f}" if last_px else "—",
            "Costs":     f"{float(run.get('spread_pct', 0.0) or 0.0):.2f}% + {float(run.get('slippage_pct', 0.0) or 0.0):.2f}%",
            "Open Position": (f"{open_t['direction']} @ {open_t['entry_price']:.4f}"
                              if open_t else "None"),
            "Closed Trades": trades_cnt,
            "Win Rate":  f"{wins/trades_cnt*100:.0f}%" if trades_cnt else "—",
            "Total P&L": f"${total_pnl:+,.2f}",
            "Active":    "✅" if run.get("active") else "⏸",
        })

    if summary_rows:
        st.dataframe(pd.DataFrame(summary_rows), width='stretch')

    st.divider()

    # ── Per-symbol detail tabs ────────────────────────────────────────────────
    sym_list = list(runs.keys())

    # Allow jumping to a specific symbol from portfolio
    jump_sym = st.session_state.pop("ft_jump_symbol", None)
    default_tab = sym_list.index(jump_sym) if jump_sym and jump_sym in sym_list else 0

    if sym_list:
        tabs = st.tabs([f"{'🟢' if runs[s].get('active') else '⏸'} {s}"
                        for s in sym_list])

        for i, (tab, symbol) in enumerate(zip(tabs, sym_list)):
            with tab:
                run    = runs[symbol]
                prices = st.session_state[_CACHE].get(symbol)
                open_t = st.session_state[_OPEN].get(symbol)

                # Per-symbol controls
                c1, c2, c3, c4 = st.columns(4)
                if c1.button(f"🔄 Refresh {symbol}", key=f"ft_ref_{symbol}"):
                    _run_tick(symbol, run, closed_this_session)
                    st.rerun()
                if c2.button(f"{'⏸ Pause' if run.get('active') else '▶ Resume'}",
                              key=f"ft_toggle_{symbol}"):
                    st.session_state[_RUNS][symbol]["active"] = not run.get("active", True)
                    st.rerun()
                if c3.button(f"❌ Close Open Trade",
                              key=f"ft_close_{symbol}",
                              disabled=open_t is None):
                    if prices is not None and open_t:
                        xp  = float(prices.iloc[-1]["close"])
                        ep  = open_t["entry_price"]
                        d   = open_t["direction"]
                        lev = open_t["leverage"]
                        raw = (xp - ep) / ep * (1 if d == "Long" else -1)
                        ret = raw * lev * 100
                        gross_pnl = open_t["capital"] * ret / 100
                        cost = _trade_cost(open_t)
                        pnl = gross_pnl - cost
                        open_t.update({
                            "outcome": "Manual close", "exit_price": xp,
                            "exit_time": prices.iloc[-1]["date"],
                            "leveraged_return_%": round(ret, 3),
                            "gross_pnl": round(gross_pnl, 2),
                            "cost": round(cost, 2),
                            "pnl": round(pnl, 2),
                        })
                        closed_this_session.append(open_t)
                        _save_trade_to_db(open_t, symbol, run["strategy_id"])
                        st.session_state[_OPEN][symbol] = None
                        st.rerun()
                if c4.button(f"🗑️ Remove {symbol}", key=f"ft_remove_{symbol}"):
                    del st.session_state[_RUNS][symbol]
                    st.session_state[_OPEN].pop(symbol, None)
                    st.session_state[_CACHE].pop(symbol, None)
                    st.session_state[_EQUITY].pop(symbol, None)
                    st.rerun()

                if prices is None:
                    st.info(f"Click **🔄 Refresh {symbol}** to fetch first bar.")
                    continue

                latest = prices.iloc[-1]

                # Status
                if open_t:
                    curr   = float(latest["close"])
                    ep     = open_t["entry_price"]
                    lev    = open_t["leverage"]
                    d      = open_t["direction"]
                    unreal = (curr - ep) / ep * (1 if d == "Long" else -1) * lev * 100
                    col    = "green" if unreal >= 0 else "red"
                    tp_str = f"{open_t['take_profit']:.4f}" if open_t.get("take_profit") else "—"
                    st.markdown(
                        f'<div style="border:1px solid #2a2d3e;border-radius:8px;'
                        f'padding:8px 14px;">'
                        f'<b>Open:</b> {d} @ <code>{ep:.4f}</code> · '
                        f'SL <code>{open_t["stop_loss"]:.4f}</code> · '
                        f'TP <code>{tp_str}</code> · '
                        f'Unrealised: <span style="color:{col};font-weight:bold">'
                        f'{unreal:+.2f}%</span></div>',
                        unsafe_allow_html=True,
                    )

                # Metrics
                sym_closed_list = [t for t in closed_this_session
                                   if t.get("symbol") == symbol and t.get("pnl") is not None]
                if sym_closed_list:
                    wins_sym = len([t for t in sym_closed_list if (t.get("pnl",0) > 0)])
                    render_metrics_row({
                        "Closed": len(sym_closed_list),
                        "Win Rate": f"{wins_sym/len(sym_closed_list)*100:.0f}%",
                        "Total P&L": f"${sum(t.get('pnl',0) for t in sym_closed_list):+,.2f}",
                    })

                # Price chart
                st.altair_chart(_price_chart(prices, closed_this_session,
                                             open_t, st.session_state[_SIGNALS], symbol),
                                width='stretch')

                # RSI chart
                if run["strategy_id"] in ("rsi_threshold", "atr_rsi",
                                           "vwap_rsi", "bollinger_rsi", "ema_trend_rsi"):
                    p_cfg     = run.get("params", {})
                    rsi_p     = int(p_cfg.get("rsi_period", 9))
                    try:
                        buy_lvls  = [float(x) for x in
                                     str(p_cfg.get("buy_levels","30")).replace(";",",").split(",")
                                     if x.strip()] or [30]
                        sell_lvls = [float(x) for x in
                                     str(p_cfg.get("sell_levels","70")).replace(";",",").split(",")
                                     if x.strip()] or [70]
                    except Exception:
                        buy_lvls, sell_lvls = [30], [70]
                    st.altair_chart(
                        rsi_chart(prices, rsi_p, buy_lvls, sell_lvls)
                        .configure_view(strokeOpacity=0)
                        .configure_axis(**_AXIS)
                        .configure_title(**_TITLE),
                        width='stretch',
                    )

                # Equity curve
                eq_hist = st.session_state[_EQUITY].get(symbol, [])
                if len(eq_hist) >= 2:
                    st.altair_chart(_equity_chart(eq_hist, run["capital"], symbol),
                                    width='stretch')

                # Signal log
                with st.expander("📡 Signals", expanded=False):
                    sym_sigs = [s for s in st.session_state[_SIGNALS]
                                if s.get("symbol") == symbol]
                    if sym_sigs:
                        st.dataframe(pd.DataFrame(sym_sigs)
                                     .sort_values("date", ascending=False),
                                     width='stretch')
                    else:
                        st.info("No signals yet.")

                # Trade log
                with st.expander("📋 Trades", expanded=False):
                    if sym_closed_list:
                        st.dataframe(pd.DataFrame(sym_closed_list),
                                     width='stretch')
                    else:
                        st.info("No closed trades yet.")

    # Auto-refresh mechanism
    if auto and any(r.get("active") for r in runs.values()):
        import time
        min_interval = min(
            int(_interval_td(r["interval"]).total_seconds())
            for r in runs.values() if r.get("active")
        )
        time.sleep(max(min_interval, 60))
        st.rerun()

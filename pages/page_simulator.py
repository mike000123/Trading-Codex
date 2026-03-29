"""
pages/page_simulator.py
────────────────────────
Historical Trade Outcome Simulator  (migrated from the original app.py)
Two modes:
  1. Given TP/SL → find first hit date
  2. Given desired profit → suggest TP/SL
"""
from __future__ import annotations

from typing import Optional, Tuple

import pandas as pd
import streamlit as st

from core.models import Direction, TradeOutcome
from risk.manager import RiskManager
from config.settings import settings
from ui.components import render_mode_banner, render_data_source_selector
from ui.charts import price_chart


# ─── Pure simulation logic (ported from original app.py) ─────────────────

def _leveraged_return(entry: float, exit_p: float, lev: float, direction: Direction) -> float:
    raw = (exit_p - entry) / entry
    if direction == Direction.SHORT:
        raw = -raw
    return raw * lev * 100


def _simulate_trade(
    data: pd.DataFrame, entry_date: pd.Timestamp,
    direction: Direction, leverage: float,
    take_price: float, stop_price: float,
) -> dict:
    subset = data[data["date"] >= entry_date]
    if subset.empty:
        return {"outcome": TradeOutcome.NO_DATA, "date": None, "exit_price": None, "ret_pct": None, "notes": "No rows from entry date."}

    entry_price = float(subset.iloc[0]["close"])
    for _, row in subset.iterrows():
        high, low = float(row["high"]), float(row["low"])
        hit_tp = high >= take_price if direction == Direction.LONG else low <= take_price
        hit_sl = low <= stop_price if direction == Direction.LONG else high >= stop_price
        if hit_tp and hit_sl:
            return {"outcome": TradeOutcome.AMBIGUOUS, "date": row["date"], "exit_price": None, "ret_pct": None,
                    "notes": "TP and SL both touched in the same candle."}
        if hit_tp:
            return {"outcome": TradeOutcome.TAKE_PROFIT, "date": row["date"], "exit_price": take_price,
                    "ret_pct": _leveraged_return(entry_price, take_price, leverage, direction), "notes": ""}
        if hit_sl:
            return {"outcome": TradeOutcome.STOP_LOSS, "date": row["date"], "exit_price": stop_price,
                    "ret_pct": _leveraged_return(entry_price, stop_price, leverage, direction), "notes": ""}

    return {"outcome": TradeOutcome.OPEN, "date": None, "exit_price": None, "ret_pct": None,
            "notes": "Neither threshold hit in available data."}


def _suggest_levels(
    data: pd.DataFrame, entry_date: pd.Timestamp,
    direction: Direction, leverage: float, desired_pct: float,
) -> Tuple[Optional[float], Optional[float], Optional[pd.Timestamp], str]:
    subset = data[data["date"] >= entry_date]
    if subset.empty:
        return None, None, None, "No rows from entry date."
    entry_price = float(subset.iloc[0]["close"])
    move = desired_pct / 100 / leverage
    if direction == Direction.LONG:
        take = entry_price * (1 + move)
        hits = subset[subset["high"] >= take]
    else:
        take = entry_price * (1 - move)
        hits = subset[subset["low"] <= take]
    if hits.empty:
        return None, None, None, "Desired profit target never reached in this window."
    first_hit = hits.iloc[0]["date"]
    pre = subset[subset["date"] < first_hit]
    if direction == Direction.LONG:
        stop = float(pre["low"].min()) * 0.999 if not pre.empty else entry_price * 0.99
    else:
        stop = float(pre["high"].max()) * 1.001 if not pre.empty else entry_price * 1.01
    return take, stop, first_hit, "Stop set just beyond worst adverse move before first TP hit."


# ─── Page render ─────────────────────────────────────────────────────────

def render() -> None:
    render_mode_banner()
    st.title("📊 Historical Trade Outcome Simulator")
    st.caption("Load OHLC data, then test leveraged long/short scenarios from any historical date.")

    prices = render_data_source_selector()

    if prices is None:
        st.info("← Select a data source in the sidebar to begin.")
        return

    symbol = st.session_state.get("loaded_symbol", "DATA")
    st.success(f"**{symbol}** — {len(prices)} bars · {prices['date'].min().date()} → {prices['date'].max().date()}")

    with st.expander("📋 Data Preview", expanded=False):
        st.dataframe(prices.tail(50), use_container_width=True)

    st.divider()

    controls, results = st.columns([1.1, 0.9])

    with controls:
        mode = st.radio(
            "Workflow",
            ["1) Given TP/SL → find outcome date", "2) Given desired profit → suggest TP/SL"],
            key="sim_mode",
        )
        entry_date = st.date_input(
            "Entry date",
            value=prices["date"].min().date(),
            min_value=prices["date"].min().date(),
            max_value=prices["date"].max().date(),
            key="sim_entry_date",
        )
        direction = Direction(st.selectbox("Direction", ["Long", "Short"], key="sim_direction"))
        leverage  = st.number_input("Leverage", 1.0, 500.0, 10.0, 1.0, key="sim_leverage")
        capital   = st.number_input("Capital ($)", 0.0, value=1000.0, step=100.0, key="sim_capital")
        max_loss_pct = 50.0
        st.caption(f"Max stop-loss cap: **{max_loss_pct}%** of capital (enforced automatically)")

        entry_ts = pd.Timestamp(entry_date)
        entry_slice = prices[prices["date"] >= entry_ts]
        entry_close = float(entry_slice.iloc[0]["close"]) if not entry_slice.empty else None

        if entry_close:
            move_limit = max_loss_pct / leverage
            if direction == Direction.LONG:
                floor = entry_close * (1 - move_limit / 100)
                st.caption(f"Entry close: **{entry_close:.4f}** · Minimum SL floor: **{floor:.4f}**")
            else:
                ceiling = entry_close * (1 + move_limit / 100)
                st.caption(f"Entry close: **{entry_close:.4f}** · Maximum SL ceiling: **{ceiling:.4f}**")

        risk = RiskManager(settings.risk)

        # ── Mode 1 ──────────────────────────────────────────────────────────
        if mode.startswith("1"):
            c1, c2 = st.columns(2)
            default_tp = round(entry_close * (1.02 if direction == Direction.LONG else 0.98), 4) if entry_close else 105.0
            default_sl = round(entry_close * (0.98 if direction == Direction.LONG else 1.02), 4) if entry_close else 95.0
            take_price = c1.number_input("Take-profit price", 0.0, value=default_tp, key="sim_tp")
            stop_price = c2.number_input("Stop-loss price",   0.0, value=default_sl, key="sim_sl")

            if st.button("▶ Run Simulation", type="primary", key="sim_run1"):
                check = risk.check(
                    direction=direction, entry_price=entry_close,
                    take_profit=take_price, stop_loss=stop_price,
                    leverage=leverage, capital_requested=capital,
                )
                eff_sl = check.adjusted_sl or stop_price
                result = _simulate_trade(prices, entry_ts, direction, leverage, take_price, eff_sl)

                with results:
                    st.subheader("Result")
                    _show_result(result, capital, eff_sl != stop_price, stop_price, eff_sl)
                    fig = price_chart(
                        prices,
                        entry_date=entry_slice.iloc[0]["date"] if not entry_slice.empty else None,
                        exit_date=result.get("date"),
                        take_profit=take_price,
                        stop_loss=eff_sl,
                        title=f"{symbol} – Simulation",
                    )
                    st.plotly_chart(fig, use_container_width=True)

        # ── Mode 2 ──────────────────────────────────────────────────────────
        else:
            desired_pct = st.number_input(
                "Desired capital profit (%)", 0.1, 10000.0, 20.0, 0.1, key="sim_desired"
            )
            if st.button("💡 Suggest Levels", type="primary", key="sim_run2"):
                take, stop, hit_date, note = _suggest_levels(
                    prices, entry_ts, direction, leverage, desired_pct
                )
                with results:
                    st.subheader("Suggested Levels")
                    if take is None:
                        st.error(note)
                    else:
                        check = risk.check(
                            direction=direction, entry_price=entry_close,
                            take_profit=take, stop_loss=stop,
                            leverage=leverage, capital_requested=capital,
                        )
                        eff_sl = check.adjusted_sl or stop
                        sim = _simulate_trade(prices, entry_ts, direction, leverage, take, eff_sl)
                        _show_levels(take, eff_sl, stop, entry_close, leverage, direction, capital, sim)
                        fig = price_chart(
                            prices,
                            entry_date=entry_slice.iloc[0]["date"] if not entry_slice.empty else None,
                            exit_date=sim.get("date"),
                            take_profit=take, stop_loss=eff_sl,
                            title=f"{symbol} – Suggested Levels",
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        st.info(note)


def _show_result(result: dict, capital: float, was_capped: bool, orig_sl: float, eff_sl: float) -> None:
    outcome = result["outcome"]
    ret = result.get("ret_pct")
    cols = st.columns(2)
    cols[0].metric("Outcome", outcome.value if hasattr(outcome, "value") else str(outcome))
    cols[1].metric("Exit Date", str(result["date"].date()) if result["date"] else "N/A")
    cols[0].metric("Exit Price", f"{result['exit_price']:.4f}" if result["exit_price"] else "N/A")
    cols[1].metric("Leveraged Return", f"{ret:.2f}%" if ret is not None else "N/A",
                   delta=f"{ret:.2f}%" if ret is not None else None)
    if ret is not None:
        final = capital * (1 + ret / 100)
        pnl = final - capital
        st.metric("Final Portfolio Value", f"${final:,.2f}", delta=f"${pnl:+,.2f}")
    if was_capped:
        st.warning(f"SL adjusted {orig_sl:.4f} → {eff_sl:.4f} (max-loss cap applied)")
    if result.get("notes"):
        st.warning(result["notes"])


def _show_levels(
    take: float, eff_sl: float, orig_sl: float,
    entry: float, leverage: float, direction: Direction,
    capital: float, sim: dict,
) -> None:
    tp_move = (take - entry) / entry * (1 if direction == Direction.LONG else -1) * 100
    sl_move = (eff_sl - entry) / entry * (1 if direction == Direction.LONG else -1) * 100
    tp_ret = tp_move * leverage
    sl_ret = sl_move * leverage

    c1, c2 = st.columns(2)
    c1.metric("Take-Profit", f"{take:.4f}", delta=f"{tp_ret:+.2f}% leveraged")
    c2.metric("Stop-Loss", f"{eff_sl:.4f}", delta=f"{sl_ret:+.2f}% leveraged")
    if abs(eff_sl - orig_sl) > 1e-6:
        st.warning(f"SL adjusted {orig_sl:.4f} → {eff_sl:.4f} (max-loss cap)")
    ret = sim.get("ret_pct")
    outcome = sim.get("outcome")
    st.metric("Simulated Outcome", outcome.value if hasattr(outcome, "value") else str(outcome))
    if ret is not None:
        final = capital * (1 + ret / 100)
        st.metric("Final Portfolio Value", f"${final:,.2f}", delta=f"${final - capital:+,.2f}")

"""
pages/page_backtest.py  —  Walk-forward backtester.
Charts: price with entry+exit markers, RSI with entry+exit markers, equity curve, per-trade P&L bar.
All Altair, zero Plotly.
"""
from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from config.settings import settings
from reporting.backtest import BacktestEngine
from risk.manager import RiskManager
from strategies import list_strategies, get_strategy
from ui.components import (
    render_mode_banner, render_data_source_selector,
    render_strategy_params, render_metrics_row,
)
from ui.charts import equity_curve_chart, pnl_distribution

_GREEN = "#26a69a"
_RED   = "#ef5350"
_BLUE  = "#4a9eff"
_GOLD  = "#ffd54f"
_GREY  = "#9e9eb8"
_AXIS  = dict(gridColor="#1e2130", labelColor="#c9d8f5", titleColor="#c9d8f5")


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _calc_rsi(series: pd.Series, period: int) -> pd.Series:
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs       = avg_gain / avg_loss.replace(0, float("nan"))
    return 100 - (100 / (1 + rs))


def _bar_label(prices: pd.DataFrame) -> str:
    if len(prices) < 2:
        return "bars"
    delta = (prices["date"].iloc[1] - prices["date"].iloc[0]).total_seconds()
    return {60: "1-min", 300: "5-min", 900: "15-min", 1800: "30-min",
            3600: "1-hour", 86400: "1-day", 604800: "1-week"}.get(int(delta), f"{int(delta//60)}-min")


def _trade_events(trades) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split all trades into entry_df and exit_df DataFrames for charting.
    Returns (entry_df, exit_df).
    """
    entries, exits = [], []
    for i, t in enumerate(trades):
        direction = t.direction.value if hasattr(t.direction, "value") else str(t.direction)
        outcome   = t.outcome.value   if hasattr(t.outcome,   "value") else str(t.outcome)
        ret       = t.leveraged_return_pct
        label     = f"T{i+1}"

        entries.append({
            "date":        pd.Timestamp(t.entry_time),
            "price":       t.entry_price,
            "direction":   direction,
            "outcome":     outcome,
            "return_pct":  ret,
            "trade_n":     label,
        })
        if t.exit_time and t.exit_price is not None:
            exits.append({
                "date":       pd.Timestamp(t.exit_time),
                "price":      t.exit_price,
                "direction":  direction,
                "outcome":    outcome,
                "return_pct": ret,
                "trade_n":    label,
                "win":        (ret or 0) > 0,
            })

    return pd.DataFrame(entries), pd.DataFrame(exits)


# ─── Price chart ─────────────────────────────────────────────────────────────

def _price_with_trades(prices: pd.DataFrame, trades, symbol: str) -> alt.LayerChart:
    base = (
        alt.Chart(prices).mark_line(color=_BLUE, strokeWidth=1.2)
        .encode(
            x=alt.X("date:T", title="Date / Time"),
            y=alt.Y("close:Q", title="Price", scale=alt.Scale(zero=False)),
            tooltip=["date:T", alt.Tooltip("close:Q", format=".4f")],
        )
    )
    layers = [base]

    if not trades:
        return (alt.layer(*layers)
                .properties(title=f"{symbol} – Price · Entries ▲▼  Exits ✕", height=280)
                .configure_view(strokeOpacity=0).configure_axis(**_AXIS)
                .configure_title(color="#c9d8f5"))

    entry_df, exit_df = _trade_events(trades)

    # ── Entries ──────────────────────────────────────────────────────────────
    if not entry_df.empty:
        long_e  = entry_df[entry_df["direction"] == "Long"].copy()
        short_e = entry_df[entry_df["direction"] == "Short"].copy()

        tt_entry = [
            "date:T", "trade_n:N", "direction:N",
            alt.Tooltip("price:Q", format=".4f", title="Entry"),
            "outcome:N",
            alt.Tooltip("return_pct:Q", format=".2f", title="Return %"),
        ]
        if not long_e.empty:
            long_e["y"] = long_e["price"] * 0.997
            layers.append(
                alt.Chart(long_e)
                .mark_point(shape="triangle-up", size=100, filled=True, color=_GREEN)
                .encode(x="date:T", y="y:Q", tooltip=tt_entry)
            )
        if not short_e.empty:
            short_e["y"] = short_e["price"] * 1.003
            layers.append(
                alt.Chart(short_e)
                .mark_point(shape="triangle-down", size=100, filled=True, color=_RED)
                .encode(x="date:T", y="y:Q", tooltip=tt_entry)
            )

    # ── Exits ────────────────────────────────────────────────────────────────
    if not exit_df.empty:
        tt_exit = [
            "date:T", "trade_n:N", "direction:N",
            alt.Tooltip("price:Q", format=".4f", title="Exit"),
            "outcome:N",
            alt.Tooltip("return_pct:Q", format=".2f", title="Return %"),
        ]
        win_ex  = exit_df[exit_df["win"] == True].copy()
        loss_ex = exit_df[exit_df["win"] == False].copy()

        for sub, col in [(win_ex, _GREEN), (loss_ex, _RED)]:
            if not sub.empty:
                layers.append(
                    alt.Chart(sub)
                    .mark_point(shape="cross", size=90, strokeWidth=2, color=col)
                    .encode(x="date:T", y="price:Q", tooltip=tt_exit)
                )

    return (
        alt.layer(*layers)
        .properties(title=f"{symbol} – Price · Entries ▲▼  Exits ✕", height=280)
        .configure_view(strokeOpacity=0)
        .configure_axis(**_AXIS)
        .configure_title(color="#c9d8f5")
    )


# ─── RSI chart ───────────────────────────────────────────────────────────────

def _rsi_with_trades(
    prices: pd.DataFrame, trades,
    period: int, oversold: float, overbought: float,
    bar_label: str, symbol: str,
) -> alt.LayerChart:
    rsi_vals = _calc_rsi(prices["close"], period).rename("rsi")
    df = pd.concat([prices[["date"]], rsi_vals], axis=1).dropna()

    rsi_line = (
        alt.Chart(df).mark_line(color=_GOLD, strokeWidth=1.5)
        .encode(
            x=alt.X("date:T", title="Date / Time"),
            y=alt.Y("rsi:Q", title="RSI", scale=alt.Scale(domain=[0, 100])),
            tooltip=["date:T", alt.Tooltip("rsi:Q", format=".2f")],
        )
    )

    ob_df = pd.DataFrame({"y": [overbought], "label": [f"OB {overbought:.0f}"]})
    os_df = pd.DataFrame({"y": [oversold],   "label": [f"OS {oversold:.0f}"]})
    ob_rule = alt.Chart(ob_df).mark_rule(color=_RED,   strokeDash=[5, 3], strokeWidth=1.2).encode(y="y:Q")
    os_rule = alt.Chart(os_df).mark_rule(color=_GREEN, strokeDash=[5, 3], strokeWidth=1.2).encode(y="y:Q")
    ob_text = alt.Chart(ob_df).mark_text(align="left", dx=4, dy=-7, fontSize=10, color=_RED  ).encode(y="y:Q", x=alt.value(4), text="label:N")
    os_text = alt.Chart(os_df).mark_text(align="left", dx=4, dy=-7, fontSize=10, color=_GREEN).encode(y="y:Q", x=alt.value(4), text="label:N")

    ob_band = alt.Chart(pd.DataFrame({"y1": [overbought], "y2": [100]})).mark_rect(color=_RED,   opacity=0.05).encode(y="y1:Q", y2="y2:Q")
    os_band = alt.Chart(pd.DataFrame({"y1": [0],          "y2": [oversold]})).mark_rect(color=_GREEN, opacity=0.05).encode(y="y1:Q", y2="y2:Q")

    layers = [ob_band, os_band, rsi_line, ob_rule, os_rule, ob_text, os_text]

    # ── Entry + exit markers snapped to nearest RSI bar ───────────────────────
    if trades:
        entry_df, exit_df = _trade_events(trades)

        def _snap_rsi(ts: pd.Timestamp) -> float:
            idx = df["date"].searchsorted(ts)
            idx = min(idx, len(df) - 1)
            v   = df.iloc[idx]["rsi"]
            return float(v) if not pd.isna(v) else 50.0

        def _snap_date(ts: pd.Timestamp) -> pd.Timestamp:
            idx = df["date"].searchsorted(ts)
            return df.iloc[min(idx, len(df) - 1)]["date"]

        tt_fields = [
            "date:T", "trade_n:N", "direction:N",
            alt.Tooltip("rsi_val:Q", format=".1f", title="RSI"),
            "event:N",
            alt.Tooltip("return_pct:Q", format=".2f", title="Return %"),
        ]

        # Entries
        if not entry_df.empty:
            entry_df["rsi_val"]  = entry_df["date"].apply(_snap_rsi)
            entry_df["date"]     = entry_df["date"].apply(_snap_date)
            entry_df["event"]    = "Entry"

            long_e  = entry_df[entry_df["direction"] == "Long"].copy()
            short_e = entry_df[entry_df["direction"] == "Short"].copy()

            if not long_e.empty:
                long_e["y"] = long_e["rsi_val"] - 5
                layers.append(
                    alt.Chart(long_e)
                    .mark_point(shape="triangle-up", size=80, filled=True, color=_GREEN)
                    .encode(x="date:T", y="y:Q", tooltip=tt_fields)
                )
            if not short_e.empty:
                short_e["y"] = short_e["rsi_val"] + 5
                layers.append(
                    alt.Chart(short_e)
                    .mark_point(shape="triangle-down", size=80, filled=True, color=_RED)
                    .encode(x="date:T", y="y:Q", tooltip=tt_fields)
                )

        # Exits  ← NEW
        if not exit_df.empty:
            exit_df["rsi_val"] = exit_df["date"].apply(_snap_rsi)
            exit_df["date"]    = exit_df["date"].apply(_snap_date)
            exit_df["event"]   = "Exit"

            win_ex  = exit_df[exit_df["win"] == True].copy()
            loss_ex = exit_df[exit_df["win"] == False].copy()

            for sub, col in [(win_ex, _GREEN), (loss_ex, _RED)]:
                if not sub.empty:
                    layers.append(
                        alt.Chart(sub)
                        .mark_point(shape="cross", size=80, strokeWidth=2, color=col)
                        .encode(x="date:T", y="rsi_val:Q", tooltip=tt_fields)
                    )

    return (
        alt.layer(*layers)
        .properties(title=f"{symbol} – RSI ({period}) · bar = {bar_label}  ·  ▲▼ Entry  ✕ Exit", height=260)
        .configure_view(strokeOpacity=0)
        .configure_axis(**_AXIS)
        .configure_title(color="#c9d8f5")
    )


# ─── Equity curve built from individual trade P&Ls ────────────────────────────

def _equity_from_trades(trades, starting_equity: float, symbol: str) -> alt.LayerChart:
    """
    Build an equity curve from closed-trade P&Ls (more meaningful than bar-by-bar).
    Also draws a horizontal baseline at starting equity.
    """
    closed = [t for t in trades if t.pnl is not None and t.exit_time is not None]
    if not closed:
        return alt.Chart(pd.DataFrame()).mark_line()

    rows = []
    eq   = starting_equity
    # Opening point
    rows.append({"date": pd.Timestamp(closed[0].entry_time), "equity": eq, "trade_n": "Start", "pnl": 0.0})
    for i, t in enumerate(sorted(closed, key=lambda x: x.exit_time)):
        eq += t.pnl
        ret  = t.leveraged_return_pct or 0
        rows.append({
            "date":     pd.Timestamp(t.exit_time),
            "equity":   round(eq, 2),
            "trade_n":  f"T{i+1}",
            "pnl":      round(t.pnl, 2),
            "ret_pct":  round(ret, 3),
            "outcome":  t.outcome.value if hasattr(t.outcome, "value") else str(t.outcome),
        })

    eq_df = pd.DataFrame(rows)

    area = (
        alt.Chart(eq_df)
        .mark_area(line={"color": _BLUE, "strokeWidth": 2},
                   color=alt.Gradient(
                       gradient="linear",
                       stops=[alt.GradientStop(color="rgba(74,158,255,0.25)", offset=0),
                              alt.GradientStop(color="rgba(74,158,255,0.0)",  offset=1)],
                       x1=1, x2=1, y1=1, y2=0))
        .encode(
            x=alt.X("date:T", title="Date / Time"),
            y=alt.Y("equity:Q", title="Equity ($)", scale=alt.Scale(zero=False)),
            tooltip=[
                "date:T", "trade_n:N",
                alt.Tooltip("equity:Q",  format="$,.2f", title="Equity"),
                alt.Tooltip("pnl:Q",     format="+$,.2f", title="Trade P&L"),
                alt.Tooltip("ret_pct:Q", format=".2f",   title="Return %"),
                "outcome:N",
            ],
        )
    )

    # Baseline at starting equity
    base_df   = pd.DataFrame({"y": [starting_equity]})
    base_rule = (alt.Chart(base_df)
                 .mark_rule(color=_GREY, strokeDash=[4, 4], strokeWidth=1)
                 .encode(y="y:Q"))

    # Dot at each trade exit
    dots = (
        alt.Chart(eq_df.iloc[1:])   # skip the "Start" synthetic row
        .mark_point(size=50, filled=True)
        .encode(
            x="date:T",
            y="equity:Q",
            color=alt.condition(
                alt.datum.pnl > 0,
                alt.value(_GREEN),
                alt.value(_RED),
            ),
            tooltip=[
                "date:T", "trade_n:N",
                alt.Tooltip("equity:Q",  format="$,.2f", title="Equity"),
                alt.Tooltip("pnl:Q",     format="+$,.2f", title="Trade P&L"),
                alt.Tooltip("ret_pct:Q", format=".2f",   title="Return %"),
                "outcome:N",
            ],
        )
    )

    return (
        alt.layer(base_rule, area, dots)
        .properties(title=f"{symbol} – Equity Curve (per closed trade)", height=280)
        .configure_view(strokeOpacity=0)
        .configure_axis(**_AXIS)
        .configure_title(color="#c9d8f5")
    )


# ─── Page ─────────────────────────────────────────────────────────────────────

def render() -> None:
    render_mode_banner()
    st.title("⏪ Backtester")

    prices = render_data_source_selector()
    if prices is None:
        st.info("← Select a data source in the sidebar to begin.")
        return

    symbol    = st.session_state.get("loaded_symbol", "DATA")
    bar_label = _bar_label(prices)
    st.success(f"**{symbol}** — {len(prices)} bars · bar size: **{bar_label}**")
    st.caption(
        f"ℹ️ RSI period = number of **{bar_label} bars**. "
        "Intraday suggestions: 5-min → try 7–9 · 1-min → try 9–14 · 1-day → classic 14."
    )

    st.divider()

    strategies  = list_strategies()
    strat_names = {s["name"]: s["id"] for s in strategies}

    col_cfg, col_risk = st.columns(2)
    with col_cfg:
        selected_name     = st.selectbox("Strategy", list(strat_names.keys()), key="bt_strategy")
        selected_id       = strat_names[selected_name]
        leverage          = st.number_input("Leverage", 1.0, 100.0, 1.0, 0.5, key="bt_lev")
        capital_per_trade = st.number_input("Capital per trade ($)", 100.0, value=1000.0, key="bt_cap")
        starting_equity   = st.number_input("Starting equity ($)", 1000.0, value=10_000.0, key="bt_equity")
        direction_filter  = st.selectbox("Direction filter", ["Both", "Long only", "Short only"], key="bt_dir")

    with col_risk:
        st.markdown("**Risk Controls**")
        use_risk = st.checkbox("Apply risk manager", value=True, key="bt_risk")
        max_loss = st.slider("Max loss per trade (% of capital)", 5, 100, 50, key="bt_maxloss")

    st.divider()
    params = render_strategy_params(selected_id)

    if st.button("▶ Run Backtest", type="primary", key="bt_run"):
        cls      = get_strategy(selected_id)
        strategy = cls(params=params)

        errors = strategy.validate_params()
        if errors:
            for e in errors: st.error(e)
            return

        from config.settings import RiskConfig
        from core.models import Direction as Dir

        risk_cfg = RiskConfig(
            max_capital_per_trade_pct=100.0,
            max_daily_loss_pct=100.0,
            max_open_positions=999,
            default_max_loss_pct_of_capital=float(max_loss),
        )
        rm = RiskManager(risk_cfg) if use_risk else None

        dir_filter = None
        if direction_filter == "Long only":  dir_filter = Dir.LONG
        if direction_filter == "Short only": dir_filter = Dir.SHORT

        engine = BacktestEngine(strategy, risk_manager=rm, direction_filter=dir_filter)

        with st.spinner("Running backtest…"):
            result = engine.run(
                data=prices, symbol=symbol,
                leverage=leverage,
                capital_per_trade=capital_per_trade,
                starting_equity=starting_equity,
            )

        closed = [t for t in result.trades if t.leveraged_return_pct is not None]

        # ── Summary ───────────────────────────────────────────────────────────
        st.subheader("📊 Results")
        s = result.summary()
        render_metrics_row({
            "Total Trades": s["Total Trades"],
            "Win Rate":     s["Win Rate"],
            "Total Return": s["Total Return"],
            "Max Drawdown": s["Max Drawdown"],
            "Sharpe Ratio": s["Sharpe Ratio"],
            "Avg Win":      s["Avg Win"],
            "Avg Loss":     s["Avg Loss"],
        })

        st.divider()

        # ── Price chart ───────────────────────────────────────────────────────
        st.markdown("#### 📈 Price — Entries ▲▼ · Exits ✕")
        st.altair_chart(_price_with_trades(prices, result.trades, symbol), use_container_width=True)

        # ── RSI chart (RSI strategy only) ────────────────────────────────────
        if selected_id == "rsi_threshold":
            period     = int(params.get("rsi_period", 14))
            oversold   = float(params.get("oversold", 30))
            overbought = float(params.get("overbought", 70))
            st.markdown(f"#### 📉 RSI ({period})  —  ▲▼ Entry · ✕ Exit")
            st.altair_chart(
                _rsi_with_trades(prices, result.trades, period, oversold, overbought, bar_label, symbol),
                use_container_width=True,
            )

        # ── Equity curve (per closed trade) ──────────────────────────────────
        if closed:
            st.markdown("#### 💰 Equity Curve — one dot per closed trade")
            st.altair_chart(_equity_from_trades(result.trades, starting_equity, symbol), use_container_width=True)

        # ── Per-trade P&L bars ────────────────────────────────────────────────
        if closed:
            trades_df = pd.DataFrame([
                {
                    "symbol":               t.symbol,
                    "direction":            t.direction.value,
                    "entry_price":          t.entry_price,
                    "exit_price":           t.exit_price,
                    "outcome":              t.outcome.value if t.outcome else None,
                    "leveraged_return_pct": t.leveraged_return_pct,
                    "pnl":                  t.pnl,
                    "entry_time":           t.entry_time,
                    "exit_time":            t.exit_time,
                }
                for t in closed
            ])
            st.markdown("#### 📊 Per-Trade Return")
            st.altair_chart(pnl_distribution(trades_df), use_container_width=True)

            with st.expander("📋 Trade Log", expanded=False):
                st.dataframe(trades_df.sort_values("entry_time", ascending=False), use_container_width=True)

        # ── Save to DB ────────────────────────────────────────────────────────
        try:
            from db.database import Database
            db = Database(settings.db_path)
            for t in result.trades:
                db.save_trade(t)
            st.caption(f"✓ {len(result.trades)} trades saved to database.")
        except Exception as e:
            st.caption(f"DB save skipped: {e}")

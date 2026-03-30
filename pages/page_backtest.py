"""
pages/page_backtest.py
Walk-forward backtester with interactive chart layer toggles.
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
from ui.charts import equity_curve_chart, pnl_distribution, rsi_chart

_GREEN  = "#26a69a"
_RED    = "#ef5350"
_BLUE   = "#4a9eff"
_GOLD   = "#ffd54f"
_GREY   = "#9e9eb8"
_ORANGE = "#ff9800"
_AXIS   = dict(gridColor="#2a2d3e", labelColor="#d0d4f0", titleColor="#d0d4f0",
               labelFontSize=12, titleFontSize=13)
_TITLE  = dict(color="#e8eaf6", fontSize=14, fontWeight="bold")


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _calc_rsi(series: pd.Series, period: int) -> pd.Series:
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs       = avg_gain / avg_loss.replace(0, float("nan"))
    return 100 - (100 / (1 + rs))


def _parse_levels(raw) -> list[float]:
    if isinstance(raw, (int, float)):
        return [float(raw)]
    if isinstance(raw, (list, tuple)):
        return sorted(float(v) for v in raw)
    parts = str(raw).replace(";", ",").split(",")
    return sorted(float(p.strip()) for p in parts if p.strip())


def _bar_label(prices: pd.DataFrame) -> str:
    if len(prices) < 2:
        return "bars"
    delta = (prices["date"].iloc[1] - prices["date"].iloc[0]).total_seconds()
    return {60:"1-min",300:"5-min",900:"15-min",1800:"30-min",
            3600:"1-hour",86400:"1-day",604800:"1-week"}.get(int(delta), f"{int(delta//60)}-min")


def _trade_events(trades) -> tuple[pd.DataFrame, pd.DataFrame]:
    entries, exits = [], []
    for i, t in enumerate(trades):
        direction = t.direction.value if hasattr(t.direction, "value") else str(t.direction)
        outcome   = t.outcome.value   if hasattr(t.outcome,   "value") else str(t.outcome)
        ret       = t.leveraged_return_pct
        label     = f"T{i+1}"
        entries.append({"date": pd.Timestamp(t.entry_time), "price": t.entry_price,
                         "direction": direction, "outcome": outcome,
                         "return_pct": ret, "trade_n": label})
        if t.exit_time and t.exit_price is not None:
            exits.append({"date": pd.Timestamp(t.exit_time), "price": t.exit_price,
                          "direction": direction, "outcome": outcome,
                          "return_pct": ret, "trade_n": label,
                          "win": (ret or 0) > 0})
    return pd.DataFrame(entries), pd.DataFrame(exits)


def _outcome_color(outcome: str) -> str:
    """Map outcome string to a display colour."""
    if "Take-profit" in outcome:  return _GREEN
    if "Stop-loss"   in outcome:  return _RED
    if "Counter"     in outcome:  return _ORANGE
    if "Ambiguous"   in outcome:  return _GREY
    return _BLUE


# ─── Price chart ─────────────────────────────────────────────────────────────

def _price_chart(
    prices: pd.DataFrame, trades, symbol: str,
    show_long_entries: bool, show_short_entries: bool,
    show_win_exits: bool, show_loss_exits: bool, show_signal_exits: bool,
) -> alt.LayerChart:

    base = (alt.Chart(prices).mark_line(color=_BLUE, strokeWidth=1.4)
            .encode(x=alt.X("date:T", title="Date / Time", axis=alt.Axis(**_AXIS)),
                    y=alt.Y("close:Q", title="Price", scale=alt.Scale(zero=False),
                            axis=alt.Axis(**_AXIS)),
                    tooltip=["date:T", alt.Tooltip("close:Q", format=".4f")]))
    layers = [base]

    if not trades:
        return (alt.layer(*layers)
                .properties(title=alt.TitleParams(f"{symbol} – Price", **_TITLE), height=340)
                .configure_view(strokeOpacity=0).configure_axis(**_AXIS).configure_title(**_TITLE))

    entry_df, exit_df = _trade_events(trades)

    tt_e = ["date:T","trade_n:N","direction:N",
            alt.Tooltip("price:Q", format=".4f", title="Entry"),
            "outcome:N", alt.Tooltip("return_pct:Q", format=".2f", title="Return %")]

    if not entry_df.empty:
        long_e  = entry_df[entry_df["direction"]=="Long"].copy()
        short_e = entry_df[entry_df["direction"]=="Short"].copy()
        if show_long_entries and not long_e.empty:
            long_e["y"] = long_e["price"] * 0.997
            layers.append(alt.Chart(long_e)
                          .mark_point(shape="triangle-up", size=120, filled=True, color=_GREEN)
                          .encode(x="date:T", y="y:Q", tooltip=tt_e))
        if show_short_entries and not short_e.empty:
            short_e["y"] = short_e["price"] * 1.003
            layers.append(alt.Chart(short_e)
                          .mark_point(shape="triangle-down", size=120, filled=True, color=_RED)
                          .encode(x="date:T", y="y:Q", tooltip=tt_e))

    if not exit_df.empty:
        tt_x = ["date:T","trade_n:N","direction:N",
                alt.Tooltip("price:Q", format=".4f", title="Exit"),
                "outcome:N", alt.Tooltip("return_pct:Q", format=".2f", title="Return %")]

        # Split exits by outcome type
        win_ex    = exit_df[(exit_df["win"]==True)  & ~exit_df["outcome"].str.contains("Counter", na=False)]
        loss_ex   = exit_df[(exit_df["win"]==False) & ~exit_df["outcome"].str.contains("Counter", na=False)]
        signal_ex = exit_df[exit_df["outcome"].str.contains("Counter", na=False)]

        if show_win_exits and not win_ex.empty:
            layers.append(alt.Chart(win_ex)
                          .mark_point(shape="cross", size=110, strokeWidth=2.5, color=_GREEN)
                          .encode(x="date:T", y="price:Q", tooltip=tt_x))
        if show_loss_exits and not loss_ex.empty:
            layers.append(alt.Chart(loss_ex)
                          .mark_point(shape="cross", size=110, strokeWidth=2.5, color=_RED)
                          .encode(x="date:T", y="price:Q", tooltip=tt_x))
        if show_signal_exits and not signal_ex.empty:
            layers.append(alt.Chart(signal_ex)
                          .mark_point(shape="cross", size=110, strokeWidth=2.5, color=_ORANGE)
                          .encode(x="date:T", y="price:Q", tooltip=tt_x))

    return (alt.layer(*layers)
            .properties(title=alt.TitleParams(
                f"{symbol} – Price  ·  ▲ Long entry  ▼ Short entry  ✕ Exit", **_TITLE), height=340)
            .configure_view(strokeOpacity=0).configure_axis(**_AXIS).configure_title(**_TITLE))


# ─── RSI chart ───────────────────────────────────────────────────────────────

def _rsi_chart(
    prices: pd.DataFrame, trades,
    period: int, buy_levels: list[float], sell_levels: list[float],
    bar_label: str, symbol: str,
    show_long_entries: bool, show_short_entries: bool,
    show_win_exits: bool, show_loss_exits: bool, show_signal_exits: bool,
) -> alt.LayerChart:

    rsi_s = _calc_rsi(prices["close"], period).rename("rsi")
    df    = pd.concat([prices[["date"]], rsi_s], axis=1).dropna()

    rsi_line = (alt.Chart(df).mark_line(color=_GOLD, strokeWidth=2.0)
                .encode(x=alt.X("date:T", title="Date / Time", axis=alt.Axis(**_AXIS)),
                        y=alt.Y("rsi:Q", title="RSI", scale=alt.Scale(domain=[0,100]),
                                axis=alt.Axis(**_AXIS)),
                        tooltip=["date:T", alt.Tooltip("rsi:Q", format=".2f")]))
    layers = [rsi_line]

    for lvl in buy_levels:
        lvl_df = pd.DataFrame({"y":[lvl],"label":[f"OS {lvl:.0f}"]})
        layers.append(alt.Chart(lvl_df).mark_rule(color=_GREEN, strokeDash=[5,3], strokeWidth=1.5).encode(y="y:Q"))
        layers.append(alt.Chart(lvl_df).mark_text(align="left", dx=4, dy=-7, fontSize=12,
                                                   color=_GREEN, fontWeight="bold")
                      .encode(y="y:Q", x=alt.value(4), text="label:N"))
    for lvl in sell_levels:
        lvl_df = pd.DataFrame({"y":[lvl],"label":[f"OB {lvl:.0f}"]})
        layers.append(alt.Chart(lvl_df).mark_rule(color=_RED, strokeDash=[5,3], strokeWidth=1.5).encode(y="y:Q"))
        layers.append(alt.Chart(lvl_df).mark_text(align="left", dx=4, dy=-7, fontSize=12,
                                                   color=_RED, fontWeight="bold")
                      .encode(y="y:Q", x=alt.value(4), text="label:N"))
    if buy_levels:
        layers.append(alt.Chart(pd.DataFrame({"y1":[0],"y2":[min(buy_levels)]}))
                      .mark_rect(color=_GREEN, opacity=0.07).encode(y="y1:Q", y2="y2:Q"))
    if sell_levels:
        layers.append(alt.Chart(pd.DataFrame({"y1":[max(sell_levels)],"y2":[100]}))
                      .mark_rect(color=_RED, opacity=0.07).encode(y="y1:Q", y2="y2:Q"))

    if trades:
        entry_df, exit_df = _trade_events(trades)

        def _snap(ts: pd.Timestamp):
            idx = min(df["date"].searchsorted(ts), len(df)-1)
            row = df.iloc[idx]
            return float(row["rsi"]) if not pd.isna(row["rsi"]) else 50.0, row["date"]

        tt = ["date:T","trade_n:N","direction:N",
              alt.Tooltip("rsi_val:Q", format=".1f", title="RSI"),
              "outcome:N", alt.Tooltip("return_pct:Q", format=".2f", title="Return %")]

        if not entry_df.empty:
            entry_df[["rsi_val","date"]] = pd.DataFrame(
                [_snap(r) for r in entry_df["date"]], columns=["rsi_val","date"])
            entry_df["event"] = "Entry"
            long_e  = entry_df[entry_df["direction"]=="Long"].copy()
            short_e = entry_df[entry_df["direction"]=="Short"].copy()
            if show_long_entries and not long_e.empty:
                long_e["y"] = long_e["rsi_val"] - 5
                layers.append(alt.Chart(long_e)
                              .mark_point(shape="triangle-up", size=90, filled=True, color=_GREEN)
                              .encode(x="date:T", y="y:Q", tooltip=tt))
            if show_short_entries and not short_e.empty:
                short_e["y"] = short_e["rsi_val"] + 5
                layers.append(alt.Chart(short_e)
                              .mark_point(shape="triangle-down", size=90, filled=True, color=_RED)
                              .encode(x="date:T", y="y:Q", tooltip=tt))

        if not exit_df.empty:
            exit_df[["rsi_val","date"]] = pd.DataFrame(
                [_snap(r) for r in exit_df["date"]], columns=["rsi_val","date"])
            exit_df["event"] = "Exit"
            win_ex    = exit_df[(exit_df["win"]==True)  & ~exit_df["outcome"].str.contains("Counter", na=False)]
            loss_ex   = exit_df[(exit_df["win"]==False) & ~exit_df["outcome"].str.contains("Counter", na=False)]
            signal_ex = exit_df[exit_df["outcome"].str.contains("Counter", na=False)]

            if show_win_exits and not win_ex.empty:
                layers.append(alt.Chart(win_ex)
                              .mark_point(shape="cross", size=90, strokeWidth=2.5, color=_GREEN)
                              .encode(x="date:T", y="rsi_val:Q", tooltip=tt))
            if show_loss_exits and not loss_ex.empty:
                layers.append(alt.Chart(loss_ex)
                              .mark_point(shape="cross", size=90, strokeWidth=2.5, color=_RED)
                              .encode(x="date:T", y="rsi_val:Q", tooltip=tt))
            if show_signal_exits and not signal_ex.empty:
                layers.append(alt.Chart(signal_ex)
                              .mark_point(shape="cross", size=90, strokeWidth=2.5, color=_ORANGE)
                              .encode(x="date:T", y="rsi_val:Q", tooltip=tt))

    return (alt.layer(*layers)
            .properties(title=alt.TitleParams(
                f"{symbol} – RSI ({period})  ·  bar={bar_label}  ·  "
                f"Buy≤{buy_levels}  Sell≥{sell_levels}", **_TITLE), height=320)
            .configure_view(strokeOpacity=0).configure_axis(**_AXIS).configure_title(**_TITLE))


# ─── Equity curve ─────────────────────────────────────────────────────────────

def _equity_from_trades(trades, starting_equity: float, symbol: str) -> alt.LayerChart:
    closed = [t for t in trades if t.pnl is not None and t.exit_time is not None]
    if not closed:
        return alt.Chart(pd.DataFrame()).mark_line()

    rows, eq = [], starting_equity
    rows.append({"date": pd.Timestamp(closed[0].entry_time), "equity": eq,
                 "trade_n": "Start", "pnl": 0.0, "ret_pct": 0.0, "outcome": "—"})
    for i, t in enumerate(sorted(closed, key=lambda x: x.exit_time)):
        eq += t.pnl
        outcome = t.outcome.value if hasattr(t.outcome,"value") else str(t.outcome)
        rows.append({"date": pd.Timestamp(t.exit_time), "equity": round(eq,2),
                     "trade_n": f"T{i+1}", "pnl": round(t.pnl,2),
                     "ret_pct": round(t.leveraged_return_pct or 0,3),
                     "outcome": outcome})

    eq_df = pd.DataFrame(rows)
    area  = (alt.Chart(eq_df)
             .mark_area(line={"color":_BLUE,"strokeWidth":2},
                        color=alt.Gradient(gradient="linear",
                            stops=[alt.GradientStop(color="rgba(74,158,255,0.25)",offset=0),
                                   alt.GradientStop(color="rgba(74,158,255,0.0)", offset=1)],
                            x1=1,x2=1,y1=1,y2=0))
             .encode(x=alt.X("date:T",title="Date / Time",axis=alt.Axis(**_AXIS)),
                     y=alt.Y("equity:Q",title="Equity ($)",scale=alt.Scale(zero=False),
                             axis=alt.Axis(**_AXIS)),
                     tooltip=["date:T","trade_n:N",
                              alt.Tooltip("equity:Q",  format="$,.2f", title="Equity"),
                              alt.Tooltip("pnl:Q",     format="+$,.2f",title="Trade P&L ($)"),
                              alt.Tooltip("ret_pct:Q", format=".2f",   title="Return % of capital"),
                              "outcome:N"]))
    baseline = (alt.Chart(pd.DataFrame({"y":[starting_equity]}))
                .mark_rule(color=_GREY, strokeDash=[4,4], strokeWidth=1.2).encode(y="y:Q"))
    dots = (alt.Chart(eq_df.iloc[1:])
            .mark_point(size=60, filled=True)
            .encode(x="date:T", y="equity:Q",
                    color=alt.condition(alt.datum.pnl > 0, alt.value(_GREEN), alt.value(_RED)),
                    tooltip=["date:T","trade_n:N",
                             alt.Tooltip("equity:Q",  format="$,.2f", title="Equity"),
                             alt.Tooltip("pnl:Q",     format="+$,.2f",title="Trade P&L ($)"),
                             alt.Tooltip("ret_pct:Q", format=".2f",   title="Return % of capital"),
                             "outcome:N"]))
    return (alt.layer(baseline, area, dots)
            .properties(title=alt.TitleParams(f"{symbol} – Equity (per closed trade)", **_TITLE), height=300)
            .configure_view(strokeOpacity=0).configure_axis(**_AXIS).configure_title(**_TITLE))


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
        f"ℹ️ RSI period = number of **{bar_label}** bars.  "
        "Default is **9** — responsive for intraday. "
        "Shorter (7–9) = less lag, more signals. Longer (14+) = smoother, fewer signals. "
        "For daily bars the classic is 14."
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
        direction_filter  = st.selectbox("Direction filter", ["Both","Long only","Short only"], key="bt_dir")

    with col_risk:
        st.markdown("**Risk Controls**")
        use_risk = st.checkbox("Apply risk manager", value=True, key="bt_risk")
        max_loss = st.slider("Max loss per trade (% of capital)", 5, 100, 50, key="bt_maxloss")
        st.markdown("---")
        counter_signal_exit = st.checkbox(
            "Counter-signal exit",
            value=True,
            key="bt_counter",
            help=(
                "When ON: a SELL signal while Long (or BUY while Short) closes the "
                "current trade immediately and opens the reverse position. "
                "When OFF: trades only exit on SL or TP price hit."
            ),
        )

    st.divider()
    params = render_strategy_params(selected_id)

    if st.button("▶ Run Backtest", type="primary", key="bt_run"):
        cls      = get_strategy(selected_id)
        strategy = cls(params=params)
        errors   = strategy.validate_params()
        if errors:
            for e in errors: st.error(e)
            return

        from config.settings import RiskConfig
        from core.models import Direction as Dir

        risk_cfg = RiskConfig(max_capital_per_trade_pct=100.0, max_daily_loss_pct=100.0,
                              max_open_positions=999,
                              default_max_loss_pct_of_capital=float(max_loss))
        rm = RiskManager(risk_cfg) if use_risk else None

        dir_filter = None
        if direction_filter == "Long only":  dir_filter = Dir.LONG
        if direction_filter == "Short only": dir_filter = Dir.SHORT

        engine = BacktestEngine(strategy, risk_manager=rm, direction_filter=dir_filter,
                                counter_signal_exit=counter_signal_exit)
        with st.spinner("Running backtest…"):
            result = engine.run(data=prices, symbol=symbol, leverage=leverage,
                                capital_per_trade=capital_per_trade,
                                starting_equity=starting_equity)

        closed = [t for t in result.trades if t.leveraged_return_pct is not None]

        # ── Summary ───────────────────────────────────────────────────────────
        st.subheader("📊 Results")
        s = result.summary()
        render_metrics_row({"Total Trades":s["Total Trades"],"Win Rate":s["Win Rate"],
                            "Total Return":s["Total Return"],"Max Drawdown":s["Max Drawdown"],
                            "Sharpe Ratio":s["Sharpe Ratio"],"Avg Win":s["Avg Win"],
                            "Avg Loss":s["Avg Loss"]})
        st.divider()

        # ── Outcome breakdown ────────────────────────────────────────────────
        if closed:
            from collections import Counter
            outcome_counts = Counter(
                (t.outcome.value if hasattr(t.outcome,"value") else str(t.outcome))
                for t in closed
            )
            cols = st.columns(len(outcome_counts))
            for col, (label, cnt) in zip(cols, outcome_counts.items()):
                col.metric(label, cnt)
            st.caption(
                "**PnL** = profit/loss in $ (capital × return%).  "
                "**leveraged_return_pct** = % return on your invested capital after leverage.  "
                "**Counter-signal exit** = trade closed by an opposing RSI signal (not a price TP/SL hit)."
            )
            st.divider()

        # ── Chart visibility toggles ─────────────────────────────────────────
        st.markdown("#### 🎛️ Chart Layers")
        tog_col1, tog_col2, tog_col3, tog_col4, tog_col5 = st.columns(5)
        show_long   = tog_col1.checkbox("▲ Long entries",    value=True,  key="show_long")
        show_short  = tog_col2.checkbox("▼ Short entries",   value=True,  key="show_short")
        show_win_x  = tog_col3.checkbox("✕ Win exits",       value=True,  key="show_win_x")
        show_loss_x = tog_col4.checkbox("✕ Loss exits",      value=True,  key="show_loss_x")
        show_sig_x  = tog_col5.checkbox("✕ Signal exits",    value=True,  key="show_sig_x",
                                         help="Orange ✕ = trade closed by counter-signal (not price SL/TP)")

        # ── Price chart ───────────────────────────────────────────────────────
        st.markdown("#### 📈 Price")
        st.altair_chart(
            _price_chart(prices, result.trades, symbol,
                         show_long, show_short, show_win_x, show_loss_x, show_sig_x),
            use_container_width=True,
        )

        # ── RSI chart ─────────────────────────────────────────────────────────
        if selected_id == "rsi_threshold":
            period      = int(params.get("rsi_period", 9))
            buy_levels  = _parse_levels(params.get("buy_levels",  "30"))
            sell_levels = _parse_levels(params.get("sell_levels", "70"))
            tp_disabled = float(params.get("tp_pct", 3.0)) == 0

            st.markdown(
                f"#### 📉 RSI ({period})"
                + ("  &nbsp;·&nbsp; ⚠️ TP disabled — counter-signal or SL exit only" if tp_disabled else "")
            )
            st.altair_chart(
                _rsi_chart(prices, result.trades, period, buy_levels, sell_levels,
                           bar_label, symbol,
                           show_long, show_short, show_win_x, show_loss_x, show_sig_x),
                use_container_width=True,
            )

        # ── Equity curve ──────────────────────────────────────────────────────
        if closed:
            st.markdown("#### 💰 Equity Curve — one dot per closed trade")
            st.altair_chart(_equity_from_trades(result.trades, starting_equity, symbol),
                            use_container_width=True)

        # ── Per-trade P&L bars ────────────────────────────────────────────────
        if closed:
            trades_df = pd.DataFrame([{
                "symbol":               t.symbol,
                "direction":            t.direction.value,
                "entry_price":          t.entry_price,
                "exit_price":           t.exit_price,
                "outcome":              t.outcome.value if t.outcome else None,
                "leveraged_return_pct": t.leveraged_return_pct,
                "pnl":                  t.pnl,
                "entry_time":           t.entry_time,
                "exit_time":            t.exit_time,
                "notes":                t.notes,
            } for t in closed])

            st.markdown("#### 📊 Per-Trade Return")
            st.altair_chart(pnl_distribution(trades_df), use_container_width=True)

            with st.expander("📋 Trade Log", expanded=False):
                display_df = trades_df.rename(columns={
                    "leveraged_return_pct": "return_pct (%)",
                    "pnl": "PnL ($)",
                })
                st.caption(
                    "**return_pct (%)** = leveraged return on capital. "
                    "**PnL ($)** = dollar profit/loss = capital × return_pct / 100.  "
                    "Outcome **Counter-signal exit** = closed by opposing RSI crossing, not a price level."
                )
                st.dataframe(display_df.sort_values("entry_time", ascending=False),
                             use_container_width=True)

        try:
            from db.database import Database
            db = Database(settings.db_path)
            for t in result.trades: db.save_trade(t)
            st.caption(f"✓ {len(result.trades)} trades saved to database.")
        except Exception as e:
            st.caption(f"DB save skipped: {e}")

"""
pages/page_backtest.py
Walk-forward backtester.

Rendering pattern:
  1. Run button sets session_state["bt_result"] + sets bt_ran=True
  2. st.rerun() triggers a clean re-render
  3. On re-render: prices auto-loaded from session_state["loaded_data"]
  4. Results rendered from session_state — always visible, survives checkbox clicks
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
from ui.charts import pnl_distribution

_GREEN  = "#26a69a"
_RED    = "#ef5350"
_BLUE   = "#4a9eff"
_GOLD   = "#ffd54f"
_GREY   = "#9e9eb8"
_ORANGE = "#ff9800"
_AXIS   = dict(gridColor="#2a2d3e", labelColor="#d0d4f0", titleColor="#d0d4f0",
               labelFontSize=12, titleFontSize=13)
_TITLE  = dict(color="#e8eaf6", fontSize=14, fontWeight="bold")

_OUTCOME_COLOR = {
    "TP hit":               _GREEN,
    "SL hit":               _RED,
    "RSI overbought exit":  _ORANGE,
    "RSI oversold exit":    _ORANGE,
    "Counter-signal exit":  _ORANGE,
    "Ambiguous candle":     _GREY,
    "Open":                 _BLUE,
}

_MAX_CHART_PTS = 5_000   # max points sent to Altair for price / RSI charts


def _downsample(df: pd.DataFrame, max_pts: int = _MAX_CHART_PTS) -> pd.DataFrame:
    if len(df) <= max_pts:
        return df
    step = max(1, len(df) // max_pts)
    return df.iloc[::step].reset_index(drop=True)


def _is_signal_exit(outcome: str) -> bool:
    return any(k in outcome for k in ("overbought", "oversold", "Counter"))


def _calc_rsi(series: pd.Series, period: int) -> pd.Series:
    d  = series.diff()
    g  = d.clip(lower=0)
    l  = (-d).clip(lower=0)
    ag = g.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    al = l.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    return 100 - (100 / (1 + ag / al.replace(0, float("nan"))))


def _parse_levels(raw) -> list[float]:
    if isinstance(raw, (int, float)): return [float(raw)]
    s = str(raw).strip().lower()
    if s in ("", "none", "off", "-"): return []
    try:
        return sorted(float(p.strip()) for p in s.replace(";",",").split(",") if p.strip())
    except ValueError:
        return []


def _bar_label(prices: pd.DataFrame) -> str:
    if len(prices) < 2: return "bars"
    delta = (prices["date"].iloc[1] - prices["date"].iloc[0]).total_seconds()
    return {60:"1-min",300:"5-min",900:"15-min",1800:"30-min",
            3600:"1-hour",86400:"1-day"}.get(int(delta), f"{int(delta//60)}-min")


def _trade_events(trades) -> tuple[pd.DataFrame, pd.DataFrame]:
    entries, exits = [], []
    for i, t in enumerate(trades):
        direction = t.direction.value if hasattr(t.direction,"value") else str(t.direction)
        outcome   = t.outcome.value   if hasattr(t.outcome,  "value") else str(t.outcome)
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


# ─── Charts ───────────────────────────────────────────────────────────────────

def _price_chart(prices, trades, symbol,
                 show_long, show_short, show_tp, show_sl, show_sig):
    base = (alt.Chart(prices).mark_line(color=_BLUE, strokeWidth=1.2)
            .encode(x=alt.X("date:T", title="Date / Time", axis=alt.Axis(**_AXIS)),
                    y=alt.Y("close:Q", title="Price", scale=alt.Scale(zero=False),
                            axis=alt.Axis(**_AXIS)),
                    tooltip=["date:T", alt.Tooltip("close:Q", format=".4f")]))
    layers = [base]
    if not trades:
        return (alt.layer(*layers)
                .properties(title=alt.TitleParams(f"{symbol} – Price", **_TITLE), height=320)
                .configure_view(strokeOpacity=0).configure_axis(**_AXIS).configure_title(**_TITLE))

    entry_df, exit_df = _trade_events(trades)
    tt_e = ["date:T","trade_n:N","direction:N",
            alt.Tooltip("price:Q",format=".4f",title="Entry"),"outcome:N",
            alt.Tooltip("return_pct:Q",format=".2f",title="Return %")]

    if not entry_df.empty:
        long_e  = entry_df[entry_df["direction"]=="Long"].copy()
        short_e = entry_df[entry_df["direction"]=="Short"].copy()
        if show_long and not long_e.empty:
            long_e["y"] = long_e["price"] * 0.997
            layers.append(alt.Chart(long_e)
                .mark_point(shape="triangle-up",size=120,filled=True,color=_GREEN)
                .encode(x="date:T",y="y:Q",tooltip=tt_e))
        if show_short and not short_e.empty:
            short_e["y"] = short_e["price"] * 1.003
            layers.append(alt.Chart(short_e)
                .mark_point(shape="triangle-down",size=120,filled=True,color=_RED)
                .encode(x="date:T",y="y:Q",tooltip=tt_e))

    if not exit_df.empty:
        tt_x = ["date:T","trade_n:N","direction:N",
                alt.Tooltip("price:Q",format=".4f",title="Exit"),"outcome:N",
                alt.Tooltip("return_pct:Q",format=".2f",title="Return %")]
        tp_ex  = exit_df[exit_df["outcome"]=="TP hit"]
        sl_ex  = exit_df[exit_df["outcome"]=="SL hit"]
        sig_ex = exit_df[exit_df["outcome"].apply(_is_signal_exit)]
        if show_tp  and not tp_ex.empty:
            layers.append(alt.Chart(tp_ex)
                .mark_point(shape="cross",size=110,strokeWidth=2.5,color=_GREEN)
                .encode(x="date:T",y="price:Q",tooltip=tt_x))
        if show_sl  and not sl_ex.empty:
            layers.append(alt.Chart(sl_ex)
                .mark_point(shape="cross",size=110,strokeWidth=2.5,color=_RED)
                .encode(x="date:T",y="price:Q",tooltip=tt_x))
        if show_sig and not sig_ex.empty:
            layers.append(alt.Chart(sig_ex)
                .mark_point(shape="cross",size=110,strokeWidth=2.5,color=_ORANGE)
                .encode(x="date:T",y="price:Q",tooltip=tt_x))

    return (alt.layer(*layers)
            .properties(title=alt.TitleParams(
                f"{symbol} – Price  ▲ Long  ▼ Short  ✕ Exit", **_TITLE), height=320)
            .configure_view(strokeOpacity=0).configure_axis(**_AXIS).configure_title(**_TITLE))


def _rsi_chart(prices, trades, period, buy_levels, sell_levels, bar_label, symbol,
               show_long, show_short, show_tp, show_sl, show_sig):
    rsi_s = _calc_rsi(prices["close"], period).rename("rsi")
    df    = pd.concat([prices[["date"]], rsi_s], axis=1).dropna()

    rsi_line = (alt.Chart(df).mark_line(color=_GOLD, strokeWidth=1.8)
                .encode(x=alt.X("date:T",title="Date / Time",axis=alt.Axis(**_AXIS)),
                        y=alt.Y("rsi:Q",title="RSI",scale=alt.Scale(domain=[0,100]),
                                axis=alt.Axis(**_AXIS)),
                        tooltip=["date:T",alt.Tooltip("rsi:Q",format=".2f")]))
    layers = [rsi_line]

    for lvl in buy_levels:
        ldf = pd.DataFrame({"y":[lvl],"label":[f"OS {lvl:.0f}"]})
        layers += [
            alt.Chart(ldf).mark_rule(color=_GREEN,strokeDash=[5,3],strokeWidth=1.5).encode(y="y:Q"),
            alt.Chart(ldf).mark_text(align="left",dx=4,dy=-7,fontSize=12,color=_GREEN,fontWeight="bold")
                .encode(y="y:Q",x=alt.value(4),text="label:N"),
        ]
    for lvl in sell_levels:
        ldf = pd.DataFrame({"y":[lvl],"label":[f"OB {lvl:.0f}"]})
        layers += [
            alt.Chart(ldf).mark_rule(color=_RED,strokeDash=[5,3],strokeWidth=1.5).encode(y="y:Q"),
            alt.Chart(ldf).mark_text(align="left",dx=4,dy=-7,fontSize=12,color=_RED,fontWeight="bold")
                .encode(y="y:Q",x=alt.value(4),text="label:N"),
        ]
    if buy_levels:
        layers.append(alt.Chart(pd.DataFrame({"y1":[0],"y2":[min(buy_levels)]}))
                      .mark_rect(color=_GREEN,opacity=0.07).encode(y="y1:Q",y2="y2:Q"))
    if sell_levels:
        layers.append(alt.Chart(pd.DataFrame({"y1":[max(sell_levels)],"y2":[100]}))
                      .mark_rect(color=_RED,opacity=0.07).encode(y="y1:Q",y2="y2:Q"))

    def _snap(ts):
        idx = min(df["date"].searchsorted(pd.Timestamp(ts)), len(df)-1)
        row = df.iloc[idx]
        return (float(row["rsi"]) if not pd.isna(row["rsi"]) else 50.0), row["date"]

    if trades:
        entry_df, exit_df = _trade_events(trades)
        tt = ["date:T","trade_n:N","direction:N",
              alt.Tooltip("rsi_val:Q",format=".1f",title="RSI"),"outcome:N",
              alt.Tooltip("return_pct:Q",format=".2f",title="Return %")]

        if not entry_df.empty:
            entry_df[["rsi_val","date"]] = pd.DataFrame(
                [_snap(r) for r in entry_df["date"]], columns=["rsi_val","date"])
            long_e  = entry_df[entry_df["direction"]=="Long"].copy()
            short_e = entry_df[entry_df["direction"]=="Short"].copy()
            if show_long and not long_e.empty:
                long_e["y"] = long_e["rsi_val"] - 5
                layers.append(alt.Chart(long_e)
                    .mark_point(shape="triangle-up",size=90,filled=True,color=_GREEN)
                    .encode(x="date:T",y="y:Q",tooltip=tt))
            if show_short and not short_e.empty:
                short_e["y"] = short_e["rsi_val"] + 5
                layers.append(alt.Chart(short_e)
                    .mark_point(shape="triangle-down",size=90,filled=True,color=_RED)
                    .encode(x="date:T",y="y:Q",tooltip=tt))

        if not exit_df.empty:
            exit_df[["rsi_val","date"]] = pd.DataFrame(
                [_snap(r) for r in exit_df["date"]], columns=["rsi_val","date"])
            tp_ex  = exit_df[exit_df["outcome"]=="TP hit"]
            sl_ex  = exit_df[exit_df["outcome"]=="SL hit"]
            sig_ex = exit_df[exit_df["outcome"].apply(_is_signal_exit)]
            if show_tp  and not tp_ex.empty:
                layers.append(alt.Chart(tp_ex)
                    .mark_point(shape="cross",size=90,strokeWidth=2.5,color=_GREEN)
                    .encode(x="date:T",y="rsi_val:Q",tooltip=tt))
            if show_sl  and not sl_ex.empty:
                layers.append(alt.Chart(sl_ex)
                    .mark_point(shape="cross",size=90,strokeWidth=2.5,color=_RED)
                    .encode(x="date:T",y="rsi_val:Q",tooltip=tt))
            if show_sig and not sig_ex.empty:
                layers.append(alt.Chart(sig_ex)
                    .mark_point(shape="cross",size=90,strokeWidth=2.5,color=_ORANGE)
                    .encode(x="date:T",y="rsi_val:Q",tooltip=tt))

    return (alt.layer(*layers)
            .properties(title=alt.TitleParams(
                f"{symbol} – RSI ({period})  Buy≤{buy_levels}  Sell≥{sell_levels}", **_TITLE),
                height=300)
            .configure_view(strokeOpacity=0).configure_axis(**_AXIS).configure_title(**_TITLE))


def _equity_chart(trades, starting_equity, symbol):
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
                     "ret_pct": round(t.leveraged_return_pct or 0, 3),
                     "outcome": outcome})

    eq_df = pd.DataFrame(rows)
    tt_eq = ["date:T","trade_n:N",
             alt.Tooltip("equity:Q",format="$,.2f",title="Equity"),
             alt.Tooltip("pnl:Q",format="+$,.2f",title="PnL ($)"),
             alt.Tooltip("ret_pct:Q",format=".2f",title="Return %"),
             "outcome:N"]
    area = (alt.Chart(eq_df)
            .mark_area(line={"color":_BLUE,"strokeWidth":2},
                       color=alt.Gradient(gradient="linear",
                           stops=[alt.GradientStop(color="rgba(74,158,255,0.25)",offset=0),
                                  alt.GradientStop(color="rgba(74,158,255,0.0)", offset=1)],
                           x1=1,x2=1,y1=1,y2=0))
            .encode(x=alt.X("date:T",title="Date / Time",axis=alt.Axis(**_AXIS)),
                    y=alt.Y("equity:Q",title="Equity ($)",scale=alt.Scale(zero=False),
                            axis=alt.Axis(**_AXIS)), tooltip=tt_eq))
    baseline = (alt.Chart(pd.DataFrame({"y":[starting_equity]}))
                .mark_rule(color=_GREY,strokeDash=[4,4],strokeWidth=1.2).encode(y="y:Q"))
    dots = (alt.Chart(eq_df.iloc[1:]).mark_point(size=65,filled=True)
            .encode(x="date:T",y="equity:Q",
                    color=alt.condition(alt.datum.pnl > 0,alt.value(_GREEN),alt.value(_RED)),
                    tooltip=tt_eq))
    return (alt.layer(baseline,area,dots)
            .properties(title=alt.TitleParams(f"{symbol} – Equity (per closed trade)",**_TITLE),height=300)
            .configure_view(strokeOpacity=0).configure_axis(**_AXIS).configure_title(**_TITLE))


# ─── Page ─────────────────────────────────────────────────────────────────────

def render() -> None:
    render_mode_banner()
    st.title("⏪ Backtester")

    # ── Load data ─────────────────────────────────────────────────────────────
    prices = render_data_source_selector()

    # Ensure loaded_data is always in session_state if prices came back
    if prices is not None:
        st.session_state["bt_prices_live"] = prices
        st.session_state["bt_symbol_live"] = st.session_state.get("loaded_symbol","DATA")

    # Use live prices if available, otherwise nothing to configure
    prices = st.session_state.get("bt_prices_live")

    if prices is None:
        st.info("← Select a data source in the sidebar to begin.")
        # Still show any previous results
        if "bt_result" in st.session_state:
            st.divider()
            _show_results()
        return

    symbol    = st.session_state.get("bt_symbol_live", "DATA")
    bar_label = _bar_label(prices)
    st.success(f"**{symbol}** — {len(prices):,} bars · bar size: **{bar_label}**")
    st.divider()

    # ── Config form ───────────────────────────────────────────────────────────
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
            "Counter-signal exit", value=True, key="bt_counter",
            help="When ON: opposing RSI signal closes the current trade and opens reverse.")
        st.markdown("---")
        st.markdown("**Transaction Costs**")
        st.caption(
            "UVXY realistic costs: spread ~0.06%, slippage ~0.02%.  \n"
            "Leave at 0 for gross return (no costs), set realistic values "
            "to see net return.  \n"
            "⚠️ At 50k trades/year, even 0.06% spread = strategy killer."
        )
        spread_pct   = st.number_input("Spread % (round-trip)", 0.0, 2.0, 0.06,
                                        step=0.01, format="%.2f", key="bt_spread",
                                        help="Bid-ask spread cost, round trip. UVXY ≈ 0.06%")
        slippage_pct = st.number_input("Slippage % (round-trip)", 0.0, 2.0, 0.02,
                                        step=0.01, format="%.2f", key="bt_slip",
                                        help="Execution slippage. UVXY 1-min ≈ 0.02%")
        commission   = st.number_input("Commission per trade ($)", 0.0, 10.0, 0.0,
                                        step=0.10, format="%.2f", key="bt_comm",
                                        help="Flat $ per trade. Alpaca is free but spread/slippage apply.")

    st.divider()
    params = render_strategy_params(selected_id, leverage=leverage,
                                    max_capital_loss_pct=float(max_loss))

    # ── Run button ────────────────────────────────────────────────────────────
    run_clicked = st.button("▶ Run Backtest", type="primary", key="bt_run")

    # ── Show any existing results BEFORE potentially running again ────────────
    if "bt_result" in st.session_state:
        st.divider()
        _show_results()

    # ── Execute run if button was clicked ─────────────────────────────────────
    if run_clicked:
        cls      = get_strategy(selected_id)
        strategy = cls(params=params)
        errors   = strategy.validate_params()
        if errors:
            for e in errors:
                st.error(e)
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

        engine = BacktestEngine(strategy, risk_manager=rm,
                                direction_filter=dir_filter,
                                counter_signal_exit=counter_signal_exit,
                                spread_pct=float(spread_pct),
                                slippage_pct=float(slippage_pct),
                                commission_per_trade=float(commission))
        n_bars     = len(prices)
        speed_note = "a few seconds" if n_bars < 200_000 else "~30-60 seconds"

        with st.spinner(f"Running backtest on {n_bars:,} bars — {speed_note}…"):
            result = engine.run(data=prices, symbol=symbol, leverage=leverage,
                                capital_per_trade=capital_per_trade,
                                starting_equity=starting_equity)

        # Persist result and all rendering context
        st.session_state["bt_result"]          = result
        st.session_state["bt_symbol"]          = symbol
        st.session_state["bt_bar_label"]       = bar_label
        st.session_state["bt_selected_id"]     = selected_id
        st.session_state["bt_params"]          = dict(params)
        st.session_state["bt_starting_equity"] = float(starting_equity)

        try:
            from db.database import Database
            db = Database(settings.db_path)
            for t in result.trades:
                db.save_trade(t)
            st.session_state["bt_db_msg"] = f"✓ {len(result.trades)} trades saved."
        except Exception as e:
            st.session_state["bt_db_msg"] = f"DB save skipped: {e}"

        # Trigger a clean re-render — this time prices will load from
        # session_state["bt_prices_live"] so the page won't return early
        st.rerun()


def _show_results() -> None:
    """Render backtest results from session_state."""
    if "bt_result" not in st.session_state:
        return

    result        = st.session_state["bt_result"]
    symbol_r      = st.session_state.get("bt_symbol", "DATA")
    bar_label_r   = st.session_state.get("bt_bar_label", "bars")
    selected_id_r = st.session_state.get("bt_selected_id", "")
    params_r      = st.session_state.get("bt_params", {})
    start_eq      = st.session_state.get("bt_starting_equity", 10_000.0)
    prices_r      = st.session_state.get("bt_prices_live")

    closed = [t for t in result.trades if t.leveraged_return_pct is not None]

    # ── Summary metrics ───────────────────────────────────────────────────────
    st.subheader("📊 Results")
    s = result.summary()
    render_metrics_row({
        "Total Trades":   s["Total Trades"],
        "Win Rate":       s["Win Rate"],
        "Total Return":   s["Total Return"],
        "Max Drawdown":   s["Max Drawdown"],
        "Sharpe Ratio":   s["Sharpe Ratio"],
        "Avg Win":        s["Avg Win"],
        "Avg Loss":       s["Avg Loss"],
    })

    if closed:
        from collections import Counter
        outcome_counts = Counter(
            t.outcome.value if hasattr(t.outcome,"value") else str(t.outcome)
            for t in closed
        )
        st.markdown("**Exit breakdown:**")
        cols = st.columns(min(len(outcome_counts), 5))
        for col, (label, cnt) in zip(cols, sorted(outcome_counts.items())):
            col.metric(label, cnt)

    st.caption(
        "📖 **PnL ($)** = dollar profit/loss · **Return %** = leveraged return on capital · "
        "**TP hit** = price target reached · **SL hit** = stop hit · "
        "**RSI exits** = RSI threshold crossed"
    )
    st.divider()

    # ── Layer toggles ─────────────────────────────────────────────────────────
    st.markdown("#### 🎛️ Chart Layers")
    c1,c2,c3,c4,c5 = st.columns(5)
    show_long  = c1.checkbox("▲ Long entries",  value=True, key="show_long")
    show_short = c2.checkbox("▼ Short entries", value=True, key="show_short")
    show_tp    = c3.checkbox("✕ TP exits",      value=True, key="show_tp_x")
    show_sl    = c4.checkbox("✕ SL exits",      value=True, key="show_sl_x")
    show_sig   = c5.checkbox("✕ Signal exits",  value=True, key="show_sig_x")

    # ── Price chart ───────────────────────────────────────────────────────────
    if prices_r is not None:
        prices_plot = _downsample(prices_r)
        n_bars      = len(prices_r)
        label_extra = (f"  ·  *{len(prices_plot):,} of {n_bars:,} bars shown*"
                       if len(prices_plot) < n_bars else "")
        st.markdown(f"#### 📈 Price{label_extra}")
        st.altair_chart(
            _price_chart(prices_plot, result.trades, symbol_r,
                         show_long, show_short, show_tp, show_sl, show_sig),
            use_container_width=True,
        )

        # ── RSI chart ─────────────────────────────────────────────────────────
        rsi_strategies = ("rsi_threshold","atr_rsi","vwap_rsi",
                          "bollinger_rsi","ema_trend_rsi")
        if selected_id_r in rsi_strategies:
            period      = int(params_r.get("rsi_period", 9))
            buy_levels  = _parse_levels(params_r.get("buy_levels",  "30"))
            sell_levels = _parse_levels(params_r.get("sell_levels", "70"))
            tp_disabled = float(params_r.get("tp_pct", 3.0)) == 0
            notes = []
            if not buy_levels:  notes.append("🚫 No Long entries")
            if not sell_levels: notes.append("🚫 No Short entries")
            if tp_disabled:     notes.append("⚠️ TP=0")
            note_str = ("  ·  " + "  ·  ".join(notes)) if notes else ""
            st.markdown(f"#### 📉 RSI ({period}){note_str}")
            st.altair_chart(
                _rsi_chart(prices_plot, result.trades, period, buy_levels, sell_levels,
                           bar_label_r, symbol_r,
                           show_long, show_short, show_tp, show_sl, show_sig),
                use_container_width=True,
            )
    else:
        st.info("ℹ️ Price chart not available — reload data to see charts.")

    # ── Equity curve ──────────────────────────────────────────────────────────
    if closed:
        st.markdown("#### 💰 Equity Curve")
        st.altair_chart(_equity_chart(result.trades, start_eq, symbol_r),
                        use_container_width=True)

    # ── Per-trade distribution ────────────────────────────────────────────────
    if closed:
        trades_df = pd.DataFrame([{
            "symbol":              t.symbol,
            "direction":          t.direction.value,
            "entry_price":        t.entry_price,
            "exit_price":         t.exit_price,
            "outcome":            t.outcome.value if t.outcome else None,
            "leveraged_return_pct": t.leveraged_return_pct,
            "pnl":                t.pnl,
            "entry_time":         t.entry_time,
            "exit_time":          t.exit_time,
        } for t in closed])

        st.markdown("#### 📊 Per-Trade Return")
        st.altair_chart(pnl_distribution(trades_df), use_container_width=True)

        with st.expander("📋 Trade Log", expanded=False):
            st.dataframe(
                trades_df.rename(columns={
                    "leveraged_return_pct": "return_pct (%)",
                    "pnl": "PnL ($)",
                }).sort_values("entry_time", ascending=False),
                use_container_width=True,
            )

    if "bt_db_msg" in st.session_state:
        st.caption(st.session_state["bt_db_msg"])

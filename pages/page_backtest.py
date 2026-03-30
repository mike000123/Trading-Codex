"""
pages/page_backtest.py
───────────────────────
Walk-forward backtester with indicator chart + trade entry/exit markers.
All charts use Altair (no Plotly).
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
_AXIS  = dict(gridColor="#1e2130", labelColor="#c9d8f5", titleColor="#c9d8f5")


# ─── RSI indicator chart with trade markers ──────────────────────────────────

def _calc_rsi(series: pd.Series, period: int) -> pd.Series:
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs       = avg_gain / avg_loss.replace(0, float("nan"))
    return 100 - (100 / (1 + rs))


def _rsi_with_trades_chart(
    prices: pd.DataFrame,
    trades,           # list[TradeRecord]
    period: int,
    oversold: float,
    overbought: float,
    bar_label: str,
    symbol: str,
) -> alt.LayerChart:
    """
    RSI line + threshold bands + BUY/SELL trade entry markers.
    bar_label  – e.g. "5-min bars" or "1-day bars", shown in caption.
    """
    rsi_vals = _calc_rsi(prices["close"], period).rename("rsi")
    df = pd.concat([prices[["date"]], rsi_vals], axis=1).dropna()

    # ── RSI line ─────────────────────────────────────────────────────────────
    rsi_line = (
        alt.Chart(df)
        .mark_line(color=_GOLD, strokeWidth=1.5)
        .encode(
            x=alt.X("date:T", title="Date / Time"),
            y=alt.Y("rsi:Q", title="RSI",
                    scale=alt.Scale(domain=[0, 100])),
            tooltip=["date:T", alt.Tooltip("rsi:Q", format=".2f")],
        )
    )

    # ── Oversold / overbought threshold lines ────────────────────────────────
    ob_df = pd.DataFrame({"y": [overbought], "label": [f"OB {overbought:.0f}"]})
    os_df = pd.DataFrame({"y": [oversold],   "label": [f"OS {oversold:.0f}"]})

    ob_rule = (alt.Chart(ob_df)
               .mark_rule(color=_RED, strokeDash=[5, 3], strokeWidth=1.2)
               .encode(y="y:Q"))
    os_rule = (alt.Chart(os_df)
               .mark_rule(color=_GREEN, strokeDash=[5, 3], strokeWidth=1.2)
               .encode(y="y:Q"))
    ob_text = (alt.Chart(ob_df)
               .mark_text(align="left", dx=4, dy=-7, fontSize=10, color=_RED)
               .encode(y="y:Q", x=alt.value(4), text="label:N"))
    os_text = (alt.Chart(os_df)
               .mark_text(align="left", dx=4, dy=-7, fontSize=10, color=_GREEN)
               .encode(y="y:Q", x=alt.value(4), text="label:N"))

    # ── Shaded bands ─────────────────────────────────────────────────────────
    ob_band_df = pd.DataFrame({"y1": [overbought], "y2": [100]})
    os_band_df = pd.DataFrame({"y1": [0], "y2": [oversold]})
    ob_band = (alt.Chart(ob_band_df)
               .mark_rect(color=_RED, opacity=0.05)
               .encode(y="y1:Q", y2="y2:Q"))
    os_band = (alt.Chart(os_band_df)
               .mark_rect(color=_GREEN, opacity=0.05)
               .encode(y="y1:Q", y2="y2:Q"))

    layers = [ob_band, os_band, rsi_line, ob_rule, os_rule, ob_text, os_text]

    # ── Trade entry markers ───────────────────────────────────────────────────
    # Join each trade's entry_time to the nearest RSI value so we know the y position
    if trades:
        trade_rows = []
        rsi_lookup = df.set_index("date")["rsi"]
        for t in trades:
            entry_ts = pd.Timestamp(t.entry_time)
            # Snap to nearest available RSI bar
            idx = df["date"].searchsorted(entry_ts)
            idx = min(idx, len(df) - 1)
            rsi_at_entry = float(df.iloc[idx]["rsi"]) if not pd.isna(df.iloc[idx]["rsi"]) else 50.0
            date_at_entry = df.iloc[idx]["date"]

            direction = t.direction.value if hasattr(t.direction, "value") else str(t.direction)
            outcome   = t.outcome.value   if hasattr(t.outcome,   "value") else str(t.outcome)
            trade_rows.append({
                "date":      date_at_entry,
                "rsi":       rsi_at_entry,
                "direction": direction,
                "outcome":   outcome,
                "entry_price": t.entry_price,
                "exit_price":  t.exit_price,
                "return_pct":  t.leveraged_return_pct,
            })

        if trade_rows:
            tm_df = pd.DataFrame(trade_rows)

            # Separate long (BUY ▲) and short (SELL ▼) entries
            long_df  = tm_df[tm_df["direction"] == "Long"].copy()
            short_df = tm_df[tm_df["direction"] == "Short"].copy()

            colour_scale = alt.Scale(
                domain=["Long", "Short"],
                range=[_GREEN, _RED],
            )

            if not long_df.empty:
                long_df["y_marker"] = long_df["rsi"] - 4   # below the RSI line
                layers.append(
                    alt.Chart(long_df)
                    .mark_point(shape="triangle-up", size=80, filled=True)
                    .encode(
                        x="date:T",
                        y=alt.Y("y_marker:Q"),
                        color=alt.value(_GREEN),
                        tooltip=[
                            "date:T",
                            alt.Tooltip("rsi:Q", format=".1f", title="RSI"),
                            "direction:N", "outcome:N",
                            alt.Tooltip("entry_price:Q", format=".4f"),
                            alt.Tooltip("return_pct:Q", format=".2f", title="Return %"),
                        ],
                    )
                )

            if not short_df.empty:
                short_df["y_marker"] = short_df["rsi"] + 4  # above the RSI line
                layers.append(
                    alt.Chart(short_df)
                    .mark_point(shape="triangle-down", size=80, filled=True)
                    .encode(
                        x="date:T",
                        y=alt.Y("y_marker:Q"),
                        color=alt.value(_RED),
                        tooltip=[
                            "date:T",
                            alt.Tooltip("rsi:Q", format=".1f", title="RSI"),
                            "direction:N", "outcome:N",
                            alt.Tooltip("entry_price:Q", format=".4f"),
                            alt.Tooltip("return_pct:Q", format=".2f", title="Return %"),
                        ],
                    )
                )

    return (
        alt.layer(*layers)
        .properties(
            title=f"{symbol} – RSI ({period}) · bar = {bar_label}",
            height=260,
        )
        .configure_view(strokeOpacity=0)
        .configure_axis(**_AXIS)
        .configure_title(color="#c9d8f5")
    )


def _price_with_trades_chart(
    prices: pd.DataFrame,
    trades,
    symbol: str,
) -> alt.LayerChart:
    """Close price line with entry (▲/▼) and exit (✕) markers."""
    base = (
        alt.Chart(prices)
        .mark_line(color=_BLUE, strokeWidth=1.5)
        .encode(
            x=alt.X("date:T", title="Date / Time"),
            y=alt.Y("close:Q", title="Price", scale=alt.Scale(zero=False)),
            tooltip=["date:T", alt.Tooltip("close:Q", format=".4f")],
        )
    )
    layers = [base]

    if trades:
        entry_rows, exit_rows = [], []
        for t in trades:
            direction = t.direction.value if hasattr(t.direction, "value") else str(t.direction)
            outcome   = t.outcome.value   if hasattr(t.outcome,   "value") else str(t.outcome)
            entry_rows.append({
                "date":  pd.Timestamp(t.entry_time),
                "price": t.entry_price,
                "direction": direction,
                "outcome": outcome,
                "return_pct": t.leveraged_return_pct,
            })
            if t.exit_time and t.exit_price:
                color = _GREEN if (t.leveraged_return_pct or 0) > 0 else _RED
                exit_rows.append({
                    "date":  pd.Timestamp(t.exit_time),
                    "price": t.exit_price,
                    "outcome": outcome,
                    "return_pct": t.leveraged_return_pct,
                    "color": color,
                })

        if entry_rows:
            en_df = pd.DataFrame(entry_rows)
            long_en  = en_df[en_df["direction"] == "Long"].copy()
            short_en = en_df[en_df["direction"] == "Short"].copy()
            if not long_en.empty:
                long_en["y"] = long_en["price"] * 0.997
                layers.append(
                    alt.Chart(long_en)
                    .mark_point(shape="triangle-up", size=90, filled=True, color=_GREEN)
                    .encode(
                        x="date:T", y="y:Q",
                        tooltip=["date:T",
                                 alt.Tooltip("price:Q", format=".4f", title="Entry"),
                                 "direction:N", "outcome:N",
                                 alt.Tooltip("return_pct:Q", format=".2f", title="Return %")],
                    )
                )
            if not short_en.empty:
                short_en["y"] = short_en["price"] * 1.003
                layers.append(
                    alt.Chart(short_en)
                    .mark_point(shape="triangle-down", size=90, filled=True, color=_RED)
                    .encode(
                        x="date:T", y="y:Q",
                        tooltip=["date:T",
                                 alt.Tooltip("price:Q", format=".4f", title="Entry"),
                                 "direction:N", "outcome:N",
                                 alt.Tooltip("return_pct:Q", format=".2f", title="Return %")],
                    )
                )

        if exit_rows:
            ex_df = pd.DataFrame(exit_rows)
            # Colour exits green=profit, red=loss
            win_ex  = ex_df[ex_df["return_pct"].fillna(0) > 0]
            loss_ex = ex_df[ex_df["return_pct"].fillna(0) <= 0]
            for sub, col in [(win_ex, _GREEN), (loss_ex, _RED)]:
                if not sub.empty:
                    layers.append(
                        alt.Chart(sub)
                        .mark_point(shape="cross", size=80, filled=True, color=col, strokeWidth=2)
                        .encode(
                            x="date:T", y="price:Q",
                            tooltip=["date:T",
                                     alt.Tooltip("price:Q", format=".4f", title="Exit"),
                                     "outcome:N",
                                     alt.Tooltip("return_pct:Q", format=".2f", title="Return %")],
                        )
                    )

    return (
        alt.layer(*layers)
        .properties(title=f"{symbol} – Price with Trade Entries/Exits", height=280)
        .configure_view(strokeOpacity=0)
        .configure_axis(**_AXIS)
        .configure_title(color="#c9d8f5")
    )


# ─── Bar label helper ─────────────────────────────────────────────────────────

def _bar_label(prices: pd.DataFrame) -> str:
    """Infer a human-readable bar-size string from the data timestamps."""
    if len(prices) < 2:
        return "bars"
    delta = (prices["date"].iloc[1] - prices["date"].iloc[0]).total_seconds()
    mapping = {
        60: "1-min", 300: "5-min", 900: "15-min", 1800: "30-min",
        3600: "1-hour", 86400: "1-day", 604800: "1-week",
    }
    return mapping.get(int(delta), f"{int(delta//60)}-min")


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
        f"ℹ️ RSI period counts **{bar_label} bars**. "
        "For intraday data (5-min bars) try period 7–9; for 1-day bars the classic 14 applies."
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

        closed_trades = [t for t in result.trades if t.leveraged_return_pct is not None]

        # ── Summary metrics ───────────────────────────────────────────────────
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

        # ── Price chart with trade markers ────────────────────────────────────
        st.markdown("#### 📈 Price — Entries ▲▼ · Exits ✕")
        st.altair_chart(
            _price_with_trades_chart(prices, result.trades, symbol),
            use_container_width=True,
        )

        # ── RSI chart with trade markers (only for RSI strategy) ─────────────
        if selected_id == "rsi_threshold":
            period     = int(params.get("rsi_period", 14))
            oversold   = float(params.get("oversold", 30))
            overbought = float(params.get("overbought", 70))

            st.markdown(
                f"#### 📉 RSI ({period}) — bar = {bar_label}\n"
                f"▲ = Long entry &nbsp;&nbsp; ▼ = Short entry"
            )
            st.altair_chart(
                _rsi_with_trades_chart(
                    prices, result.trades,
                    period, oversold, overbought,
                    bar_label, symbol,
                ),
                use_container_width=True,
            )
        elif selected_id in ("ma_crossover", "macd_crossover"):
            st.info(
                "💡 Tip: Switch to the 🔬 Strategy Lab page to see MA/MACD indicator "
                "values plotted with signals on the price chart."
            )

        # ── Equity curve ──────────────────────────────────────────────────────
        if not result.equity_curve.empty:
            st.markdown("#### 💰 Equity Curve")
            st.altair_chart(
                equity_curve_chart(result.equity_curve, f"{symbol} Equity Curve"),
                use_container_width=True,
            )

        # ── P&L distribution ──────────────────────────────────────────────────
        if closed_trades:
            trades_df = pd.DataFrame([
                {
                    "symbol": t.symbol,
                    "direction": t.direction.value,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "outcome": t.outcome.value if t.outcome else None,
                    "leveraged_return_pct": t.leveraged_return_pct,
                    "pnl": t.pnl,
                    "entry_time": t.entry_time,
                    "exit_time": t.exit_time,
                }
                for t in closed_trades
            ])

            st.markdown("#### 📊 P&L Distribution")
            st.altair_chart(pnl_distribution(trades_df), use_container_width=True)

            with st.expander("📋 Trade Log", expanded=False):
                st.dataframe(
                    trades_df.sort_values("entry_time", ascending=False),
                    use_container_width=True,
                )

        # ── Save to DB ────────────────────────────────────────────────────────
        try:
            from db.database import Database
            db = Database(settings.db_path)
            for t in result.trades:
                db.save_trade(t)
            st.caption(f"✓ {len(result.trades)} trades saved to database.")
        except Exception as e:
            st.caption(f"DB save skipped: {e}")

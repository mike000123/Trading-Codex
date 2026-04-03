"""
pages/page_strategy_lab.py
───────────────────────────
Strategy Lab: inspect signals for any loaded dataset + strategy combo.
All charts use Altair (no Plotly).
"""
from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from strategies import list_strategies, get_strategy
from ui.components import render_mode_banner, render_data_source_selector, render_strategy_params
from ui.charts import price_chart, rsi_chart
from core.models import SignalAction

_GREEN = "#26a69a"
_RED   = "#ef5350"


def render() -> None:
    render_mode_banner()
    st.title("🔬 Strategy Lab")
    st.caption("Inspect strategy signals on historical data. No orders are placed here.")

    prices = render_data_source_selector()
    if prices is None:
        st.info("← Select a data source in the sidebar to begin.")
        return

    symbol = st.session_state.get("loaded_symbol", "DATA")
    st.success(f"**{symbol}** — {len(prices)} bars")
    st.divider()

    strategies = list_strategies()
    strat_names = {s["name"]: s["id"] for s in strategies}

    col_strat, col_params = st.columns([0.35, 0.65])
    with col_strat:
        selected_name = st.selectbox("Strategy", list(strat_names.keys()), key="lab_strategy")
        selected_id   = strat_names[selected_name]
        strat_info    = next(s for s in strategies if s["id"] == selected_id)
        st.caption(strat_info["description"])

    with col_params:
        params = render_strategy_params(selected_id)

    if st.button("🔍 Analyse Full Window", type="primary", key="lab_run"):
        cls      = get_strategy(selected_id)
        strategy = cls(params=params)

        errors = strategy.validate_params()
        if errors:
            for e in errors:
                st.error(e)
            return

        min_bars = max(params.get("slow_period", params.get("rsi_period", 14)), 10) + 5
        all_signals = []
        for i in range(min_bars, len(prices)):
            window = prices.iloc[:i + 1].copy()
            sig = strategy.generate_signal(window, symbol)
            all_signals.append({
                "date":   prices.iloc[i]["date"],
                "close":  prices.iloc[i]["close"],
                "action": sig.action.value,
                "confidence": sig.confidence,
                "suggested_tp": sig.suggested_tp,
                "suggested_sl": sig.suggested_sl,
                **sig.metadata,
            })

        signals_df = pd.DataFrame(all_signals)

        latest = strategy.generate_signal(prices, symbol)
        colour  = {"BUY": "green", "SELL": "red", "HOLD": "gray"}[latest.action.value]
        st.markdown(
            f"""<div style="border:2px solid {colour};border-radius:8px;padding:12px 20px;margin:8px 0;">
              <strong>Latest Signal:</strong>
              <span style="color:{colour};font-size:1.4rem;font-weight:700;margin-left:10px;">
                {latest.action.value}
              </span>
              &nbsp;|&nbsp; Confidence: <strong>{latest.confidence:.0%}</strong>
              &nbsp;|&nbsp; TP: <code>{f"{latest.suggested_tp:.4f}" if latest.suggested_tp else "—"}</code>
              &nbsp;|&nbsp; SL: <code>{f"{latest.suggested_sl:.4f}" if latest.suggested_sl else "—"}</code>
            </div>""",
            unsafe_allow_html=True,
        )

        st.markdown("#### 📈 Price + Signals")

        # Base price line
        base = (
            alt.Chart(prices)
            .mark_line(color="#4a9eff")
            .encode(
                x=alt.X("date:T", title="Date"),
                y=alt.Y("close:Q", title="Close", scale=alt.Scale(zero=False)),
                tooltip=["date:T", "close:Q"],
            )
        )
        layers = [base]

        buy_rows  = signals_df[signals_df["action"] == "BUY"][["date", "close"]].copy()
        sell_rows = signals_df[signals_df["action"] == "SELL"][["date", "close"]].copy()

        if not buy_rows.empty:
            buy_rows["y"] = buy_rows["close"] * 0.995
            layers.append(
                alt.Chart(buy_rows)
                .mark_point(shape="triangle-up", size=120, color=_GREEN, filled=True)
                .encode(x="date:T", y=alt.Y("y:Q"), tooltip=["date:T", "close:Q"])
            )
        if not sell_rows.empty:
            sell_rows["y"] = sell_rows["close"] * 1.005
            layers.append(
                alt.Chart(sell_rows)
                .mark_point(shape="triangle-down", size=120, color=_RED, filled=True)
                .encode(x="date:T", y=alt.Y("y:Q"), tooltip=["date:T", "close:Q"])
            )

        chart = (
            alt.layer(*layers)
            .properties(title=f"{symbol} – {selected_name} Signals", height=320)
            .configure_view(strokeOpacity=0)
            .configure_axis(gridColor="#1e2130", labelColor="#c9d8f5", titleColor="#c9d8f5")
            .configure_title(color="#c9d8f5")
        )
        st.altair_chart(chart, use_container_width=True)

        if selected_id == "rsi_threshold":
            period = int(params.get("rsi_period", 14))
            
            buy_lvls  = [float(x) for x in str(params.get("buy_levels","30")).replace(";",",").split(",") if x.strip()]
            sell_lvls = [float(x) for x in str(params.get("sell_levels","70")).replace(";",",").split(",") if x.strip()]
            st.altair_chart(
                rsi_chart(prices, period, buy_lvls, sell_lvls)
                .configure_view(strokeOpacity=0)
                .configure_axis(gridColor="#2a2d3e", labelColor="#d0d4f0",
                                titleColor="#d0d4f0", labelFontSize=12, titleFontSize=13)
                .configure_title(color="#e8eaf6", fontSize=14, fontWeight="bold"),
                use_container_width=True,
            )

        with st.expander("📋 All Signals", expanded=False):
            active = signals_df[signals_df["action"] != "HOLD"].copy()
            if active.empty:
                st.info("No BUY/SELL signals generated on this dataset with these parameters.")
            else:
                st.dataframe(active.sort_values("date", ascending=False), use_container_width=True)
                st.caption(
                    f"{len(active)} signals · "
                    f"{len(active[active['action']=='BUY'])} BUY · "
                    f"{len(active[active['action']=='SELL'])} SELL"
                )

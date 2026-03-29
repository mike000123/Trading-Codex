"""
pages/page_strategy_lab.py
───────────────────────────
Strategy Lab: inspect signals for any loaded dataset + strategy combo.
Shows the indicator, current signal, and annotated price chart.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from strategies import list_strategies, get_strategy
from ui.components import render_mode_banner, render_data_source_selector, render_strategy_params
from ui.charts import price_chart, rsi_chart
from core.models import SignalAction


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

    # ── Strategy selector ────────────────────────────────────────────────────
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

    # ── Run strategy on full data window ────────────────────────────────────
    if st.button("🔍 Analyse Full Window", type="primary", key="lab_run"):
        cls = get_strategy(selected_id)
        strategy = cls(params=params)

        errors = strategy.validate_params()
        if errors:
            for e in errors:
                st.error(e)
            return

        # Walk-forward: collect signal at each bar
        all_signals = []
        min_bars = max(params.get("slow_period", params.get("rsi_period", 14)), 10) + 5

        for i in range(min_bars, len(prices)):
            window = prices.iloc[:i + 1].copy()
            sig = strategy.generate_signal(window, symbol)
            all_signals.append({
                "date": prices.iloc[i]["date"],
                "close": prices.iloc[i]["close"],
                "action": sig.action.value,
                "confidence": sig.confidence,
                "suggested_tp": sig.suggested_tp,
                "suggested_sl": sig.suggested_sl,
                **sig.metadata,
            })

        signals_df = pd.DataFrame(all_signals)

        # ── Latest signal ────────────────────────────────────────────────────
        latest = strategy.generate_signal(prices, symbol)
        colour = {"BUY": "green", "SELL": "red", "HOLD": "gray"}[latest.action.value]
        st.markdown(
            f"""
            <div style="border:2px solid {colour};border-radius:8px;padding:12px 20px;margin:8px 0;">
              <strong>Latest Signal:</strong>
              <span style="color:{colour};font-size:1.4rem;font-weight:700;margin-left:10px;">
                {latest.action.value}
              </span>
              &nbsp;|&nbsp; Confidence: <strong>{latest.confidence:.0%}</strong>
              &nbsp;|&nbsp; TP: <code>{f"{latest.suggested_tp:.4f}" if latest.suggested_tp else "—"}</code>
              &nbsp;|&nbsp; SL: <code>{f"{latest.suggested_sl:.4f}" if latest.suggested_sl else "—"}</code>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### 📈 Price + Signals")
        # Mark buy/sell signals on chart
        buy_dates  = signals_df[signals_df["action"] == "BUY"]["date"]
        sell_dates = signals_df[signals_df["action"] == "SELL"]["date"]

        import plotly.graph_objects as go
        fig = price_chart(prices, title=f"{symbol} – {selected_name} Signals")

        if not buy_dates.empty:
            buy_prices = prices[prices["date"].isin(buy_dates)]["close"]
            fig.add_trace(go.Scatter(
                x=buy_dates, y=buy_prices * 0.995,
                mode="markers", name="BUY",
                marker=dict(symbol="triangle-up", size=12, color="#26a69a"),
            ))
        if not sell_dates.empty:
            sell_prices = prices[prices["date"].isin(sell_dates)]["close"]
            fig.add_trace(go.Scatter(
                x=sell_dates, y=sell_prices * 1.005,
                mode="markers", name="SELL",
                marker=dict(symbol="triangle-down", size=12, color="#ef5350"),
            ))

        st.plotly_chart(fig, use_container_width=True)

        # ── RSI sub-chart (if RSI strategy) ─────────────────────────────────
        if selected_id == "rsi_threshold":
            period = int(params.get("rsi_period", 14))
            st.plotly_chart(rsi_chart(prices, period), use_container_width=True)

        # ── Signal table ─────────────────────────────────────────────────────
        with st.expander("📋 All Signals", expanded=False):
            active = signals_df[signals_df["action"] != "HOLD"].copy()
            if active.empty:
                st.info("No BUY/SELL signals generated on this dataset with these parameters.")
            else:
                st.dataframe(active.sort_values("date", ascending=False), use_container_width=True)
                st.caption(f"{len(active)} signals · {len(active[active['action']=='BUY'])} BUY · {len(active[active['action']=='SELL'])} SELL")

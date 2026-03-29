"""
ui/components.py
────────────────
Reusable Streamlit widget blocks.
Import these into any page to avoid repeating layout code.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd
import streamlit as st

from config.settings import TradingMode, settings
from ui.themes import mode_badge


def render_mode_banner() -> None:
    """Show a sticky top banner indicating current trading mode."""
    mode = settings.trading_mode.value
    badge = mode_badge(mode)
    if settings.is_live():
        st.markdown(
            f"""
            <div style="background:#4a1a1a;border:2px solid #ff7043;border-radius:8px;
                        padding:10px 18px;margin-bottom:12px;">
              ⚠️ &nbsp;<strong>LIVE TRADING MODE</strong> &nbsp;{badge}&nbsp;
              — Real money is at risk. All orders will be sent to Alpaca live endpoint.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""<div style="padding:6px 14px;margin-bottom:8px;">
            Mode: {badge}</div>""",
            unsafe_allow_html=True,
        )


def render_data_source_selector() -> Optional[pd.DataFrame]:
    """
    Sidebar widget to pick data source and return a loaded DataFrame.
    Returns None if data is not yet loaded.
    """
    st.sidebar.subheader("📡 Data Source")
    source = st.sidebar.radio("Source", ["Yahoo Finance", "CSV Upload", "Alpaca"], key="data_source")

    data: Optional[pd.DataFrame] = None

    if source == "Yahoo Finance":
        ticker = st.sidebar.text_input("Ticker", value="AAPL", key="yf_ticker")
        interval = st.sidebar.selectbox(
            "Interval",
            ["1m", "5m", "15m", "30m", "1h", "1d", "1wk"],
            index=5,
            key="yf_interval",
        )
        col1, col2 = st.sidebar.columns(2)
        start = col1.date_input("Start", value=(pd.Timestamp.today() - pd.Timedelta(days=365)).date(), key="yf_start")
        end   = col2.date_input("End",   value=pd.Timestamp.today().date(), key="yf_end")

        if st.sidebar.button("Fetch Data", type="primary", key="yf_fetch"):
            from data.ingestion import load_from_ticker
            with st.spinner("Fetching..."):
                try:
                    data = load_from_ticker(ticker, interval, pd.Timestamp(start), pd.Timestamp(end))
                    st.session_state["loaded_data"] = data
                    st.session_state["loaded_symbol"] = ticker.upper()
                    st.sidebar.success(f"✓ {len(data)} bars loaded")
                except Exception as e:
                    st.sidebar.error(str(e))

    elif source == "CSV Upload":
        uploaded = st.sidebar.file_uploader("Upload OHLCV CSV", type=["csv"], key="csv_upload")
        if uploaded:
            from data.ingestion import load_from_csv
            try:
                data = load_from_csv(uploaded)
                st.session_state["loaded_data"] = data
                st.session_state["loaded_symbol"] = uploaded.name.split(".")[0].upper()
            except Exception as e:
                st.sidebar.error(str(e))

    elif source == "Alpaca":
        if not settings.alpaca.has_paper_credentials():
            st.sidebar.warning("No Alpaca credentials found. Add them to .env")
        else:
            symbol = st.sidebar.text_input("Symbol", value="AAPL", key="alp_symbol")
            tf = st.sidebar.selectbox("Timeframe", ["1Day", "1Hour", "15Min", "5Min"], key="alp_tf")
            col1, col2 = st.sidebar.columns(2)
            start = col1.date_input("Start", value=(pd.Timestamp.today() - pd.Timedelta(days=90)).date(), key="alp_start")
            end   = col2.date_input("End",   value=pd.Timestamp.today().date(), key="alp_end")

            if st.sidebar.button("Fetch from Alpaca", key="alp_fetch"):
                from data.ingestion import load_from_alpaca_history
                creds = settings.alpaca
                key = creds.paper_api_key if not settings.is_live() else creds.live_api_key
                sec = creds.paper_secret_key if not settings.is_live() else creds.live_secret_key
                with st.spinner("Fetching from Alpaca..."):
                    try:
                        data = load_from_alpaca_history(
                            symbol, tf,
                            pd.Timestamp(start), pd.Timestamp(end),
                            key, sec, paper=not settings.is_live(),
                        )
                        st.session_state["loaded_data"] = data
                        st.session_state["loaded_symbol"] = symbol.upper()
                        st.sidebar.success(f"✓ {len(data)} bars from Alpaca")
                    except Exception as e:
                        st.sidebar.error(str(e))

    return st.session_state.get("loaded_data")


def render_strategy_params(strategy_id: str) -> dict:
    """Render a form for a strategy's default params and return filled values."""
    from strategies import get_strategy
    cls = get_strategy(strategy_id)
    instance = cls()
    defaults = instance.default_params()

    st.subheader(f"⚙️ {cls.name} Parameters")
    st.caption(cls.description)

    filled: dict = {}
    cols = st.columns(2)
    for i, (param, default) in enumerate(defaults.items()):
        col = cols[i % 2]
        with col:
            if isinstance(default, bool):
                filled[param] = st.checkbox(param, value=default, key=f"param_{strategy_id}_{param}")
            elif isinstance(default, int):
                filled[param] = st.number_input(param, value=default, step=1, key=f"param_{strategy_id}_{param}")
            elif isinstance(default, float):
                filled[param] = st.number_input(param, value=default, format="%.2f", key=f"param_{strategy_id}_{param}")
            elif isinstance(default, str) and param == "ma_type":
                filled[param] = st.selectbox(param, ["ema", "sma"], index=0, key=f"param_{strategy_id}_{param}")
            else:
                filled[param] = st.text_input(param, value=str(default), key=f"param_{strategy_id}_{param}")

    return filled


def render_metrics_row(metrics: dict) -> None:
    """Render a row of st.metric boxes from a dict."""
    cols = st.columns(len(metrics))
    for col, (label, value) in zip(cols, metrics.items()):
        with col:
            if isinstance(value, tuple) and len(value) == 2:
                st.metric(label, value[0], delta=value[1])
            else:
                st.metric(label, value)


def live_trade_confirm_dialog(symbol: str, direction: str, capital: float) -> bool:
    """Show a mandatory confirmation before any live order is submitted."""
    st.error("⚠️ **LIVE TRADING WARNING**")
    st.markdown(
        f"You are about to submit a **LIVE** order:\n"
        f"- Symbol: `{symbol}`\n"
        f"- Direction: `{direction}`\n"
        f"- Capital: `${capital:,.2f}`\n\n"
        f"**This will use real money.**"
    )
    col1, col2 = st.columns(2)
    confirmed = col1.button("✅ Confirm Live Order", type="primary")
    col2.button("❌ Cancel")
    return confirmed

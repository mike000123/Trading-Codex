"""
ui/components.py  —  Reusable Streamlit widget blocks.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd
import streamlit as st

from config.settings import TradingMode, settings
from ui.themes import mode_badge


def render_mode_banner() -> None:
    mode  = settings.trading_mode.value
    badge = mode_badge(mode)
    if settings.is_live():
        st.markdown(
            f"""<div style="background:#4a1a1a;border:2px solid #ff7043;border-radius:8px;
                padding:10px 18px;margin-bottom:12px;">
              ⚠️ &nbsp;<strong>LIVE TRADING MODE</strong> &nbsp;{badge}&nbsp;
              — Real money is at risk.
            </div>""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div style="padding:6px 14px;margin-bottom:8px;">Mode: {badge}</div>',
            unsafe_allow_html=True,
        )


def render_data_source_selector() -> Optional[pd.DataFrame]:
    st.sidebar.subheader("📡 Data Source")
    source = st.sidebar.radio("Source", ["Yahoo Finance", "CSV Upload", "Alpaca"], key="data_source")
    data: Optional[pd.DataFrame] = None

    if source == "Yahoo Finance":
        ticker   = st.sidebar.text_input("Ticker", value="AAPL", key="yf_ticker")
        interval = st.sidebar.selectbox("Interval",
            ["1m", "5m", "15m", "30m", "1h", "1d", "1wk"], index=5, key="yf_interval")
        col1, col2 = st.sidebar.columns(2)
        start = col1.date_input("Start", value=(pd.Timestamp.today() - pd.Timedelta(days=365)).date(), key="yf_start")
        end   = col2.date_input("End",   value=pd.Timestamp.today().date(), key="yf_end")
        if st.sidebar.button("Fetch Data", type="primary", key="yf_fetch"):
            from data.ingestion import load_from_ticker
            with st.spinner("Fetching…"):
                try:
                    data = load_from_ticker(ticker, interval, pd.Timestamp(start), pd.Timestamp(end))
                    st.session_state["loaded_data"]   = data
                    st.session_state["loaded_symbol"] = ticker.upper()
                    st.sidebar.success(f"✓ {len(data)} bars")
                except Exception as e:
                    st.sidebar.error(str(e))

    elif source == "CSV Upload":
        uploaded = st.sidebar.file_uploader("Upload OHLCV CSV", type=["csv"], key="csv_upload")
        if uploaded:
            from data.ingestion import load_from_csv
            try:
                data = load_from_csv(uploaded)
                st.session_state["loaded_data"]   = data
                st.session_state["loaded_symbol"] = uploaded.name.split(".")[0].upper()
            except Exception as e:
                st.sidebar.error(str(e))

    elif source == "Alpaca":
        if not settings.alpaca.has_paper_credentials():
            st.sidebar.warning("No Alpaca credentials. Add to .env")
        else:
            symbol = st.sidebar.text_input("Symbol", value="AAPL", key="alp_symbol")
            tf     = st.sidebar.selectbox("Timeframe", ["1Day", "1Hour", "15Min", "5Min"], key="alp_tf")
            col1, col2 = st.sidebar.columns(2)
            start = col1.date_input("Start", value=(pd.Timestamp.today()-pd.Timedelta(days=90)).date(), key="alp_start")
            end   = col2.date_input("End",   value=pd.Timestamp.today().date(), key="alp_end")
            if st.sidebar.button("Fetch from Alpaca", key="alp_fetch"):
                from data.ingestion import load_from_alpaca_history
                creds = settings.alpaca
                key   = creds.paper_api_key    if not settings.is_live() else creds.live_api_key
                sec   = creds.paper_secret_key if not settings.is_live() else creds.live_secret_key
                with st.spinner("Fetching…"):
                    try:
                        data = load_from_alpaca_history(symbol, tf,
                            pd.Timestamp(start), pd.Timestamp(end),
                            key, sec, paper=not settings.is_live())
                        st.session_state["loaded_data"]   = data
                        st.session_state["loaded_symbol"] = symbol.upper()
                        st.sidebar.success(f"✓ {len(data)} bars from Alpaca")
                    except Exception as e:
                        st.sidebar.error(str(e))

    return st.session_state.get("loaded_data")


# ─── Per-param metadata for richer UI rendering ───────────────────────────────
_PARAM_META: dict[str, dict] = {
    # RSI strategy
    "rsi_period":  {"label": "RSI Period",          "help": "Number of bars for RSI calculation. Default 9 (intraday 1-min/5-min). Shorter = faster reaction, more noise. Longer = smoother, more lag. Common: 9 (intraday), 14 (daily)."},
    "buy_levels":  {"label": "Buy Levels (OS)",      "help": "Comma-separated RSI levels. BUY when RSI crosses below. e.g. '25, 30'"},
    "sell_levels": {"label": "Sell Levels (OB)",     "help": "Comma-separated RSI levels. SELL when RSI crosses above. e.g. '70, 75'"},
    "tp_pct":      {"label": "Take-Profit % (price, 0=off)","help": "Raw price move % to take profit. Capital gain = this × leverage. Set to 0 to disable (SL-only exit)."},
    "sl_pct":      {"label": "Stop-Loss % (price move)",  "help": "Raw price move % that triggers SL. Capital loss = this × leverage. RiskManager caps it further if needed."},
    # MA crossover
    "fast_period": {"label": "Fast MA Period",  "help": ""},
    "slow_period": {"label": "Slow MA Period",  "help": ""},
    "ma_type":     {"label": "MA Type",         "help": "ema or sma"},
    # MACD
    "fast_period": {"label": "MACD Fast",       "help": ""},
    "slow_period": {"label": "MACD Slow",       "help": ""},
    "signal_period":{"label": "Signal Period",  "help": ""},
    # Fixed level
    "direction":        {"label": "Direction",          "help": "Long or Short"},
    "signal_frequency": {"label": "Signal Frequency",   "help": "first_bar = one trade then hold; every_bar = re-signal each bar"},
}


def render_strategy_params(strategy_id: str) -> dict:
    """Render param form with labels, help text, and type-aware widgets."""
    from strategies import get_strategy
    cls      = get_strategy(strategy_id)
    instance = cls()
    defaults = instance.default_params()

    st.subheader(f"⚙️ {cls.name} Parameters")
    st.caption(cls.description)

    # Special note for RSI optional TP
    if strategy_id == "rsi_threshold":
        st.info("💡 **Buy Levels / Sell Levels** accept multiple comma-separated values, "
                "e.g. `25, 30` for buy or `70, 75` for sell.  \n"
                "Set **Take-Profit % = 0** to disable TP and exit on SL only.")

    filled: dict = {}
    cols = st.columns(2)
    for i, (param, default) in enumerate(defaults.items()):
        col  = cols[i % 2]
        meta = _PARAM_META.get(param, {})
        label = meta.get("label", param)
        help_ = meta.get("help", "") or None
        key   = f"param_{strategy_id}_{param}"

        with col:
            if isinstance(default, bool):
                filled[param] = st.checkbox(label, value=default, key=key, help=help_)

            elif param in ("buy_levels", "sell_levels"):
                # Always text input for comma-separated thresholds
                filled[param] = st.text_input(label, value=str(default), key=key, help=help_)

            elif param == "ma_type":
                filled[param] = st.selectbox(label, ["ema", "sma"], index=0, key=key, help=help_)

            elif param == "direction":
                filled[param] = st.selectbox(label, ["Long", "Short"], index=0, key=key, help=help_)

            elif param == "signal_frequency":
                filled[param] = st.selectbox(label, ["first_bar", "every_bar"], index=0, key=key, help=help_)

            elif isinstance(default, int):
                filled[param] = st.number_input(label, value=int(default), step=1, key=key, help=help_)

            elif isinstance(default, float):
                filled[param] = st.number_input(label, value=float(default), format="%.2f",
                                                 min_value=0.0, key=key, help=help_)
            else:
                filled[param] = st.text_input(label, value=str(default), key=key, help=help_)

    return filled


def render_metrics_row(metrics: dict) -> None:
    cols = st.columns(len(metrics))
    for col, (label, value) in zip(cols, metrics.items()):
        with col:
            if isinstance(value, tuple) and len(value) == 2:
                st.metric(label, value[0], delta=value[1])
            else:
                st.metric(label, value)


def live_trade_confirm_dialog(symbol: str, direction: str, capital: float) -> bool:
    st.error("⚠️ **LIVE TRADING WARNING**")
    st.markdown(
        f"You are about to submit a **LIVE** order:\n"
        f"- Symbol: `{symbol}`\n- Direction: `{direction}`\n- Capital: `${capital:,.2f}`\n\n"
        f"**This will use real money.**"
    )
    col1, col2 = st.columns(2)
    confirmed = col1.button("✅ Confirm Live Order", type="primary")
    col2.button("❌ Cancel")
    return confirmed

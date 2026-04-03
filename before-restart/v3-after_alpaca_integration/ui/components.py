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
        ticker   = st.sidebar.text_input("Ticker", value="UVXY", key="yf_ticker")
        interval = st.sidebar.selectbox("Interval",
            ["1m", "5m", "15m", "30m", "1h", "1d", "1wk"], index=0, key="yf_interval")  # default 1m
        col1, col2 = st.sidebar.columns(2)
        start = col1.date_input("Start", value=pd.Timestamp("2026-03-23").date(), key="yf_start")
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
            symbol = st.sidebar.text_input("Symbol", value="UVXY", key="alp_symbol")
            tf     = st.sidebar.selectbox("Timeframe",
                         ["1Min","5Min","15Min","30Min","1Hour","1Day"],
                         index=0, key="alp_tf")
            col1, col2 = st.sidebar.columns(2)
            # Alpaca has ~5 years of 1-min data — default start 2 years back
            start = col1.date_input("Start",
                value=(pd.Timestamp.today()-pd.Timedelta(days=730)).date(),
                key="alp_start")
            end   = col2.date_input("End", value=pd.Timestamp.today().date(), key="alp_end")
            st.sidebar.caption(
                "💡 **Alpaca data ranges** (SIP feed, free):  \n"
                "**1Min** → ~5 years back · **5Min+** → ~5 years back  \n"
                "For UVXY: use **1Min**, start **2020-01-01** to get ~6 years.  \n"
                "⚠️ Keep date ranges under 3 months at a time for 1Min "
                "to avoid very large downloads. Load multiple batches.")
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
    "buy_levels":  {"label": "Buy Levels (OS)",      "help": "Comma-separated RSI thresholds for LONG entry (RSI crosses below). e.g. '25, 30'. Leave BLANK to disable Long trades entirely."},
    "sell_levels": {"label": "Sell Levels (OB)",     "help": "Comma-separated RSI thresholds for SHORT entry (RSI crosses above). e.g. '70, 75'. Leave BLANK to disable Short trades entirely."},
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
    # ATR-based params (shared)
    "atr_period":     {"label": "ATR Period",          "help": "Lookback for Average True Range (volatility measure). Default 14."},
    "atr_sl_mult":    {"label": "SL × ATR",            "help": "Stop-loss = this × ATR from entry. 1.5 = 1.5 ATR away. Adapts to volatility automatically."},
    "atr_tp_mult":    {"label": "TP × ATR",            "help": "Take-profit = this × ATR from entry. 2.5 gives ~1.67 R:R with 1.5 SL mult."},
    "tp_disabled":    {"label": "Disable TP",          "help": "If checked, no price TP — exit via counter-signal or SL only."},
    # VWAP+RSI
    "rsi_oversold":   {"label": "RSI Oversold",        "help": "RSI level to trigger BUY. Slightly higher than pure RSI strategy because VWAP acts as extra filter."},
    "rsi_overbought": {"label": "RSI Overbought",      "help": "RSI level to trigger SELL."},
    # Bollinger
    "bb_period":      {"label": "BB Period",           "help": "Bollinger Bands SMA period. Default 20."},
    "bb_std":         {"label": "BB Std Devs",         "help": "Standard deviations for upper/lower bands. Default 2.0."},
    "sl_band_mult":   {"label": "SL beyond band",      "help": "SL = outer band ± (this × band width). Default 0.2."},
    "require_cross":  {"label": "Require band cross",  "help": "If checked, price must break through the band, not just touch it."},
    # EMA Crossover + RSI + Trend Filter (GC=F)
    "fast_ema":    {"label": "Fast EMA",       "help": "Fast EMA period (default 9). Golden cross above slow EMA = buy signal."},
    "slow_ema":    {"label": "Slow EMA",       "help": "Slow EMA period (default 21). Death cross below fast EMA = sell signal."},
    "trend_ema":   {"label": "Trend EMA",      "help": "Long-term trend filter (default 200). Set to 0 to disable. Longs only above, shorts only below."},
    "rsi_gate":       {"label": "RSI Gate",          "help": "RSI must be above this for longs, below for shorts (default 50). Raise to 55 for stricter momentum filter."},
    "atr_min_filter": {"label": "Min ATR filter",     "help": "Skip signal if ATR < this value (default 0=off). Set e.g. 0.5 for GC=F to avoid trading in tight choppy ranges. ATR is in price units (dollars for gold)."},
    # Fixed level
    "direction":        {"label": "Direction",          "help": "Long or Short"},
    "signal_frequency": {"label": "Signal Frequency",   "help": "first_bar = one trade then hold; every_bar = re-signal each bar"},
}


def render_strategy_params(strategy_id: str, leverage: float = 1.0,
                           max_capital_loss_pct: float = 50.0) -> dict:
    """
    Render param form with labels, help text, and type-aware widgets.

    Args:
        strategy_id          – registered strategy ID
        leverage             – current leverage setting (used to derive max SL)
        max_capital_loss_pct – platform cap on capital loss per trade (default 50%)
                               eToro / Alpaca: 50%. This caps sl_pct to
                               max_capital_loss_pct / leverage as a price move %.
    """
    from strategies import get_strategy
    cls      = get_strategy(strategy_id)
    instance = cls()
    defaults = instance.default_params()

    # ── Derived leverage constraint ────────────────────────────────────────────
    # Max price move % before capital loss cap is hit:
    #   sl_pct_max = max_capital_loss_pct / leverage
    # e.g. 50% cap, leverage 5× → max 10% price drop allowed
    max_sl_price_pct = max_capital_loss_pct / max(leverage, 1.0)

    st.subheader(f"⚙️ {cls.name} Parameters")
    st.caption(cls.description)

    # ── Strategy-specific info boxes ───────────────────────────────────────────
    if strategy_id == "rsi_threshold":
        st.info(
            "💡 **Buy Levels / Sell Levels** accept comma-separated values (e.g. `25, 30`).  \n"
            "Leave blank to disable that direction. Set TP % = 0 for counter-signal exit only.  \n"
            "🎯 Recommended thresholds: **UVXY** → buy=20 / sell=80 · "
            "**GC=F** → buy=30 / sell=70  \n"
            "*(RSI scale is universal — thresholds are instrument-specific, not strategy-specific)*"
        )
    elif strategy_id == "vwap_rsi":
        st.info(
            "📊 **VWAP + RSI** — Best for GC=F intraday (1-min / 5-min bars).  \n"
            "Enters when price crosses VWAP **and** RSI confirms direction.  \n"
            "ATR-based stops adapt automatically to gold's intraday volatility.  \n"
            "🎯 Recommended thresholds: **GC=F** → oversold 30 / overbought 70 · "
            "**UVXY** → oversold 20 / overbought 80 (more extreme = fewer, cleaner signals)"
        )
    elif strategy_id == "bollinger_rsi":
        st.info(
            "📉 **Bollinger + RSI** — Best for UVXY after VIX spikes (mean reversion).  \n"
            "Short when price hits upper Bollinger Band **and** RSI is overbought.  \n"
            "Target = middle band (mean reversion). Keep hold time < 2 days on UVXY.  \n"
            "🎯 Recommended thresholds: **UVXY** → oversold 20 / overbought 80 · "
            "**GC=F** → oversold 30 / overbought 70"
        )
    elif strategy_id == "ema_trend_rsi":
        st.info(
            "📈 **EMA Crossover + RSI + Trend Filter** — Purpose-built for GC=F 1-min/5-min.  \n"
            "**Entry:** 9/21 EMA golden/death cross · **RSI gate:** > 50 confirms direction · "
            "**200 EMA filter:** trades only with the broader trend.  \n"
            "**Tuning guide:**  \n"
            "• Near-zero return → raise `atr_tp_mult` (3.0–4.0) so wins outweigh losses  \n"
            "• Too many losing trades → set `atr_min_filter` to 0.3–0.8 (skip choppy periods)  \n"
            "• Too few trades → set `trend_ema=0` to remove the 200 EMA filter  \n"
            "• Faster signals → reduce `slow_ema` from 21 to 13  \n"
            "🎯 GC=F recommended: trend_ema=200, rsi_gate=50, atr_tp=3.0, atr_min_filter=0.5"
        )
    elif strategy_id == "atr_rsi":
        st.info(
            "🎯 **ATR-Adaptive RSI** — Works for both GC=F and UVXY.  \n"
            "Stops and targets scale automatically with current volatility (ATR).  \n"
            "More robust than fixed-% stops across different market conditions.  \n"
            "🎯 Recommended thresholds: **UVXY** → buy_levels=20 / sell_levels=80 · "
            "**GC=F** → buy_levels=30 / sell_levels=70"
        )

    # ── Leverage constraint warning ────────────────────────────────────────────
    if leverage > 1.0:
        st.warning(
            f"⚡ **Leverage {leverage:.1f}×** active.  "
            f"Platform cap: max **{max_capital_loss_pct:.0f}% capital loss** per trade.  \n"
            f"→ Maximum allowed **price move SL = {max_sl_price_pct:.2f}%** "
            f"(= {max_capital_loss_pct:.0f}% ÷ {leverage:.1f}).  \n"
            f"Any `sl_pct` or ATR-mult value implying a larger price drop will be "
            f"auto-clamped by the Risk Manager."
        )

    # ── Widget rendering ───────────────────────────────────────────────────────
    # Use strategy's defaults but reset when strategy changes
    # (Streamlit persists widget state by key; include strategy_id in key so
    #  switching strategies resets to the new strategy's defaults)
    filled: dict = {}
    cols = st.columns(2)
    for i, (param, default) in enumerate(defaults.items()):
        col   = cols[i % 2]
        meta  = _PARAM_META.get(param, {})
        label = meta.get("label", param)
        help_ = meta.get("help", "") or None
        # Key includes strategy_id → forces reset when strategy changes
        key   = f"p_{strategy_id}_{param}"

        # For sl_pct: show derived max and add constraint context to label
        if param == "sl_pct":
            label = f"Stop-Loss % (price move) · max {max_sl_price_pct:.2f}%"
            help_ = (
                f"Raw price move % that triggers SL. Capital loss = this × leverage. "
                f"With {leverage:.1f}× leverage and {max_capital_loss_pct:.0f}% cap, "
                f"maximum allowed = {max_sl_price_pct:.2f}%. "
                f"Risk Manager will clamp any larger value automatically."
            )

        with col:
            if isinstance(default, bool):
                filled[param] = st.checkbox(label, value=default, key=key, help=help_)

            elif param in ("buy_levels", "sell_levels"):
                filled[param] = st.text_input(label, value=str(default), key=key, help=help_)

            elif param == "ma_type":
                filled[param] = st.selectbox(label, ["ema", "sma"], index=0, key=key, help=help_)

            elif param == "direction":
                filled[param] = st.selectbox(label, ["Long", "Short"], index=0, key=key, help=help_)

            elif param == "signal_frequency":
                filled[param] = st.selectbox(label, ["first_bar", "every_bar"], index=0, key=key, help=help_)

            elif isinstance(default, int):
                filled[param] = st.number_input(label, value=int(default), step=1,
                                                  key=key, help=help_)

            elif isinstance(default, float):
                # For sl_pct: cap the max_value in the widget itself
                if param == "sl_pct":
                    capped_default = min(float(default), max_sl_price_pct)
                    filled[param] = st.number_input(
                        label, value=round(capped_default, 4),
                        format="%.4f", min_value=0.0001,
                        max_value=round(max_sl_price_pct, 4),
                        key=key, help=help_,
                    )
                else:
                    filled[param] = st.number_input(label, value=float(default),
                                                     format="%.2f", min_value=0.0,
                                                     key=key, help=help_)
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
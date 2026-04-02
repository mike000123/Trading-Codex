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
        from data.cache import DataCache
        _yc = DataCache()

        ticker   = st.sidebar.text_input("Ticker", value="UVXY", key="yf_ticker")
        interval = st.sidebar.selectbox("Interval",
            ["1m", "5m", "15m", "30m", "1h", "1d", "1wk"], index=0, key="yf_interval")
        col1, col2 = st.sidebar.columns(2)
        start = col1.date_input("Start", value=pd.Timestamp("2026-03-23").date(), key="yf_start")
        end   = col2.date_input("End",   value=pd.Timestamp.today().date(), key="yf_end")

        start_ts = pd.Timestamp(start)
        end_ts   = pd.Timestamp(end)

        # Auto-load from cache if available
        _yf_key = f"yf_loaded_{ticker}_{interval}_{start}_{end}"
        if _yf_key not in st.session_state:
            cached_df = _yc.load("yfinance", ticker.upper(), interval)
            if cached_df is not None and not cached_df.empty:
                mask = ((cached_df["date"] >= start_ts) & (cached_df["date"] <= end_ts))
                in_range = cached_df[mask]
                if not in_range.empty:
                    st.session_state["loaded_data"]   = in_range.reset_index(drop=True)
                    st.session_state["loaded_symbol"] = ticker.upper()
                    st.session_state[_yf_key]         = True
                    st.sidebar.success(f"✓ {len(in_range):,} bars from cache")

        if st.sidebar.button("🔄 Fetch / Update", type="primary", key="yf_fetch"):
            from data.ingestion import load_from_ticker
            with st.spinner("Fetching…"):
                try:
                    data = load_from_ticker(ticker, interval, start_ts, end_ts)
                    st.session_state["loaded_data"]   = data
                    st.session_state["loaded_symbol"] = ticker.upper()
                    for k in list(st.session_state.keys()):
                        if k.startswith(f"yf_loaded_{ticker}_{interval}"):
                            del st.session_state[k]
                    st.sidebar.success(f"✓ {len(data):,} bars")
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
            from data.cache import DataCache
            _ac = DataCache()

            symbol = st.sidebar.text_input("Symbol", value="UVXY", key="alp_symbol")
            tf     = st.sidebar.selectbox("Timeframe",
                         ["1Min","5Min","15Min","30Min","1Hour","1Day"],
                         index=0, key="alp_tf")
            col1, col2 = st.sidebar.columns(2)
            start = col1.date_input("Start",
                value=(pd.Timestamp.today()-pd.Timedelta(days=730)).date(),
                key="alp_start")
            end   = col2.date_input("End", value=pd.Timestamp.today().date(), key="alp_end")

            start_ts = pd.Timestamp(start)
            end_ts   = pd.Timestamp(end)

            # ── Auto-load from cache if available ─────────────────────────────
            # Build a cache-state key so we only re-load when symbol/tf/dates change
            _cache_key = f"alp_loaded_{symbol}_{tf}_{start}_{end}"
            if _cache_key not in st.session_state:
                cached_df = _ac.load("alpaca", symbol, tf)
                if cached_df is not None and not cached_df.empty:
                    # Filter to requested range
                    mask = ((cached_df["date"] >= start_ts) &
                            (cached_df["date"] <= end_ts))
                    in_range = cached_df[mask]
                    if not in_range.empty:
                        st.session_state["loaded_data"]   = in_range.reset_index(drop=True)
                        st.session_state["loaded_symbol"] = symbol.upper()
                        st.session_state[_cache_key]      = True
                        st.sidebar.success(
                            f"✓ Loaded {len(in_range):,} bars from local cache  \n"
                            f"({in_range['date'].iloc[0].date()} → "
                            f"{in_range['date'].iloc[-1].date()})"
                        )

            # Show cache status for this symbol/tf
            existing = _ac.load("alpaca", symbol, tf)
            if existing is not None and not existing.empty:
                last_cached = existing["date"].max()
                st.sidebar.caption(
                    f"💾 Cache: **{symbol} {tf}** · {len(existing):,} bars  \n"
                    f"{existing['date'].min().date()} → {last_cached.date()}  \n"
                    f"Press **Fetch** to append new bars since {last_cached.date()}"
                )
            else:
                st.sidebar.caption(
                    "💡 **Alpaca SIP feed** — free, ~5 years of 1-min history.  \n"
                    "For UVXY: start **2020-01-01** to get ~6 years.")

            if st.sidebar.button("🔄 Fetch / Update from Alpaca", key="alp_fetch"):
                from data.ingestion import load_from_alpaca_history
                creds = settings.alpaca
                key_  = creds.paper_api_key    if not settings.is_live() else creds.live_api_key
                sec_  = creds.paper_secret_key if not settings.is_live() else creds.live_secret_key
                with st.spinner("Checking for new bars…"):
                    try:
                        data = load_from_alpaca_history(symbol, tf,
                            start_ts, end_ts, key_, sec_,
                            paper=not settings.is_live())
                        st.session_state["loaded_data"]   = data
                        st.session_state["loaded_symbol"] = symbol.upper()
                        # Invalidate cache-state key so next load re-checks
                        for k in list(st.session_state.keys()):
                            if k.startswith(f"alp_loaded_{symbol}_{tf}"):
                                del st.session_state[k]
                        st.sidebar.success(
                            f"✓ {len(data):,} bars  \n"
                            f"({data['date'].iloc[0].date()} → "
                            f"{data['date'].iloc[-1].date()})"
                        )
                    except Exception as e:
                        st.sidebar.error(str(e))

    # ── Cache status panel ─────────────────────────────────────────────────
    from data.cache import DataCache
    cache = DataCache()
    cached_list = cache.list_cached()
    if cached_list:
        with st.sidebar.expander(f"💾 Local Cache ({len(cached_list)} datasets)", expanded=False):
            for entry in cached_list:
                col_a, col_b = st.columns([0.8, 0.2])
                col_a.caption(
                    f"**{entry['symbol']}** · {entry['source']} · {entry['timeframe']}  \n"
                    f"{entry['from']} → {entry['to']} · "
                    f"{entry['bars']:,} bars · {entry['size_kb']} KB"
                )
                if col_b.button("🗑", key=f"del_{entry['source']}_{entry['symbol']}_{entry['timeframe']}",
                                help="Delete this cache entry"):
                    cache.delete(entry['source'], entry['symbol'].replace('=','_'), entry['timeframe'])
                    st.rerun()

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
    "sl_band_mult":        {"label": "SL beyond band",       "help": "SL = outer band ± (mult × band width). Default 0.2."},
    "require_cross":       {"label": "Require band cross",   "help": "Price must cross band, not just touch. Reduces signals ~40%."},
    "min_band_width_pct":  {"label": "Min band width %",     "help": "Skip if band < X% of price. Default 2.0%."},
    "min_rr_ratio":        {"label": "Min R:R ratio",        "help": "Skip if TP distance / SL distance < X. Default 1.5."},
    "cooldown_bars":       {"label": "Cooldown bars",        "help": "Min bars between entries. Default 5."},
    "min_atr_pct":         {"label": "Min ATR % of price",   "help": "Skip ALL signals when ATR < X% of price (too flat to trade profitably). Default 0.3%."},
    # Spike regime
    "spike_atr_mult":      {"label": "Spike ATR mult",       "help": "Fast spike: ATR > X × ATR_MA. Default 2.0."},
    "rise_lookback":       {"label": "Rise lookback bars",   "help": "Gradual rise check window. Default 1170 = 3 trading days. If price > rise_pct% above N-bar low → spike → no shorts."},
    "rise_pct":            {"label": "Rise % threshold",     "help": "Price > X% above rise_lookback low → rising trend → suppress shorts. Default 5%. Catches gradual multi-day UVXY spikes."},
    # Post-spike regime
    "spike_high_window":   {"label": "Spike high window",    "help": "Rolling high window for spike detection. Default 1170 = 3 days."},
    "spike_ema_mult":      {"label": "Spike EMA mult",       "help": "Spike if 3d-high > X × 5d-EMA. Default 1.5."},
    "peak_drop_pct":       {"label": "Peak drop % trigger",  "help": "Price dropped X% from 3d-high → post-spike → aggressive shorts. Default 8%."},
    "reversion_tp_pct":    {"label": "Reversion TP %",       "help": "Post-spike short TP = entry × (1 - X%). Default 15%."},
    "reversion_sl_pct":    {"label": "Reversion SL %",       "help": "Post-spike short SL = entry × (1 + X%). Default 5%."},
    "reversion_rsi_min":   {"label": "Reversion RSI min",    "help": "Don't short post-spike if RSI < X (already oversold). Default 40."},
    "fast_ema":            {"label": "Fast EMA",       "help": "Fast EMA period (default 9). Golden cross above slow EMA = buy signal."},
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
            "📉 **Bollinger + RSI (4-Regime)** — Full UVXY spike cycle strategy.  \n"
            "**Four regimes (independently detected):**  \n"
            "• 🟢 **Normal** — Bollinger mean reversion, both directions  \n"
            "• 🟡 **Spike** — Fast (ATR) OR gradual (price > 5% above 3d-low): NO shorts  \n"
            "• 🔴 **Post-spike** — 3d-high > 1.5× EMA AND dropped 8%: AGGRESSIVE SHORTS  \n"
            "• ⏸️ **Drift** — ATR < 0.3% of price: all paused  \n"
            "🎯 Key params: rise_pct=5 (gradual rise sensitivity), "
            "peak_drop_pct=8 (post-spike entry), reversion_tp_pct=15 (short target)"
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

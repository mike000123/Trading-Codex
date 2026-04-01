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
    source = st.sidebar.radio("Source", ["Yahoo Finance", "CSV Upload", "Alpaca"], key="data_source", index=2)
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
    "reversion_tp_pct":    {"label": "Reversion TP %",       "help": "Post-spike short hard-floor TP = entry × (1 - X%). Acts as max-profit cap even with dynamic exits enabled. Default 15%."},
    "reversion_sl_pct":    {"label": "Reversion SL %",       "help": "Post-spike short SL = entry × (1 + X%). Default 5%."},
    "reversion_rsi_min":   {"label": "Reversion RSI min",    "help": "Don't short post-spike if RSI < X (already oversold). Default 40."},
    # Dynamic post-spike exit params
    "dyn_use_regime_exit": {"label": "Dyn exit: regime flip", "help": "✅ Close post-spike short when post_spike regime turns False (price recovered vs rolling high/EMA). Usually the most reliable signal. Default on."},
    "dyn_rsi_rev_floor":   {"label": "Dyn exit: RSI floor",   "help": "RSI reversal exit (signal B): RSI trough must reach ≤ this value before a rebound counts. Default 35. Lower = requires deeper flush before exiting."},
    "dyn_rsi_rev_rise":    {"label": "Dyn exit: RSI rise pts", "help": "RSI reversal exit (signal B): RSI must rise this many points from the trough to trigger exit. Set to 0 to disable (default OFF). Use ≥20 on 5-min UVXY to avoid noise."},
    "dyn_ema_fast":        {"label": "Dyn exit: EMA fast",    "help": "EMA cross exit (signal C): fast EMA span (bars). Set to 0 to disable (default OFF). On 5-min UVXY data spans of 5/13 fire every ~26 bars — far too noisy. Use ≥50/130 if enabling."},
    "dyn_ema_slow":        {"label": "Dyn exit: EMA slow",    "help": "EMA cross exit (signal C): slow EMA span (bars). Only used when dyn_ema_fast > 0. Default 130 (~10hrs on 5-min bars)."},
    "dyn_atr_collapse":    {"label": "Dyn exit: ATR collapse", "help": "ATR collapse exit (signal D): exit when current ATR drops below this fraction of ATR-at-entry. 0.55 = 55%. Set to 0 to disable. Catches moves that have exhausted volatility."},
    "fast_ema":            {"label": "Fast EMA",       "help": "Fast EMA period (default 9). Golden cross above slow EMA = buy signal."},
    "slow_ema":    {"label": "Slow EMA",       "help": "Slow EMA period (default 21). Death cross below fast EMA = sell signal."},
    "trend_ema":   {"label": "Trend EMA",      "help": "Long-term trend filter (default 200). Set to 0 to disable. Longs only above, shorts only below."},
    "rsi_gate":       {"label": "RSI Gate",          "help": "RSI must be above this for longs, below for shorts (default 50). Raise to 55 for stricter momentum filter."},
    "atr_min_filter": {"label": "Min ATR filter",     "help": "Skip signal if ATR < this value (default 0=off). Set e.g. 0.5 for GC=F to avoid trading in tight choppy ranges. ATR is in price units (dollars for gold)."},
    # Fixed level
    "direction":        {"label": "Direction",          "help": "Long or Short"},
    "signal_frequency": {"label": "Signal Frequency",   "help": "first_bar = one trade then hold; every_bar = re-signal each bar"},
    # UVXY Auto — shared
    "spike_momentum_bars":   {"label": "Spike momentum bars",    "help": "Bars to look back for price momentum confirmation on spike entries. Default 12 (~1 hour)."},
    "spike_momentum_pct":    {"label": "Spike momentum %",       "help": "Price must be ≥ X% above N bars ago to confirm upward momentum. Default 1.5%."},
    "spike_rsi_ob":          {"label": "Spike RSI OB gate",      "help": "Don't enter spike long if RSI already above this. Default 75."},
    "spike_sl_pct":          {"label": "Spike SL %",             "help": "Hard stop below spike long entry. Keep tight — spikes reverse fast. Default 5%."},
    "spike_atr_trail":       {"label": "Spike ATR trail mult",   "help": "Trailing SL for spike longs = peak_high - X×ATR. Tighter than decay. Default 1.5."},
    "spike_atr_exit_mult":   {"label": "Spike ATR exit thresh",  "help": "Exit long when ATR drops below X × ATR_MA (spike unwinding). Default 1.3."},
    "spike_cooldown":        {"label": "Spike cooldown bars",    "help": "Min bars between spike long entries. Default 78 (~2 hours on 5-min)."},
    "spike_max_entries":     {"label": "Max spike entries",      "help": "Max long entries per spike event. Default 2."},
    "ps_high_window":        {"label": "Post-spike high window", "help": "Rolling high window for post-spike detection. Default 1170 (~3 days)."},
    "ps_ema_span":           {"label": "Post-spike EMA span",    "help": "Long EMA anchor for spike detection. Default 1950 (~5 days)."},
    "ps_ema_mult":           {"label": "Post-spike EMA mult",    "help": "Spike if rolling high > EMA × mult. Default 1.5."},
    "ps_drop_pct":           {"label": "Post-spike drop %",      "help": "Enter short after price drops X% from spike high. Default 8%."},
    "ps_rsi_min":            {"label": "Post-spike RSI min",     "help": "Skip post-spike short if RSI < this (already oversold). Default 40."},
    "ps_tp_pct":             {"label": "Post-spike TP %",        "help": "Post-spike short TP = entry × (1 - X%). Default 15%."},
    "ps_sl_pct":             {"label": "Post-spike SL %",        "help": "Post-spike short SL = entry × (1 + X%). Default 5%."},
    "ps_max_entries":        {"label": "Max post-spike entries", "help": "Max short entries per post-spike event. Default 2."},
    "ps_cooldown_mult":      {"label": "Post-spike cooldown ×",  "help": "Cooldown = rsi_period × this value. Default 10."},
    "decay_ema_period":      {"label": "Decay EMA period",       "help": "EMA for slope detection. Default 1950 (~5 days on 5-min data)."},
    "decay_slope_lb":        {"label": "Decay slope lookback",   "help": "Compare EMA now vs this many bars ago. Default 780 (~2 days)."},
    "decay_slope_min_pct":   {"label": "Decay min slope %",      "help": "EMA must fall ≥ X% over lookback to qualify as downtrend. Default 0.3%."},
    "decay_floor":           {"label": "Decay floor price ($)",  "help": "Structural UVXY equilibrium TP target. Default $40. Update after reverse splits."},
    "decay_floor_buf":       {"label": "Decay floor buffer %",   "help": "Don't enter if price < floor × (1 + X%). Default 8%."},
    "decay_rsi_os":          {"label": "Decay RSI OS gate",      "help": "Skip decay short if RSI already oversold (< this). Default 32."},
    "decay_sl_pct":          {"label": "Decay hard SL %",        "help": "Insurance SL above entry. Wide — ATR trail should close first. Default 12%."},
    "decay_atr_trail":       {"label": "Decay ATR trail mult",   "help": "Wide trailing SL for slow grind = lowest_low + X×ATR. Default 4.5."},
    "decay_cooldown":        {"label": "Decay cooldown bars",    "help": "Min bars between decay short entries. Default 780 (~2 days)."},
    "decay_max_entries":     {"label": "Max decay entries",      "help": "Max short entries per continuous decay period. Default 12."},
    "bb_sl_mult":            {"label": "BB SL beyond band",      "help": "Normal regime SL = outer band ± (mult × band width). Default 0.2."},
    "bb_min_width_pct":      {"label": "BB min width %",         "help": "Skip normal signal if band < X% of price. Default 2%."},
    "bb_min_rr":             {"label": "BB min R:R",             "help": "Skip normal signal if TP/SL distance ratio < this. Default 1.5."},
    "bb_require_cross":      {"label": "BB require cross",       "help": "Price must cross band (not just touch) for normal regime entry."},
    "bb_cooldown":           {"label": "BB cooldown bars",       "help": "Min bars between normal regime entries. Default 5."},
    "slope_lookback":        {"label": "Slope lookback bars",   "help": "Compare EMA now vs this many bars ago to measure slope. 390 = 1 trading day. Must be < decay_ema_period."},
    "slope_min_pct":         {"label": "Min slope % drop",      "help": "EMA must have fallen at least this % over the lookback window to qualify as a downtrend. Default 0.5%."},
    "floor_price":           {"label": "Floor price ($)",        "help": "Structural equilibrium target for UVXY (~$40). The TP for each decay short. Update if UVXY does a reverse split."},
    "floor_buffer_pct":      {"label": "Floor buffer %",        "help": "Don't enter if price is within X% above the floor. Prevents entries too close to the target. Default 10%."},
    "rsi_os_gate":           {"label": "RSI oversold gate",     "help": "Skip entry if RSI is already below this (already oversold — wait for bounce). Default 35."},
    "max_entries_per_decay": {"label": "Max entries per decay", "help": "Maximum number of short entries during a single continuous decay period. Default 6."},
    "decay_sl_pct":          {"label": "Hard SL %",             "help": "Insurance stop-loss: entry × (1 + X%). ATR trailing SL should close trades first; this is the backstop. Default 6%."},
    "atr_trail_mult":        {"label": "ATR trail multiplier",  "help": "Trailing SL distance = X × ATR above the lowest low reached. Tightens as price falls and ATR compresses. Default 2.5."},
    # Spike Long
    "atr_ma_period":         {"label": "ATR MA period",         "help": "Baseline ATR moving average period. ATR is compared vs this MA to detect spikes. Default 20 bars."},
    "atr_exit_mult":         {"label": "ATR exit threshold",    "help": "Exit long when ATR drops below X × ATR_MA — spike is unwinding. Must be < spike_atr_mult. Default 1.3."},
    "momentum_bars":         {"label": "Momentum lookback",     "help": "Price must be ≥ momentum_pct% above price N bars ago. Confirms price is actually rising during the ATR spike. Default 12 bars."},
    "momentum_pct":          {"label": "Momentum % threshold",  "help": "Minimum price rise % over momentum_bars to qualify as upward momentum. Default 1.5%."},
    "rsi_ob_gate":           {"label": "RSI overbought gate",   "help": "Skip spike long entry if RSI already above this (spike already overbought). Default 75."},
    "max_entries_per_spike": {"label": "Max entries per spike", "help": "Maximum long entries during a single spike event. Default 2 to avoid over-pyramiding."},
    "spike_sl_pct":          {"label": "Spike SL %",            "help": "Hard stop-loss below entry: entry × (1 - X%). Keep tight — spikes reverse violently. Default 5%."},
    "spike_tp_pct":          {"label": "Spike TP % (0=off)",    "help": "Hard take-profit above entry. Set to 0 to rely on ATR trailing stop only (recommended). Default 0."},
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
            "🔄 **Bollinger + RSI (Full Cycle)** — One strategy, full UVXY lifecycle.  \n"
            "Automatically detects the current regime every bar — no manual switching needed.  \n\n"
            "**Six regimes (priority order):**  \n"
            "• ⏸️ **Drift** — ATR too low: all signals paused  \n"
            "• 📈 **Spike** — ATR expanding + upward momentum → **LONG** with ATR trailing stop  \n"
            "• 🔴 **Post-spike** — dropped 8% from high >> EMA → **SHORT** (aggressive, 15% TP)  \n"
            "• 📉 **Decay** — EMA declining + price below EMA → **SHORT** toward $40 floor with wide ATR trail  \n"
            "• 🟢 **Normal** — Bollinger mean reversion, both directions  \n\n"
            "**Full UVXY cycle:** Quiet → Spike UP (long) → Post-spike (short) → Weeks of decay (short) → Quiet  \n\n"
            "🎯 Key params: `decay_floor=40` (update after reverse splits) · "
            "`spike_sl_pct=5` (keep tight) · `decay_atr_trail=4.5` (wide for slow grind) · "
            "`reversion_tp_pct=15` (post-spike cap)"
        )
    elif strategy_id == "uvxy_auto":
        st.info(
            "🤖 **UVXY Auto (Full Cycle)** — One strategy, all regimes. "
            "Automatically detects the current market phase every bar and trades accordingly — "
            "no manual strategy switching needed.  \n\n"
            "**Regime hierarchy (auto-detected):**  \n"
            "• ⏸️ **Drift** — ATR too low, all signals paused  \n"
            "• 📈 **Spike** — ATR expanding + upward momentum → **LONG** with ATR trailing stop  \n"
            "• 🔴 **Post-spike** — dropped 8% from high >> long EMA → **SHORT** (aggressive, 15% TP)  \n"
            "• 📉 **Decay** — EMA declining + price below EMA → **SHORT** toward $40 floor with wide ATR trail  \n"
            "• 🟢 **Normal** — Bollinger mean reversion, both directions  \n\n"
            "**Full UVXY cycle coverage:**  \n"
            "Grind → Spike UP (long) → Post-spike (short) → Weeks of decay (short) → Grind  \n\n"
            "🎯 Key params to tune: `decay_floor=40` (update after reverse splits) · "
            "`spike_sl_pct=5` (keep tight) · `decay_atr_trail=4.5` (wide for slow grind)"
        )
    elif strategy_id == "trend_decay":
        st.info(
            "📉 **UVXY Trend Decay** — Shorts the prolonged post-spike contango decay "
            "toward structural equilibrium (~$40).  \n"
            "**When it activates:** medium EMA declining ≥ 0.5% per day AND price below EMA "
            "AND not in a spike AND price still > 10% above floor.  \n"
            "**Exit:** ATR trailing SL trails price down (locks in profit as decay continues) "
            "+ hard floor TP at $40 + regime exit when EMA flattens.  \n"
            "**Pairs with:** Bollinger+RSI (handles the first few days post-spike); "
            "this strategy takes over for the weeks/months of sustained decay that follow.  \n"
            "🎯 Key params: `floor_price=40` (update after reverse splits) · "
            "`decay_ema_period=780` (2-day EMA) · `atr_trail_mult=2.5`"
        )
    elif strategy_id == "spike_long":
        st.info(
            "📈 **UVXY Spike Long** — Rides the violent upward leg of VIX spikes.  \n"
            "**Entry:** ATR expands above `spike_atr_mult × ATR_MA` (spike confirmed) "
            "AND price is ≥ 1.5% above its level 12 bars ago (momentum confirmed) "
            "AND RSI < 75 (not already overbought).  \n"
            "**Exit:** ATR trailing stop (SL = peak_high - 1.5×ATR) tightens as spike "
            "matures; also exits when ATR contracts back below `atr_exit_mult × ATR_MA` "
            "(spike is unwinding).  \n"
            "⚠️ **Risk:** UVXY spikes reverse violently — keep `spike_sl_pct` ≤ 5% "
            "and `max_entries_per_spike` ≤ 2.  \n"
            "🎯 Key params: `spike_atr_mult=2.0` · `atr_exit_mult=1.3` · "
            "`atr_trail_mult=1.5` · `spike_sl_pct=5`"
        )
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

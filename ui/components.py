"""
ui/components.py  —  Reusable Streamlit widget blocks.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd
import streamlit as st

from config.settings import settings
from ui.themes import mode_badge


def render_mode_banner() -> None:
    mode = settings.trading_mode.value
    badge = mode_badge(mode)
    if settings.is_live():
        st.markdown(
            f"""<div style="background:#4a1a1a;border:2px solid #ff7043;border-radius:8px;
                padding:10px 18px;margin-bottom:12px;">
              ⚠️ &nbsp;<strong>LIVE TRADING MODE</strong> &nbsp;{badge}&nbsp;
              - Real money is at risk.
            </div>""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(f'<div style="padding:6px 14px;margin-bottom:8px;">Mode: {badge}</div>', unsafe_allow_html=True)


def render_data_source_selector() -> Optional[pd.DataFrame]:
    st.sidebar.subheader("📡 Data Source")
    source = st.sidebar.radio("Source", ["Yahoo Finance", "CSV Upload", "Alpaca"], index=2, key="data_source")
    data: Optional[pd.DataFrame] = None

    if source == "Yahoo Finance":
        from data.cache import DataCache

        _yc = DataCache()
        ticker = st.sidebar.text_input("Ticker", value="UVXY", key="yf_ticker")
        interval = st.sidebar.selectbox("Interval", ["1m", "5m", "15m", "30m", "1h", "1d", "1wk"], index=0, key="yf_interval")
        col1, col2 = st.sidebar.columns(2)
        start = col1.date_input("Start", value=pd.Timestamp("2026-03-23").date(), key="yf_start")
        end = col2.date_input("End", value=pd.Timestamp.today().date(), key="yf_end")
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        end_exclusive = end_ts + pd.Timedelta(days=1)

        _yf_key = f"yf_loaded_{ticker}_{interval}_{start}_{end}"
        if _yf_key not in st.session_state:
            cached_df = _yc.load("yfinance", ticker.upper(), interval)
            if cached_df is not None and not cached_df.empty:
                mask = (cached_df["date"] >= start_ts) & (cached_df["date"] < end_exclusive)
                in_range = cached_df[mask]
                if not in_range.empty:
                    st.session_state["loaded_data"] = in_range.reset_index(drop=True)
                    st.session_state["loaded_symbol"] = ticker.upper()
                    st.session_state[_yf_key] = True
                    st.sidebar.success(f"✓ {len(in_range):,} bars from cache")

        if st.sidebar.button("🔄 Fetch / Update", type="primary", key="yf_fetch"):
            from data.ingestion import load_from_ticker

            with st.spinner("Fetching…"):
                try:
                    data = load_from_ticker(ticker, interval, start_ts, end_exclusive)
                    st.session_state["loaded_data"] = data
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
                st.session_state["loaded_data"] = data
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
            tf = st.sidebar.selectbox("Timeframe", ["1Min", "5Min", "15Min", "30Min", "1Hour", "1Day"], index=0, key="alp_tf")
            cached_df = _ac.load("alpaca", symbol, tf)
            if cached_df is not None and not cached_df.empty:
                default_start_date = pd.Timestamp(cached_df["date"].min()).date()
                default_end_date = pd.Timestamp(cached_df["date"].max()).date()
            else:
                default_start_date = (pd.Timestamp.today() - pd.Timedelta(days=730)).date()
                default_end_date = pd.Timestamp.today().date()
            col1, col2 = st.sidebar.columns(2)
            start = col1.date_input("Start", value=default_start_date, key="alp_start")
            end = col2.date_input("End", value=default_end_date, key="alp_end")
            start_ts = pd.Timestamp(start)
            end_ts = pd.Timestamp(end)
            end_exclusive = end_ts + pd.Timedelta(days=1)

            _cache_key = f"alp_loaded_{symbol}_{tf}_{start}_{end}"
            if _cache_key not in st.session_state:
                if cached_df is not None and not cached_df.empty:
                    mask = (cached_df["date"] >= start_ts) & (cached_df["date"] < end_exclusive)
                    in_range = cached_df[mask]
                    if not in_range.empty:
                        st.session_state["loaded_data"] = in_range.reset_index(drop=True)
                        st.session_state["loaded_symbol"] = symbol.upper()
                        st.session_state[_cache_key] = True
                        st.sidebar.success(
                            f"✓ Loaded {len(in_range):,} bars from local cache  \n"
                            f"({in_range['date'].iloc[0].date()} → {in_range['date'].iloc[-1].date()})"
                        )
                elif _ac.exists("alpaca", symbol, tf):
                    st.sidebar.warning(
                        "⚠️ The local intraday cache appears corrupted and was ignored. "
                        "Press Fetch / Update from Alpaca to rebuild it with full timestamps."
                    )

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
                    "For UVXY: start **2020-01-01** to get ~6 years."
                )

            if st.sidebar.button("🔄 Fetch / Update from Alpaca", key="alp_fetch"):
                from data.ingestion import load_from_alpaca_history

                creds = settings.alpaca
                key_ = creds.paper_api_key if not settings.is_live() else creds.live_api_key
                sec_ = creds.paper_secret_key if not settings.is_live() else creds.live_secret_key
                with st.spinner("Checking for new bars…"):
                    try:
                        data = load_from_alpaca_history(symbol, tf, start_ts, end_exclusive, key_, sec_, paper=not settings.is_live())
                        st.session_state["loaded_data"] = data
                        st.session_state["loaded_symbol"] = symbol.upper()
                        for k in list(st.session_state.keys()):
                            if k.startswith(f"alp_loaded_{symbol}_{tf}"):
                                del st.session_state[k]
                        st.sidebar.success(
                            f"✓ {len(data):,} bars  \n"
                            f"({data['date'].iloc[0].date()} → {data['date'].iloc[-1].date()})"
                        )
                    except Exception as e:
                        st.sidebar.error(str(e))

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
                if col_b.button("🗑", key=f"del_{entry['source']}_{entry['symbol']}_{entry['timeframe']}", help="Delete this cache entry"):
                    cache.delete(entry["source"], entry["symbol"].replace("=", "_"), entry["timeframe"])
                    st.rerun()

    return st.session_state.get("loaded_data")


_PARAM_META: dict[str, dict] = {
    "rsi_period": {"label": "RSI Period", "help": "Number of bars for RSI calculation."},
    "rsi_oversold": {"label": "RSI Oversold", "help": "RSI level to trigger BUY."},
    "rsi_overbought": {"label": "RSI Overbought", "help": "RSI level to trigger SELL."},
    "bb_period": {"label": "BB Period", "help": "Bollinger Bands SMA period."},
    "bb_std": {"label": "BB Std Devs", "help": "Standard deviations for upper/lower bands."},
    "sl_band_mult": {"label": "SL beyond band", "help": "SL = outer band ± (mult × band width)."},
    "require_cross": {"label": "Require band cross", "help": "Price must cross band, not just touch."},
    "min_band_width_pct": {"label": "Min band width %", "help": "Skip if band < X% of price."},
    "min_rr_ratio": {"label": "Min R:R ratio", "help": "Skip if TP distance / SL distance < X."},
    "cooldown_bars": {"label": "Cooldown bars", "help": "Min bars between entries."},
    "min_atr_pct": {"label": "Min ATR % of price", "help": "Skip signals when ATR < X% of price."},
    "spike_gap_pct": {"label": "Spike gap %", "help": "Single-bar % jump that starts a spike episode immediately."},
    "grad_spike_lookback": {"label": "Grad spike lookback", "help": "Lookback window for slower multi-day spike build-ups."},
    "grad_spike_pct": {"label": "Grad spike %", "help": "Rise above rolling low that starts a spike episode."},
    "rise_lookback": {"label": "Rise lookback bars", "help": "Gradual rise check window."},
    "rise_pct": {"label": "Rise % threshold", "help": "Price > X% above lookback low suppresses shorts."},
    "spike_long_sl_pct": {"label": "Spike long SL %", "help": "Hard stop for the dedicated gap-spike long leg."},
    "spike_long_trail_pct": {"label": "Spike long trail %", "help": "Percent trailing stop for spike longs."},
    "spike_long_max": {"label": "Spike long max", "help": "Maximum dedicated spike-long entries per episode."},
    "spike_long_cooldown": {"label": "Spike long cooldown", "help": "Bars between dedicated spike-long entries."},
    "spike_lockout_bars": {"label": "Spike lockout bars", "help": "Bars to wait after a spike episode ends before another spike-specific trade can fire."},
    "spike_long_min_rsi": {"label": "Spike long min RSI", "help": "Dedicated spike longs only fire when RSI is at least this high."},
    "spike_long_min_peak_pct": {"label": "Spike long min peak %", "help": "Minimum episode expansion above the spike base before spike-long logic can activate."},
    "spike_long_min_atr_pct": {"label": "Spike long min ATR %", "help": "Dedicated spike longs only fire when ATR as a percent of price is at least this high."},
    "spike_profile_shock_gap_pct": {"label": "Shock gap %", "help": "Gap size that classifies a spike as a shock-style spike instead of a persistent one."},
    "spike_profile_shock_peak_pct": {"label": "Shock peak %", "help": "Episode expansion that upgrades a spike to shock-style behavior."},
    "spike_rollover_watch_bars": {"label": "Rollover watch bars", "help": "Bars since peak before a persistent spike can be treated as rolling over."},
    "spike_rollover_watch_peak_pct": {"label": "Rollover watch peak %", "help": "Minimum persistent spike size before rollover-watch logic is allowed."},
    "spike_rollover_fast_ema_tol": {"label": "Rollover EMA tol", "help": "How close price must be to or below the fast EMA before a persistent spike is treated as rolling over."},
    "persistent_rollover_long_rsi_max": {"label": "Rollover long RSI max", "help": "During persistent-rollover spikes, normal longs are only allowed if RSI is at or below this deeper-reset level."},
    "persistent_spread_block_pct": {"label": "Persistent spread block %", "help": "When the fast EMA stays this far above the slow EMA during a persistent spike, normal longs require a deeper RSI reset."},
    "persistent_spread_deep_rsi_max": {"label": "Persistent deep RSI max", "help": "Deeper RSI cap required for normal longs when persistent spike EMA spread is still highly stretched."},
    "persistent_prepeak_block_peak_pct": {"label": "Pre-peak block peak %", "help": "During persistent spike build-ups, once peak expansion exceeds this level the strategy can stop taking normal longs if the move already looks too stretched."},
    "persistent_prepeak_block_rsi_min": {"label": "Pre-peak block RSI min", "help": "Minimum RSI needed before the persistent pre-peak guard blocks normal longs. Higher values make the block apply only to more overheated ramps."},
    "persistent_rebound_trap_peak_pct": {"label": "Rebound trap peak %", "help": "Minimum total spike expansion before the strategy treats a persistent-spike pullback as a mature rebound trap instead of a fresh dip-buying opportunity."},
    "persistent_rebound_trap_drawdown_pct": {"label": "Rebound trap drawdown %", "help": "How far price must pull back from the running spike high before the mature persistent-spike rebound trap turns on."},
    "persistent_active_long_day_max": {"label": "Persistent long/day max", "help": "Maximum number of normal longs allowed per day while a persistent spike is active."},
    "spike_momentum_max": {"label": "Spike momo max", "help": "Maximum breakout-style momentum longs allowed per spike episode."},
    "spike_momo_atr_mult": {"label": "Spike momo ATR mult", "help": "ATR must be at least this multiple of its recent average before the breakout momentum long is allowed."},
    "spike_momo_momentum_bars": {"label": "Spike momo momentum bars", "help": "Lookback used to measure short-horizon price momentum for the breakout spike-long leg."},
    "spike_momo_momentum_pct": {"label": "Spike momo momentum %", "help": "Price must be up by at least this percent versus the momentum lookback bar before a breakout spike long can trigger."},
    "spike_momo_min_bar_pct": {"label": "Spike momo min bar %", "help": "Minimum single-bar expansion required for the dedicated spike momentum long."},
    "spike_momo_min_rsi": {"label": "Spike momo min RSI", "help": "Momentum spike longs require RSI at or above this level."},
    "spike_momo_max_rsi": {"label": "Spike momo max RSI", "help": "Momentum spike longs are blocked once RSI rises above this level to avoid chasing an already-exhausted blowoff."},
    "spike_momo_min_peak_pct": {"label": "Spike momo min peak %", "help": "Momentum spike longs only activate once the episode has already expanded by at least this much."},
    "spike_momo_min_atr_pct": {"label": "Spike momo min ATR %", "help": "Minimum ATR as a percent of price required for the breakout momentum long."},
    "spike_momo_trail_pct": {"label": "Spike momo trail %", "help": "Trailing stop for the dedicated momentum spike long leg."},
    "spike_momo_sl_pct": {"label": "Spike momo SL %", "help": "Hard stop for the dedicated momentum spike long leg."},
    "spike_momo_cooldown": {"label": "Spike momo cooldown", "help": "Bars between dedicated momentum spike-long entries."},
    "spike_momo_max": {"label": "Spike momo max", "help": "Maximum momentum spike-long entries per spike episode."},
    "psshort_drop_pct": {"label": "Reversal drop %", "help": "How far price must fall from the spike peak before the episode can flip from spike to decay."},
    "psshort_sl_pct": {"label": "Post-spike short SL %", "help": "Hard stop for the first confirmed post-spike short."},
    "psshort_trail_pct": {"label": "Post-spike trail %", "help": "Percent trailing stop for the first confirmed post-spike short."},
    "psshort_max": {"label": "Post-spike max", "help": "Maximum number of initial reversal shorts per spike episode."},
    "event_target_short_enabled": {"label": "Event target short", "help": "Enable the optional one-short-per-spike module that enters after a confirmed peak and aims for a large portion of the spike unwind."},
    "event_target_anchor_lookback": {"label": "Event anchor lookback", "help": "How far back the strategy looks from the spike peak to find the start price used for the event-target unwind calculation."},
    "event_target_max_rise_bars": {"label": "Event max rise bars", "help": "Maximum number of bars allowed between the event anchor and the peak. Lower values restrict the module to sharper, more self-contained spike episodes."},
    "event_target_min_peak_pct": {"label": "Event min peak %", "help": "Minimum rise from the event anchor to the peak for a spike episode to qualify for the event-target short."},
    "event_target_completion_pct": {"label": "Event completion %", "help": "How much of the spike unwind the event short tries to capture. Higher values hold longer and target deeper retracements toward the spike start price."},
    "event_target_confirm_drop_pct": {"label": "Event confirm drop %", "help": "Minimum drop from the peak required before the event-target short is allowed to open. This helps avoid shorting before the down-leg is truly underway."},
    "event_target_sl_pct": {"label": "Event short SL %", "help": "Hard stop for the event-target short, measured above the short entry price."},
    "psshort_cooldown": {"label": "Post-spike cooldown", "help": "Bars between confirmed post-spike short entries."},
    "psshort_window": {"label": "Spike episode window", "help": "How long a spike episode stays active after onset for reversal/decay logic."},
    "spike_reversal_atr_frac": {"label": "ATR cooloff frac", "help": "Spike must cool to this fraction of peak ATR before shorts unlock."},
    "spike_reversal_ema_fast": {"label": "Reversal EMA fast", "help": "Fast EMA used to confirm the spike has rolled over."},
    "spike_reversal_ema_slow": {"label": "Reversal EMA slow", "help": "Slow EMA used to confirm decay trend after a spike."},
    "spike_reversal_min_bars": {"label": "Reversal min bars", "help": "Minimum bars after the peak before the first decay short can trigger."},
    "spike_reversal_min_peak_pct": {"label": "Reversal min peak %", "help": "Minimum full spike size before post-spike shorts are allowed."},
    "decay_reentry_rsi": {"label": "Decay re-entry RSI", "help": "Only short decay bounces after RSI has reset upward to at least this level."},
    "decay_bounce_min_pct": {"label": "Decay bounce min %", "help": "Minimum rebound off the local decay low before a continuation short setup can arm."},
    "decay_bounce_fail_pct": {"label": "Decay bounce fail %", "help": "How much the rebound must roll over from its bounce high before the continuation short triggers."},
    "decay_bounce_cooldown": {"label": "Decay bounce cooldown", "help": "Bars between continuation-short entries during the same decay episode."},
    "decay_bounce_max": {"label": "Decay bounce max", "help": "Maximum continuation-short entries inside one decay episode."},
    "decay_cooldown": {"label": "Decay cooldown", "help": "Bars between decay re-entry shorts."},
    "decay_max_entries": {"label": "Decay max entries", "help": "Maximum decay re-entry shorts per episode."},
    "decay_floor": {"label": "Decay floor", "help": "Skip decay shorts below this absolute price."},
    "low_price_chop_price": {"label": "Low-price chop price", "help": "Below this price, the strategy treats UVXY as being in a late low-price chop regime and suppresses some normal trades to avoid whipsaw clusters."},
    "low_price_chop_bandwidth_pct": {"label": "Low-price chop BW %", "help": "Maximum Bollinger band width percent for the low-price chop guard. Narrower low-price action below this threshold is treated as whipsaw noise."},
}


def render_strategy_params(strategy_id: str, leverage: float = 1.0, max_capital_loss_pct: float = 50.0) -> dict:
    from strategies import get_strategy

    cls = get_strategy(strategy_id)
    instance = cls()
    defaults = instance.default_params()
    max_sl_price_pct = max_capital_loss_pct / max(leverage, 1.0)

    st.subheader(f"⚙️ {cls.name} Parameters")
    st.caption(cls.description)

    if strategy_id == "bollinger_rsi":
        st.info(
            "📉 **Bollinger + RSI (Spike-Aware)** — UVXY mean reversion plus explicit spike episode handling.  \n"
            "**Episode flow:** `idle → spike → decay → idle`  \n"
            "• 🟢 **Normal** — Bollinger mean reversion outside spike episodes  \n"
            "• 🟡 **Spike** — rare dedicated spike-long logic plus a momentum long for true expansion bars; normal shorts suppressed  \n"
            "• 🔴 **Decay** — conservative reversal-confirmed shorts plus decay re-entries on bearish bounce setups  \n"
            "• ⏸️ **Drift** — ATR too low, so all signals pause  \n"
            "• 🛑 **Lockout** — after a spike episode ends, spike-specific trades stay disabled for a cooldown window to avoid pseudo-spike churn"
        )

    if leverage > 1.0:
        st.warning(
            f"⚡ **Leverage {leverage:.1f}×** active.  "
            f"Platform cap: max **{max_capital_loss_pct:.0f}% capital loss** per trade.  \n"
            f"→ Maximum allowed **price move SL = {max_sl_price_pct:.2f}%**."
        )

    filled: dict = {}

    prominent_bools = ["require_cross", "event_target_short_enabled"]
    checkbox_params = [param for param in prominent_bools if param in defaults]
    if checkbox_params:
        checkbox_cols = st.columns(len(checkbox_params))
        for idx, param in enumerate(checkbox_params):
            default = defaults[param]
            meta = _PARAM_META.get(param, {})
            label = meta.get("label", param)
            help_ = meta.get("help", "") or None
            key = f"p_{strategy_id}_{param}"
            with checkbox_cols[idx]:
                filled[param] = st.checkbox(label, value=bool(default), key=key, help=help_)

    st.markdown(
            """
        <style>
        div[data-testid="stExpander"] details summary p {
            color: #FFFFFF !important;
            font-weight: 700 !important;
        }
        div[data-testid="stExpander"] label p,
        div[data-testid="stExpander"] .stNumberInput label p,
        div[data-testid="stExpander"] .stTextInput label p,
        div[data-testid="stExpander"] .stCheckbox label p {
            color: #FFFFFF !important;
            font-weight: 700 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    grouped_params = [(param, default) for param, default in defaults.items() if param not in checkbox_params]
    with st.expander("Model Parameters", expanded=False):
        cols = st.columns(4)
        for i, (param, default) in enumerate(grouped_params):
            col = cols[i % 4]
            meta = _PARAM_META.get(param, {})
            label = meta.get("label", param)
            help_ = meta.get("help", "") or None
            full_label = f"{label} ({param})"
            key = f"p_{strategy_id}_{param}"
            with col:
                head_a, head_b = st.columns([0.88, 0.12], vertical_alignment="center")
                with head_a:
                    st.markdown(
                        (
                                "<div style='color:#FFFFFF;font-weight:700;font-size:0.95rem;margin-bottom:0.20rem;'>"
                                f"{label} <span style='color:#C9D4F0;font-weight:500;'>({param})</span>"
                                "</div>"
                        ),
                        unsafe_allow_html = True,
                    )
                if help_:
                    with head_b:
                        with st.popover("?", use_container_width=True):
                            st.caption(help_)
                if isinstance(default, bool):
                    filled[param] = st.checkbox("Enabled", value=default, key=key, label_visibility="collapsed")
                elif isinstance(default, int):
                    filled[param] = st.number_input(label, value=int(default), step=1, key=key,
                                                    label_visibility="collapsed")
                elif isinstance(default, float):
                    filled[param] = st.number_input(label, value=float(default), format="%.2f", min_value=0.0, key=key, label_visibility="collapsed")
                else:
                    filled[param] = st.text_input(label, value=str(default), key=key, label_visibility="collapsed")
    return filled


def render_metrics_row(metrics: dict) -> None:
    cols = st.columns(len(metrics))
    for col, (label, value) in zip(cols, metrics.items()):
        with col:
            st.metric(label, value[0], delta=value[1]) if isinstance(value, tuple) and len(value) == 2 else st.metric(label, value)


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

"""
ui/components.py  —  Reusable Streamlit widget blocks.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd
import streamlit as st

from config.symbol_profiles import context_label, resolve_context_symbol
from config.settings import settings
from strategies import get_strategy, list_strategies
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


def _resolve_companion_strategy_id(explicit_strategy_id: str | None = None) -> str | None:
    if explicit_strategy_id:
        return explicit_strategy_id
    available = list_strategies()
    name_to_id = {item["name"]: item["id"] for item in available}
    for key in ("bt_strategy", "pt_strategy"):
        selected_name = st.session_state.get(key)
        if selected_name in name_to_id:
            return name_to_id[selected_name]
    return None


def _companion_strategy(explicit_strategy_id: str | None = None):
    strategy_id = _resolve_companion_strategy_id(explicit_strategy_id)
    if not strategy_id:
        return None
    try:
        cls = get_strategy(strategy_id)
        return cls(params={})
    except Exception:
        return None


def _render_companion_hint(
    symbol: str,
    source: str,
    interval: str | None,
    explicit_strategy_id: str | None = None,
) -> None:
    strategy = _companion_strategy(explicit_strategy_id)
    if strategy is None:
        return

    symbol_u = symbol.strip().upper()
    companion_lines: list[str] = []
    if hasattr(strategy, "companion_contexts"):
        for context_key in strategy.companion_contexts(symbol_u, source=source, interval=interval) or []:
            resolved = resolve_context_symbol(symbol_u, context_key)
            if resolved and resolved.strip().upper() != symbol_u:
                companion_lines.append(f"{context_label(context_key)}: {resolved.strip().upper()}")
    elif hasattr(strategy, "companion_symbols"):
        companions = strategy.companion_symbols(symbol_u, source=source, interval=interval) or []
        companion_lines.extend([sym for sym in companions if sym and sym.strip().upper() != symbol_u])

    if not companion_lines:
        return
    st.sidebar.info(
        "Companion market context for this symbol: "
        f"**{' | '.join(companion_lines)}**\n\n"
        "When you fetch the main symbol, these caches will also be checked and only missing bars will be appended."
    )


def _is_intraday_interval(interval: str | None) -> bool:
    if not interval:
        return False
    key = str(interval).lower()
    return any(token in key for token in ("min", "hour", "m", "h")) and "day" not in key and "wk" not in key


def _alpaca_safe_request_end(end_exclusive: pd.Timestamp, interval: str | None) -> tuple[pd.Timestamp, bool]:
    """Avoid recent SIP bars that Alpaca accounts may not be allowed to query."""
    requested_end = pd.Timestamp(end_exclusive).tz_localize(None)
    if not _is_intraday_interval(interval):
        return requested_end, False

    safe_end = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(minutes=30)
    if requested_end > safe_end:
        return safe_end, True
    return requested_end, False


def render_data_source_selector(strategy_id: str | None = None) -> Optional[pd.DataFrame]:
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
        _render_companion_hint(ticker, "yfinance", interval, strategy_id)

        _yf_key = f"yf_loaded_{ticker}_{interval}_{start}_{end}"
        if _yf_key not in st.session_state:
            cached_df = _yc.load("yfinance", ticker.upper(), interval)
            if cached_df is not None and not cached_df.empty:
                mask = (cached_df["date"] >= start_ts) & (cached_df["date"] < end_exclusive)
                in_range = cached_df[mask]
                if not in_range.empty:
                    st.session_state["loaded_data"] = in_range.reset_index(drop=True)
                    st.session_state["loaded_symbol"] = ticker.upper()
                    st.session_state["loaded_source"] = "yfinance"
                    st.session_state["loaded_interval"] = interval
                    st.session_state["loaded_start"] = start_ts
                    st.session_state["loaded_end"] = pd.Timestamp(in_range["date"].max())
                    st.session_state[_yf_key] = True
                    st.sidebar.success(f"✓ {len(in_range):,} bars from cache")

        if st.sidebar.button("🔄 Fetch / Update", type="primary", key="yf_fetch"):
            from data.ingestion import load_from_ticker, prefetch_strategy_companions

            with st.spinner("Fetching…"):
                try:
                    data = load_from_ticker(ticker, interval, start_ts, end_exclusive)
                    st.session_state["loaded_data"] = data
                    st.session_state["loaded_symbol"] = ticker.upper()
                    st.session_state["loaded_source"] = "yfinance"
                    st.session_state["loaded_interval"] = interval
                    st.session_state["loaded_start"] = start_ts
                    st.session_state["loaded_end"] = pd.Timestamp(data["date"].max()) if not data.empty else end_exclusive
                    for k in list(st.session_state.keys()):
                        if k.startswith(f"yf_loaded_{ticker}_{interval}"):
                            del st.session_state[k]
                    warmed = prefetch_strategy_companions(
                        _companion_strategy(strategy_id),
                        primary_symbol=ticker.upper(),
                        source="yfinance",
                        interval=interval,
                        start=start_ts,
                        end=end_exclusive,
                    )
                    st.sidebar.success(f"✓ {len(data):,} bars")
                    if warmed:
                        st.sidebar.caption(f"Companion cache ready: {', '.join(warmed)}")
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
                st.session_state["loaded_source"] = "csv"
                st.session_state["loaded_interval"] = None
                st.session_state["loaded_start"] = pd.Timestamp(data["date"].min()) if not data.empty else None
                st.session_state["loaded_end"] = pd.Timestamp(data["date"].max()) if not data.empty else None
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
            _render_companion_hint(symbol, "alpaca", tf, strategy_id)

            _cache_key = f"alp_loaded_{symbol}_{tf}_{start}_{end}"
            if _cache_key not in st.session_state:
                if cached_df is not None and not cached_df.empty:
                    mask = (cached_df["date"] >= start_ts) & (cached_df["date"] < end_exclusive)
                    in_range = cached_df[mask]
                    if not in_range.empty:
                        st.session_state["loaded_data"] = in_range.reset_index(drop=True)
                        st.session_state["loaded_symbol"] = symbol.upper()
                        st.session_state["loaded_source"] = "alpaca"
                        st.session_state["loaded_interval"] = tf
                        st.session_state["loaded_start"] = start_ts
                        st.session_state["loaded_end"] = pd.Timestamp(in_range["date"].max())
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
                    "💡 **Alpaca SIP feed** — ~5 years of 1-min history, with recent intraday bars delayed by account permissions.  \n"
                    "For UVXY: start **2020-01-01** to get ~6 years."
                )

            if st.sidebar.button("🔄 Fetch / Update from Alpaca", key="alp_fetch"):
                from data.ingestion import load_from_alpaca_history, prefetch_strategy_companions

                creds = settings.alpaca
                key_ = creds.paper_api_key if not settings.is_live() else creds.live_api_key
                sec_ = creds.paper_secret_key if not settings.is_live() else creds.live_secret_key
                with st.spinner("Checking for new bars…"):
                    try:
                        request_end, capped_recent = _alpaca_safe_request_end(end_exclusive, tf)
                        if request_end <= start_ts:
                            raise ValueError(
                                "Selected Alpaca end time is too recent for SIP intraday data. "
                                "Choose an earlier end date or try again after the market data delay has passed."
                            )
                        if capped_recent:
                            st.sidebar.warning(
                                "Alpaca SIP does not allow this account to query very recent intraday bars. "
                                f"Fetch capped at {request_end:%Y-%m-%d %H:%M} UTC; existing cache will still be appended only for missing bars."
                            )

                        data = load_from_alpaca_history(symbol, tf, start_ts, request_end, key_, sec_, paper=not settings.is_live())
                        st.session_state["loaded_data"] = data
                        st.session_state["loaded_symbol"] = symbol.upper()
                        st.session_state["loaded_source"] = "alpaca"
                        st.session_state["loaded_interval"] = tf
                        st.session_state["loaded_start"] = start_ts
                        st.session_state["loaded_end"] = pd.Timestamp(data["date"].max()) if not data.empty else request_end
                        for k in list(st.session_state.keys()):
                            if k.startswith(f"alp_loaded_{symbol}_{tf}"):
                                del st.session_state[k]
                        warmed = prefetch_strategy_companions(
                            _companion_strategy(strategy_id),
                            primary_symbol=symbol.upper(),
                            source="alpaca",
                            interval=tf,
                            start=start_ts,
                            end=request_end,
                        )
                        st.sidebar.success(
                            f"✓ {len(data):,} bars  \n"
                            f"({data['date'].iloc[0].date()} → {data['date'].iloc[-1].date()})"
                        )
                        if warmed:
                            st.sidebar.caption(f"Companion cache ready: {', '.join(warmed)}")
                    except Exception as e:
                        msg = str(e)
                        if "subscription does not permit querying recent SIP data" in msg:
                            st.sidebar.error(
                                "Alpaca rejected the request because the selected end time is too recent for SIP intraday data. "
                                "Choose an earlier end date, or try again after the market data delay has passed."
                            )
                        else:
                            st.sidebar.error(msg)

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
    "normal_long_enabled": {"label": "Normal longs enabled", "help": "Enable the baseline Bollinger/RSI long mean-reversion entries. Some symbol presets disable this when the instrument is better handled by sparse momentum rules."},
    "normal_short_enabled": {"label": "Normal shorts enabled", "help": "Enable the baseline Bollinger/RSI short mean-reversion entries outside active spike regimes."},
    "trend_bias_long_enabled": {"label": "Trend bias long", "help": "Enable the long-bias trend-continuation leg that buys constructive pullbacks in already-established uptrends."},
    "trend_bias_fast_ema": {"label": "Trend fast EMA", "help": "Fast EMA used by the long-bias trend module to define support reclamation."},
    "trend_bias_slow_ema": {"label": "Trend slow EMA", "help": "Slow EMA used to confirm that the broader trend is still pointed upward."},
    "trend_bias_lookback_bars": {"label": "Trend lookback bars", "help": "Lookback used to measure the recent high and pullback depth for the long-bias trend module."},
    "trend_bias_min_retrace_pct": {"label": "Trend min retrace %", "help": "Minimum pullback from the recent high before a long-bias trend entry is allowed."},
    "trend_bias_min_momentum_120": {"label": "Trend min 120-bar momentum %", "help": "Minimum medium-horizon momentum required before the trend-continuation leg can buy the pullback."},
    "trend_bias_min_atr_pct": {"label": "Trend min ATR %", "help": "Minimum ATR as a percent of price required for the long-bias trend entry."},
    "trend_bias_min_rsi": {"label": "Trend min RSI", "help": "Lower RSI bound for the trend-continuation entry window."},
    "trend_bias_max_rsi": {"label": "Trend max RSI", "help": "Upper RSI bound for the trend-continuation entry window."},
    "trend_bias_trail_pct": {"label": "Trend trail %", "help": "Percent trailing stop used once the long-bias trend trade is open."},
    "trend_bias_sl_pct": {"label": "Trend long SL %", "help": "Initial stop for the long-bias trend trade, measured below entry."},
    "trend_bias_cooldown": {"label": "Trend long cooldown", "help": "Bars to wait before another long-bias trend entry is allowed."},
    "trend_context_score_enabled": {"label": "Trend context score", "help": "Use a weighted SLV/GDX/VIXY-style companion score to qualify long-bias trend entries."},
    "trend_context_min_score": {"label": "Trend context min score", "help": "Minimum weighted companion score required before the long-bias trend leg can enter."},
    "trend_peer_strength_weight": {"label": "Trend peer weight", "help": "Score contribution from the precious-metals peer, such as SLV, for the trend-continuation leg."},
    "trend_miners_strength_weight": {"label": "Trend miners weight", "help": "Score contribution from the miners proxy, such as GDX, for the trend-continuation leg."},
    "trend_riskoff_strength_weight": {"label": "Trend risk-off weight", "help": "Score contribution from a risk-off proxy, such as VIXY, when it supports the gold trend."},
    "intraday_pullback_short_enabled": {"label": "Intraday pullback short", "help": "Enable a short module that fades intraday overbought bursts only after downside confirmation. Useful for symbols like UVXY when RSI spikes into exhaustion and then starts rolling over."},
    "intraday_pullback_rsi_trigger": {"label": "Pullback RSI trigger", "help": "Recent RSI must have reached at least this level before the intraday pullback short can arm."},
    "intraday_pullback_rsi_fade_pts": {"label": "Pullback RSI fade pts", "help": "Minimum number of RSI points price must fade from the recent RSI peak before the pullback short is allowed."},
    "intraday_pullback_lookback_bars": {"label": "Pullback lookback bars", "help": "Lookback used to measure the recent high and the highest RSI for the intraday pullback-short setup."},
    "intraday_pullback_drop_pct": {"label": "Pullback drop %", "help": "Price must already be this far below the recent intraday high before the pullback short is allowed."},
    "intraday_pullback_min_atr_pct": {"label": "Pullback min ATR %", "help": "Minimum ATR as a percent of price required before the intraday pullback-short setup is allowed."},
    "intraday_pullback_sl_pct": {"label": "Pullback short SL %", "help": "Hard stop for the intraday pullback-short module, measured above the short entry price."},
    "intraday_pullback_tp_pct": {"label": "Pullback short TP %", "help": "Initial target for the intraday pullback-short module, measured below the short entry price."},
    "intraday_pullback_trail_pct": {"label": "Pullback short trail %", "help": "Percent trailing stop used once the intraday pullback short is open."},
    "intraday_pullback_cooldown": {"label": "Pullback short cooldown", "help": "Bars to wait before allowing another intraday pullback short."},
    "spike_gap_pct": {"label": "Spike gap %", "help": "Single-bar % jump that starts a spike episode immediately."},
    "grad_spike_lookback": {"label": "Grad spike lookback", "help": "Lookback window for slower multi-day spike build-ups."},
    "grad_spike_pct": {"label": "Grad spike %", "help": "Rise above rolling low that starts a spike episode."},
    "grad_spike_rearm_bars": {"label": "Grad spike rearm bars", "help": "Quiet-period bars required before a new gradual-spike onset can trigger again after the previous episode."},
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
    "spike_atr_mult": {"label": "Spike ATR mult", "help": "ATR must rise above this multiple of its recent average before the strategy treats volatility as spike-active."},
    "spike_sl_pct": {"label": "Spike SL %", "help": "Hard stop for the core spike long leg that trades inside active spike episodes."},
    "spike_atr_trail": {"label": "Spike ATR trail", "help": "ATR-based trailing-stop multiple for the core spike long leg."},
    "spike_max_entries": {"label": "Spike max entries", "help": "Maximum number of core spike-long entries allowed during a single spike episode."},
    "spike_cooldown": {"label": "Spike cooldown", "help": "Bars to wait between core spike-long entries during the same episode."},
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
    "event_target_persistent_confirm_drop_pct": {"label": "Event persistent drop %", "help": "Stricter peak-drop confirmation used for slower persistent-style spikes before the event-target short can open."},
    "event_target_sl_pct": {"label": "Event short SL %", "help": "Hard stop for the event-target short, measured above the short entry price."},
    "event_target_profit_giveback_frac": {"label": "Event profit giveback frac", "help": "After the event short has meaningful open profit, close it if price gives back this fraction of the best unrealized gain."},
    "event_target_profit_giveback_min_pct": {"label": "Event giveback min %", "help": "Minimum open profit required before the event short's giveback-protection exit becomes active."},
    "spy_selloff_assist_ret_30": {"label": "Benchmark selloff 30-bar %", "help": "The companion equity benchmark's 30-bar return must be at or below this level before UVXY spike-momentum longs can use the selloff-assist path."},
    "spy_selloff_assist_ret_120": {"label": "Benchmark selloff 120-bar %", "help": "The companion equity benchmark's 120-bar return must be at or below this level before UVXY spike-momentum longs can use the selloff-assist path."},
    "spy_rebound_block_ret_30": {"label": "Benchmark rebound 30-bar %", "help": "If the companion equity benchmark rebounds by at least this much over 30 bars and reclaims its EMAs, UVXY longs are blocked during active spike phases."},
    "spy_rebound_block_ret_120": {"label": "Benchmark rebound 120-bar %", "help": "Medium-horizon rebound filter used with the 30-bar benchmark rebound rule to avoid buying UVXY while equities are already stabilizing."},
    "dollar_strength_block_ret_30": {"label": "Dollar strength block 30-bar %", "help": "Block gold momentum longs when the dollar benchmark is rising by at least this much over 30 bars and is in an uptrend. Neutral high values disable this filter."},
    "dollar_strength_block_ret_120": {"label": "Dollar strength block 120-bar %", "help": "Medium-horizon dollar-strength threshold used with the 30-bar dollar filter for GLD-style presets."},
    "rates_weakness_block_ret_30": {"label": "Rates weakness block 30-bar %", "help": "Block gold momentum longs when the rates benchmark is falling by at least this much over 30 bars and is in a downtrend. Neutral very-low values disable this filter."},
    "rates_weakness_block_ret_120": {"label": "Rates weakness block 120-bar %", "help": "Medium-horizon rates-weakness threshold used with the 30-bar rates filter for GLD-style presets."},
    "long_rates_weakness_block_ret_30": {"label": "Long-rates weakness block 30-bar %", "help": "Block gold momentum longs when the long-rates benchmark is falling by at least this much over 30 bars and is in a downtrend."},
    "long_rates_weakness_block_ret_120": {"label": "Long-rates weakness block 120-bar %", "help": "Medium-horizon long-rates weakness threshold used with the 30-bar long-rates filter for GLD-style presets."},
    "gold_macro_score_enabled": {"label": "Gold macro score enabled", "help": "Use a weighted macro-risk score for gold-style symbols instead of simple one-indicator blocking."},
    "gold_macro_block_score": {"label": "Gold macro block score", "help": "Block GLD-style momentum longs when the weighted dollar/rates macro-risk score reaches this level."},
    "dollar_strength_weight": {"label": "Dollar strength weight", "help": "Weight assigned to dollar-strength risk in the GLD macro-risk score."},
    "rates_weakness_weight": {"label": "Rates weakness weight", "help": "Weight assigned to intermediate-rates weakness risk in the GLD macro-risk score."},
    "long_rates_weakness_weight": {"label": "Long-rates weakness weight", "help": "Weight assigned to long-rates weakness risk in the GLD macro-risk score. Leave neutral unless the symbol profile maps a long-rates benchmark."},
    "gold_macro_regime_enabled": {"label": "Gold macro regime", "help": "Enable a slower bidirectional macro regime for gold-style symbols. It classifies the background as bullish, neutral, or bearish and uses that as bias for both longs and shorts."},
    "gold_macro_regime_fast_bars": {"label": "Gold regime fast bars", "help": "Shorter slow-macro lookback used to measure the background regime of the gold companion proxies."},
    "gold_macro_regime_slow_bars": {"label": "Gold regime slow bars", "help": "Longer slow-macro lookback used to classify the background gold regime from companion proxies."},
    "gold_macro_regime_bullish_score": {"label": "Gold regime bullish score", "help": "Minimum weighted macro score required to classify the gold backdrop as bullish."},
    "gold_macro_regime_bearish_score": {"label": "Gold regime bearish score", "help": "Maximum weighted macro score required to classify the gold backdrop as bearish."},
    "gold_regime_dollar_weight": {"label": "Gold regime dollar weight", "help": "How strongly the dollar benchmark contributes to the slow gold macro regime. Dollar strength is bearish for gold; dollar weakness is bullish."},
    "gold_regime_rates_weight": {"label": "Gold regime rates weight", "help": "How strongly the rates benchmark contributes to the slow gold macro regime. Bond strength / lower yields are treated as supportive for gold."},
    "gold_regime_long_rates_weight": {"label": "Gold regime long-rates weight", "help": "Optional weight for a long-duration rates benchmark in the slow gold macro regime."},
    "gold_regime_peer_weight": {"label": "Gold regime peer weight", "help": "How strongly the precious-metals peer, such as SLV, contributes to the slow gold macro regime."},
    "gold_regime_miners_weight": {"label": "Gold regime miners weight", "help": "How strongly the gold-miners proxy, such as GDX, contributes to the slow gold macro regime."},
    "gold_regime_riskoff_weight": {"label": "Gold regime risk-off weight", "help": "How strongly the risk-off proxy, such as VIXY, contributes to the slow gold macro regime."},
    "gold_peer_confirm_enabled": {"label": "Precious peer confirmation", "help": "Require the precious-metals peer, such as SLV for GLD, to be in an uptrend before opening gold momentum longs."},
    "gold_peer_confirm_ret_30": {"label": "Precious peer confirm 30-bar %", "help": "Minimum 30-bar return required from the precious-metals peer when peer confirmation is enabled."},
    "gold_peer_confirm_ret_120": {"label": "Precious peer confirm 120-bar %", "help": "Minimum 120-bar return required from the precious-metals peer when peer confirmation is enabled."},
    "gold_miners_confirm_enabled": {"label": "Miners confirmation", "help": "Require the gold-miners proxy, such as GDX for GLD, to be in an uptrend before opening gold momentum longs."},
    "gold_miners_confirm_ret_30": {"label": "Miners confirm 30-bar %", "help": "Minimum 30-bar return required from the miners proxy when miners confirmation is enabled."},
    "gold_miners_confirm_ret_120": {"label": "Miners confirm 120-bar %", "help": "Minimum 120-bar return required from the miners proxy when miners confirmation is enabled."},
    "gold_riskoff_override_enabled": {"label": "Risk-off override", "help": "Allow a strong risk-off proxy move, such as VIXY rising, to override hostile dollar/rates macro filters for gold momentum longs."},
    "gold_riskoff_ret_30": {"label": "Risk-off override 30-bar %", "help": "Minimum 30-bar return required from the risk-off proxy before it can override the gold macro-risk filter."},
    "gold_riskoff_ret_120": {"label": "Risk-off override 120-bar %", "help": "Minimum 120-bar return required from the risk-off proxy before it can override the gold macro-risk filter."},
    "gold_context_assist_enabled": {"label": "Gold context assist", "help": "Allow strong SLV/GDX/VIXY-style context to help GLD momentum entries qualify with slightly softer internal thresholds."},
    "gold_context_assist_min_score": {"label": "Gold context assist min score", "help": "Minimum combined confirmation score from the gold companion proxies before the context-assist entry path can activate."},
    "gold_peer_strength_weight": {"label": "Precious peer strength weight", "help": "Score contribution when the precious-metals peer, such as SLV, confirms GLD strength."},
    "gold_miners_strength_weight": {"label": "Miners strength weight", "help": "Score contribution when the gold-miners proxy, such as GDX, confirms GLD strength."},
    "gold_riskoff_strength_weight": {"label": "Risk-off strength weight", "help": "Score contribution when the risk-off proxy, such as VIXY, confirms a risk-off backdrop that may support gold."},
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
    "spike_high_window": {"label": "Spike high window", "help": "Rolling lookback used to track recent episode highs for spike-strength and peak-drop calculations."},
    "spike_ema_mult": {"label": "Spike EMA mult", "help": "Price extension multiplier versus the long spike EMA used by the broader spike-state filters."},
    "spike_ema_span": {"label": "Spike EMA span", "help": "Long EMA span used by spike-state filters that measure extended spike conditions."},
    "peak_drop_pct": {"label": "Peak drop %", "help": "Additional peak-drop threshold used by legacy spike-to-decay filters. Very high values effectively disable this path."},
    "reversion_tp_pct": {"label": "Reversion TP %", "help": "Target percentage for the early reversal short leg once a spike first flips into decay."},
    "reversion_sl_pct": {"label": "Reversion SL %", "help": "Stop percentage for the early reversal short leg once a spike first flips into decay."},
    "reversion_rsi_min": {"label": "Reversion RSI min", "help": "Minimum RSI reset required before the early reversal short leg is allowed to trigger."},
    "decay_ema_period": {"label": "Decay EMA period", "help": "EMA length used to define the broader decay trend after a spike peak."},
    "decay_slope_lb": {"label": "Decay slope lookback", "help": "Lookback used to measure the EMA slope for decay-trend detection."},
    "decay_slope_min_pct": {"label": "Decay slope min %", "help": "Minimum negative EMA slope required by the broader decay filter. Very high values effectively disable this path."},
    "decay_atr_trail": {"label": "Decay ATR trail", "help": "ATR-based trailing-stop multiple used for decay short legs."},
    "decay_sl_pct": {"label": "Decay SL %", "help": "Hard stop percentage used for decay continuation and re-entry shorts."},
    "decay_cooldown": {"label": "Decay cooldown", "help": "Bars between decay re-entry shorts."},
    "decay_max_entries": {"label": "Decay max entries", "help": "Maximum decay re-entry shorts per episode."},
    "decay_floor": {"label": "Decay floor", "help": "Skip decay shorts below this absolute price."},
    "low_price_chop_price": {"label": "Low-price chop price", "help": "Below this price, the strategy treats UVXY as being in a late low-price chop regime and suppresses some normal trades to avoid whipsaw clusters."},
    "low_price_chop_bandwidth_pct": {"label": "Low-price chop BW %", "help": "Maximum Bollinger band width percent for the low-price chop guard. Narrower low-price action below this threshold is treated as whipsaw noise."},
}

_PARAM_GROUPS: list[tuple[str, str, list[str]]] = [
    (
        "Core Mean Reversion",
        "Baseline Bollinger/RSI entry, reset, and reward-to-risk controls outside special spike handling.",
        [
            "bb_period",
            "bb_std",
            "rsi_period",
            "rsi_oversold",
            "rsi_overbought",
            "sl_band_mult",
            "min_band_width_pct",
            "min_rr_ratio",
            "cooldown_bars",
            "min_atr_pct",
            "normal_long_enabled",
            "normal_short_enabled",
        ],
    ),
    (
        "Spike Lifecycle",
        "How the strategy detects, classifies, and manages spike episodes before switching into rollover or decay logic.",
        [
            "spike_gap_pct",
            "grad_spike_lookback",
            "grad_spike_pct",
            "grad_spike_rearm_bars",
            "rise_lookback",
            "rise_pct",
            "spike_lockout_bars",
            "spike_profile_shock_gap_pct",
            "spike_profile_shock_peak_pct",
            "spike_rollover_watch_bars",
            "spike_rollover_watch_peak_pct",
            "spike_rollover_fast_ema_tol",
            "persistent_rollover_long_rsi_max",
            "persistent_spread_block_pct",
            "persistent_spread_deep_rsi_max",
            "persistent_prepeak_block_peak_pct",
            "persistent_prepeak_block_rsi_min",
            "persistent_rebound_trap_peak_pct",
            "persistent_rebound_trap_drawdown_pct",
            "persistent_active_long_day_max",
        ],
    ),
    (
        "Momentum & Up-Leg Capture",
        "Dedicated long-side rules for explosive upside moves, including breakout, volatility, and trailing-stop behavior.",
        [
            "spike_long_sl_pct",
            "spike_long_trail_pct",
            "spike_long_max",
            "spike_long_cooldown",
            "spike_long_min_rsi",
            "spike_long_min_peak_pct",
            "spike_long_min_atr_pct",
            "spike_momentum_max",
            "spike_momo_atr_mult",
            "spike_momo_momentum_bars",
            "spike_momo_momentum_pct",
            "spike_momo_min_bar_pct",
            "spike_momo_min_rsi",
            "spike_momo_max_rsi",
            "spike_momo_min_peak_pct",
            "spike_momo_min_atr_pct",
            "spike_momo_trail_pct",
            "spike_momo_sl_pct",
            "spike_momo_cooldown",
            "spike_momo_max",
        ],
    ),
    (
        "Trend Continuation",
        "Stay with strong longer-lasting uptrends by buying controlled pullbacks that reclaim trend support with supportive companion context.",
        [
            "trend_bias_fast_ema",
            "trend_bias_slow_ema",
            "trend_bias_lookback_bars",
            "trend_bias_min_retrace_pct",
            "trend_bias_min_momentum_120",
            "trend_bias_min_atr_pct",
            "trend_bias_min_rsi",
            "trend_bias_max_rsi",
            "trend_bias_trail_pct",
            "trend_bias_sl_pct",
            "trend_bias_cooldown",
            "trend_context_score_enabled",
            "trend_context_min_score",
            "trend_peer_strength_weight",
            "trend_miners_strength_weight",
            "trend_riskoff_strength_weight",
        ],
    ),
    (
        "Intraday Pullback Short",
        "Fade overbought intraday bursts only after RSI has rolled over, price has pulled back, and downside confirmation is present.",
        [
            "intraday_pullback_rsi_trigger",
            "intraday_pullback_rsi_fade_pts",
            "intraday_pullback_lookback_bars",
            "intraday_pullback_drop_pct",
            "intraday_pullback_min_atr_pct",
            "intraday_pullback_sl_pct",
            "intraday_pullback_tp_pct",
            "intraday_pullback_trail_pct",
            "intraday_pullback_cooldown",
        ],
    ),
    (
        "Post-Peak & Event Short",
        "Rules for confirming the top, entering post-peak shorts, and managing decay or large unwind targets.",
        [
            "psshort_drop_pct",
            "psshort_sl_pct",
            "psshort_trail_pct",
            "psshort_max",
            "psshort_cooldown",
            "psshort_window",
            "spike_reversal_atr_frac",
            "spike_reversal_ema_fast",
            "spike_reversal_ema_slow",
            "spike_reversal_min_bars",
            "spike_reversal_min_peak_pct",
            "event_target_anchor_lookback",
            "event_target_max_rise_bars",
            "event_target_min_peak_pct",
            "event_target_completion_pct",
            "event_target_confirm_drop_pct",
            "event_target_persistent_confirm_drop_pct",
            "event_target_sl_pct",
            "event_target_profit_giveback_frac",
            "event_target_profit_giveback_min_pct",
            "decay_reentry_rsi",
            "decay_bounce_min_pct",
            "decay_bounce_fail_pct",
            "decay_bounce_cooldown",
            "decay_bounce_max",
            "decay_cooldown",
            "decay_max_entries",
            "decay_floor",
            "spike_high_window",
            "spike_ema_mult",
            "spike_ema_span",
            "peak_drop_pct",
            "reversion_tp_pct",
            "reversion_sl_pct",
            "reversion_rsi_min",
            "decay_ema_period",
            "decay_slope_lb",
            "decay_slope_min_pct",
            "decay_atr_trail",
            "decay_sl_pct",
        ],
    ),
    (
        "External Context",
        "Companion-market filters and assists that use the symbol profile's benchmark context when one is available.",
        [
            "spy_selloff_assist_ret_30",
            "spy_selloff_assist_ret_120",
            "spy_rebound_block_ret_30",
            "spy_rebound_block_ret_120",
            "dollar_strength_block_ret_30",
            "dollar_strength_block_ret_120",
            "rates_weakness_block_ret_30",
            "rates_weakness_block_ret_120",
            "long_rates_weakness_block_ret_30",
            "long_rates_weakness_block_ret_120",
            "gold_macro_score_enabled",
            "gold_macro_block_score",
            "dollar_strength_weight",
            "rates_weakness_weight",
            "long_rates_weakness_weight",
            "gold_macro_regime_enabled",
            "gold_macro_regime_fast_bars",
            "gold_macro_regime_slow_bars",
            "gold_macro_regime_bullish_score",
            "gold_macro_regime_bearish_score",
            "gold_regime_dollar_weight",
            "gold_regime_rates_weight",
            "gold_regime_long_rates_weight",
            "gold_regime_peer_weight",
            "gold_regime_miners_weight",
            "gold_regime_riskoff_weight",
            "gold_peer_confirm_enabled",
            "gold_peer_confirm_ret_30",
            "gold_peer_confirm_ret_120",
            "gold_miners_confirm_enabled",
            "gold_miners_confirm_ret_30",
            "gold_miners_confirm_ret_120",
            "gold_riskoff_override_enabled",
            "gold_riskoff_ret_30",
            "gold_riskoff_ret_120",
            "gold_context_assist_enabled",
            "gold_context_assist_min_score",
            "gold_peer_strength_weight",
            "gold_miners_strength_weight",
            "gold_riskoff_strength_weight",
        ],
    ),
    (
        "Late-Regime Cleanup",
        "Filters that suppress noisy trades in weak late-stage, low-price chop conditions.",
        [
            "low_price_chop_price",
            "low_price_chop_bandwidth_pct",
        ],
    ),
]


def render_strategy_params(
    strategy_id: str,
    leverage: float = 1.0,
    max_capital_loss_pct: float = 50.0,
    symbol: str | None = None,
    source: str | None = None,
    interval: str | None = None,
) -> dict:
    from strategies import get_strategy

    cls = get_strategy(strategy_id)
    instance = cls()
    active_symbol = symbol or st.session_state.get("loaded_symbol")
    active_source = source or st.session_state.get("loaded_source")
    active_interval = interval or st.session_state.get("loaded_interval")
    defaults = instance.effective_default_params(
        symbol=active_symbol,
        source=active_source,
        interval=active_interval,
    )
    max_sl_price_pct = max_capital_loss_pct / max(leverage, 1.0)
    reset_key = f"reset_params_{strategy_id}"
    defaults_signature = (
        strategy_id,
        str(active_symbol or ""),
        str(active_source or ""),
        str(active_interval or ""),
        tuple(defaults.items()),
    )
    signature_key = f"p_{strategy_id}_defaults_signature"
    if st.session_state.get(signature_key) != defaults_signature:
        for param, default in defaults.items():
            st.session_state[f"p_{strategy_id}_{param}"] = default
        st.session_state[signature_key] = defaults_signature

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

    header_a, header_b = st.columns([0.82, 0.18], vertical_alignment="center")
    with header_b:
        if st.button("↺ Reset to defaults", key=reset_key, use_container_width=True):
            for param, default in defaults.items():
                st.session_state[f"p_{strategy_id}_{param}"] = default
            st.session_state[signature_key] = defaults_signature
            st.rerun()

    filled: dict = {}

    prominent_bools = ["require_cross", "event_target_short_enabled", "intraday_pullback_short_enabled"]
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
        remaining = {param for param, _ in grouped_params}

        def _render_param(param: str, default):
            meta = _PARAM_META.get(param, {})
            label = meta.get("label", param)
            help_ = meta.get("help", "") or None
            key = f"p_{strategy_id}_{param}"
            head_a, head_b = st.columns([0.88, 0.12], vertical_alignment="center")
            with head_a:
                st.markdown(
                    (
                        "<div style='color:#FFFFFF;font-weight:700;font-size:0.95rem;margin-bottom:0.20rem;'>"
                        f"{label} <span style='color:#C9D4F0;font-weight:500;'>({param})</span>"
                        "</div>"
                    ),
                    unsafe_allow_html=True,
                )
            if help_:
                with head_b:
                    with st.popover("?", use_container_width=True):
                        st.caption(help_)
            if isinstance(default, bool):
                filled[param] = st.checkbox("Enabled", value=default, key=key, label_visibility="collapsed")
            elif isinstance(default, int):
                filled[param] = st.number_input(label, value=int(default), step=1, key=key, label_visibility="collapsed")
            elif isinstance(default, float):
                min_value = None if default < 0 else 0.0
                filled[param] = st.number_input(
                    label,
                    value=float(default),
                    format="%.2f",
                    min_value=min_value,
                    key=key,
                    label_visibility="collapsed",
                )
            else:
                filled[param] = st.text_input(label, value=str(default), key=key, label_visibility="collapsed")

        for title, caption, params_in_group in _PARAM_GROUPS:
            group_items = [(param, defaults[param]) for param in params_in_group if param in defaults and param in remaining]
            if not group_items:
                continue
            st.markdown(
                f"<div style='color:#E8EAF6;font-weight:800;font-size:1.0rem;margin-top:0.6rem;margin-bottom:0.5rem;'>{title}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div style='color:#D7E3FF;font-size:0.92rem;margin-top:-0.15rem;margin-bottom:0.55rem;'>{caption}</div>",
                unsafe_allow_html=True,
            )
            cols = st.columns(4)
            for i, (param, default) in enumerate(group_items):
                with cols[i % 4]:
                    _render_param(param, default)
                remaining.discard(param)

        other_items = [(param, defaults[param]) for param in defaults if param in remaining]
        if other_items:
            st.markdown(
                "<div style='color:#E8EAF6;font-weight:800;font-size:1.0rem;margin-top:0.6rem;margin-bottom:0.5rem;'>Other</div>",
                unsafe_allow_html=True,
            )
            cols = st.columns(4)
            for i, (param, default) in enumerate(other_items):
                with cols[i % 4]:
                    _render_param(param, default)
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

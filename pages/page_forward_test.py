"""
pages/page_forward_test.py
───────────────────────────
Forward Test — live data, simulated orders, no broker.

Sits between backtester (historical) and paper trading (broker-connected).

How it works:
  1. You configure a strategy + symbol + interval
  2. Click "Start Forward Test" — the app fetches the latest N bars from
     Yahoo Finance and evaluates the strategy signal RIGHT NOW
  3. If a signal fires, a simulated trade is opened and stored locally
  4. Click "Refresh" (or enable auto-refresh) to fetch the next bar and
     check for exit conditions on open simulated trades
  5. Full trade log, equity curve, and signal history are shown live

Key differences vs Backtester:
  - Uses CURRENT market data, not historical replay
  - Each refresh = one new bar arriving (simulates live feed)
  - Signals fire in real time — you see exactly what the strategy
    would do if connected to a broker right now
  - No order is ever sent anywhere — 100% local simulation

Key differences vs Paper Trading:
  - No Alpaca connection required
  - Works on any yfinance symbol including GC=F, UVXY
  - Designed for strategy validation before going live
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from typing import Optional

import altair as alt
import pandas as pd
import streamlit as st

from config.settings import settings
from core.models import Direction, SignalAction, TradeOutcome
from data.ingestion import load_from_ticker
from risk.manager import RiskManager, RiskCheckResult
from strategies import list_strategies, get_strategy
from ui.components import render_mode_banner, render_strategy_params, render_metrics_row
from ui.charts import rsi_chart

_GREEN  = "#26a69a"
_RED    = "#ef5350"
_BLUE   = "#4a9eff"
_GOLD   = "#ffd54f"
_ORANGE = "#ff9800"
_GREY   = "#9e9eb8"
_AXIS   = dict(gridColor="#2a2d3e", labelColor="#d0d4f0", titleColor="#d0d4f0",
               labelFontSize=12, titleFontSize=13)
_TITLE  = dict(color="#e8eaf6", fontSize=14, fontWeight="bold")

# ─── Session state keys ───────────────────────────────────────────────────────
_KEY_TRADES    = "ft_trades"
_KEY_SIGNALS   = "ft_signals"
_KEY_PRICES    = "ft_prices"
_KEY_CONFIG    = "ft_config"
_KEY_OPEN      = "ft_open_trade"
_KEY_EQUITY    = "ft_equity_history"
_KEY_RUNNING   = "ft_running"


def _init_state() -> None:
    for key, default in [
        (_KEY_TRADES,   []),
        (_KEY_SIGNALS,  []),
        (_KEY_PRICES,   None),
        (_KEY_CONFIG,   {}),
        (_KEY_OPEN,     None),
        (_KEY_EQUITY,   []),
        (_KEY_RUNNING,  False),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default


def _reset_state() -> None:
    st.session_state[_KEY_TRADES]  = []
    st.session_state[_KEY_SIGNALS] = []
    st.session_state[_KEY_PRICES]  = None
    st.session_state[_KEY_OPEN]    = None
    st.session_state[_KEY_EQUITY]  = []
    st.session_state[_KEY_RUNNING] = True


def _interval_to_timedelta(interval: str) -> timedelta:
    mapping = {"1m": timedelta(minutes=1), "2m": timedelta(minutes=2),
               "5m": timedelta(minutes=5), "15m": timedelta(minutes=15),
               "30m": timedelta(minutes=30), "1h": timedelta(hours=1),
               "1d": timedelta(days=1)}
    return mapping.get(interval, timedelta(minutes=5))


def _fetch_latest(symbol: str, interval: str, lookback_bars: int) -> pd.DataFrame:
    """Fetch the most recent lookback_bars bars for the symbol."""
    delta     = _interval_to_timedelta(interval)
    # Add buffer for weekends/holidays
    buffer    = max(lookback_bars * 3, 500)
    end       = pd.Timestamp.now()
    start     = end - delta * buffer
    data      = load_from_ticker(symbol, interval, start, end)
    return data.tail(lookback_bars).reset_index(drop=True)


def _calc_leveraged_return(entry: float, exit_p: float,
                            leverage: float, direction: Direction) -> float:
    raw = (exit_p - entry) / entry
    if direction == Direction.SHORT:
        raw = -raw
    return raw * leverage * 100


def _check_exit(trade: dict, bar: pd.Series) -> dict:
    """Check if an open simulated trade hit SL or TP on this bar."""
    high = float(bar["high"])
    low  = float(bar["low"])
    tp   = trade.get("take_profit")
    sl   = trade["stop_loss"]
    ep   = trade["entry_price"]
    lev  = trade["leverage"]
    d    = trade["direction"]

    hit_sl = hit_tp = False
    if d == "Long":
        hit_sl = low  <= sl
        hit_tp = tp is not None and high >= tp
    else:
        hit_sl = high >= sl
        hit_tp = tp is not None and low  <= tp

    if hit_sl and hit_tp:
        trade["outcome"]   = "Ambiguous candle"
        trade["exit_time"] = bar["date"]
    elif hit_sl:
        trade["outcome"]            = "SL hit"
        trade["exit_price"]         = sl
        trade["exit_time"]          = bar["date"]
        trade["leveraged_return_%"] = _calc_leveraged_return(ep, sl, lev, d)
    elif hit_tp:
        trade["outcome"]            = "TP hit"
        trade["exit_price"]         = tp
        trade["exit_time"]          = bar["date"]
        trade["leveraged_return_%"] = _calc_leveraged_return(ep, tp, lev, d)
    return trade


# ─── Chart helpers ────────────────────────────────────────────────────────────

def _price_chart(prices: pd.DataFrame, trades: list, signals: list,
                 symbol: str) -> alt.LayerChart:
    base = (alt.Chart(prices).mark_line(color=_BLUE, strokeWidth=1.4)
            .encode(x=alt.X("date:T", title="Date / Time", axis=alt.Axis(**_AXIS)),
                    y=alt.Y("close:Q", title="Price", scale=alt.Scale(zero=False),
                            axis=alt.Axis(**_AXIS)),
                    tooltip=["date:T", alt.Tooltip("close:Q", format=".4f")]))
    layers = [base]

    if signals:
        sig_df = pd.DataFrame(signals)
        buy_s  = sig_df[sig_df["action"] == "BUY"].copy()
        sell_s = sig_df[sig_df["action"] == "SELL"].copy()
        tt     = ["date:T", "action:N", alt.Tooltip("close:Q", format=".4f", title="Price"),
                  "strategy:N"]
        if not buy_s.empty:
            buy_s["y"] = buy_s["close"] * 0.997
            layers.append(alt.Chart(buy_s)
                          .mark_point(shape="triangle-up", size=120, filled=True, color=_GREEN)
                          .encode(x="date:T", y="y:Q", tooltip=tt))
        if not sell_s.empty:
            sell_s["y"] = sell_s["close"] * 1.003
            layers.append(alt.Chart(sell_s)
                          .mark_point(shape="triangle-down", size=120, filled=True, color=_RED)
                          .encode(x="date:T", y="y:Q", tooltip=tt))

    if trades:
        closed = [t for t in trades if t.get("exit_price")]
        if closed:
            ex_df = pd.DataFrame(closed)
            win   = ex_df[ex_df["leveraged_return_%"].fillna(0) > 0]
            loss  = ex_df[ex_df["leveraged_return_%"].fillna(0) <= 0]
            tt_x  = ["exit_time:T", "outcome:N",
                     alt.Tooltip("exit_price:Q", format=".4f"),
                     alt.Tooltip("leveraged_return_%:Q", format=".2f", title="Return %")]
            for sub, col in [(win, _GREEN), (loss, _RED)]:
                if not sub.empty:
                    layers.append(alt.Chart(sub.rename(columns={"exit_time":"date",
                                                                  "exit_price":"price"}))
                                  .mark_point(shape="cross", size=100,
                                              strokeWidth=2.5, color=col)
                                  .encode(x="date:T", y="price:Q", tooltip=tt_x))

    return (alt.layer(*layers)
            .properties(title=alt.TitleParams(
                f"{symbol} – Forward Test  ·  ▲ BUY  ▼ SELL  ✕ Exit", **_TITLE),
                height=320)
            .configure_view(strokeOpacity=0)
            .configure_axis(**_AXIS)
            .configure_title(**_TITLE))


def _equity_chart(equity_history: list, starting_capital: float,
                  symbol: str) -> alt.LayerChart:
    if len(equity_history) < 2:
        return alt.Chart(pd.DataFrame()).mark_line()

    df       = pd.DataFrame(equity_history)
    baseline = (alt.Chart(pd.DataFrame({"y": [starting_capital]}))
                .mark_rule(color=_GREY, strokeDash=[4, 4], strokeWidth=1)
                .encode(y="y:Q"))
    line     = (alt.Chart(df)
                .mark_area(line={"color": _BLUE, "strokeWidth": 2},
                           color=alt.Gradient(gradient="linear",
                               stops=[alt.GradientStop(color="rgba(74,158,255,0.2)", offset=0),
                                      alt.GradientStop(color="rgba(74,158,255,0.0)", offset=1)],
                               x1=1, x2=1, y1=1, y2=0))
                .encode(x=alt.X("time:T", title="Time", axis=alt.Axis(**_AXIS)),
                        y=alt.Y("equity:Q", title="Equity ($)",
                                scale=alt.Scale(zero=False), axis=alt.Axis(**_AXIS)),
                        tooltip=["time:T",
                                 alt.Tooltip("equity:Q", format="$,.2f"),
                                 alt.Tooltip("pnl:Q",    format="+$,.2f", title="Last P&L")]))
    dots     = (alt.Chart(df)
                .mark_point(size=50, filled=True)
                .encode(x="time:T", y="equity:Q",
                        color=alt.condition(alt.datum.pnl > 0,
                                            alt.value(_GREEN), alt.value(_RED)),
                        tooltip=["time:T",
                                 alt.Tooltip("equity:Q", format="$,.2f"),
                                 alt.Tooltip("pnl:Q",    format="+$,.2f", title="Last P&L")]))
    return (alt.layer(baseline, line, dots)
            .properties(title=alt.TitleParams(
                f"{symbol} – Simulated Equity (forward test)", **_TITLE), height=260)
            .configure_view(strokeOpacity=0)
            .configure_axis(**_AXIS)
            .configure_title(**_TITLE))


# ─── Page ─────────────────────────────────────────────────────────────────────

def render() -> None:
    _init_state()
    render_mode_banner()

    st.title("🔭 Forward Test")
    st.caption(
        "Run your strategy on **live market data** with **simulated orders** — "
        "no broker connection needed. Each refresh fetches the latest bar and "
        "evaluates the strategy in real time."
    )

    st.info(
        "**How this differs from Backtester and Paper Trading:**  \n"
        "• **Backtester** — replays historical data you already have  \n"
        "• **Forward Test** ← you are here — fetches live data, simulates locally, no broker  \n"
        "• **Paper Trading** — connects to Alpaca paper account, sends real API orders"
    )

    st.divider()

    # ── Configuration ─────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        symbol   = st.text_input("Symbol", value="GC=F", key="ft_symbol").upper()
        interval = st.selectbox("Interval",
                                ["1m","2m","5m","15m","30m","1h","1d"],
                                index=0, key="ft_interval")
    with col2:
        lookback = st.number_input("Warm-up bars", min_value=50,
                                   max_value=1000, value=200, step=50,
                                   key="ft_lookback",
                                   help="How many recent bars to load for indicator warm-up. "
                                        "Must be > your slowest indicator period.")
        capital  = st.number_input("Capital per trade ($)", min_value=10.0,
                                   value=1000.0, key="ft_capital")
    with col3:
        leverage = st.number_input("Leverage", min_value=1.0,
                                   max_value=100.0, value=1.0, step=0.5,
                                   key="ft_leverage")
        max_loss = st.slider("Max capital loss %", 5, 100, 50,
                             key="ft_maxloss")

    strategies  = list_strategies()
    strat_names = {s["name"]: s["id"] for s in strategies}
    selected_name = st.selectbox("Strategy", list(strat_names.keys()),
                                 key="ft_strategy")
    selected_id   = strat_names[selected_name]

    params = render_strategy_params(selected_id, leverage=leverage,
                                    max_capital_loss_pct=float(max_loss))

    st.divider()

    # ── Control buttons ───────────────────────────────────────────────────────
    col_start, col_refresh, col_stop, col_close = st.columns(4)

    start_clicked   = col_start.button("▶ Start Forward Test",
                                       type="primary", key="ft_start")
    refresh_clicked = col_refresh.button("🔄 Fetch Next Bar",
                                         key="ft_refresh",
                                         disabled=not st.session_state[_KEY_RUNNING])
    stop_clicked    = col_stop.button("⏹ Stop",
                                      key="ft_stop",
                                      disabled=not st.session_state[_KEY_RUNNING])
    close_clicked   = col_close.button("❌ Close Open Trade",
                                       key="ft_close_open",
                                       disabled=st.session_state[_KEY_OPEN] is None)

    # Auto-refresh toggle
    auto_refresh = st.checkbox(
        "Auto-refresh every minute",
        value=False, key="ft_auto",
        help="When checked, the page re-fetches data automatically. "
             "Streamlit re-runs on each interaction — use this to simulate "
             "a live bar arriving."
    )

    if auto_refresh and st.session_state[_KEY_RUNNING]:
        import time
        st.caption(f"⏱ Auto-refresh active — last update: {datetime.now().strftime('%H:%M:%S')}")

    # ── Start ─────────────────────────────────────────────────────────────────
    if start_clicked:
        _reset_state()
        st.session_state[_KEY_CONFIG] = {
            "symbol": symbol, "interval": interval,
            "lookback": lookback, "capital": capital,
            "leverage": leverage, "max_loss": max_loss,
            "strategy_id": selected_id, "params": dict(params),
            "started_at": datetime.now().isoformat(),
        }
        st.rerun()

    # ── Stop ──────────────────────────────────────────────────────────────────
    if stop_clicked:
        st.session_state[_KEY_RUNNING] = False
        st.success("Forward test stopped. Results preserved below.")

    # ── Nothing started yet ───────────────────────────────────────────────────
    if not st.session_state[_KEY_RUNNING] and not st.session_state[_KEY_TRADES]:
        st.info("Configure your strategy above and click **▶ Start Forward Test**.")
        return

    # ── Fetch + evaluate (on start, refresh, or auto-refresh) ─────────────────
    cfg = st.session_state[_KEY_CONFIG]
    should_fetch = (start_clicked or refresh_clicked or
                    (auto_refresh and st.session_state[_KEY_RUNNING]))

    if should_fetch and st.session_state[_KEY_RUNNING]:
        with st.spinner(f"Fetching latest {cfg['lookback']} bars for {cfg['symbol']}…"):
            try:
                prices = _fetch_latest(cfg["symbol"], cfg["interval"],
                                       cfg["lookback"])
                st.session_state[_KEY_PRICES] = prices
            except Exception as e:
                st.error(f"Data fetch failed: {e}")
                return

        prices     = st.session_state[_KEY_PRICES]
        latest_bar = prices.iloc[-1]
        latest_ts  = latest_bar["date"]

        # ── Check open trade exit ──────────────────────────────────────────
        open_trade = st.session_state[_KEY_OPEN]
        if open_trade is not None:
            open_trade = _check_exit(open_trade, latest_bar)
            if open_trade.get("outcome") not in (None, "Open"):
                pnl = (open_trade["capital"] *
                       open_trade.get("leveraged_return_%", 0) / 100)
                open_trade["pnl"] = round(pnl, 2)
                st.session_state[_KEY_TRADES].append(open_trade)
                # Update equity
                prev_eq = (st.session_state[_KEY_EQUITY][-1]["equity"]
                           if st.session_state[_KEY_EQUITY]
                           else cfg["capital"])
                st.session_state[_KEY_EQUITY].append({
                    "time":   latest_ts,
                    "equity": round(prev_eq + pnl, 2),
                    "pnl":    round(pnl, 2),
                })
                st.session_state[_KEY_OPEN] = None
                open_trade = None

        # ── Generate signal ────────────────────────────────────────────────
        if open_trade is None:
            cls      = get_strategy(cfg["strategy_id"])
            strategy = cls(params=cfg["params"])
            signal   = strategy.generate_signal(prices, cfg["symbol"])

            sig_row = {
                "date":     latest_ts,
                "action":   signal.action.value,
                "close":    float(latest_bar["close"]),
                "strategy": cls.name,
                "tp":       signal.suggested_tp,
                "sl":       signal.suggested_sl,
                "rsi":      signal.metadata.get("rsi"),
            }
            st.session_state[_KEY_SIGNALS].append(sig_row)

            # Open simulated trade if signal fired
            if signal.action != SignalAction.HOLD and signal.suggested_sl is not None:
                risk = RiskManager(settings.risk)
                direction = (Direction.LONG if signal.action == SignalAction.BUY
                             else Direction.SHORT)
                check = risk.check(
                    direction=direction,
                    entry_price=float(latest_bar["close"]),
                    take_profit=signal.suggested_tp,
                    stop_loss=signal.suggested_sl,
                    leverage=cfg["leverage"],
                    capital_requested=cfg["capital"],
                )
                if check.approved:
                    eff_sl = check.adjusted_sl or signal.suggested_sl
                    st.session_state[_KEY_OPEN] = {
                        "id":          str(uuid.uuid4())[:8],
                        "symbol":      cfg["symbol"],
                        "direction":   direction.value,
                        "entry_price": float(latest_bar["close"]),
                        "take_profit": signal.suggested_tp,
                        "stop_loss":   eff_sl,
                        "leverage":    cfg["leverage"],
                        "capital":     cfg["capital"],
                        "entry_time":  latest_ts,
                        "strategy":    cls.name,
                        "outcome":     "Open",
                        "exit_price":  None,
                        "exit_time":   None,
                    }

    # ── Manual close open trade ────────────────────────────────────────────────
    if close_clicked and st.session_state[_KEY_OPEN] is not None:
        prices = st.session_state.get(_KEY_PRICES)
        if prices is not None:
            t   = st.session_state[_KEY_OPEN]
            ep  = t["entry_price"]
            xp  = float(prices.iloc[-1]["close"])
            d   = t["direction"]
            lev = t["leverage"]
            raw = (xp - ep) / ep * (1 if d == "Long" else -1)
            ret = raw * lev * 100
            pnl = t["capital"] * ret / 100
            t.update({"outcome": "Manual close", "exit_price": xp,
                      "exit_time": prices.iloc[-1]["date"],
                      "leveraged_return_%": round(ret, 3),
                      "pnl": round(pnl, 2)})
            st.session_state[_KEY_TRADES].append(t)
            st.session_state[_KEY_OPEN] = None
            st.rerun()

    # ── Display ────────────────────────────────────────────────────────────────
    prices = st.session_state.get(_KEY_PRICES)
    if prices is None:
        return

    latest = prices.iloc[-1]

    # Status bar
    open_trade = st.session_state[_KEY_OPEN]
    status_col1, status_col2, status_col3, status_col4 = st.columns(4)
    status_col1.metric("Symbol", cfg["symbol"])
    status_col2.metric("Last Price", f"{latest['close']:.4f}")
    status_col3.metric("Last Bar Time",
                       pd.Timestamp(latest["date"]).strftime("%H:%M:%S"))
    status_col4.metric("Open Position",
                       f"{open_trade['direction']} @ {open_trade['entry_price']:.4f}"
                       if open_trade else "None")

    # Open trade details
    if open_trade:
        tp_str = f"{open_trade['take_profit']:.4f}" if open_trade.get("take_profit") else "—"
        sl_str = f"{open_trade['stop_loss']:.4f}"
        curr   = float(latest["close"])
        ep     = open_trade["entry_price"]
        lev    = open_trade["leverage"]
        d      = open_trade["direction"]
        raw    = (curr - ep) / ep * (1 if d == "Long" else -1)
        unreal = raw * lev * 100
        colour = "green" if unreal >= 0 else "red"
        st.markdown(
            f'<div style="border:1px solid #2a2d3e;border-radius:8px;padding:10px 16px;">'
            f'<b>Open Trade:</b> {d} {cfg["symbol"]} · '
            f'Entry: <code>{ep:.4f}</code> · '
            f'SL: <code>{sl_str}</code> · TP: <code>{tp_str}</code> · '
            f'Unrealised: <span style="color:{colour};font-weight:bold">'
            f'{unreal:+.2f}%</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # Summary metrics
    closed = [t for t in st.session_state[_KEY_TRADES]
              if t.get("leveraged_return_%") is not None]
    if closed:
        total_pnl  = sum(t["pnl"] for t in closed)
        wins       = [t for t in closed if t.get("leveraged_return_%", 0) > 0]
        win_rate   = len(wins) / len(closed) * 100
        avg_ret    = sum(t["leveraged_return_%"] for t in closed) / len(closed)
        render_metrics_row({
            "Closed Trades": len(closed),
            "Win Rate":      f"{win_rate:.1f}%",
            "Total P&L":     f"${total_pnl:+,.2f}",
            "Avg Return":    f"{avg_ret:+.2f}%",
        })

    # Price + signals chart
    st.markdown("#### 📈 Price")
    st.altair_chart(
        _price_chart(prices, st.session_state[_KEY_TRADES],
                     st.session_state[_KEY_SIGNALS], cfg["symbol"]),
        use_container_width=True,
    )

    # RSI chart for RSI-based strategies
    if cfg.get("strategy_id") in ("rsi_threshold", "atr_rsi",
                                   "vwap_rsi", "bollinger_rsi",
                                   "ema_trend_rsi"):
        p_cfg = cfg.get("params", {})
        rsi_p = int(p_cfg.get("rsi_period", p_cfg.get("rsi_period", 9)))
        buys  = p_cfg.get("buy_levels", "30")
        sells = p_cfg.get("sell_levels", "70")
        try:
            buy_lvls  = [float(x) for x in str(buys).replace(";",",").split(",")
                         if x.strip()] or [30]
            sell_lvls = [float(x) for x in str(sells).replace(";",",").split(",")
                         if x.strip()] or [70]
        except Exception:
            buy_lvls, sell_lvls = [30], [70]

        st.markdown(f"#### 📉 RSI ({rsi_p})")
        st.altair_chart(
            rsi_chart(prices, rsi_p, buy_lvls, sell_lvls)
            .configure_view(strokeOpacity=0)
            .configure_axis(**_AXIS)
            .configure_title(**_TITLE),
            use_container_width=True,
        )

    # Equity curve
    if len(st.session_state[_KEY_EQUITY]) >= 2:
        st.markdown("#### 💰 Simulated Equity")
        st.altair_chart(
            _equity_chart(st.session_state[_KEY_EQUITY],
                          cfg["capital"], cfg["symbol"]),
            use_container_width=True,
        )

    # Signal log
    with st.expander("📡 Signal Log", expanded=False):
        if st.session_state[_KEY_SIGNALS]:
            sig_df = pd.DataFrame(st.session_state[_KEY_SIGNALS])
            st.dataframe(sig_df.sort_values("date", ascending=False),
                         use_container_width=True)
        else:
            st.info("No signals generated yet.")

    # Trade log
    with st.expander("📋 Trade Log", expanded=False):
        if st.session_state[_KEY_TRADES]:
            tr_df = pd.DataFrame(st.session_state[_KEY_TRADES])
            show_cols = [c for c in ["symbol","direction","entry_price","exit_price",
                                      "outcome","leveraged_return_%","pnl",
                                      "entry_time","exit_time","strategy"]
                         if c in tr_df.columns]
            st.dataframe(tr_df[show_cols].sort_values("entry_time",
                                                        ascending=False),
                         use_container_width=True)
        else:
            st.info("No closed trades yet.")

    # Auto-refresh mechanism
    if auto_refresh and st.session_state[_KEY_RUNNING]:
        import time
        interval_secs = int(_interval_to_timedelta(cfg["interval"]).total_seconds())
        refresh_every = max(interval_secs, 60)  # minimum 60s to avoid API rate limits
        time.sleep(refresh_every)
        st.rerun()

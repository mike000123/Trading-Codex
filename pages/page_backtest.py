"""
pages/page_backtest.py
Walk-forward backtester.
"""
from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from config.settings import settings
from data.fair_value import compute_gld_fair_value_diagnostics, fair_value_cache_fingerprint
from data.ingestion import prepare_strategy_data
from reporting.backtest import BacktestEngine
from risk.manager import RiskManager
from strategies import get_strategy, list_strategies
from ui.charts import pnl_distribution
from ui.components import (
    render_data_source_selector,
    render_metrics_row,
    render_mode_banner,
    render_strategy_params,
)

_GREEN = "#26a69a"
_RED = "#ef5350"
_BLUE = "#4a9eff"
_GOLD = "#ffd54f"
_ORANGE = "#ff9800"
_PURPLE = "#ab47bc"
_AXIS = dict(
    gridColor="#2a2d3e",
    labelColor="#d0d4f0",
    titleColor="#d0d4f0",
    labelFontSize=12,
    titleFontSize=13,
)
_TITLE = dict(color="#e8eaf6", fontSize=14, fontWeight="bold")
_MAX_CHART_PTS = 5_000


def _downsample(df: pd.DataFrame, max_pts: int = _MAX_CHART_PTS) -> pd.DataFrame:
    if len(df) <= max_pts:
        return df
    step = max(1, len(df) // max_pts)
    return df.iloc[::step].reset_index(drop=True)


def _is_signal_exit(outcome: str) -> bool:
    return any(k in outcome for k in ("overbought", "oversold", "Counter"))


def _calc_rsi(series: pd.Series, period: int) -> pd.Series:
    d = series.diff()
    g = d.clip(lower=0)
    l = (-d).clip(lower=0)
    ag = g.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    al = l.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return 100 - (100 / (1 + ag / al.replace(0, float("nan"))))


def _parse_levels(raw) -> list[float]:
    if isinstance(raw, (int, float)):
        return [float(raw)]
    s = str(raw).strip().lower()
    if s in ("", "none", "off", "-"):
        return []
    try:
        return sorted(float(p.strip()) for p in s.replace(";", ",").split(",") if p.strip())
    except ValueError:
        return []


def _bar_label(prices: pd.DataFrame) -> str:
    if len(prices) < 2:
        return "bars"

    # Use the dominant spacing, not the first two rows. Intraday data can start
    # with a missing/opening gap, which made 1-min caches display as "4-min".
    dates = pd.to_datetime(prices["date"], errors="coerce").dropna().sort_values()
    if len(dates) < 2:
        return "bars"
    diffs = dates.diff().dropna().dt.total_seconds()
    diffs = diffs[diffs > 0]
    if diffs.empty:
        return "bars"

    mode = diffs.mode()
    delta = int(round(float(mode.iloc[0] if not mode.empty else diffs.median())))
    return {
        60: "1-min",
        300: "5-min",
        900: "15-min",
        1800: "30-min",
        3600: "1-hour",
        86400: "1-day",
    }.get(delta, f"{max(1, int(round(delta / 60)))}-min")


def _trade_regime_from_notes(notes: str) -> str:
    raw = notes or ""
    if "regime=" not in raw:
        return "unknown"
    return raw.split("regime=")[1].split(" | ")[0]


def _trade_regime(trade) -> str:
    return _trade_regime_from_notes(getattr(trade, "notes", "") or "")


def _trade_events(trades) -> tuple[pd.DataFrame, pd.DataFrame]:
    entries, exits = [], []
    for i, t in enumerate(trades):
        direction = t.direction.value if hasattr(t.direction, "value") else str(t.direction)
        outcome = t.outcome.value if hasattr(t.outcome, "value") else str(t.outcome)
        ret = t.leveraged_return_pct
        label = f"T{i + 1}"
        entries.append(
            {
                "date": pd.Timestamp(t.entry_time),
                "price": t.entry_price,
                "direction": direction,
                "outcome": outcome,
                "return_pct": ret,
                "trade_n": label,
            }
        )
        if t.exit_time and t.exit_price is not None:
            exits.append(
                {
                    "date": pd.Timestamp(t.exit_time),
                    "price": t.exit_price,
                    "direction": direction,
                    "outcome": outcome,
                    "return_pct": ret,
                    "trade_n": label,
                    "win": (ret or 0) > 0,
                }
            )
    return pd.DataFrame(entries), pd.DataFrame(exits)


def _window_trade_summary(trades, start: str, end: str) -> dict:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) + pd.Timedelta(days=1)
    subset = [
        t
        for t in trades
        if t.leveraged_return_pct is not None and start_ts <= pd.Timestamp(t.entry_time) < end_ts
    ]
    if not subset:
        return {
            "Trades": 0,
            "PnL ($)": 0.0,
            "Win Rate": "0.0%",
            "Avg Win %": "0.00%",
            "Avg Loss %": "0.00%",
            "Top Regime": "—",
        }
    wins = [t for t in subset if (t.leveraged_return_pct or 0) > 0]
    losses = [t for t in subset if (t.leveraged_return_pct or 0) <= 0]
    regime_counts = pd.Series([_trade_regime(t) for t in subset]).value_counts()
    avg_win = sum((t.leveraged_return_pct or 0) for t in wins) / len(wins) if wins else 0.0
    avg_loss = sum((t.leveraged_return_pct or 0) for t in losses) / len(losses) if losses else 0.0
    return {
        "Trades": len(subset),
        "PnL ($)": round(sum((t.pnl or 0) for t in subset), 2),
        "Win Rate": f"{(len(wins) / len(subset) * 100):.1f}%",
        "Avg Win %": f"{avg_win:.2f}%",
        "Avg Loss %": f"{avg_loss:.2f}%",
        "Top Regime": regime_counts.index[0] if not regime_counts.empty else "—",
    }


def _window_price_move(prices: pd.DataFrame | None, start: str | None, end: str | None) -> tuple[float | None, float | None]:
    if prices is None or prices.empty or start is None or end is None:
        return None, None
    date_s = pd.to_datetime(prices["date"])
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) + pd.Timedelta(days=1)
    subset = prices[(date_s >= start_ts) & (date_s < end_ts)]
    if subset.empty:
        return None, None
    low = float(subset["close"].min())
    high = float(subset["close"].max())
    return low, ((high / max(low, 1e-9)) - 1.0) * 100.0


def _capture_efficiency(pnl: float, move_pct: float | None, baseline: float = 1000.0) -> str:
    if move_pct is None or move_pct <= 0:
        return "—"
    theoretical = baseline * (move_pct / 100.0)
    if theoretical <= 0:
        return "—"
    return f"{(pnl / theoretical * 100.0):.1f}%"


def _spike_structure(prices: pd.DataFrame | None, start: str, end: str) -> dict:
    if prices is None or prices.empty:
        return {
            "Start": start,
            "Peak": "—",
            "End": end,
            "Up Longs": 0,
            "Up Shorts": 0,
            "Post Longs": 0,
            "Post Shorts": 0,
        }
    date_s = pd.to_datetime(prices["date"])
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) + pd.Timedelta(days=1)
    subset = prices[(date_s >= start_ts) & (date_s < end_ts)].copy()
    if subset.empty:
        return {
            "Start": start,
            "Peak": "—",
            "End": end,
            "Up Longs": 0,
            "Up Shorts": 0,
            "Post Longs": 0,
            "Post Shorts": 0,
        }
    peak_idx = subset["close"].astype(float).idxmax()
    peak_ts = pd.Timestamp(prices.loc[peak_idx, "date"])
    return {
        "Start": start_ts.strftime("%Y-%m-%d"),
        "Peak": peak_ts.strftime("%Y-%m-%d %H:%M"),
        "End": pd.Timestamp(end).strftime("%Y-%m-%d"),
        "_peak_ts": peak_ts,
    }


def _comparison_report(prices, trades) -> pd.DataFrame:
    windows = [
        ("Full Sample", None, None),
        ("Aug 2024 Spike", "2024-08-01", "2024-09-15"),
        ("Apr 2025 Spike", "2025-04-01", "2025-05-15"),
    ]
    rows = []
    closed = [t for t in trades if t.leveraged_return_pct is not None]
    for label, start, end in windows:
        if start is None:
            wins = [t for t in closed if (t.leveraged_return_pct or 0) > 0]
            losses = [t for t in closed if (t.leveraged_return_pct or 0) <= 0]
            regime_counts = pd.Series([_trade_regime(t) for t in closed]).value_counts()
            rows.append(
                {
                    "Window": label,
                    "Trades": len(closed),
                    "PnL ($)": round(sum((t.pnl or 0) for t in closed), 2),
                    "Start": "—",
                    "Peak": "—",
                    "End": "—",
                    "Win Rate": f"{(len(wins) / len(closed) * 100):.1f}%" if closed else "0.0%",
                    "Avg Win %": f"{(sum((t.leveraged_return_pct or 0) for t in wins) / len(wins)):.2f}%" if wins else "0.00%",
                    "Avg Loss %": f"{(sum((t.leveraged_return_pct or 0) for t in losses) / len(losses)):.2f}%" if losses else "0.00%",
                    "Spike Move %": "—",
                    "Capture Eff.": "—",
                    "Up PnL ($)": "—",
                    "Up Longs": "—",
                    "Up Shorts": "—",
                    "Post PnL ($)": "—",
                    "Post Longs": "—",
                    "Post Shorts": "—",
                    "Top Regime": regime_counts.index[0] if not regime_counts.empty else "—",
                }
            )
        else:
            row = _window_trade_summary(trades, start, end)
            structure = _spike_structure(prices, start, end)
            start_ts = pd.Timestamp(start)
            end_ts = pd.Timestamp(end) + pd.Timedelta(days=1)
            peak_ts = structure.pop("_peak_ts")
            subset = [
                t for t in trades
                if t.leveraged_return_pct is not None and start_ts <= pd.Timestamp(t.entry_time) < end_ts
            ]
            up_trades = [t for t in subset if pd.Timestamp(t.entry_time) <= peak_ts]
            post_trades = [t for t in subset if pd.Timestamp(t.entry_time) > peak_ts]
            _low, move_pct = _window_price_move(prices, start, end)
            row["Spike Move %"] = f"{move_pct:.1f}%" if move_pct is not None else "—"
            row["Capture Eff."] = _capture_efficiency(float(row["PnL ($)"]), move_pct)
            row["Start"] = structure["Start"]
            row["Peak"] = structure["Peak"]
            row["End"] = structure["End"]
            row["Up PnL ($)"] = round(sum((t.pnl or 0) for t in up_trades), 2)
            row["Up Longs"] = sum(1 for t in up_trades if getattr(t.direction, "value", str(t.direction)) == "Long")
            row["Up Shorts"] = sum(1 for t in up_trades if getattr(t.direction, "value", str(t.direction)) == "Short")
            row["Post PnL ($)"] = round(sum((t.pnl or 0) for t in post_trades), 2)
            row["Post Longs"] = sum(1 for t in post_trades if getattr(t.direction, "value", str(t.direction)) == "Long")
            row["Post Shorts"] = sum(1 for t in post_trades if getattr(t.direction, "value", str(t.direction)) == "Short")
            row["Window"] = label
            rows.append(row)
    return pd.DataFrame(rows)


def _price_chart(prices, trades, symbol, show_long, show_short, show_tp, show_sl, show_trail, show_sig):
    base = alt.Chart(prices).mark_line(color=_BLUE, strokeWidth=1.2).encode(
        x=alt.X("date:T", title="Date / Time", axis=alt.Axis(**_AXIS)),
        y=alt.Y("close:Q", title="Price", scale=alt.Scale(zero=False), axis=alt.Axis(**_AXIS)),
        tooltip=["date:T", alt.Tooltip("close:Q", format=".4f")],
    )
    layers = [base]
    if not trades:
        return (
            alt.layer(*layers)
            .properties(title=alt.TitleParams(f"{symbol} – Price", **_TITLE), height=320)
            .configure_view(strokeOpacity=0)
            .configure_axis(**_AXIS)
            .configure_title(**_TITLE)
        )
    entry_df, exit_df = _trade_events(trades)
    tt_e = [
        "date:T",
        "trade_n:N",
        "direction:N",
        alt.Tooltip("price:Q", format=".4f", title="Entry"),
        "outcome:N",
        alt.Tooltip("return_pct:Q", format=".2f", title="Return %"),
    ]
    if not entry_df.empty:
        long_e = entry_df[entry_df["direction"] == "Long"].copy()
        short_e = entry_df[entry_df["direction"] == "Short"].copy()
        if show_long and not long_e.empty:
            long_e["y"] = long_e["price"] * 0.997
            layers.append(
                alt.Chart(long_e)
                .mark_point(shape="triangle-up", size=120, filled=True, color=_GREEN)
                .encode(x="date:T", y="y:Q", tooltip=tt_e)
            )
        if show_short and not short_e.empty:
            short_e["y"] = short_e["price"] * 1.003
            layers.append(
                alt.Chart(short_e)
                .mark_point(shape="triangle-down", size=120, filled=True, color=_RED)
                .encode(x="date:T", y="y:Q", tooltip=tt_e)
            )
    if not exit_df.empty:
        tt_x = [
            "date:T",
            "trade_n:N",
            "direction:N",
            alt.Tooltip("price:Q", format=".4f", title="Exit"),
            "outcome:N",
            alt.Tooltip("return_pct:Q", format=".2f", title="Return %"),
        ]
        tp_ex = exit_df[exit_df["outcome"] == "TP hit"]
        sl_ex = exit_df[exit_df["outcome"] == "SL hit"]
        trail_ex = exit_df[exit_df["outcome"] == "Trail stop"]
        sig_ex = exit_df[exit_df["outcome"].apply(_is_signal_exit)]
        if show_tp and not tp_ex.empty:
            layers.append(
                alt.Chart(tp_ex)
                .mark_point(shape="cross", size=110, strokeWidth=2.5, color=_GREEN)
                .encode(x="date:T", y="price:Q", tooltip=tt_x)
            )
        if show_sl and not sl_ex.empty:
            layers.append(
                alt.Chart(sl_ex)
                .mark_point(shape="cross", size=110, strokeWidth=2.5, color=_RED)
                .encode(x="date:T", y="price:Q", tooltip=tt_x)
            )
        if show_trail and not trail_ex.empty:
            layers.append(
                alt.Chart(trail_ex)
                .mark_point(shape="diamond", size=100, filled=True, color=_PURPLE)
                .encode(x="date:T", y="price:Q", tooltip=tt_x)
            )
        if show_sig and not sig_ex.empty:
            layers.append(
                alt.Chart(sig_ex)
                .mark_point(shape="cross", size=110, strokeWidth=2.5, color=_ORANGE)
                .encode(x="date:T", y="price:Q", tooltip=tt_x)
            )
    return (
        alt.layer(*layers)
        .properties(title=alt.TitleParams(f"{symbol} – Price  ▲ Long  ▼ Short  ✕ Exit", **_TITLE), height=320)
        .configure_view(strokeOpacity=0)
        .configure_axis(**_AXIS)
        .configure_title(**_TITLE)
    )


def _rsi_chart(prices, trades, period, buy_levels, sell_levels, symbol, show_long, show_short, show_tp, show_sl, show_trail, show_sig):
    rsi_s = _calc_rsi(prices["close"], period).rename("rsi")
    df = pd.concat([prices[["date"]], rsi_s], axis=1).dropna()
    rsi_line = alt.Chart(df).mark_line(color=_GOLD, strokeWidth=1.8).encode(
        x=alt.X("date:T", title="Date / Time", axis=alt.Axis(**_AXIS)),
        y=alt.Y("rsi:Q", title="RSI", scale=alt.Scale(domain=[0, 100]), axis=alt.Axis(**_AXIS)),
        tooltip=["date:T", alt.Tooltip("rsi:Q", format=".2f")],
    )
    layers = [rsi_line]
    for lvl in buy_levels:
        ldf = pd.DataFrame({"y": [lvl], "label": [f"OS {lvl:.0f}"]})
        layers += [
            alt.Chart(ldf).mark_rule(color=_GREEN, strokeDash=[5, 3], strokeWidth=1.5).encode(y="y:Q"),
            alt.Chart(ldf).mark_text(align="left", dx=4, dy=-7, fontSize=12, color=_GREEN, fontWeight="bold").encode(y="y:Q", x=alt.value(4), text="label:N"),
        ]
    for lvl in sell_levels:
        ldf = pd.DataFrame({"y": [lvl], "label": [f"OB {lvl:.0f}"]})
        layers += [
            alt.Chart(ldf).mark_rule(color=_RED, strokeDash=[5, 3], strokeWidth=1.5).encode(y="y:Q"),
            alt.Chart(ldf).mark_text(align="left", dx=4, dy=-7, fontSize=12, color=_RED, fontWeight="bold").encode(y="y:Q", x=alt.value(4), text="label:N"),
        ]
    if buy_levels:
        layers.append(alt.Chart(pd.DataFrame({"y1": [0], "y2": [min(buy_levels)]})).mark_rect(color=_GREEN, opacity=0.07).encode(y="y1:Q", y2="y2:Q"))
    if sell_levels:
        layers.append(alt.Chart(pd.DataFrame({"y1": [max(sell_levels)], "y2": [100]})).mark_rect(color=_RED, opacity=0.07).encode(y="y1:Q", y2="y2:Q"))

    def _snap(ts):
        idx = min(df["date"].searchsorted(pd.Timestamp(ts)), len(df) - 1)
        row = df.iloc[idx]
        return (float(row["rsi"]) if not pd.isna(row["rsi"]) else 50.0), row["date"]

    if trades:
        entry_df, exit_df = _trade_events(trades)
        tt = ["date:T", "trade_n:N", "direction:N", alt.Tooltip("rsi_val:Q", format=".1f", title="RSI"), "outcome:N", alt.Tooltip("return_pct:Q", format=".2f", title="Return %")]
        if not entry_df.empty:
            entry_df[["rsi_val", "date"]] = pd.DataFrame([_snap(r) for r in entry_df["date"]], columns=["rsi_val", "date"])
            long_e = entry_df[entry_df["direction"] == "Long"].copy()
            short_e = entry_df[entry_df["direction"] == "Short"].copy()
            if show_long and not long_e.empty:
                long_e["y"] = long_e["rsi_val"] - 5
                layers.append(alt.Chart(long_e).mark_point(shape="triangle-up", size=90, filled=True, color=_GREEN).encode(x="date:T", y="y:Q", tooltip=tt))
            if show_short and not short_e.empty:
                short_e["y"] = short_e["rsi_val"] + 5
                layers.append(alt.Chart(short_e).mark_point(shape="triangle-down", size=90, filled=True, color=_RED).encode(x="date:T", y="y:Q", tooltip=tt))
        if not exit_df.empty:
            exit_df[["rsi_val", "date"]] = pd.DataFrame([_snap(r) for r in exit_df["date"]], columns=["rsi_val", "date"])
            tp_ex = exit_df[exit_df["outcome"] == "TP hit"]
            sl_ex = exit_df[exit_df["outcome"] == "SL hit"]
            trail_ex = exit_df[exit_df["outcome"] == "Trail stop"]
            sig_ex = exit_df[exit_df["outcome"].apply(_is_signal_exit)]
            if show_tp and not tp_ex.empty:
                layers.append(alt.Chart(tp_ex).mark_point(shape="cross", size=90, strokeWidth=2.5, color=_GREEN).encode(x="date:T", y="rsi_val:Q", tooltip=tt))
            if show_sl and not sl_ex.empty:
                layers.append(alt.Chart(sl_ex).mark_point(shape="cross", size=90, strokeWidth=2.5, color=_RED).encode(x="date:T", y="rsi_val:Q", tooltip=tt))
            if show_trail and not trail_ex.empty:
                layers.append(alt.Chart(trail_ex).mark_point(shape="diamond", size=80, filled=True, color=_PURPLE).encode(x="date:T", y="rsi_val:Q", tooltip=tt))
            if show_sig and not sig_ex.empty:
                layers.append(alt.Chart(sig_ex).mark_point(shape="cross", size=90, strokeWidth=2.5, color=_ORANGE).encode(x="date:T", y="rsi_val:Q", tooltip=tt))
    return (
        alt.layer(*layers)
        .properties(title=alt.TitleParams(f"{symbol} – RSI ({period})  Buy≤{buy_levels}  Sell≥{sell_levels}", **_TITLE), height=300)
        .configure_view(strokeOpacity=0)
        .configure_axis(**_AXIS)
        .configure_title(**_TITLE)
    )


def _equity_chart(equity_curve: pd.DataFrame, symbol: str):
    if equity_curve is None or equity_curve.empty:
        return alt.Chart(pd.DataFrame()).mark_line()
    eq_df = equity_curve.copy()
    eq_df["date"] = pd.to_datetime(eq_df["date"])
    line = alt.Chart(eq_df).mark_line(color=_BLUE, strokeWidth=2).encode(
        x=alt.X("date:T", title="Date / Time", axis=alt.Axis(**_AXIS)),
        y=alt.Y("equity:Q", title="Equity ($)", scale=alt.Scale(zero=False), axis=alt.Axis(**_AXIS)),
        tooltip=["date:T", alt.Tooltip("equity:Q", format="$,.2f", title="Portfolio Value")],
    )
    return alt.layer(line).properties(title=alt.TitleParams(f"{symbol} – Portfolio Equity", **_TITLE), height=300).configure_view(strokeOpacity=0).configure_axis(**_AXIS).configure_title(**_TITLE)


@st.cache_data(show_spinner=False)
def _cached_gld_fair_value(start: str | None, end: str | None, cache_fingerprint: str):
    diagnostics = compute_gld_fair_value_diagnostics(start=start, end=end)
    if diagnostics is None:
        return None
    return {
        "frame": diagnostics.frame,
        "stats": diagnostics.stats,
        "model": diagnostics.model,
        "symbol": diagnostics.symbol,
    }


def _fair_value_chart(frame: pd.DataFrame, symbol: str):
    plot_df = frame.copy()
    value_cols = ["actual", "fair_value"]
    if "structural_fair_value" in plot_df.columns and plot_df["structural_fair_value"].notna().any():
        value_cols.append("structural_fair_value")
    value_df = plot_df.melt(
        id_vars=["date"],
        value_vars=value_cols,
        var_name="series",
        value_name="price",
    )
    value_df["series"] = value_df["series"].map(
        {
            "actual": f"{symbol} actual",
            "fair_value": f"{symbol} fair value",
            "structural_fair_value": f"{symbol} structural fair value",
        }
    )
    return (
        alt.Chart(value_df)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("date:T", title="Date", axis=alt.Axis(**_AXIS)),
            y=alt.Y("price:Q", title="Price", scale=alt.Scale(zero=False), axis=alt.Axis(**_AXIS)),
            color=alt.Color(
                "series:N",
                scale=alt.Scale(
                    domain=[f"{symbol} actual", f"{symbol} fair value", f"{symbol} structural fair value"],
                    range=[_BLUE, _GOLD, _ORANGE],
                ),
                legend=alt.Legend(title=None, labelColor="#d0d4f0"),
            ),
            tooltip=["date:T", "series:N", alt.Tooltip("price:Q", format=".2f")],
        )
        .properties(title=alt.TitleParams(f"{symbol} – Actual vs Fair Value (optimized slow macro fit)", **_TITLE), height=320)
        .configure_view(strokeOpacity=0)
        .configure_axis(**_AXIS)
        .configure_title(**_TITLE)
    )


def _fair_gap_chart(frame: pd.DataFrame, symbol: str):
    plot_df = frame.copy()
    plot_df["gap_sign"] = plot_df["fair_gap_pct"].apply(lambda x: "Undervalued" if x > 0 else "Overvalued")
    bars = (
        alt.Chart(plot_df)
        .mark_bar(opacity=0.75)
        .encode(
            x=alt.X("date:T", title="Date", axis=alt.Axis(**_AXIS)),
            y=alt.Y("fair_gap_pct:Q", title="Fair Gap %", axis=alt.Axis(**_AXIS)),
            color=alt.Color(
                "gap_sign:N",
                scale=alt.Scale(domain=["Undervalued", "Overvalued"], range=[_GREEN, _RED]),
                legend=alt.Legend(title=None, labelColor="#d0d4f0"),
            ),
            tooltip=[
                "date:T",
                alt.Tooltip("fair_gap_pct:Q", format=".2f", title="Gap %"),
                alt.Tooltip("actual:Q", format=".2f", title=f"{symbol} actual"),
                alt.Tooltip("fair_value:Q", format=".2f", title="Fair value"),
            ],
        )
    )
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#cfd8dc", opacity=0.5).encode(y="y:Q")
    return (
        alt.layer(bars, zero)
        .properties(title=alt.TitleParams(f"{symbol} – Fair Value Gap", **_TITLE), height=180)
        .configure_view(strokeOpacity=0)
        .configure_axis(**_AXIS)
        .configure_title(**_TITLE)
    )


def render() -> None:
    render_mode_banner()
    st.title("⏪ Backtester")
    st.session_state.setdefault("bt_strategy", "Bollinger + RSI (Spike-Aware)")
    prices = render_data_source_selector()
    if prices is not None:
        st.session_state["bt_prices_live"] = prices
        st.session_state["bt_symbol_live"] = st.session_state.get("loaded_symbol", "DATA")
    prices = st.session_state.get("bt_prices_live")
    if prices is None:
        st.info("← Select a data source in the sidebar to begin.")
        if "bt_result" in st.session_state:
            st.divider()
            _show_results()
        return

    symbol = st.session_state.get("bt_symbol_live", "DATA")
    bar_label = _bar_label(prices)
    st.success(f"**{symbol}** — {len(prices):,} bars · bar size: **{bar_label}**")
    st.divider()

    strategies = list_strategies()
    strat_names = {s["name"]: s["id"] for s in strategies}
    strategy_names = list(strat_names.keys())
    default_strategy_name = "Bollinger + RSI (Spike-Aware)"
    if st.session_state.get("bt_strategy") not in strategy_names:
        st.session_state["bt_strategy"] = (
            default_strategy_name if default_strategy_name in strategy_names else strategy_names[0]
        )
    col_cfg, col_risk = st.columns(2)
    with col_cfg:
        selected_name = st.selectbox("Strategy", strategy_names, key="bt_strategy")
        selected_id = strat_names[selected_name]
        leverage = st.number_input("Leverage", 1.0, 100.0, 1.0, 0.5, key="bt_lev")
        capital_per_trade = st.number_input("Capital per trade ($)", 100.0, value=1000.0, key="bt_cap")
        starting_equity = st.number_input("Starting equity ($)", 1000.0, value=1000.0, key="bt_equity")
        direction_filter = st.selectbox("Direction filter", ["Both", "Long only", "Short only"], key="bt_dir")
    with col_risk:
        st.markdown("**Risk Controls**")
        use_risk = st.checkbox("Apply risk manager", value=True, key="bt_risk")
        max_loss = st.slider("Max loss per trade (% of capital)", 5, 100, 50, key="bt_maxloss")
        st.markdown("---")
        counter_signal_exit = st.checkbox(
            "Counter-signal exit",
            value=True,
            key="bt_counter",
            help="When ON: opposing RSI signal closes the current trade and opens reverse.",
        )
        st.markdown("---")
        st.markdown("**Transaction Costs**")
        st.caption("UVXY realistic costs: spread ~0.06%, slippage ~0.02%.  \nLeave at 0 for gross return.")
        spread_pct = st.number_input("Spread % (round-trip)", 0.0, 2.0, 0.06, step=0.01, format="%.2f", key="bt_spread")
        slippage_pct = st.number_input("Slippage % (round-trip)", 0.0, 2.0, 0.02, step=0.01, format="%.2f", key="bt_slip")
        commission = st.number_input("Commission per trade ($)", 0.0, 10.0, 0.0, step=0.10, format="%.2f", key="bt_comm")

    st.divider()
    params = render_strategy_params(selected_id, leverage=leverage, max_capital_loss_pct=float(max_loss))
    run_clicked = st.button("▶ Run Backtest", type="primary", key="bt_run")
    if "bt_result" in st.session_state:
        st.divider()
        _show_results()

    if run_clicked:
        cls = get_strategy(selected_id)
        strategy = cls(params=params)
        errors = strategy.validate_params()
        if errors:
            for e in errors:
                st.error(e)
            return
        from config.settings import RiskConfig
        from core.models import Direction as Dir

        risk_cfg = RiskConfig(
            max_capital_per_trade_pct=100.0,
            max_daily_loss_pct=100.0,
            max_open_positions=999,
            default_max_loss_pct_of_capital=float(max_loss),
        )
        rm = RiskManager(risk_cfg) if use_risk else None
        dir_filter = None
        if direction_filter == "Long only":
            dir_filter = Dir.LONG
        if direction_filter == "Short only":
            dir_filter = Dir.SHORT
        engine = BacktestEngine(
            strategy,
            risk_manager=rm,
            direction_filter=dir_filter,
            counter_signal_exit=counter_signal_exit,
            spread_pct=float(spread_pct),
            slippage_pct=float(slippage_pct),
            commission_per_trade=float(commission),
        )
        with st.spinner(f"Preparing context and running backtest on {len(prices):,} bars…"):
            prepared_prices = prepare_strategy_data(
                prices,
                strategy,
                primary_symbol=symbol,
                source=st.session_state.get("loaded_source"),
                interval=st.session_state.get("loaded_interval"),
                start=st.session_state.get("loaded_start"),
                end=st.session_state.get("loaded_end"),
            )
            result = engine.run(
                data=prepared_prices,
                symbol=symbol,
                leverage=leverage,
                capital_per_trade=capital_per_trade,
                starting_equity=starting_equity,
            )
        st.session_state["bt_result"] = result
        st.session_state["bt_symbol"] = symbol
        st.session_state["bt_bar_label"] = bar_label
        st.session_state["bt_selected_id"] = selected_id
        st.session_state["bt_params"] = dict(params)
        st.session_state["bt_starting_equity"] = float(starting_equity)
        st.session_state["bt_cost_settings"] = {
            "spread_pct": float(spread_pct),
            "slippage_pct": float(slippage_pct),
            "commission": float(commission),
        }
        try:
            from db.database import Database

            db = Database(settings.db_path)
            for t in result.trades:
                db.save_trade(t)
            st.session_state["bt_db_msg"] = f"✓ {len(result.trades)} trades saved."
        except Exception as e:
            st.session_state["bt_db_msg"] = f"DB save skipped: {e}"
        st.divider()
        _show_results()
        return


def _show_results() -> None:
    if "bt_result" not in st.session_state:
        return
    result = st.session_state["bt_result"]
    symbol_r = st.session_state.get("bt_symbol", "DATA")
    selected_id_r = st.session_state.get("bt_selected_id", "")
    params_r = st.session_state.get("bt_params", {})
    costs_r = st.session_state.get("bt_cost_settings", {})
    prices_r = st.session_state.get("bt_prices_live")
    closed = [t for t in result.trades if t.leveraged_return_pct is not None]

    st.subheader("📊 Results")
    s = result.summary()
    render_metrics_row(
        {
            "Total Trades": s["Total Trades"],
            "Win Rate": s["Win Rate"],
            "Total Return": s["Total Return"],
            "Max Drawdown": s["Max Drawdown"],
            "Sharpe Ratio": s["Sharpe Ratio"],
            "Avg Win": s["Avg Win"],
            "Avg Loss": s["Avg Loss"],
        }
    )
    if closed:
        from collections import Counter

        outcome_counts = Counter(t.outcome.value if hasattr(t.outcome, "value") else str(t.outcome) for t in closed)
        st.markdown("**Exit breakdown:**")
        cols = st.columns(min(len(outcome_counts), 5))
        for col, (label, cnt) in zip(cols, sorted(outcome_counts.items())):
            col.metric(label, cnt)
        gross_pnl = sum((t.capital_allocated or 0) * ((t.leveraged_return_pct or 0) / 100.0) for t in closed)
        net_pnl = sum((t.pnl or 0) for t in closed)
        deducted_cost = gross_pnl - net_pnl
        spread_used = float(costs_r.get("spread_pct", 0.0) or 0.0)
        slippage_used = float(costs_r.get("slippage_pct", 0.0) or 0.0)
        commission_used = float(costs_r.get("commission", 0.0) or 0.0)
        st.caption(
            f"Applied costs: spread {spread_used:.2f}% + slippage {slippage_used:.2f}% "
            f"+ commission {commission_used:.2f} dollars/trade = approx. "
            f"{deducted_cost:,.2f} dollars deducted from gross closed-trade PnL."
        )
    st.caption("📖 **PnL ($)** = dollar profit/loss · **Return %** = leveraged return on capital · **TP hit** = price target reached · **SL hit** = stop hit · **Trail stop** = trailing stop exit · **RSI exits** = RSI threshold crossed")
    if closed:
        st.subheader("🧪 Spike Comparison Report")
        st.caption("This table compares the full backtest with the two key UVXY spike windows, shows the defined start/peak/end dates for each spike, counts long/short trades before and after the peak, and reports how much of the raw spike move the strategy captured.")
        st.dataframe(_comparison_report(prices_r, result.trades), use_container_width=True, hide_index=True)
    st.divider()

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    show_long = c1.checkbox("▲ Long entries", value=True, key="show_long")
    show_short = c2.checkbox("▼ Short entries", value=True, key="show_short")
    show_tp = c3.checkbox("✕ TP exits", value=True, key="show_tp_x")
    show_sl = c4.checkbox("✕ SL exits", value=True, key="show_sl_x")
    show_trail = c5.checkbox("◆ Trail exits", value=True, key="show_trail_x")
    show_sig = c6.checkbox("✕ Signal exits", value=True, key="show_sig_x")

    if prices_r is not None:
        prices_plot = _downsample(prices_r)
        n_bars = len(prices_r)
        label_extra = f"  ·  *{len(prices_plot):,} of {n_bars:,} bars shown*" if len(prices_plot) < n_bars else ""
        st.markdown(f"#### 📈 Price{label_extra}")
        st.altair_chart(_price_chart(prices_plot, result.trades, symbol_r, show_long, show_short, show_tp, show_sl, show_trail, show_sig), use_container_width=True)
        if selected_id_r in ("rsi_threshold", "atr_rsi", "vwap_rsi", "bollinger_rsi", "ema_trend_rsi"):
            period = int(params_r.get("rsi_period", 9))
            buy_levels = _parse_levels(params_r.get("buy_levels", "30"))
            sell_levels = _parse_levels(params_r.get("sell_levels", "70"))
            st.markdown(f"#### 📉 RSI ({period})")
            st.altair_chart(_rsi_chart(prices_plot, result.trades, period, buy_levels, sell_levels, symbol_r, show_long, show_short, show_tp, show_sl, show_trail, show_sig), use_container_width=True)
    else:
        st.info("ℹ️ Price chart not available — reload data to see charts.")

    if closed:
        st.markdown("#### 💰 Equity Curve")
        st.altair_chart(_equity_chart(result.equity_curve, symbol_r), use_container_width=True)

    if closed:
        trades_df = pd.DataFrame(
            [
                {
                    "symbol": t.symbol,
                    "regime": _trade_regime(t),
                    "direction": t.direction.value,
                    "capital_allocated": t.capital_allocated,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "outcome": t.outcome.value if t.outcome else None,
                    "leveraged_return_pct": t.leveraged_return_pct,
                    "pnl": t.pnl,
                    "entry_time": t.entry_time,
                    "exit_time": t.exit_time,
                    "notes": t.notes,
                }
                for t in closed
            ]
        )
        st.markdown("#### 📊 Per-Trade Return")
        st.altair_chart(pnl_distribution(trades_df), use_container_width=True)
        with st.expander("📋 Trade Log", expanded=False):
            st.dataframe(
                trades_df.rename(
                    columns={
                        "capital_allocated": "capital_allocated ($)",
                        "leveraged_return_pct": "return_pct (%)",
                        "pnl": "PnL ($)",
                    }
                ).sort_values("entry_time", ascending=False),
                use_container_width=True,
            )

    if "bt_db_msg" in st.session_state:
        st.caption(st.session_state["bt_db_msg"])

    if symbol_r.upper() == "GLD":
        st.markdown("#### 🪙 Macro Fair Value")
        st.caption(
            "This is a slow diagnostic model for GLD only. It fits an optimized monthly fair-value proxy from cached macro and peer series, "
            "so we can judge the macro layer by fit quality before using it for trading bias."
        )
        trigger_key = "bt_gld_fair_value_visible"
        if st.button("Build / Refresh Fair Value Diagnostics", key="bt_gld_fair_value_btn"):
            st.session_state[trigger_key] = True
        if st.session_state.get(trigger_key):
            with st.spinner("Fitting GLD fair-value curve from cached slow macro data…"):
                fair_payload = _cached_gld_fair_value(
                    str(st.session_state.get("loaded_start")) if st.session_state.get("loaded_start") is not None else None,
                    str(st.session_state.get("loaded_end")) if st.session_state.get("loaded_end") is not None else None,
                    fair_value_cache_fingerprint(),
                )
            if fair_payload:
                fair_stats = fair_payload["stats"]
                model = fair_payload["model"]
                render_metrics_row(
                    {
                        "Correlation": f"{fair_stats['corr']:.3f}",
                        "R²": f"{fair_stats['r2']:.3f}",
                        "MAE Gap": f"{fair_stats['mae_pct']:.2f}%",
                        "RMSE Gap": f"{fair_stats['rmse_pct']:.2f}%",
                        "Direction Hit": f"{fair_stats['directional_hit'] * 100:.1f}%",
                    }
                )
                if model.get("model_type") == "two_layer":
                    st.caption(
                        "Monthly slow fair-value proxy optimized as a structural layer plus a market-adjustment layer. "
                        f"Best fit: structural set `{model['structural_set']}`, market set `{model['market_set']}`, "
                        f"z-window `{model['z_window']}` months, structural fit window `{model['structural_fit_window']}` months, "
                        f"market fit window `{model['market_fit_window']}` months, ridge α `{model['ridge_alpha']:.2f}`, "
                        f"smoothing span `{model['smooth_span']}` months."
                    )
                else:
                    st.caption(
                        "Monthly slow fair-value proxy optimized as a blended macro-plus-market fit. "
                        f"Best fit: feature set `{model['feature_set']}`, z-window `{model['z_window']}` months, "
                        f"fit window `{model['fit_window']}` months, ridge α `{model['ridge_alpha']:.2f}`, "
                        f"smoothing span `{model['smooth_span']}` months."
                    )
                optional_sources = model.get("optional_sources") or {}
                if optional_sources:
                    pretty_sources = ", ".join(f"`{k}` from `{v}`" for k, v in sorted(optional_sources.items()))
                    st.caption(f"Optional proxy inputs currently available: {pretty_sources}.")
                else:
                    st.caption(
                        "Optional official ETF / central-bank proxy files are not loaded yet. "
                        "Current fit is using only the cached macro and market proxies."
                    )
                st.altair_chart(_fair_value_chart(fair_payload["frame"], symbol_r), use_container_width=True)
                st.altair_chart(_fair_gap_chart(fair_payload["frame"], symbol_r), use_container_width=True)
            else:
                st.warning("Could not build GLD fair-value diagnostics from the cached slow datasets.")

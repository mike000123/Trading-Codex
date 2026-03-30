"""
ui/charts.py  —  pure Altair, zero Plotly imports.
"""
from __future__ import annotations

from typing import Optional

import altair as alt
import pandas as pd

_GREEN = "#26a69a"
_RED   = "#ef5350"
_BLUE  = "#4a9eff"
_GOLD  = "#ffd54f"
_GREY  = "#9e9eb8"

_AXIS_CFG = dict(gridColor="#1e2130", labelColor="#c9d8f5", titleColor="#c9d8f5")


def price_chart(
    data: pd.DataFrame,
    entry_date=None,
    exit_date=None,
    take_profit: Optional[float] = None,
    stop_loss: Optional[float] = None,
    title: str = "Price",
) -> alt.LayerChart:
    """Altair close-line chart with TP/SL rules and entry/exit markers."""
    base = (
        alt.Chart(data)
        .mark_line(color=_BLUE)
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("close:Q", title="Close", scale=alt.Scale(zero=False)),
            tooltip=["date:T", "close:Q"],
        )
    )
    layers: list = [base]

    if take_profit is not None:
        tp_df = pd.DataFrame({"y": [take_profit], "label": [f"TP {take_profit:.4f}"]})
        layers.append(alt.Chart(tp_df).mark_rule(color=_GREEN, strokeDash=[4, 4]).encode(y="y:Q"))
        layers.append(
            alt.Chart(tp_df).mark_text(color=_GREEN, align="left", dx=4, dy=-6, fontSize=11)
            .encode(y="y:Q", x=alt.value(4), text="label:N")
        )

    if stop_loss is not None:
        sl_df = pd.DataFrame({"y": [stop_loss], "label": [f"SL {stop_loss:.4f}"]})
        layers.append(alt.Chart(sl_df).mark_rule(color=_RED, strokeDash=[4, 4]).encode(y="y:Q"))
        layers.append(
            alt.Chart(sl_df).mark_text(color=_RED, align="left", dx=4, dy=-6, fontSize=11)
            .encode(y="y:Q", x=alt.value(4), text="label:N")
        )

    rows = []
    if entry_date is not None:
        rows.append({"date": pd.Timestamp(entry_date), "label": "Entry"})
    if exit_date is not None:
        rows.append({"date": pd.Timestamp(exit_date), "label": "Exit"})

    if rows:
        m_df = pd.DataFrame(rows)
        cs   = alt.Scale(domain=["Entry", "Exit"], range=[_GREEN, _RED])
        layers.append(
            alt.Chart(m_df).mark_rule(strokeWidth=2)
            .encode(x="date:T", color=alt.Color("label:N", scale=cs, legend=None))
        )
        layers.append(
            alt.Chart(m_df).mark_text(dy=-10, fontWeight="bold", fontSize=12)
            .encode(x="date:T", y=alt.value(20), text="label:N",
                    color=alt.Color("label:N", scale=cs, legend=None))
        )

    return (
        alt.layer(*layers)
        .properties(title=title, height=320)
        .configure_view(strokeOpacity=0)
        .configure_axis(**_AXIS_CFG)
        .configure_title(color="#c9d8f5")
    )


def equity_curve_chart(equity_df: pd.DataFrame, title: str = "Equity Curve") -> alt.Chart:
    if equity_df.empty:
        return alt.Chart(pd.DataFrame()).mark_line()

    return (
        alt.Chart(equity_df)
        .mark_area(
            line={"color": _BLUE},
            color=alt.Gradient(
                gradient="linear",
                stops=[
                    alt.GradientStop(color="rgba(74,158,255,0.3)", offset=0),
                    alt.GradientStop(color="rgba(74,158,255,0.0)", offset=1),
                ],
                x1=1, x2=1, y1=1, y2=0,
            ),
        )
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("equity:Q", title="Equity ($)", scale=alt.Scale(zero=False)),
            tooltip=["date:T", alt.Tooltip("equity:Q", format="$,.2f")],
        )
        .properties(title=title, height=280)
        .configure_view(strokeOpacity=0)
        .configure_axis(**_AXIS_CFG)
        .configure_title(color="#c9d8f5")
    )


def rsi_chart(data: pd.DataFrame, period: int = 14) -> alt.LayerChart:
    delta    = data["close"].diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs       = avg_gain / avg_loss.replace(0, float("nan"))
    rsi_vals = (100 - (100 / (1 + rs))).rename("rsi")
    df       = pd.concat([data[["date"]], rsi_vals], axis=1).dropna()

    rsi_line = (
        alt.Chart(df).mark_line(color=_GOLD)
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("rsi:Q", title="RSI", scale=alt.Scale(domain=[0, 100])),
            tooltip=["date:T", "rsi:Q"],
        )
    )
    ob_df = pd.DataFrame({"y": [70], "label": ["Overbought 70"]})
    os_df = pd.DataFrame({"y": [30], "label": ["Oversold 30"]})
    ob_rule = alt.Chart(ob_df).mark_rule(color=_RED,   strokeDash=[4, 4]).encode(y="y:Q")
    os_rule = alt.Chart(os_df).mark_rule(color=_GREEN, strokeDash=[4, 4]).encode(y="y:Q")

    return (
        alt.layer(rsi_line, ob_rule, os_rule)
        .properties(title=f"RSI ({period})", height=180)
        .configure_view(strokeOpacity=0)
        .configure_axis(**_AXIS_CFG)
        .configure_title(color="#c9d8f5")
    )


def pnl_distribution(trades_df: pd.DataFrame) -> alt.LayerChart:
    """
    Per-trade waterfall bar chart — one bar per trade, ordered by entry time.
    Green = win, Red = loss. Avoids histogram bin-collapse when all returns are similar.
    """
    if trades_df.empty or "leveraged_return_pct" not in trades_df.columns:
        return alt.Chart(pd.DataFrame({"x": [], "y": []})).mark_bar()

    df = trades_df.dropna(subset=["leveraged_return_pct"]).copy()
    if df.empty:
        return alt.Chart(pd.DataFrame()).mark_bar()

    # Sort by entry time if available
    if "entry_time" in df.columns:
        df = df.sort_values("entry_time").reset_index(drop=True)
    df["trade_n"]  = [f"T{i+1}" for i in range(len(df))]
    df["result"]   = df["leveraged_return_pct"].apply(lambda v: "Win" if v >= 0 else "Loss")
    df["ret_fmt"]  = df["leveraged_return_pct"].round(3)

    # Zero reference line
    zero_df = pd.DataFrame({"y": [0]})
    zero_rule = alt.Chart(zero_df).mark_rule(color=_GREY, strokeDash=[3, 3]).encode(y="y:Q")

    bars = (
        alt.Chart(df)
        .mark_bar(width={"band": 0.7})
        .encode(
            x=alt.X("trade_n:N", sort=None, title="Trade #",
                    axis=alt.Axis(labelAngle=0, labelColor="#c9d8f5", titleColor="#c9d8f5")),
            y=alt.Y("leveraged_return_pct:Q", title="Leveraged Return %",
                    axis=alt.Axis(labelColor="#c9d8f5", titleColor="#c9d8f5")),
            color=alt.Color(
                "result:N",
                scale=alt.Scale(domain=["Win", "Loss"], range=[_GREEN, _RED]),
                legend=alt.Legend(title="Result", labelColor="#c9d8f5", titleColor="#c9d8f5"),
            ),
            tooltip=[
                alt.Tooltip("trade_n:N",               title="Trade"),
                alt.Tooltip("ret_fmt:Q",                title="Return %",    format=".3f"),
                alt.Tooltip("entry_price:Q",            title="Entry price", format=".4f"),
                alt.Tooltip("exit_price:Q",             title="Exit price",  format=".4f"),
                alt.Tooltip("outcome:N",                title="Outcome"),
                alt.Tooltip("direction:N",              title="Direction"),
            ] if "entry_price" in df.columns else [
                alt.Tooltip("trade_n:N",   title="Trade"),
                alt.Tooltip("ret_fmt:Q",   title="Return %", format=".3f"),
            ],
        )
    )

    text = (
        alt.Chart(df)
        .mark_text(dy=-8, fontSize=9, color="#c9d8f5")
        .encode(
            x=alt.X("trade_n:N", sort=None),
            y=alt.Y("leveraged_return_pct:Q"),
            text=alt.Text("ret_fmt:Q", format=".1f"),
        )
    )

    return (
        alt.layer(zero_rule, bars, text)
        .properties(title="Per-Trade Leveraged Return (%)", height=260)
        .configure_view(strokeOpacity=0)
        .configure_axis(gridColor="#1e2130")
        .configure_title(color="#c9d8f5")
    )


def portfolio_allocation_pie(positions: list[dict]) -> alt.Chart:
    if not positions:
        return alt.Chart(pd.DataFrame()).mark_arc()

    df = pd.DataFrame({
        "symbol": [p["symbol"] for p in positions],
        "value":  [abs(p.get("capital_allocated", 1)) for p in positions],
    })

    return (
        alt.Chart(df)
        .mark_arc(innerRadius=50)
        .encode(
            theta=alt.Theta("value:Q"),
            color=alt.Color("symbol:N", legend=alt.Legend(labelColor="#c9d8f5")),
            tooltip=["symbol:N", "value:Q"],
        )
        .properties(title="Position Allocation", height=240)
        .configure_title(color="#c9d8f5")
    )

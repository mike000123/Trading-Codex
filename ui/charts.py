"""
ui/charts.py  —  Pure Altair. Zero Plotly.
Design fixes: larger fonts, brighter axis text, bigger chart heights.
"""
from __future__ import annotations

from typing import Optional

import altair as alt
import pandas as pd

_GREEN = "#2faa6a"
_RED   = "#c64242"
_BLUE  = "#d4af37"          # primary line colour, repurposed as gold under MRMI Gold
_GOLD  = "#e8c566"
_GREY  = "#a89c80"
_BG    = "#0c0d14"          # chart-wide background (matches app body)
_VIEW  = "#181a25"          # plot-area fill (matches card background)
# RSI threshold zones — muted bronze for OB, muted sage for OS. Avoid pure
# red/green so the bands sit comfortably alongside the gold line.
_OB_ZONE = "#b9871e"
_OS_ZONE = "#5a8a64"
_TITLE_FONT = dict(font="Cinzel, serif")

# Shared axis config — gold-on-charcoal to match the MRMI Gold theme. The
# variables are still named the way other modules expect them so no caller
# changes are required.
_AXIS = dict(
    gridColor="rgba(212,175,55,0.18)",
    labelColor="#a89c80",
    titleColor="#e8c566",
    labelFontSize=12,
    titleFontSize=13,
)
_TITLE_CFG = dict(color="#e8c566", fontSize=14, fontWeight="bold")


def _base_layout(title: str, height: int) -> dict:
    return dict(title=alt.TitleParams(title, **_TITLE_CFG), height=height)


def price_chart(
    data: pd.DataFrame,
    entry_date=None,
    exit_date=None,
    take_profit: Optional[float] = None,
    stop_loss: Optional[float] = None,
    title: str = "Price",
) -> alt.LayerChart:
    # Area-fill gradient below the line — subtle gold-to-transparent.
    area = (
        alt.Chart(data).mark_area(
            line={"color": _BLUE, "strokeWidth": 1.5},
            color=alt.Gradient(
                gradient="linear",
                stops=[
                    alt.GradientStop(color="rgba(212,175,55,0.45)", offset=0),
                    alt.GradientStop(color="rgba(212,175,55,0.00)", offset=1),
                ],
                x1=1, x2=1, y1=0, y2=1,
            ),
        )
        .encode(
            x=alt.X("date:T", title="Date", axis=alt.Axis(**_AXIS)),
            y=alt.Y("close:Q", title="Close", scale=alt.Scale(zero=False),
                    axis=alt.Axis(**_AXIS)),
            tooltip=["date:T", alt.Tooltip("close:Q", format=".4f")],
        )
    )
    layers: list = [area]

    if take_profit is not None:
        tp_df = pd.DataFrame({"y": [take_profit], "label": [f"TP {take_profit:.4f}"]})
        layers.append(alt.Chart(tp_df).mark_rule(color=_GREEN, strokeDash=[4,4], strokeWidth=1.5).encode(y="y:Q"))
        layers.append(alt.Chart(tp_df).mark_text(color=_GREEN, align="left", dx=4, dy=-7, fontSize=12)
                      .encode(y="y:Q", x=alt.value(4), text="label:N"))

    if stop_loss is not None:
        sl_df = pd.DataFrame({"y": [stop_loss], "label": [f"SL {stop_loss:.4f}"]})
        layers.append(alt.Chart(sl_df).mark_rule(color=_RED, strokeDash=[4,4], strokeWidth=1.5).encode(y="y:Q"))
        layers.append(alt.Chart(sl_df).mark_text(color=_RED, align="left", dx=4, dy=-7, fontSize=12)
                      .encode(y="y:Q", x=alt.value(4), text="label:N"))

    rows = []
    if entry_date is not None:
        rows.append({"date": pd.Timestamp(entry_date), "label": "Entry"})
    if exit_date is not None:
        rows.append({"date": pd.Timestamp(exit_date), "label": "Exit"})
    if rows:
        m_df = pd.DataFrame(rows)
        cs   = alt.Scale(domain=["Entry","Exit"], range=[_GREEN, _RED])
        layers.append(alt.Chart(m_df).mark_rule(strokeWidth=2)
                      .encode(x="date:T", color=alt.Color("label:N", scale=cs, legend=None)))
        layers.append(alt.Chart(m_df).mark_text(dy=-10, fontWeight="bold", fontSize=13)
                      .encode(x="date:T", y=alt.value(18), text="label:N",
                              color=alt.Color("label:N", scale=cs, legend=None)))

    return (alt.layer(*layers)
            .properties(**_base_layout(title, 360))
            .configure(background=_BG)
            .configure_view(fill=_VIEW, strokeOpacity=0)
            .configure_axis(**_AXIS)
            .configure_title(**_TITLE_CFG))


def equity_curve_chart(equity_df: pd.DataFrame, title: str = "Equity Curve") -> alt.Chart:
    if equity_df.empty:
        return alt.Chart(pd.DataFrame()).mark_line()
    return (
        alt.Chart(equity_df)
        .mark_area(line={"color": _BLUE, "strokeWidth": 2},
                   color=alt.Gradient(gradient="linear",
                       stops=[alt.GradientStop(color="rgba(212,175,55,0.45)", offset=0),
                              alt.GradientStop(color="rgba(212,175,55,0.00)", offset=1)],
                       x1=1, x2=1, y1=1, y2=0))
        .encode(
            x=alt.X("date:T", title="Date", axis=alt.Axis(**_AXIS)),
            y=alt.Y("equity:Q", title="Equity ($)", scale=alt.Scale(zero=False),
                    axis=alt.Axis(**_AXIS)),
            tooltip=["date:T", alt.Tooltip("equity:Q", format="$,.2f")],
        )
        .properties(**_base_layout(title, 300))
        .configure(background=_BG)
        .configure_view(fill=_VIEW, strokeOpacity=0)
        .configure_axis(**_AXIS)
        .configure_title(**_TITLE_CFG)
    )


def rsi_chart(
    data: pd.DataFrame,
    period: int = 14,
    buy_levels: Optional[list[float]] = None,
    sell_levels: Optional[list[float]] = None,
) -> alt.LayerChart:
    """
    RSI chart with configurable threshold lines.
    buy_levels  – list of oversold  thresholds (green dashes)
    sell_levels – list of overbought thresholds (red dashes)
    Falls back to [30] / [70] if not supplied.
    """
    buy_levels  = buy_levels  or [30]
    sell_levels = sell_levels or [70]

    delta    = data["close"].diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs       = avg_gain / avg_loss.replace(0, float("nan"))
    rsi_vals = (100 - (100 / (1 + rs))).rename("rsi")
    df       = pd.concat([data[["date"]], rsi_vals], axis=1).dropna()

    rsi_line = (
        alt.Chart(df).mark_line(color=_GOLD, strokeWidth=1.8)
        .encode(
            x=alt.X("date:T", title="Date / Time", axis=alt.Axis(**_AXIS)),
            y=alt.Y("rsi:Q", title="RSI", scale=alt.Scale(domain=[0, 100]),
                    axis=alt.Axis(**_AXIS)),
            tooltip=["date:T", alt.Tooltip("rsi:Q", format=".2f")],
        )
    )
    layers = [rsi_line]

    # Threshold zones drawn FIRST so they sit behind everything else.
    if buy_levels:
        os_min = min(buy_levels)
        layers.insert(0, alt.Chart(pd.DataFrame({"y1":[0],"y2":[os_min]}))
                      .mark_rect(color=_OS_ZONE, opacity=0.18).encode(y="y1:Q", y2="y2:Q"))
    if sell_levels:
        ob_max = max(sell_levels)
        layers.insert(0, alt.Chart(pd.DataFrame({"y1":[ob_max],"y2":[100]}))
                      .mark_rect(color=_OB_ZONE, opacity=0.18).encode(y="y1:Q", y2="y2:Q"))

    # Threshold reference lines + labels in subtle bronze/sage so the gold
    # RSI line stays the focal point.
    for lvl in buy_levels:
        lvl_df = pd.DataFrame({"y": [lvl], "label": [f"OS {lvl:.0f}"]})
        layers.append(alt.Chart(lvl_df).mark_rule(color=_OS_ZONE, strokeDash=[5,3], strokeWidth=1.2, opacity=0.7).encode(y="y:Q"))
        layers.append(alt.Chart(lvl_df).mark_text(align="left", dx=4, dy=-7, fontSize=11, color=_OS_ZONE,
                                                   fontWeight="bold")
                      .encode(y="y:Q", x=alt.value(4), text="label:N"))

    for lvl in sell_levels:
        lvl_df = pd.DataFrame({"y": [lvl], "label": [f"OB {lvl:.0f}"]})
        layers.append(alt.Chart(lvl_df).mark_rule(color=_OB_ZONE, strokeDash=[5,3], strokeWidth=1.2, opacity=0.7).encode(y="y:Q"))
        layers.append(alt.Chart(lvl_df).mark_text(align="left", dx=4, dy=-7, fontSize=11, color=_OB_ZONE,
                                                   fontWeight="bold")
                      .encode(y="y:Q", x=alt.value(4), text="label:N"))

    return (alt.layer(*layers)
            .properties(**_base_layout(f"RSI ({period})", 260))
            .configure(background=_BG)
            .configure_view(fill=_VIEW, strokeOpacity=0)
            .configure_axis(**_AXIS)
            .configure_title(**_TITLE_CFG))


def pnl_distribution(trades_df: pd.DataFrame) -> alt.LayerChart:
    """Per-trade waterfall bar chart — one bar per trade."""
    if trades_df.empty or "leveraged_return_pct" not in trades_df.columns:
        return alt.Chart(pd.DataFrame()).mark_bar()

    df = trades_df.dropna(subset=["leveraged_return_pct"]).copy()
    if df.empty:
        return alt.Chart(pd.DataFrame()).mark_bar()

    if "entry_time" in df.columns:
        df = df.sort_values("entry_time").reset_index(drop=True)
    df["trade_n"] = [f"T{i+1}" for i in range(len(df))]
    df["result"]  = df["leveraged_return_pct"].apply(lambda v: "Win" if v >= 0 else "Loss")
    df["ret_fmt"] = df["leveraged_return_pct"].round(3)

    zero_rule = (alt.Chart(pd.DataFrame({"y": [0]}))
                 .mark_rule(color=_GREY, strokeDash=[3,3], strokeWidth=1.2).encode(y="y:Q"))

    tt = [
        alt.Tooltip("trade_n:N",               title="Trade"),
        alt.Tooltip("ret_fmt:Q",                title="Return %",   format=".3f"),
        alt.Tooltip("outcome:N",                title="Outcome"),
        alt.Tooltip("direction:N",              title="Direction"),
    ]
    if "entry_time" in df.columns:
        df["entry_time"] = pd.to_datetime(df["entry_time"], errors="coerce")
        tt.append(alt.Tooltip("entry_time:T", title="Entry time"))
    if "exit_time" in df.columns:
        df["exit_time"] = pd.to_datetime(df["exit_time"], errors="coerce")
        tt.append(alt.Tooltip("exit_time:T", title="Exit time"))
    if "entry_price" in df.columns:
        tt += [
            alt.Tooltip("entry_price:Q", title="Entry", format=".4f"),
            alt.Tooltip("exit_price:Q",  title="Exit",  format=".4f"),
        ]

    bars = (
        alt.Chart(df).mark_bar(width={"band": 0.75})
        .encode(
            x=alt.X("trade_n:N", sort=None, title="Trade #",
                    axis=alt.Axis(labelAngle=0, labelFontSize=12, titleFontSize=13,
                                  labelColor="#d0d4f0", titleColor="#d0d4f0")),
            y=alt.Y("leveraged_return_pct:Q", title="Leveraged Return %",
                    axis=alt.Axis(labelFontSize=12, titleFontSize=13,
                                  labelColor="#d0d4f0", titleColor="#d0d4f0")),
            color=alt.Color("result:N",
                scale=alt.Scale(domain=["Win","Loss"], range=[_GREEN, _RED]),
                legend=alt.Legend(title="Result", labelColor="#d0d4f0", titleColor="#d0d4f0",
                                  labelFontSize=12)),
            tooltip=tt,
        )
    )

    text = (
        alt.Chart(df).mark_text(dy=-9, fontSize=11, color="#e8eaf6", fontWeight="bold")
        .encode(x=alt.X("trade_n:N", sort=None), y="leveraged_return_pct:Q",
                text=alt.Text("ret_fmt:Q", format=".1f"))
    )

    return (alt.layer(zero_rule, bars, text)
            .properties(**_base_layout("Per-Trade Leveraged Return (%)", 300))
            .configure(background=_BG)
            .configure_view(fill=_VIEW, strokeOpacity=0)
            .configure_axis(gridColor="rgba(212,175,55,0.18)")
            .configure_title(**_TITLE_CFG))


def portfolio_allocation_pie(positions: list[dict]) -> alt.Chart:
    if not positions:
        return alt.Chart(pd.DataFrame()).mark_arc()
    df = pd.DataFrame({
        "symbol": [p["symbol"] for p in positions],
        "value":  [abs(p.get("capital_allocated", 1)) for p in positions],
    })
    _GOLD_PALETTE = ["#d4af37", "#e8c566", "#b9871e", "#a89c80", "#5a8a64", "#c64242"]
    return (
        alt.Chart(df).mark_arc(innerRadius=58, stroke="#0c0d14", strokeWidth=2)
        .encode(
            theta=alt.Theta("value:Q"),
            color=alt.Color("symbol:N",
                scale=alt.Scale(range=_GOLD_PALETTE),
                legend=alt.Legend(labelColor="#f0e8d0", titleColor="#e8c566", labelFontSize=12)),
            tooltip=["symbol:N", "value:Q"],
        )
        .properties(**_base_layout("Position Allocation", 260))
        .configure(background=_BG)
        .configure_view(fill=_VIEW, strokeOpacity=0)
        .configure_title(**_TITLE_CFG)
    )

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
CHART_WINDOW_OPTIONS = ["All", "1D", "5D", "30D", "1Y", "4Y"]
_CHART_WINDOW_LABELS = {
    "All": "All",
    "1D": "1D",
    "5D": "5D",
    "30D": "30D",
    "1Y": "1Y",
    "4Y": "4Y",
}


def _palette() -> dict:
    """Return the active theme's chart palette.

    Reads st.session_state["theme_selector"] so the palette flips at the
    moment the user changes themes, without requiring chart-function
    callers to know about the theme system. Returns the gold palette as
    the default for any non-Silver theme (Dark / Midnight Blue still get
    a gold accent on charts — most of the data primitives are theme-
    agnostic enough that this looks fine; if a per-theme accent is wanted
    later, branch on the theme name here).
    """
    theme = "MRMI Gold"
    try:
        import streamlit as _st
        theme = _st.session_state.get("theme_selector") or "MRMI Gold"
    except Exception:
        pass
    if theme == "MRMI Silver":
        return {
            "primary":     "#c0c0c8",   # _BLUE / line colour
            "primary_glow": "rgba(192,192,200,0.45)",
            "secondary":   "#cfcfd6",   # _GOLD / accent text on charts
            "axis_grid":   "rgba(192,192,200,0.18)",
            "axis_label":  "#9aa5b3",
            "axis_title":  "#cfcfd6",
            "title":       "#cfcfd6",
            "ob_zone":     "#7a7a86",   # subdued silver for overbought
            "os_zone":     "#5a8a64",   # sage green works on both themes
        }
    return {
        "primary":     "#d4af37",
        "primary_glow": "rgba(212,175,55,0.45)",
        "secondary":   "#e8c566",
        "axis_grid":   "rgba(212,175,55,0.18)",
        "axis_label":  "#a89c80",
        "axis_title":  "#e8c566",
        "title":       "#e8c566",
        "ob_zone":     "#b9871e",
        "os_zone":     "#5a8a64",
    }

def theme_chart_color(key: str = "primary") -> str:
    return _palette().get(key, "#d4af37")


def theme_axis_cfg() -> dict:
    palette = _palette()
    return dict(
        gridColor=palette["axis_grid"],
        labelColor=palette["axis_label"],
        titleColor=palette["axis_title"],
        labelFontSize=12,
        titleFontSize=13,
    )


def theme_title_cfg() -> dict:
    palette = _palette()
    return dict(color=palette["title"], fontSize=14, fontWeight="bold")


def _base_layout(title: str, height: int) -> dict:
    return dict(title=alt.TitleParams(title, **theme_title_cfg()), height=height)


def chart_window_label(option: str) -> str:
    return _CHART_WINDOW_LABELS.get(str(option), str(option))


def _chart_window_start(end_ts: pd.Timestamp, window: str) -> pd.Timestamp | None:
    w = str(window or "All").upper()
    if w == "ALL":
        return None
    if w == "1D":
        return end_ts - pd.Timedelta(days=1)
    if w == "5D":
        return end_ts - pd.Timedelta(days=5)
    if w == "30D":
        return end_ts - pd.Timedelta(days=30)
    if w == "1Y":
        return end_ts - pd.DateOffset(years=1)
    if w == "4Y":
        return end_ts - pd.DateOffset(years=4)
    return None


def filter_chart_window(data: pd.DataFrame, window: str, date_col: str = "date") -> pd.DataFrame:
    if data is None or data.empty or date_col not in data.columns:
        return data
    dates = pd.to_datetime(data[date_col], errors="coerce")
    valid = dates.dropna()
    if valid.empty:
        return data.copy()
    start_ts = _chart_window_start(valid.max(), window)
    if start_ts is None:
        return data.copy()
    mask = dates >= start_ts
    clipped = data.loc[mask.fillna(False)].copy()
    return clipped if not clipped.empty else data.tail(1).copy()


def clip_frame_to_price_window(
    frame: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    frame_date_col: str = "date",
    prices_date_col: str = "date",
) -> pd.DataFrame:
    if frame is None or frame.empty or prices is None or prices.empty:
        return frame
    if frame_date_col not in frame.columns or prices_date_col not in prices.columns:
        return frame.copy()
    price_dates = pd.to_datetime(prices[prices_date_col], errors="coerce").dropna()
    if price_dates.empty:
        return frame.copy()
    start_ts = price_dates.min()
    end_ts = price_dates.max()
    frame_dates = pd.to_datetime(frame[frame_date_col], errors="coerce")
    mask = frame_dates.between(start_ts, end_ts, inclusive="both")
    return frame.loc[mask.fillna(False)].copy()


def price_chart(
    data: pd.DataFrame,
    entry_date=None,
    exit_date=None,
    take_profit: Optional[float] = None,
    stop_loss: Optional[float] = None,
    title: str = "Price",
) -> alt.LayerChart:
    # Area-fill gradient below the line — uses active-theme palette.
    _p = _palette()
    _glow0 = _p["primary_glow"]
    _glow1 = _glow0.replace("0.45", "0.00")
    area = (
        alt.Chart(data).mark_area(
            line={"color": _p["primary"], "strokeWidth": 1.5},
            color=alt.Gradient(
                gradient="linear",
                stops=[
                    alt.GradientStop(color=_glow0, offset=0),
                    alt.GradientStop(color=_glow1, offset=1),
                ],
                x1=1, x2=1, y1=0, y2=1,
            ),
        )
        .encode(
            x=alt.X("date:T", title="Date", axis=alt.Axis(**theme_axis_cfg())),
            y=alt.Y("close:Q", title="Close", scale=alt.Scale(zero=False),
                    axis=alt.Axis(**theme_axis_cfg())),
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
            .configure_axis(**theme_axis_cfg())
            .configure_title(**theme_title_cfg()))


def equity_curve_chart(equity_df: pd.DataFrame, title: str = "Equity Curve") -> alt.Chart:
    if equity_df.empty:
        return alt.Chart(pd.DataFrame()).mark_line()
    return (
        alt.Chart(equity_df)
        .mark_area(line={"color": _palette()["primary"], "strokeWidth": 2},
                   color=alt.Gradient(gradient="linear",
                       stops=[alt.GradientStop(color=_palette()["primary_glow"], offset=0),
                              alt.GradientStop(color=_palette()["primary_glow"].replace("0.45","0.00"), offset=1)],
                       x1=1, x2=1, y1=1, y2=0))
        .encode(
            x=alt.X("date:T", title="Date", axis=alt.Axis(**theme_axis_cfg())),
            y=alt.Y("equity:Q", title="Equity ($)", scale=alt.Scale(zero=False),
                    axis=alt.Axis(**theme_axis_cfg())),
            tooltip=["date:T", alt.Tooltip("equity:Q", format="$,.2f")],
        )
        .properties(**_base_layout(title, 300))
        .configure(background=_BG)
        .configure_view(fill=_VIEW, strokeOpacity=0)
        .configure_axis(**theme_axis_cfg())
        .configure_title(**theme_title_cfg())
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

    _p = _palette()
    rsi_line = (
        alt.Chart(df).mark_line(color=_p["primary"], strokeWidth=1.8)
        .encode(
            x=alt.X("date:T", title="Date / Time", axis=alt.Axis(**theme_axis_cfg())),
            y=alt.Y("rsi:Q", title="RSI", scale=alt.Scale(domain=[0, 100]),
                    axis=alt.Axis(**theme_axis_cfg())),
            tooltip=["date:T", alt.Tooltip("rsi:Q", format=".2f")],
        )
    )
    layers = [rsi_line]

    # Threshold zones drawn FIRST so they sit behind everything else.
    if buy_levels:
        os_min = min(buy_levels)
        layers.insert(0, alt.Chart(pd.DataFrame({"y1":[0],"y2":[os_min]}))
                      .mark_rect(color=_p["os_zone"], opacity=0.18).encode(y="y1:Q", y2="y2:Q"))
    if sell_levels:
        ob_max = max(sell_levels)
        layers.insert(0, alt.Chart(pd.DataFrame({"y1":[ob_max],"y2":[100]}))
                      .mark_rect(color=_p["ob_zone"], opacity=0.18).encode(y="y1:Q", y2="y2:Q"))

    # Threshold reference lines + labels in subtle bronze/sage so the gold
    # RSI line stays the focal point.
    for lvl in buy_levels:
        lvl_df = pd.DataFrame({"y": [lvl], "label": [f"OS {lvl:.0f}"]})
        layers.append(alt.Chart(lvl_df).mark_rule(color=_p["os_zone"], strokeDash=[5,3], strokeWidth=1.2, opacity=0.7).encode(y="y:Q"))
        layers.append(alt.Chart(lvl_df).mark_text(align="left", dx=4, dy=-7, fontSize=11, color=_p["os_zone"],
                                                   fontWeight="bold")
                      .encode(y="y:Q", x=alt.value(4), text="label:N"))

    for lvl in sell_levels:
        lvl_df = pd.DataFrame({"y": [lvl], "label": [f"OB {lvl:.0f}"]})
        layers.append(alt.Chart(lvl_df).mark_rule(color=_p["ob_zone"], strokeDash=[5,3], strokeWidth=1.2, opacity=0.7).encode(y="y:Q"))
        layers.append(alt.Chart(lvl_df).mark_text(align="left", dx=4, dy=-7, fontSize=11, color=_p["ob_zone"],
                                                   fontWeight="bold")
                      .encode(y="y:Q", x=alt.value(4), text="label:N"))

    return (alt.layer(*layers)
            .properties(**_base_layout(f"RSI ({period})", 260))
            .configure(background=_BG)
            .configure_view(fill=_VIEW, strokeOpacity=0)
            .configure_axis(**theme_axis_cfg())
            .configure_title(**theme_title_cfg()))


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

    palette = _palette()
    bars = (
        alt.Chart(df).mark_bar(width={"band": 0.75})
        .encode(
            x=alt.X("trade_n:N", sort=None, title="Trade #",
                    axis=alt.Axis(labelAngle=0, labelFontSize=12, titleFontSize=13,
                                  labelColor=palette["axis_label"], titleColor=palette["axis_title"])),
            y=alt.Y("leveraged_return_pct:Q", title="Leveraged Return %",
                    axis=alt.Axis(labelFontSize=12, titleFontSize=13,
                                  labelColor=palette["axis_label"], titleColor=palette["axis_title"])),
            color=alt.Color("result:N",
                scale=alt.Scale(domain=["Win","Loss"], range=[_GREEN, _RED]),
                legend=alt.Legend(title="Result", labelColor=palette["axis_title"], titleColor=palette["title"],
                                  labelFontSize=12)),
            tooltip=tt,
        )
    )

    text = (
        alt.Chart(df).mark_text(dy=-9, fontSize=11, color=theme_chart_color("title"), fontWeight="bold")
        .encode(x=alt.X("trade_n:N", sort=None), y="leveraged_return_pct:Q",
                text=alt.Text("ret_fmt:Q", format=".1f"))
    )

    palette = _palette()
    return (alt.layer(zero_rule, bars, text)
            .properties(**_base_layout("Per-Trade Leveraged Return (%)", 300))
            .configure(background=_BG)
            .configure_view(fill=_VIEW, strokeOpacity=0)
            .configure_axis(gridColor=palette["axis_grid"], labelColor=palette["axis_label"], titleColor=palette["axis_title"])
            .configure_title(**theme_title_cfg()))


def portfolio_allocation_pie(positions: list[dict]) -> alt.Chart:
    if not positions:
        return alt.Chart(pd.DataFrame()).mark_arc()
    df = pd.DataFrame({
        "symbol": [p["symbol"] for p in positions],
        "value":  [abs(p.get("capital_allocated", 1)) for p in positions],
    })
    palette = _palette()
    themed_palette = [
        palette["primary"],
        palette["secondary"],
        palette["ob_zone"],
        palette["axis_label"],
        "#5a8a64",
        "#c64242",
    ]
    return (
        alt.Chart(df).mark_arc(innerRadius=58, stroke="#0c0d14", strokeWidth=2)
        .encode(
            theta=alt.Theta("value:Q"),
            color=alt.Color("symbol:N",
                scale=alt.Scale(range=themed_palette),
                legend=alt.Legend(labelColor=palette["axis_title"], titleColor=palette["title"], labelFontSize=12)),
            tooltip=["symbol:N", "value:Q"],
        )
        .properties(**_base_layout("Position Allocation", 260))
        .configure(background=_BG)
        .configure_view(fill=_VIEW, strokeOpacity=0)
        .configure_title(**theme_title_cfg())
    )

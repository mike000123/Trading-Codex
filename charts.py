"""
ui/charts.py
────────────
Reusable Plotly chart builders.

IMPORTANT: This file intentionally avoids add_vline(), add_hline() with
annotation_text, and any Plotly helper that internally calls
shapeannotation.axis_spanning_shape_annotation(). That function is broken
in current Plotly when the x-axis contains pandas Timestamps — it tries
_mean([x0, x1]) which fails because Timestamps don't support __radd__.

Safe replacements used throughout:
  vertical line   → add_shape(type="line") + add_annotation()
  horizontal line → add_shape(type="line") + add_annotation()
"""
from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.graph_objects as go

_GREEN = "#26a69a"
_RED   = "#ef5350"
_BLUE  = "#4a9eff"
_GOLD  = "#ffd54f"
_GREY  = "#9e9eb8"

_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#c9d8f5"),
    margin=dict(l=10, r=10, t=40, b=10),
    xaxis=dict(showgrid=True, gridcolor="#1e2130"),
    yaxis=dict(showgrid=True, gridcolor="#1e2130"),
)


def _hline(fig: go.Figure, y: float, color: str, label: str) -> None:
    """Horizontal line + right-side label. No add_hline()."""
    fig.add_shape(
        type="line", xref="paper", yref="y",
        x0=0, x1=1, y0=y, y1=y,
        line=dict(color=color, width=1, dash="dot"),
        layer="below",
    )
    fig.add_annotation(
        text=label, xref="paper", x=1.01,
        yref="y", y=y,
        showarrow=False,
        font=dict(color=color, size=10),
        xanchor="left",
    )


def _vline(fig: go.Figure, ts, color: str, label: str, anchor: str = "right") -> None:
    """Vertical line + top label. No add_vline()."""
    x_str = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
    fig.add_shape(
        type="line", xref="x", yref="paper",
        x0=x_str, x1=x_str, y0=0, y1=1,
        line=dict(color=color, width=1.5, dash="dash"),
    )
    fig.add_annotation(
        text=label, xref="x", x=x_str,
        yref="paper", y=0.97,
        showarrow=False,
        font=dict(color=color, size=11),
        bgcolor="rgba(14,17,23,0.6)",
        borderpad=3,
        xanchor=anchor,
    )


def price_chart(
    data: pd.DataFrame,
    entry_date=None,
    exit_date=None,
    take_profit: Optional[float] = None,
    stop_loss: Optional[float] = None,
    title: str = "Price",
) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=data["date"],
        open=data["open"],
        high=data["high"],
        low=data["low"],
        close=data["close"],
        name="Price",
        increasing_line_color=_GREEN,
        decreasing_line_color=_RED,
    ))

    if take_profit is not None:
        _hline(fig, take_profit, _GREEN, f"TP {take_profit:.4f}")
    if stop_loss is not None:
        _hline(fig, stop_loss, _RED, f"SL {stop_loss:.4f}")
    if entry_date is not None:
        _vline(fig, entry_date, _GREEN, "Entry", anchor="right")
    if exit_date is not None:
        _vline(fig, exit_date, _RED, "Exit", anchor="left")

    fig.update_layout(
        title=title,
        xaxis_rangeslider_visible=False,
        height=400,
        **_LAYOUT,
    )
    return fig


def equity_curve_chart(equity_df: pd.DataFrame, title: str = "Equity Curve") -> go.Figure:
    if equity_df.empty:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=equity_df["date"],
        y=equity_df["equity"],
        mode="lines",
        name="Equity",
        line=dict(color=_BLUE, width=2),
        fill="tozeroy",
        fillcolor="rgba(74,158,255,0.08)",
    ))
    fig.update_layout(title=title, height=300, **_LAYOUT)
    return fig


def rsi_chart(data: pd.DataFrame, period: int = 14) -> go.Figure:
    delta    = data["close"].diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs       = avg_gain / avg_loss.replace(0, float("nan"))
    rsi      = 100 - (100 / (1 + rs))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data["date"], y=rsi,
        mode="lines", name="RSI",
        line=dict(color=_GOLD, width=1.5),
    ))

    # Overbought / oversold bands — shapes only, no add_hline
    for y, color in [(70, _RED), (30, _GREEN)]:
        _hline(fig, y, color, str(y))

    fig.add_shape(type="rect", xref="paper", yref="y",
                  x0=0, x1=1, y0=70, y1=100,
                  fillcolor=_RED, opacity=0.04, line_width=0, layer="below")
    fig.add_shape(type="rect", xref="paper", yref="y",
                  x0=0, x1=1, y0=0, y1=30,
                  fillcolor=_GREEN, opacity=0.04, line_width=0, layer="below")

    layout = dict(**_LAYOUT)
    layout["yaxis"] = dict(range=[0, 100], showgrid=True, gridcolor="#1e2130")
    fig.update_layout(title=f"RSI ({period})", height=200, **layout)
    return fig


def pnl_distribution(trades_df: pd.DataFrame) -> go.Figure:
    if trades_df.empty or "leveraged_return_pct" not in trades_df.columns:
        return go.Figure()

    pnl    = trades_df["leveraged_return_pct"].dropna()
    colors = [_GREEN if v >= 0 else _RED for v in pnl]

    fig = go.Figure(go.Bar(x=pnl, marker_color=colors, name="P&L %"))

    # Zero line — x-axis is numeric here so safe to use a plain shape
    fig.add_shape(
        type="line", xref="x", yref="paper",
        x0=0, x1=0, y0=0, y1=1,
        line=dict(color=_GREY, width=1, dash="dash"),
    )

    fig.update_layout(
        title="Trade P&L Distribution (%)",
        xaxis_title="Return %",
        yaxis_title="Count",
        height=280,
        **_LAYOUT,
    )
    return fig


def portfolio_allocation_pie(positions: list[dict]) -> go.Figure:
    if not positions:
        return go.Figure()

    labels = [p["symbol"] for p in positions]
    values = [abs(p.get("capital_allocated", 1)) for p in positions]

    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.4,
        marker=dict(colors=[_BLUE, _GREEN, _GOLD, _GREY, _RED]),
    ))
    fig.update_layout(
        title="Position Allocation",
        height=280,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c9d8f5"),
    )
    return fig

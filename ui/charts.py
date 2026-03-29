"""
ui/charts.py
────────────
Reusable Plotly chart builders.
All functions return a plotly Figure – call st.plotly_chart(fig, use_container_width=True).
"""
from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ─── Colour palette (matches theme CSS vars where possible) ─────────────────
_GREEN = "#26a69a"
_RED   = "#ef5350"
_BLUE  = "#4a9eff"
_GOLD  = "#ffd54f"
_GREY  = "#9e9eb8"


def price_chart(
    data: pd.DataFrame,
    entry_date: Optional[pd.Timestamp] = None,
    exit_date: Optional[pd.Timestamp] = None,
    take_profit: Optional[float] = None,
    stop_loss: Optional[float] = None,
    title: str = "Price",
) -> go.Figure:
    """Candlestick chart with optional entry/exit markers and TP/SL lines."""
    fig = go.Figure()

    # Candlesticks
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

    # TP / SL horizontal lines
    if take_profit is not None:
        fig.add_hline(y=take_profit, line_dash="dot", line_color=_GREEN,
                      annotation_text=f"TP {take_profit:.4f}",
                      annotation_position="right")
    if stop_loss is not None:
        fig.add_hline(y=stop_loss, line_dash="dot", line_color=_RED,
                      annotation_text=f"SL {stop_loss:.4f}",
                      annotation_position="right")

    # Entry / Exit vertical lines
    # Plotly's add_vline requires a numeric (ms since epoch) or str, not a pandas Timestamp
    def _to_vline_x(ts) -> str:
        """Convert any timestamp-like value to an ISO string Plotly accepts."""
        if ts is None:
            return None
        if hasattr(ts, "isoformat"):
            return ts.isoformat()
        return str(ts)

    if entry_date is not None:
        fig.add_vline(x=_to_vline_x(entry_date), line_dash="dash", line_color=_GREEN,
                      annotation_text="Entry", annotation_position="top left")
    if exit_date is not None:
        fig.add_vline(x=_to_vline_x(exit_date), line_dash="dash", line_color=_RED,
                      annotation_text="Exit", annotation_position="top right")

    fig.update_layout(
        title=title,
        xaxis_rangeslider_visible=False,
        height=400,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c9d8f5"),
        xaxis=dict(showgrid=True, gridcolor="#1e2130"),
        yaxis=dict(showgrid=True, gridcolor="#1e2130"),
    )
    return fig


def equity_curve_chart(equity_df: pd.DataFrame, title: str = "Equity Curve") -> go.Figure:
    """Line chart of portfolio equity over time."""
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

    fig.update_layout(
        title=title,
        height=300,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c9d8f5"),
        xaxis=dict(showgrid=True, gridcolor="#1e2130"),
        yaxis=dict(showgrid=True, gridcolor="#1e2130"),
    )
    return fig


def rsi_chart(data: pd.DataFrame, period: int = 14) -> go.Figure:
    """RSI sub-chart with overbought/oversold bands."""
    delta = data["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    rsi = 100 - (100 / (1 + rs))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data["date"], y=rsi,
        mode="lines", name="RSI",
        line=dict(color=_GOLD, width=1.5),
    ))
    fig.add_hline(y=70, line_dash="dot", line_color=_RED, annotation_text="70")
    fig.add_hline(y=30, line_dash="dot", line_color=_GREEN, annotation_text="30")
    fig.add_hrect(y0=70, y1=100, fillcolor=_RED, opacity=0.04, line_width=0)
    fig.add_hrect(y0=0, y1=30, fillcolor=_GREEN, opacity=0.04, line_width=0)

    fig.update_layout(
        title=f"RSI ({period})",
        height=200,
        yaxis=dict(range=[0, 100], showgrid=True, gridcolor="#1e2130"),
        xaxis=dict(showgrid=True, gridcolor="#1e2130"),
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c9d8f5"),
    )
    return fig


def pnl_distribution(trades_df: pd.DataFrame) -> go.Figure:
    """Histogram of trade P&L percentages."""
    if trades_df.empty or "leveraged_return_pct" not in trades_df.columns:
        return go.Figure()

    pnl = trades_df["leveraged_return_pct"].dropna()
    colors = [_GREEN if v >= 0 else _RED for v in pnl]

    fig = go.Figure(go.Bar(x=pnl, marker_color=colors, name="P&L %"))
    fig.add_vline(x=0, line_color=_GREY, line_dash="dash")
    fig.update_layout(
        title="Trade P&L Distribution (%)",
        xaxis_title="Return %",
        yaxis_title="Count",
        height=280,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c9d8f5"),
        xaxis=dict(showgrid=True, gridcolor="#1e2130"),
        yaxis=dict(showgrid=True, gridcolor="#1e2130"),
    )
    return fig


def portfolio_allocation_pie(positions: list[dict]) -> go.Figure:
    """Pie chart of open positions by symbol."""
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

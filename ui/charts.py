"""
ui/charts.py
────────────
Reusable Plotly chart builders.
All functions return a plotly Figure – call st.plotly_chart(fig, use_container_width=True).

NOTE: We deliberately avoid add_vline() entirely. In recent Plotly versions the
annotation_position helper (shapeannotation.py) tries to compute _mean(x0, x1)
which fails when x values are pandas Timestamps. We use add_shape() + add_annotation()
instead, which bypasses that code path completely.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.graph_objects as go


# ─── Colour palette ──────────────────────────────────────────────────────────
_GREEN = "#26a69a"
_RED   = "#ef5350"
_BLUE  = "#4a9eff"
_GOLD  = "#ffd54f"
_GREY  = "#9e9eb8"


def _draw_vline(fig: go.Figure, ts, color: str, label: str, anchor: str = "right") -> None:
    """
    Draw a vertical dashed line + label without using add_vline().
    ts can be a pandas Timestamp, datetime, or string – all are normalised to ISO str.
    """
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
    """Candlestick chart with optional entry/exit markers and TP/SL lines."""
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

    # TP / SL horizontal lines (add_hline is safe – it only operates on y)
    if take_profit is not None:
        fig.add_hline(y=take_profit, line_dash="dot", line_color=_GREEN)
        fig.add_annotation(
            text=f"TP {take_profit:.4f}", xref="paper", x=1.01,
            y=take_profit, yref="y", showarrow=False,
            font=dict(color=_GREEN, size=11), xanchor="left",
        )
    if stop_loss is not None:
        fig.add_hline(y=stop_loss, line_dash="dot", line_color=_RED)
        fig.add_annotation(
            text=f"SL {stop_loss:.4f}", xref="paper", x=1.01,
            y=stop_loss, yref="y", showarrow=False,
            font=dict(color=_RED, size=11), xanchor="left",
        )

    # Entry / Exit vertical lines — safe path, no add_vline
    if entry_date is not None:
        _draw_vline(fig, entry_date, _GREEN, "Entry", anchor="right")
    if exit_date is not None:
        _draw_vline(fig, exit_date, _RED, "Exit", anchor="left")

    fig.update_layout(
        title=title,
        xaxis_rangeslider_visible=False,
        height=400,
        margin=dict(l=10, r=10, t=40, b=30),
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

    # Zero line — safe: x-axis here is numeric (return %), not datetime
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

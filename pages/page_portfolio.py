"""
pages/page_portfolio.py
────────────────────────
Portfolio overview:
  - Aggregate P&L across paper / backtest / live
  - Equity curves
  - Trade history with filters
  - Per-strategy breakdown
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from config.settings import settings
from db.database import Database
from ui.components import render_mode_banner, render_metrics_row
from ui.charts import equity_curve_chart, pnl_distribution, portfolio_allocation_pie


def _db() -> Database:
    return Database(settings.db_path)


def render() -> None:
    render_mode_banner()
    st.title("💼 Portfolio Overview")

    db = _db()

    # ── Mode tabs ────────────────────────────────────────────────────────────
    tab_paper, tab_backtest, tab_live, tab_all = st.tabs(
        ["📝 Paper", "⏪ Backtest", "🔴 Live", "🌐 All Modes"]
    )

    def _render_mode_tab(mode: str | None) -> None:
        trades = db.get_trades(mode=mode)
        if not trades:
            st.info(f"No trades recorded{f' in {mode} mode' if mode else ''}.")
            return

        df = pd.DataFrame(trades)

        # ── Summary metrics ──────────────────────────────────────────────────
        closed = df[df["outcome"].notna() & (df["outcome"] != "Open") & df["leveraged_return_pct"].notna()]
        total_pnl = closed["pnl"].sum() if "pnl" in closed.columns else 0.0
        wins  = closed[closed["leveraged_return_pct"] > 0]
        losses= closed[closed["leveraged_return_pct"] <= 0]
        win_rate = len(wins) / len(closed) * 100 if len(closed) else 0
        avg_win  = wins["leveraged_return_pct"].mean() if len(wins) else 0
        avg_loss = losses["leveraged_return_pct"].mean() if len(losses) else 0

        render_metrics_row({
            "Total Trades":   len(closed),
            "Open Positions": len(df[df["outcome"] == "Open"]),
            "Win Rate":       f"{win_rate:.1f}%",
            "Total P&L":      f"${total_pnl:,.2f}",
            "Avg Win":        f"{avg_win:.2f}%",
            "Avg Loss":       f"{avg_loss:.2f}%",
        })

        # ── P&L over time ────────────────────────────────────────────────────
        if not closed.empty and "exit_time" in closed.columns:
            eq = closed[["exit_time", "pnl"]].dropna().copy()
            eq["exit_time"] = pd.to_datetime(eq["exit_time"])
            eq = eq.sort_values("exit_time")
            eq["equity"] = 10_000 + eq["pnl"].cumsum()
            eq_chart_df = eq.rename(columns={"exit_time": "date"})
            st.altair_chart(
                equity_curve_chart(eq_chart_df, f"Cumulative P&L{' – ' + mode if mode else ''}"),
            )

        # ── P&L distribution ─────────────────────────────────────────────────
        if not closed.empty:
            st.altair_chart(pnl_distribution(closed), use_container_width=True)

        # ── Open positions allocation ─────────────────────────────────────────
        open_pos = df[df["outcome"] == "Open"].to_dict("records")
        if open_pos:
            col1, col2 = st.columns([0.4, 0.6])
            with col1:
                st.altair_chart(portfolio_allocation_pie(open_pos), use_container_width=True)
            with col2:
                st.subheader("Open Positions")
                st.dataframe(
                    pd.DataFrame(open_pos)[["symbol", "direction", "entry_price", "capital_allocated", "strategy_id"]],
                    use_container_width=True,
                )

        # ── Per-strategy breakdown ────────────────────────────────────────────
        if "strategy_id" in closed.columns and not closed.empty:
            with st.expander("📊 Per-Strategy Breakdown", expanded=False):
                breakdown = (
                    closed.groupby("strategy_id")
                    .agg(
                        trades=("id", "count"),
                        wins=("leveraged_return_pct", lambda x: (x > 0).sum()),
                        avg_return=("leveraged_return_pct", "mean"),
                        total_pnl=("pnl", "sum"),
                    )
                    .reset_index()
                )
                breakdown["win_rate"] = (breakdown["wins"] / breakdown["trades"] * 100).round(1)
                st.dataframe(breakdown, use_container_width=True)

        # ── Full trade log ────────────────────────────────────────────────────
        with st.expander("📋 Full Trade Log", expanded=False):
            cols_show = [c for c in [
                "symbol", "direction", "entry_price", "exit_price", "outcome",
                "leveraged_return_pct", "pnl", "capital_allocated", "leverage",
                "strategy_id", "mode", "entry_time", "exit_time", "notes"
            ] if c in df.columns]
            filter_sym = st.text_input("Filter by symbol", key=f"filter_sym_{mode}")
            filtered = df[df["symbol"].str.contains(filter_sym, case=False)] if filter_sym else df
            st.dataframe(filtered[cols_show].sort_values("entry_time", ascending=False), use_container_width=True)

            if st.button("📥 Export CSV", key=f"export_{mode}"):
                csv = filtered[cols_show].to_csv(index=False)
                st.download_button("Download", csv, f"trades_{mode or 'all'}.csv", "text/csv")

    with tab_paper:
        _render_mode_tab("paper")
    with tab_backtest:
        _render_mode_tab("backtest")
    with tab_live:
        _render_mode_tab("live")
    with tab_all:
        _render_mode_tab(None)

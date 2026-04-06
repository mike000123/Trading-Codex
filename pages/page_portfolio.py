"""
pages/page_portfolio.py
────────────────────────
Portfolio overview — paper, backtest, forward_test, live.
Includes unified ticker position table with navigation to source page.
"""
from __future__ import annotations

from collections import defaultdict

import altair as alt
import pandas as pd
import streamlit as st

from config.settings import settings
from db.database import Database
from ui.components import render_mode_banner, render_metrics_row
from ui.charts import equity_curve_chart, pnl_distribution, portfolio_allocation_pie

_GREEN = "#26a69a"
_RED = "#ef5350"
_BLUE = "#4a9eff"
_GREY = "#9e9eb8"
_AXIS = dict(gridColor="#2a2d3e", labelColor="#d0d4f0", titleColor="#d0d4f0",
             labelFontSize=12, titleFontSize=13)
_TITLE = dict(color="#e8eaf6", fontSize=14, fontWeight="bold")

MODE_NAV = {
    "forward_test": "🔭 Forward Test",
    "paper": "📝 Paper Trading",
    "live": "🔴 Live Trading",
}


def _db() -> Database:
    return Database(settings.db_path)


def _regime_from_notes(notes: str) -> str:
    raw = notes or ""
    if "regime=" not in raw:
        return "unknown"
    return raw.split("regime=")[1].split(" | ")[0]


def _build_ticker_table(all_trades: list[dict]) -> pd.DataFrame:
    agg: dict[tuple, dict] = defaultdict(lambda: {
        "open": 0, "closed": 0, "total_pnl": 0.0,
        "wins": 0, "mode": "", "strategy": set(),
    })

    for t in all_trades:
        sym = t.get("symbol", "?")
        mode = t.get("mode", "?")
        key = (sym, mode)
        agg[key]["mode"] = mode
        if t.get("outcome") == "Open":
            agg[key]["open"] += 1
        elif t.get("pnl") is not None:
            agg[key]["closed"] += 1
            agg[key]["total_pnl"] += t["pnl"]
            if (t.get("pnl") or 0) > 0:
                agg[key]["wins"] += 1
        if t.get("strategy_id"):
            agg[key]["strategy"].add(t["strategy_id"])

    rows = []
    for (sym, mode), data in sorted(agg.items()):
        closed = data["closed"]
        wr = f"{data['wins']/closed*100:.0f}%" if closed else "—"
        rows.append({
            "Symbol": sym,
            "Mode": mode,
            "Open": data["open"],
            "Closed Trades": closed,
            "Win Rate": wr,
            "Total P&L": round(data["total_pnl"], 2),
            "Strategies": ", ".join(sorted(data["strategy"])),
            "_nav_target": MODE_NAV.get(mode, ""),
        })
    return pd.DataFrame(rows)


def _render_mode_tab(mode: str | None, db: Database) -> None:
    trades = db.get_trades(mode=mode)
    if not trades:
        st.info(f"No trades recorded{f' in {mode} mode' if mode else ''}.")
        return

    df = pd.DataFrame(trades)
    if "notes" in df.columns:
        df["regime"] = df["notes"].fillna("").apply(_regime_from_notes)
    closed = df[df["outcome"].notna() & ~df["outcome"].isin(["Open"]) & df["leveraged_return_pct"].notna()]

    total_pnl = closed["pnl"].sum() if "pnl" in closed.columns else 0.0
    wins = closed[closed["leveraged_return_pct"] > 0]
    losses = closed[closed["leveraged_return_pct"] <= 0]
    win_rate = len(wins) / len(closed) * 100 if len(closed) else 0

    render_metrics_row({
        "Total Trades": len(closed),
        "Open Positions": len(df[df["outcome"] == "Open"]),
        "Win Rate": f"{win_rate:.1f}%",
        "Total P&L": f"${total_pnl:,.2f}",
        "Avg Win": f"{wins['leveraged_return_pct'].mean():.2f}%" if len(wins) else "—",
        "Avg Loss": f"{losses['leveraged_return_pct'].mean():.2f}%" if len(losses) else "—",
    })

    if not closed.empty and "exit_time" in closed.columns:
        eq = closed[["exit_time", "pnl"]].dropna().copy()
        eq["exit_time"] = pd.to_datetime(eq["exit_time"])
        eq = eq.sort_values("exit_time")
        eq["equity"] = 10_000 + eq["pnl"].cumsum()
        st.altair_chart(
            equity_curve_chart(eq.rename(columns={"exit_time": "date"}), f"Cumulative P&L – {mode or 'All'}"),
            use_container_width=True,
        )

    if not closed.empty:
        st.altair_chart(pnl_distribution(closed), use_container_width=True)

    open_pos = df[df["outcome"] == "Open"].to_dict("records")
    if open_pos:
        col1, col2 = st.columns([0.4, 0.6])
        with col1:
            st.altair_chart(portfolio_allocation_pie(open_pos), use_container_width=True)
        with col2:
            st.subheader("Open Positions")
            open_df = pd.DataFrame(open_pos)
            show = [c for c in ["symbol", "direction", "entry_price", "capital_allocated",
                                "strategy_id", "mode"] if c in open_df.columns]
            st.dataframe(open_df[show], use_container_width=True)

    if "strategy_id" in closed.columns and not closed.empty:
        with st.expander("📊 Per-Strategy Breakdown", expanded=False):
            bd = (
                closed.groupby("strategy_id")
                .agg(
                    trades=("id", "count"),
                    wins=("leveraged_return_pct", lambda x: (x > 0).sum()),
                    avg_return=("leveraged_return_pct", "mean"),
                    total_pnl=("pnl", "sum"),
                )
                .reset_index()
            )
            bd["win_rate"] = (bd["wins"] / bd["trades"] * 100).round(1)
            st.dataframe(bd, use_container_width=True)

    with st.expander("📋 Trade Log", expanded=False):
        cols = [c for c in [
            "symbol", "regime", "direction", "entry_price", "exit_price", "outcome",
            "leveraged_return_pct", "pnl", "capital_allocated", "leverage",
            "strategy_id", "mode", "entry_time", "exit_time", "notes"
        ] if c in df.columns]
        filt_sym = st.text_input("Filter by symbol", key=f"filt_{mode or 'all'}")
        fdf = df[df["symbol"].str.contains(filt_sym, case=False)] if filt_sym else df
        st.dataframe(fdf[cols].sort_values("entry_time", ascending=False), use_container_width=True)
        if st.button("📥 Export CSV", key=f"exp_{mode or 'all'}"):
            st.download_button("Download", fdf[cols].to_csv(index=False),
                               f"trades_{mode or 'all'}.csv", "text/csv")


def render() -> None:
    render_mode_banner()
    st.title("💼 Portfolio Overview")

    db = _db()
    st.subheader("🗂️ All Positions by Ticker")

    all_trades = db.get_trades(mode=None, limit=2000)
    ft_open = st.session_state.get("ft_open_trades", {})
    ft_runs = st.session_state.get("ft_active_runs", {})
    for sym, open_t in ft_open.items():
        if open_t:
            run = ft_runs.get(sym, {})
            all_trades.append({
                "symbol": sym,
                "mode": "forward_test",
                "outcome": "Open",
                "direction": open_t.get("direction"),
                "entry_price": open_t.get("entry_price"),
                "pnl": None,
                "leveraged_return_pct": None,
                "strategy_id": run.get("strategy_id", ""),
                "capital_allocated": open_t.get("capital"),
            })

    if all_trades:
        ticker_df = _build_ticker_table(all_trades)
        display_df = ticker_df.drop(columns=["_nav_target"])

        def _pnl_style(val):
            if isinstance(val, (int, float)):
                color = _GREEN if val > 0 else (_RED if val < 0 else _GREY)
                return f"color: {color}; font-weight: bold"
            return ""

        st.dataframe(display_df.style.applymap(_pnl_style, subset=["Total P&L"]), use_container_width=True)

        open_ticker_rows = ticker_df[ticker_df["Open"] > 0]
        if not open_ticker_rows.empty:
            st.markdown("**🔗 Navigate to open position:**")
            nav_cols = st.columns(min(len(open_ticker_rows), 4))
            for col, (_, row) in zip(nav_cols, open_ticker_rows.iterrows()):
                nav_target = row["_nav_target"]
                if nav_target and col.button(f"→ {row['Symbol']} ({row['Mode']})", key=f"nav_{row['Symbol']}_{row['Mode']}"):
                    st.session_state["nav_target"] = nav_target
                    if row["Mode"] == "forward_test":
                        st.session_state["ft_jump_symbol"] = row["Symbol"]
                    st.rerun()
    else:
        st.info("No trades recorded yet across any mode.")

    st.divider()

    tab_ft, tab_paper, tab_bt, tab_live, tab_all = st.tabs(
        ["🔭 Forward Test", "📝 Paper", "⏪ Backtest", "🔴 Live", "🌐 All"]
    )
    with tab_ft:
        _render_mode_tab("forward_test", db)
    with tab_paper:
        _render_mode_tab("paper", db)
    with tab_bt:
        _render_mode_tab("backtest", db)
    with tab_live:
        _render_mode_tab("live", db)
    with tab_all:
        _render_mode_tab(None, db)

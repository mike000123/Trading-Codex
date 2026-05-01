"""
pages/page_shadow_compare.py
─────────────────────────────
Shadow Comparison Dashboard — pair every local-sim trade with its Alpaca-paper
twin (tagged via "shadow_of=<sim_trade_id>" in notes) and quantify divergence.

This is the payoff page for Steps 1–7 of the Alpaca integration: the whole
shadow infrastructure exists so this page can answer the real question —
"would my sim numbers hold up if I went live at Alpaca?"

What's shown:
  • Summary row: #pairs, #closed, outcome-match %, total P&L delta.
  • Tab 1 "Closed pairs": aggregate stats + per-pair table + cumulative-P&L
    chart (sim vs alpaca).
  • Tab 2 "Open pairs":    still-in-flight pairs with broker_status.
  • Tab 3 "Distributions": slippage + fill-latency histograms.
  • Tab 4 "Rejections":    alpaca submissions that router-rejected.

Data source: reads from the `trades` table via Database.get_trades() filtered
by mode="paper" and mode="alpaca_paper". No extra tables, no writes.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

import altair as alt
import pandas as pd
import streamlit as st

from config.settings import settings
from db.database import Database
from reporting.shadow_compare import (
    ShadowPair,
    aggregate,
    find_pairs,
    pair_metrics,
    rejected_shadow_entries,
)


# ── Styling (matches paper trading) ──────────────────────────────────────────
_GREEN = "#2faa6a"
_RED   = "#c64242"
_BLUE  = "#d4af37"
_GOLD  = "#ffd54f"
_GREY  = "#a89c80"
_AXIS  = dict(gridColor="rgba(212,175,55,0.18)", labelColor="#a89c80", titleColor="#a89c80",
              labelFontSize=12, titleFontSize=13)
_TITLE = dict(color="#e8c566", fontSize=14, fontWeight="bold")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _db() -> Database:
    return Database(settings.db_path)


def _load_pairs(limit: int = 2000) -> tuple[list[ShadowPair], list[dict], list[dict]]:
    """Pull both trade populations and return (pairs, paper_trades, alpaca_trades)."""
    db = _db()
    paper  = db.get_trades(mode="paper",        limit=limit)
    alpaca = db.get_trades(mode="alpaca_paper", limit=limit)
    pairs  = find_pairs(paper, alpaca)
    for p in pairs:
        p.metrics = pair_metrics(p)
    return pairs, paper, alpaca


def _fmt_bps(v: Optional[float]) -> str:
    return "—" if v is None else f"{v:+.1f}"


def _fmt_sec(v: Optional[float]) -> str:
    if v is None:
        return "—"
    if abs(v) < 60:
        return f"{v:.2f}s"
    return f"{v/60:.1f}m"


def _fmt_usd(v: Optional[float]) -> str:
    return "—" if v is None else f"${v:+,.2f}"


def _to_local(value):
    if value is None:
        return None
    try:
        ts = pd.Timestamp(value)
    except Exception:
        return value
    tz = datetime.now().astimezone().tzinfo
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert(tz) if tz is not None else ts


# ── Charts ───────────────────────────────────────────────────────────────────

def _cumulative_pnl_chart(metrics_df: pd.DataFrame) -> alt.Chart:
    """Two lines: cumulative sim P&L vs cumulative alpaca P&L over time."""
    if metrics_df.empty:
        return alt.Chart(pd.DataFrame({"x": [0]})).mark_text(
            text="No closed pairs yet", color=_GREY
        ).encode(x="x:Q")

    df = metrics_df.copy()
    df = df[(df["sim_closed"]) & (df["alpaca_closed"])].sort_values("entry_time_sim")
    if df.empty:
        return alt.Chart(pd.DataFrame({"x": [0]})).mark_text(
            text="No closed pairs yet", color=_GREY
        ).encode(x="x:Q")
    df["cum_sim"]    = df["sim_pnl"].cumsum()
    df["cum_alpaca"] = df["alpaca_pnl"].cumsum()

    long_df = pd.melt(
        df[["entry_time_sim", "cum_sim", "cum_alpaca"]],
        id_vars="entry_time_sim",
        value_vars=["cum_sim", "cum_alpaca"],
        var_name="source",
        value_name="cum_pnl",
    )
    long_df["source"] = long_df["source"].map({
        "cum_sim":    "Local sim",
        "cum_alpaca": "Alpaca paper",
    })

    return (
        alt.Chart(long_df)
        .mark_line(point=True, strokeWidth=2)
        .encode(
            x=alt.X("entry_time_sim:T", title="Entry time"),
            y=alt.Y("cum_pnl:Q", title="Cumulative P&L ($)"),
            color=alt.Color(
                "source:N",
                scale=alt.Scale(domain=["Local sim", "Alpaca paper"],
                                range=[_BLUE, _GOLD]),
                legend=alt.Legend(title="Source"),
            ),
            tooltip=["source", "entry_time_sim", alt.Tooltip("cum_pnl", format="+.2f")],
        )
        .properties(height=280, title="Cumulative P&L — sim vs Alpaca paper")
        .configure(background="#0c0d14")
        .configure_view(fill="#181a25", strokeOpacity=0)
        .configure_axis(**_AXIS).configure_title(**_TITLE)
    )


def _histogram(series: pd.Series, title: str, x_label: str,
               unit_suffix: str = "") -> alt.Chart:
    vals = series.dropna()
    if vals.empty:
        return alt.Chart(pd.DataFrame({"x": [0]})).mark_text(
            text=f"No {title.lower()} data yet", color=_GREY
        ).encode(x="x:Q")
    df = pd.DataFrame({"value": vals})
    chart = (
        alt.Chart(df)
        .mark_bar(color=_BLUE, opacity=0.75)
        .encode(
            x=alt.X(
                "value:Q",
                bin=alt.Bin(maxbins=30),
                title=x_label + (f" ({unit_suffix})" if unit_suffix else ""),
            ),
            y=alt.Y("count():Q", title="Count"),
            tooltip=[alt.Tooltip("count():Q", title="Count")],
        )
        .properties(height=220, title=title)
    )
    # Vertical zero line for slippage charts
    if "slippage" in title.lower():
        zero = alt.Chart(pd.DataFrame({"v": [0]})).mark_rule(
            color=_GOLD, strokeDash=[4, 4]
        ).encode(x="v:Q")
        chart = chart + zero
    return (chart.configure(background="#0c0d14").configure_view(fill="#181a25", strokeOpacity=0).configure_axis(**_AXIS).configure_title(**_TITLE))


def _outcome_pie(metrics_df: pd.DataFrame) -> Optional[alt.Chart]:
    closed = metrics_df[
        (metrics_df["sim_closed"]) & (metrics_df["alpaca_closed"])
    ].copy()
    if closed.empty:
        return None
    closed["match_status"] = closed["outcome_match"].map({
        True: "Outcome matched", False: "Outcome diverged"
    })
    counts = closed["match_status"].value_counts().reset_index()
    counts.columns = ["match_status", "count"]
    return (
        alt.Chart(counts)
        .mark_arc(innerRadius=40)
        .encode(
            theta="count:Q",
            color=alt.Color(
                "match_status:N",
                scale=alt.Scale(
                    domain=["Outcome matched", "Outcome diverged"],
                    range=[_GREEN, _RED],
                ),
                legend=alt.Legend(title="Closed pairs"),
            ),
            tooltip=["match_status", "count"],
        )
        .properties(height=220, title="Outcome alignment")
        .configure(background="#0c0d14")
        .configure_view(fill="#181a25", strokeOpacity=0)
        .configure_title(**_TITLE)
    )


# ── Page ─────────────────────────────────────────────────────────────────────

def render() -> None:
    st.title("Shadow Comparison")
    st.caption(
        "Pairs every local-sim trade with its Alpaca-paper twin (matched via "
        "`shadow_of=<sim_trade_id>` in notes) and quantifies divergence. "
        "This is the validation layer for the sim — positive slippage = Alpaca "
        "filled WORSE than sim expected."
    )
    st.info(
        "Enable shadow mode on a Paper Trading run to populate this page. "
        "Each approved sim entry mirrors to Alpaca's paper endpoint; this page "
        "reads both populations and joins them by the shadow tag."
    )

    pairs, _paper, alpaca_trades = _load_pairs()
    rejections = rejected_shadow_entries(alpaca_trades)

    if not pairs and not rejections:
        st.warning(
            "No shadow pairs found. Run paper trading with **Send to Alpaca paper "
            "in parallel** enabled and wait for at least one entry to fire."
        )
        return

    # ── Summary row ──────────────────────────────────────────────────────────
    agg = aggregate(pairs) if pairs else {
        "n_pairs": 0, "n_closed": 0, "n_open": 0, "n_outcome_match": 0,
        "n_outcome_diverged": 0, "outcome_match_pct": None,
        "entry_slippage_bps": {"mean": None, "median": None, "n": 0,
                               "min": None, "max": None},
        "fill_latency_sec":   {"mean": None, "median": None, "n": 0,
                               "min": None, "max": None},
        "exit_slippage_bps":  {"mean": None, "median": None, "n": 0,
                               "min": None, "max": None},
        "pnl_delta_usd":      {"mean": None, "median": None, "n": 0,
                               "min": None, "max": None},
        "total_sim_pnl": 0.0, "total_alpaca_pnl": 0.0, "total_pnl_delta": 0.0,
    }

    c = st.columns(5)
    c[0].metric("Shadow pairs",     agg["n_pairs"],
                delta=f"{agg['n_closed']} closed / {agg['n_open']} open",
                delta_color="off")
    c[1].metric(
        "Outcome match",
        (f"{agg['outcome_match_pct']:.0f}%"
         if agg["outcome_match_pct"] is not None else "—"),
        delta=(
            f"{agg['n_outcome_match']}/{agg['n_outcome_match'] + agg['n_outcome_diverged']} "
            "closed"
            if (agg["n_outcome_match"] + agg["n_outcome_diverged"]) > 0 else None
        ),
        delta_color="off",
    )
    c[2].metric(
        "Mean entry slippage",
        (_fmt_bps(agg["entry_slippage_bps"]["mean"]) + " bps"
         if agg["entry_slippage_bps"]["mean"] is not None else "—"),
        delta=f"median {_fmt_bps(agg['entry_slippage_bps']['median'])}",
        delta_color="off",
    )
    c[3].metric(
        "Mean fill latency",
        _fmt_sec(agg["fill_latency_sec"]["mean"]),
        delta=f"median {_fmt_sec(agg['fill_latency_sec']['median'])}",
        delta_color="off",
    )
    total_delta = agg["total_pnl_delta"] or 0.0
    delta_color = "normal" if total_delta >= 0 else "inverse"
    c[4].metric(
        "Total P&L delta (Alpaca − sim)",
        _fmt_usd(total_delta),
        delta=(
            f"sim {_fmt_usd(agg['total_sim_pnl'])} → "
            f"alpaca {_fmt_usd(agg['total_alpaca_pnl'])}"
        ),
        delta_color="off",
    )

    if rejections:
        st.warning(
            f"⚠️ {len(rejections)} shadow entries were rejected by the Alpaca "
            "router (see **Rejections** tab). These are sim entries that would "
            "NOT have been taken at the real broker."
        )

    # ── Build the metrics DataFrame ──────────────────────────────────────────
    metrics_df = pd.DataFrame([p.metrics for p in pairs]) if pairs else pd.DataFrame()

    st.divider()
    tab_closed, tab_open, tab_dist, tab_rej = st.tabs([
        "✅ Closed pairs",
        "⏳ Open pairs",
        "📈 Distributions",
        f"❌ Rejections ({len(rejections)})",
    ])

    # ── Closed pairs ─────────────────────────────────────────────────────────
    with tab_closed:
        if metrics_df.empty:
            st.info("No pairs yet.")
        else:
            closed_df = metrics_df[
                (metrics_df["sim_closed"]) & (metrics_df["alpaca_closed"])
            ].copy()
            if closed_df.empty:
                st.info("No closed pairs yet — exits haven't fired on both sides.")
            else:
                st.altair_chart(
                    _cumulative_pnl_chart(metrics_df),
                    width="stretch",
                )

                # Detail table (formatted, limited columns for readability)
                display = closed_df.copy()
                display["sim_entry_px"]    = display["sim_entry"].apply(
                    lambda v: f"{v:.4f}" if v else "—")
                display["alpaca_entry_px"] = display["alpaca_entry"].apply(
                    lambda v: f"{v:.4f}" if v else "—")
                display["entry_slip"]      = display["entry_slippage_bps"].apply(
                    lambda v: f"{v:+.1f} bps" if v is not None else "—")
                display["exit_slip"]       = display["exit_slippage_bps"].apply(
                    lambda v: f"{v:+.1f} bps" if v is not None else "—")
                display["fill_lat"]        = display["fill_latency_sec"].apply(_fmt_sec)
                display["outcome_sim"]     = display["sim_outcome"]
                display["outcome_alpaca"]  = display["alpaca_outcome"]
                display["match"]           = display["outcome_match"].map(
                    {True: "✅", False: "❌"})
                display["pnl_sim"]         = display["sim_pnl"].apply(
                    lambda v: f"${v:+,.2f}")
                display["pnl_alpaca"]      = display["alpaca_pnl"].apply(
                    lambda v: f"${v:+,.2f}")
                display["pnl_delta"]       = display["pnl_delta_usd"].apply(
                    lambda v: f"${v:+,.2f}")
                display["entry_ts"]        = display["entry_time_sim"].apply(
                    lambda v: pd.Timestamp(_to_local(v)).strftime("%Y-%m-%d %H:%M") if v else "—"
                )

                show_cols = [
                    "entry_ts", "symbol", "direction",
                    "sim_entry_px", "alpaca_entry_px", "entry_slip",
                    "fill_lat", "outcome_sim", "outcome_alpaca", "match",
                    "exit_slip", "pnl_sim", "pnl_alpaca", "pnl_delta",
                ]
                st.markdown("**Per-pair detail (closed):**")
                st.dataframe(
                    display[show_cols].sort_values("entry_ts", ascending=False),
                    width="stretch",
                    hide_index=True,
                )

                # Divergence spotlight
                if agg["n_outcome_diverged"] > 0:
                    div = closed_df[closed_df["outcome_match"] == False]
                    div_rows = [
                        f"**{r['symbol']}** ({r['direction']}) — sim "
                        f"`{r['sim_outcome']}` vs alpaca `{r['alpaca_outcome']}` "
                        f"(Δ P&L {_fmt_usd(r['pnl_delta_usd'])})"
                        for _, r in div.iterrows()
                    ]
                    st.markdown(
                        "**⚠️ Outcome divergences:**\n\n- " +
                        "\n- ".join(div_rows)
                    )

    # ── Open pairs ───────────────────────────────────────────────────────────
    with tab_open:
        if metrics_df.empty:
            st.info("No pairs yet.")
        else:
            open_df = metrics_df[
                ~((metrics_df["sim_closed"]) & (metrics_df["alpaca_closed"]))
            ].copy()
            if open_df.empty:
                st.info("All tracked pairs have closed on both sides.")
            else:
                open_df["sim_entry_px"]    = open_df["sim_entry"].apply(
                    lambda v: f"{v:.4f}" if v else "—")
                open_df["alpaca_entry_px"] = open_df["alpaca_entry"].apply(
                    lambda v: f"{v:.4f}" if v else "—")
                open_df["entry_slip"]      = open_df["entry_slippage_bps"].apply(
                    lambda v: f"{v:+.1f} bps" if v is not None else "—")
                open_df["fill_lat"]        = open_df["fill_latency_sec"].apply(_fmt_sec)
                open_df["broker"]          = open_df["broker_status"].fillna("—")
                open_df["entry_ts"]        = open_df["entry_time_sim"].apply(
                    lambda v: pd.Timestamp(_to_local(v)).strftime("%Y-%m-%d %H:%M") if v else "—"
                )
                open_df["state"] = open_df.apply(
                    lambda r: (
                        "sim open · alpaca open"       if not r["sim_closed"] and not r["alpaca_closed"] else
                        "sim CLOSED · alpaca open"     if r["sim_closed"]     and not r["alpaca_closed"] else
                        "sim open · alpaca CLOSED"
                    ),
                    axis=1,
                )
                show = ["entry_ts", "symbol", "direction", "sim_entry_px",
                        "alpaca_entry_px", "entry_slip", "fill_lat",
                        "broker", "state"]
                st.dataframe(
                    open_df[show].sort_values("entry_ts", ascending=False),
                    width="stretch",
                    hide_index=True,
                )
                st.caption(
                    "Rows where one side is CLOSED but the other is still "
                    "Open indicate the broker TP/SL or a reconciliation path "
                    "hasn't caught up yet — it'll resolve on the next paper-"
                    "page refresh."
                )

    # ── Distributions ────────────────────────────────────────────────────────
    with tab_dist:
        if metrics_df.empty:
            st.info("No pairs yet.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.altair_chart(
                    _histogram(
                        metrics_df["entry_slippage_bps"],
                        "Entry slippage distribution",
                        "Entry slippage (positive = worse)",
                        unit_suffix="bps",
                    ),
                    width="stretch",
                )
                st.altair_chart(
                    _histogram(
                        metrics_df["fill_latency_sec"],
                        "Fill latency distribution",
                        "Broker fill delay from sim entry",
                        unit_suffix="seconds",
                    ),
                    width="stretch",
                )
            with col2:
                pie = _outcome_pie(metrics_df)
                if pie is not None:
                    st.altair_chart(pie, width="stretch")
                else:
                    st.info(
                        "Need at least one closed-on-both-sides pair to show "
                        "outcome alignment."
                    )
                st.altair_chart(
                    _histogram(
                        metrics_df["pnl_delta_usd"],
                        "P&L delta distribution",
                        "alpaca_pnl − sim_pnl",
                        unit_suffix="$",
                    ),
                    width="stretch",
                )

            # Small-print stats
            st.markdown("**Summary statistics:**")
            stat_rows = []
            for label, key in [
                ("Entry slippage (bps)", "entry_slippage_bps"),
                ("Exit slippage (bps)",  "exit_slippage_bps"),
                ("Fill latency (sec)",   "fill_latency_sec"),
                ("P&L delta ($)",        "pnl_delta_usd"),
            ]:
                s = agg[key]
                stat_rows.append({
                    "Metric": label,
                    "N":     s["n"],
                    "Mean":   f"{s['mean']:+.3f}"   if s["mean"]   is not None else "—",
                    "Median": f"{s['median']:+.3f}" if s["median"] is not None else "—",
                    "Min":    f"{s['min']:+.3f}"    if s["min"]    is not None else "—",
                    "Max":    f"{s['max']:+.3f}"    if s["max"]    is not None else "—",
                })
            st.dataframe(pd.DataFrame(stat_rows), width="stretch", hide_index=True)

    # ── Rejections ───────────────────────────────────────────────────────────
    with tab_rej:
        if not rejections:
            st.success("No shadow entries were rejected by the Alpaca router.")
        else:
            rej_df = pd.DataFrame(rejections)
            rej_df["entry_ts"] = rej_df["entry_time"].apply(
                lambda v: pd.Timestamp(_to_local(v)).strftime("%Y-%m-%d %H:%M") if v else "—"
            )
            show = ["entry_ts", "symbol", "direction", "strategy_id", "reason"]
            st.dataframe(
                rej_df[show].sort_values("entry_ts", ascending=False),
                width="stretch",
                hide_index=True,
            )
            st.caption(
                "These sim entries fired locally but were **not** submitted to "
                "Alpaca — usually because a router-level gate (PDT, fractional, "
                "RTH, capital) tripped. They represent sim trades that would "
                "not have been taken at the real broker."
            )

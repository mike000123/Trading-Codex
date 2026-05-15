"""
reporting/shadow_compare.py
───────────────────────────
Pair local-sim trades with their Alpaca-paper "shadow" twins and quantify
divergence — slippage, fill-latency, outcome alignment, P&L delta.

The shadow twin relationship is tagged at execute() time by _run_tick:
    shadow_trade.notes += " | shadow_of=<sim_trade_id>"

This module is streamlit-free so the logic is unit-testable in isolation.
The page layer pulls trades via Database.get_trades() and hands them in.

Public API:
  find_pairs(paper_trades, alpaca_trades)        → list[ShadowPair]
  pair_metrics(pair)                             → dict of per-pair metrics
  aggregate(pairs)                               → dict of rolled-up stats
  rejected_shadow_entries(alpaca_trades)         → list[dict] — the ones
                                                   Alpaca routing rejected
                                                   before a broker order
                                                   was created

Convention for signed metrics:
  entry_slippage_bps   Positive = worse for the trader (paid more on Long,
                        received less on Short).  Negative = better than
                        sim expected.
  fill_latency_sec     Broker fill time minus sim entry time. Always
                        non-negative under normal operation.
  exit_slippage_bps    Positive = worse for the trader on the exit side.
  pnl_delta_usd        alpaca.pnl - sim.pnl (negative = Alpaca did worse).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from statistics import mean, median
from typing import Optional


_SHADOW_RE = re.compile(r"shadow_of=([A-Za-z0-9\-]+)")


# ── Types ───────────────────────────────────────────────────────────────────

@dataclass
class ShadowPair:
    sim: dict                       # paper trade row
    alpaca: dict                    # alpaca_paper trade row
    metrics: dict = field(default_factory=dict)   # populated by pair_metrics()


# ── Pair discovery ──────────────────────────────────────────────────────────

def _extract_sim_id(notes: Optional[str]) -> Optional[str]:
    """Return the sim trade_id tagged in 'shadow_of=<id>' inside notes."""
    if not notes:
        return None
    m = _SHADOW_RE.search(notes)
    return m.group(1) if m else None


def find_pairs(paper_trades: list[dict], alpaca_trades: list[dict]) -> list[ShadowPair]:
    """
    Build ShadowPair objects by matching each alpaca_paper row whose notes
    carry shadow_of=<id> to the corresponding paper row by that id. Alpaca
    rows without a matching sim trade are dropped (the sim side was never
    persisted — shouldn't happen but we're defensive).
    """
    sim_by_id = {t["id"]: t for t in paper_trades if t.get("id")}
    pairs: list[ShadowPair] = []
    for a in alpaca_trades:
        sim_id = _extract_sim_id(a.get("notes"))
        if not sim_id:
            continue
        sim = sim_by_id.get(sim_id)
        if not sim:
            continue
        pairs.append(ShadowPair(sim=sim, alpaca=a))
    return pairs


# ── Per-pair metrics ────────────────────────────────────────────────────────

def _parse_dt(v) -> Optional[datetime]:
    """Parse a datetime-like value into a tz-NAIVE UTC datetime.

    Subtracting a tz-aware datetime from a tz-naive one raises TypeError.
    Some sim trade rows arrive naive (older backtest output); Alpaca's
    broker timestamps come back tz-aware ("Z" suffix). Normalise so the
    pair_metrics latency arithmetic always works.
    """
    if v is None:
        return None
    dt: Optional[datetime] = None
    if isinstance(v, datetime):
        dt = v
    else:
        try:
            dt = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
        except Exception:
            return None
    if dt is None:
        return None
    if dt.tzinfo is not None:
        try:
            dt = dt.astimezone(tz=None).replace(tzinfo=None)
        except Exception:
            dt = dt.replace(tzinfo=None)
    return dt


def _directional_bps(sim_px: float, alpaca_px: float, direction: str) -> float:
    """
    Slippage in basis points, signed so positive = worse for the trader.
    Long  → paying more than sim expected ⇒ positive is worse
    Short → receiving less than sim expected ⇒ positive is worse
    """
    if not sim_px or sim_px <= 0:
        return 0.0
    raw = (alpaca_px - sim_px) / sim_px * 10_000.0
    return -raw if direction == "Short" else raw


def pair_metrics(pair: ShadowPair) -> dict:
    """
    Compute every metric we display per pair. Missing broker data (e.g. order
    still pending) yields None — callers must filter before aggregating.
    """
    sim    = pair.sim
    alpaca = pair.alpaca
    direction = sim.get("direction") or alpaca.get("direction") or "Long"

    # Entry slippage: sim.entry_price vs alpaca.filled_avg_price (falls back
    # to alpaca.entry_price if the fill hasn't landed yet).
    sim_entry    = _safe_float(sim.get("entry_price"))
    alpaca_fill  = _safe_float(alpaca.get("filled_avg_price"))
    alpaca_entry = _safe_float(alpaca.get("entry_price"))
    alpaca_entry_px = alpaca_fill if alpaca_fill is not None else alpaca_entry

    entry_slip_bps = (
        _directional_bps(sim_entry, alpaca_entry_px, direction)
        if (sim_entry and alpaca_entry_px) else None
    )

    # Fill latency: filled_at - sim entry_time. Broker can report filled_at
    # on the submitted_at row pre-fill; we prefer broker_submitted_at in that
    # case because it reflects when we told Alpaca about the order.
    sim_entry_t  = _parse_dt(sim.get("entry_time"))
    alpaca_sub_t = _parse_dt(alpaca.get("broker_submitted_at") or alpaca.get("entry_time"))
    alpaca_fill_t = _parse_dt(alpaca.get("filled_at"))
    fill_latency_sec = None
    if sim_entry_t and alpaca_fill_t:
        fill_latency_sec = (alpaca_fill_t - sim_entry_t).total_seconds()
    submit_latency_sec = None
    if sim_entry_t and alpaca_sub_t:
        submit_latency_sec = (alpaca_sub_t - sim_entry_t).total_seconds()

    # Exit comparison — only meaningful when BOTH sides are closed
    sim_closed    = (sim.get("outcome") or "Open") != "Open"
    alpaca_closed = (alpaca.get("outcome") or "Open") != "Open"

    exit_slip_bps   = None
    exit_delta_sec  = None
    outcome_match   = None
    if sim_closed and alpaca_closed:
        sim_exit    = _safe_float(sim.get("exit_price"))
        alpaca_exit = _safe_float(alpaca.get("exit_price"))
        if sim_exit and alpaca_exit:
            # Exit convention: for a Long, a LOWER exit price is worse, so
            # flip the sign vs entry. Short uses the opposite.
            raw_bps = (alpaca_exit - sim_exit) / sim_exit * 10_000.0
            exit_slip_bps = raw_bps if direction == "Short" else -raw_bps
        sim_exit_t    = _parse_dt(sim.get("exit_time"))
        alpaca_exit_t = _parse_dt(alpaca.get("exit_time"))
        if sim_exit_t and alpaca_exit_t:
            exit_delta_sec = (alpaca_exit_t - sim_exit_t).total_seconds()
        outcome_match = sim.get("outcome") == alpaca.get("outcome")

    # P&L deltas
    sim_pnl    = _safe_float(sim.get("pnl"))    or 0.0
    alpaca_pnl = _safe_float(alpaca.get("pnl")) or 0.0
    pnl_delta_usd = alpaca_pnl - sim_pnl
    pnl_delta_pct = None
    if abs(sim_pnl) > 1e-9:
        pnl_delta_pct = (alpaca_pnl - sim_pnl) / abs(sim_pnl) * 100.0

    return {
        "sim_id":              sim.get("id"),
        "alpaca_id":           alpaca.get("id"),
        "symbol":              sim.get("symbol") or alpaca.get("symbol"),
        "direction":           direction,
        "strategy_id":         sim.get("strategy_id") or alpaca.get("strategy_id"),
        "entry_time_sim":      sim_entry_t,
        "entry_time_alpaca":   alpaca_fill_t or alpaca_sub_t,
        "sim_entry":           sim_entry,
        "alpaca_entry":        alpaca_entry_px,
        "entry_slippage_bps":  entry_slip_bps,
        "submit_latency_sec":  submit_latency_sec,
        "fill_latency_sec":    fill_latency_sec,
        "sim_exit":            _safe_float(sim.get("exit_price")),
        "alpaca_exit":         _safe_float(alpaca.get("exit_price")),
        "exit_slippage_bps":   exit_slip_bps,
        "exit_delta_sec":      exit_delta_sec,
        "sim_outcome":         sim.get("outcome"),
        "alpaca_outcome":      alpaca.get("outcome"),
        "outcome_match":       outcome_match,
        "sim_pnl":             sim_pnl,
        "alpaca_pnl":          alpaca_pnl,
        "pnl_delta_usd":       pnl_delta_usd,
        "pnl_delta_pct":       pnl_delta_pct,
        "sim_closed":          sim_closed,
        "alpaca_closed":       alpaca_closed,
        "broker_status":       alpaca.get("broker_status"),
        "broker_order_id":     alpaca.get("broker_order_id"),
    }


def _safe_float(v) -> Optional[float]:
    try:
        return float(v) if v is not None else None
    except Exception:
        return None


# ── Aggregate ───────────────────────────────────────────────────────────────

def aggregate(pairs: list[ShadowPair]) -> dict:
    """
    Roll up metrics across all pairs. Runs pair_metrics() on each pair first.
    Returns a dict the UI can unpack straight into st.metric() columns.
    """
    for p in pairs:
        if not p.metrics:
            p.metrics = pair_metrics(p)
    metrics = [p.metrics for p in pairs]

    n_total    = len(metrics)
    n_closed   = sum(1 for m in metrics if m["sim_closed"] and m["alpaca_closed"])
    n_open     = n_total - n_closed
    n_div_out  = sum(1 for m in metrics
                     if m["outcome_match"] is False)        # only counts closed
    n_out_ok   = sum(1 for m in metrics if m["outcome_match"] is True)
    closed_with_outcome = [m for m in metrics if m["outcome_match"] is not None]
    outcome_match_pct = (
        100.0 * sum(1 for m in closed_with_outcome if m["outcome_match"])
        / len(closed_with_outcome)
    ) if closed_with_outcome else None

    entry_slips  = [m["entry_slippage_bps"] for m in metrics
                    if m["entry_slippage_bps"] is not None]
    fill_latency = [m["fill_latency_sec"]   for m in metrics
                    if m["fill_latency_sec"] is not None]
    exit_slips   = [m["exit_slippage_bps"]  for m in metrics
                    if m["exit_slippage_bps"] is not None]
    pnl_deltas   = [m["pnl_delta_usd"]      for m in metrics
                    if m["pnl_delta_usd"]   is not None]

    def _stats(xs):
        if not xs:
            return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
        return {"n": len(xs), "mean": mean(xs), "median": median(xs),
                "min":  min(xs), "max":  max(xs)}

    total_sim_pnl    = sum(m["sim_pnl"]    for m in metrics if m["sim_closed"])
    total_alpaca_pnl = sum(m["alpaca_pnl"] for m in metrics if m["alpaca_closed"])

    return {
        "n_pairs":             n_total,
        "n_closed":            n_closed,
        "n_open":              n_open,
        "n_outcome_match":     n_out_ok,
        "n_outcome_diverged":  n_div_out,
        "outcome_match_pct":   outcome_match_pct,
        "entry_slippage_bps":  _stats(entry_slips),
        "fill_latency_sec":    _stats(fill_latency),
        "exit_slippage_bps":   _stats(exit_slips),
        "pnl_delta_usd":       _stats(pnl_deltas),
        "total_sim_pnl":       total_sim_pnl,
        "total_alpaca_pnl":    total_alpaca_pnl,
        "total_pnl_delta":     total_alpaca_pnl - total_sim_pnl,
    }


# ── Rejected shadow entries (never made it into a broker order) ─────────────

def rejected_shadow_entries(alpaca_trades: list[dict]) -> list[dict]:
    """
    Shadow submissions that were rejected by router-level gating (no broker
    order ever created). The entry_policy layer or RiskManager stamps
    "REJECTED: …" into the notes field, and broker_order_id stays NULL.

    We still surface these because they tell you which sim entries WOULD
    have been blocked on Alpaca — that's signal about how realistic the sim
    behaviour is.
    """
    out = []
    for t in alpaca_trades:
        notes = t.get("notes") or ""
        if "REJECTED" not in notes:
            continue
        # Extract the reason after "REJECTED:"
        reason = ""
        try:
            reason = notes.split("REJECTED:", 1)[1].split("|", 1)[0].strip()
        except Exception:
            reason = notes[:200]
        out.append({
            "symbol":      t.get("symbol"),
            "direction":   t.get("direction"),
            "strategy_id": t.get("strategy_id"),
            "entry_time":  t.get("entry_time"),
            "reason":      reason,
            "sim_id":      _extract_sim_id(notes),
        })
    return out

"""
scripts/evaluate_fair_value_regime.py
──────────────────────────────────────
A/B evaluation harness for the GLD fair-value regime feature.

Runs the same GLD candidate backtest twice per rolling window — once with
`gold_fair_value_regime_enabled=False` (control) and once with it enabled
(treatment) — keeping ALL other parameters identical. Reports per-window
and aggregate deltas on:

  • Total return (%)
  • Sharpe ratio
  • Max drawdown (%)
  • Win rate (%)
  • Total trades

Why we need this:
  The regime was left disabled because "it didn't improve results" — but
  that was never quantified across time slices. This script gives us an
  honest number so Steps B (fix regime definition) and D (time-varying
  confidence) can be judged by their delta vs. this baseline.

Usage:
  python scripts/evaluate_fair_value_regime.py
  python scripts/evaluate_fair_value_regime.py --candidate gld_best_1x_sweep_20260419
  python scripts/evaluate_fair_value_regime.py --window-months 12 --step-months 3
  python scripts/evaluate_fair_value_regime.py --full-only  # single full-range run
"""
from __future__ import annotations

import argparse
import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ── Dummy logger so we can import app modules without Streamlit / DB. ─────
def _install_dummy_logger() -> None:
    if "core.logger" in sys.modules:
        return
    logger_mod = types.ModuleType("core.logger")

    class _Dummy:
        def info(self, *args, **kwargs): pass
        def warning(self, *args, **kwargs): pass
        def error(self, *args, **kwargs): pass
        def debug(self, *args, **kwargs): pass

    logger_mod.log = _Dummy()
    sys.modules["core.logger"] = logger_mod


_install_dummy_logger()

from config.settings import RiskConfig
from config.strategy_presets.bollinger_rsi.gld_candidates import get_candidate
from data.ingestion import prepare_strategy_data
from reporting.backtest import BacktestEngine
from risk.manager import RiskManager
from strategies import get_strategy


# ── Constants ─────────────────────────────────────────────────────────────
GLD_CACHE = ROOT / "data_cache" / "alpaca" / "GLD" / "1Min.csv"
DEFAULT_START = pd.Timestamp("2024-04-04")
DEFAULT_END = pd.Timestamp("2026-04-03 23:59:59")
OUT_DIR = ROOT / "artifacts" / "optimization"


# ── Data structures ───────────────────────────────────────────────────────
@dataclass
class RunMetrics:
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate_pct: float
    total_trades: int


@dataclass
class WindowResult:
    label: str
    start: pd.Timestamp
    end: pd.Timestamp
    bars: int
    off: RunMetrics
    on: RunMetrics

    def delta(self) -> dict[str, float]:
        return {
            "d_total_return_pct": self.on.total_return_pct - self.off.total_return_pct,
            "d_sharpe_ratio":     self.on.sharpe_ratio     - self.off.sharpe_ratio,
            "d_max_drawdown_pct": self.on.max_drawdown_pct - self.off.max_drawdown_pct,
            "d_win_rate_pct":     self.on.win_rate_pct     - self.off.win_rate_pct,
            "d_total_trades":     self.on.total_trades     - self.off.total_trades,
        }


# ── Data loading ──────────────────────────────────────────────────────────
def _load_gld_prices(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if not GLD_CACHE.exists():
        raise SystemExit(f"GLD minute cache not found at {GLD_CACHE}. "
                         "Load GLD 1Min data first (via the UI or scripts/refresh_gld_cb_proxy.py).")
    prices = pd.read_csv(GLD_CACHE)
    prices["date"] = pd.to_datetime(prices["date"])
    return prices[(prices["date"] >= start) & (prices["date"] <= end)].reset_index(drop=True)


def _prepare_gld_data(prices: pd.DataFrame, resolved_params: dict[str, Any]) -> pd.DataFrame:
    """Run prepare_strategy_data with the *treatment* params so fair-value
    enrichment uses the same slope/gap thresholds the strategy will consume.

    Fair-value columns are produced regardless of whether the regime flag
    is on (the flag only gates *usage* in the strategy), so we can prep
    once per window and flip the flag between runs."""
    cls = get_strategy("bollinger_rsi")
    strategy = cls(params=dict(resolved_params))
    return prepare_strategy_data(
        prices,
        strategy,
        primary_symbol="GLD",
        source="alpaca",
        interval="1Min",
        start=prices["date"].min(),
        end=prices["date"].max(),
    )


# ── Backtest runner ───────────────────────────────────────────────────────
def _run_once(
    prepared: pd.DataFrame,
    params: dict[str, Any],
    *,
    leverage: float,
    risk_cap: float,
    spread_pct: float,
    slippage_pct: float,
    commission: float,
    starting_equity: float,
    capital_per_trade: float,
) -> RunMetrics:
    cls = get_strategy("bollinger_rsi")
    strategy = cls(params=dict(params))
    engine = BacktestEngine(
        strategy,
        risk_manager=RiskManager(
            RiskConfig(
                max_capital_per_trade_pct=100.0,
                max_daily_loss_pct=100.0,
                max_open_positions=999,
                default_max_loss_pct_of_capital=float(risk_cap),
            )
        ),
        spread_pct=spread_pct,
        slippage_pct=slippage_pct,
        commission_per_trade=commission,
    )
    result = engine.run(
        prepared,
        "GLD",
        leverage=leverage,
        capital_per_trade=capital_per_trade,
        starting_equity=starting_equity,
    )
    return RunMetrics(
        total_return_pct=float(result.total_return_pct),
        sharpe_ratio=float(result.sharpe_ratio),
        max_drawdown_pct=float(result.max_drawdown_pct),
        win_rate_pct=float(result.win_rate_pct),
        total_trades=int(result.total_trades),
    )


def _resolve_params(candidate_name: str) -> tuple[dict[str, Any], float, float]:
    """Merge candidate overlay onto base defaults → (params, leverage, risk_cap).

    The strategy's default_params are applied automatically by the engine when
    we pass a partial params dict, so we only need to include the overlay.
    But we DO want the GLD preset merged so the run matches real behavior;
    reuse the strategy class to resolve effective params."""
    candidate = get_candidate(candidate_name) if candidate_name else {}
    leverage = float(candidate.pop("leverage", 1.0))
    risk_cap = float(candidate.pop("risk_max_loss_pct_of_capital", 50.0))
    return candidate, leverage, risk_cap


# ── Window generation ─────────────────────────────────────────────────────
def _generate_windows(
    start: pd.Timestamp,
    end: pd.Timestamp,
    window_months: int,
    step_months: int,
    full_only: bool,
) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    windows: list[tuple[str, pd.Timestamp, pd.Timestamp]] = []
    if full_only:
        windows.append(("FULL", start, end))
        return windows

    cursor = start
    while True:
        w_end = cursor + pd.DateOffset(months=window_months) - pd.Timedelta(seconds=1)
        if w_end > end:
            break
        label = f"{cursor.strftime('%Y-%m')}→{w_end.strftime('%Y-%m')}"
        windows.append((label, cursor, w_end))
        cursor = cursor + pd.DateOffset(months=step_months)

    # Always include a full-range row at the end for the headline comparison.
    windows.append(("FULL", start, end))
    return windows


# ── Main evaluation loop ──────────────────────────────────────────────────
def evaluate(
    *,
    candidate: str,
    window_months: int,
    step_months: int,
    full_only: bool,
    spread_pct: float,
    slippage_pct: float,
    commission: float,
    starting_equity: float,
    capital_per_trade: float,
    out_csv: Path,
    out_md: Path,
) -> list[WindowResult]:
    candidate_params, leverage, risk_cap = _resolve_params(candidate)
    overall_prices = _load_gld_prices(DEFAULT_START, DEFAULT_END)
    if overall_prices.empty:
        raise SystemExit("No GLD prices loaded in the configured range.")

    windows = _generate_windows(
        start=overall_prices["date"].min().normalize(),
        end=overall_prices["date"].max(),
        window_months=window_months,
        step_months=step_months,
        full_only=full_only,
    )
    print(f"Evaluating {len(windows)} window(s) with candidate={candidate or '<base>'}  "
          f"leverage={leverage}x  risk_cap={risk_cap}%")

    off_params = {**candidate_params, "gold_fair_value_regime_enabled": False}
    on_params  = {**candidate_params, "gold_fair_value_regime_enabled": True}

    results: list[WindowResult] = []
    for idx, (label, w_start, w_end) in enumerate(windows, start=1):
        w_prices = overall_prices[
            (overall_prices["date"] >= w_start) & (overall_prices["date"] <= w_end)
        ].reset_index(drop=True)
        if w_prices.empty:
            print(f"  [{idx}/{len(windows)}] {label}  SKIP (no bars)")
            continue

        print(f"  [{idx}/{len(windows)}] {label}  ({len(w_prices):,} bars) … ", end="", flush=True)
        prepared = _prepare_gld_data(w_prices, on_params)
        metrics_off = _run_once(
            prepared, off_params,
            leverage=leverage, risk_cap=risk_cap,
            spread_pct=spread_pct, slippage_pct=slippage_pct, commission=commission,
            starting_equity=starting_equity, capital_per_trade=capital_per_trade,
        )
        metrics_on = _run_once(
            prepared, on_params,
            leverage=leverage, risk_cap=risk_cap,
            spread_pct=spread_pct, slippage_pct=slippage_pct, commission=commission,
            starting_equity=starting_equity, capital_per_trade=capital_per_trade,
        )
        wr = WindowResult(label=label, start=w_start, end=w_end,
                          bars=len(w_prices), off=metrics_off, on=metrics_on)
        results.append(wr)
        d = wr.delta()
        print(f"ret Δ={d['d_total_return_pct']:+.2f}%  "
              f"sharpe Δ={d['d_sharpe_ratio']:+.2f}  "
              f"dd Δ={d['d_max_drawdown_pct']:+.2f}%  "
              f"trades Δ={d['d_total_trades']:+d}")

    # ── Persist ───────────────────────────────────────────────────────────
    _write_csv(results, out_csv)
    _write_markdown(results, out_md, candidate=candidate, leverage=leverage, risk_cap=risk_cap,
                    window_months=window_months, step_months=step_months)
    def _rel(p: Path) -> str:
        try: return str(p.resolve().relative_to(ROOT))
        except ValueError: return str(p)
    print(f"\nSaved: {_rel(out_csv)}")
    print(f"Saved: {_rel(out_md)}")
    return results


# ── Output writers ────────────────────────────────────────────────────────
def _write_csv(results: Iterable[WindowResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for r in results:
        d = r.delta()
        rows.append({
            "window":               r.label,
            "start":                r.start.strftime("%Y-%m-%d"),
            "end":                  r.end.strftime("%Y-%m-%d"),
            "bars":                 r.bars,
            "off_total_return_pct": r.off.total_return_pct,
            "on_total_return_pct":  r.on.total_return_pct,
            "d_total_return_pct":   d["d_total_return_pct"],
            "off_sharpe":           r.off.sharpe_ratio,
            "on_sharpe":            r.on.sharpe_ratio,
            "d_sharpe":             d["d_sharpe_ratio"],
            "off_max_dd_pct":       r.off.max_drawdown_pct,
            "on_max_dd_pct":        r.on.max_drawdown_pct,
            "d_max_dd_pct":         d["d_max_drawdown_pct"],
            "off_win_rate_pct":     r.off.win_rate_pct,
            "on_win_rate_pct":      r.on.win_rate_pct,
            "d_win_rate_pct":       d["d_win_rate_pct"],
            "off_total_trades":     r.off.total_trades,
            "on_total_trades":      r.on.total_trades,
            "d_total_trades":       d["d_total_trades"],
        })
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_markdown(
    results: list[WindowResult],
    path: Path,
    *,
    candidate: str,
    leverage: float,
    risk_cap: float,
    window_months: int,
    step_months: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Fair-value regime A/B evaluation",
        "",
        f"- Candidate: `{candidate or '<base>'}`",
        f"- Leverage: `{leverage}x`  ·  Risk cap: `{risk_cap}%`",
        f"- Rolling window: `{window_months}` months, step `{step_months}` months",
        f"- Generated from `scripts/evaluate_fair_value_regime.py`",
        "",
        "## Per-window deltas (ON − OFF)",
        "",
        "| Window | Bars | Return Δ | Sharpe Δ | Max DD Δ | Win-rate Δ | Trades Δ |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        d = r.delta()
        lines.append(
            f"| {r.label} | {r.bars:,} | "
            f"{d['d_total_return_pct']:+.2f}% | "
            f"{d['d_sharpe_ratio']:+.3f} | "
            f"{d['d_max_drawdown_pct']:+.2f}% | "
            f"{d['d_win_rate_pct']:+.1f} pp | "
            f"{d['d_total_trades']:+d} |"
        )

    # Summary: wins / losses / mean deltas across rolling windows (excludes FULL row).
    rolling = [r for r in results if r.label != "FULL"]
    if rolling:
        n = len(rolling)
        wins_return = sum(1 for r in rolling if r.delta()["d_total_return_pct"] > 0)
        wins_sharpe = sum(1 for r in rolling if r.delta()["d_sharpe_ratio"] > 0)
        mean_d_ret = sum(r.delta()["d_total_return_pct"] for r in rolling) / n
        mean_d_sharpe = sum(r.delta()["d_sharpe_ratio"] for r in rolling) / n
        mean_d_dd = sum(r.delta()["d_max_drawdown_pct"] for r in rolling) / n
        mean_d_trades = sum(r.delta()["d_total_trades"] for r in rolling) / n
        lines += [
            "",
            "## Rolling-window summary",
            "",
            f"- Windows: **{n}**",
            f"- Return-positive windows: **{wins_return}/{n}**  ({wins_return/n*100:.0f}%)",
            f"- Sharpe-positive windows: **{wins_sharpe}/{n}**  ({wins_sharpe/n*100:.0f}%)",
            f"- Mean Δ return: **{mean_d_ret:+.2f}%**",
            f"- Mean Δ Sharpe: **{mean_d_sharpe:+.3f}**",
            f"- Mean Δ max-DD: **{mean_d_dd:+.2f}%**  (negative = improvement)",
            f"- Mean Δ trade-count: **{mean_d_trades:+.1f}**",
        ]

    # Full-range headline.
    full = next((r for r in results if r.label == "FULL"), None)
    if full:
        d = full.delta()
        lines += [
            "",
            "## Full-range headline",
            "",
            f"- Window: `{full.start.strftime('%Y-%m-%d')}` → `{full.end.strftime('%Y-%m-%d')}`  ({full.bars:,} bars)",
            f"- OFF: return `{full.off.total_return_pct:+.2f}%`  sharpe `{full.off.sharpe_ratio:.3f}`  "
            f"DD `{full.off.max_drawdown_pct:.2f}%`  win `{full.off.win_rate_pct:.1f}%`  "
            f"trades `{full.off.total_trades}`",
            f"- ON : return `{full.on.total_return_pct:+.2f}%`  sharpe `{full.on.sharpe_ratio:.3f}`  "
            f"DD `{full.on.max_drawdown_pct:.2f}%`  win `{full.on.win_rate_pct:.1f}%`  "
            f"trades `{full.on.total_trades}`",
            f"- Δ : return **{d['d_total_return_pct']:+.2f}%**  sharpe **{d['d_sharpe_ratio']:+.3f}**  "
            f"DD **{d['d_max_drawdown_pct']:+.2f}%**  win **{d['d_win_rate_pct']:+.1f} pp**  "
            f"trades **{d['d_total_trades']:+d}**",
        ]

    path.write_text("\n".join(lines), encoding="utf-8")


# ── CLI ───────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="A/B evaluation for the GLD fair-value regime.")
    parser.add_argument("--candidate", default="gld_best_1x_sweep_20260419",
                        help="Named GLD candidate preset to overlay on defaults (use '' for base only).")
    parser.add_argument("--window-months", type=int, default=12,
                        help="Rolling window length in months.")
    parser.add_argument("--step-months", type=int, default=3,
                        help="Step between rolling-window starts.")
    parser.add_argument("--full-only", action="store_true",
                        help="Run a single full-range comparison instead of rolling windows.")
    parser.add_argument("--spread-pct", type=float, default=0.06)
    parser.add_argument("--slippage-pct", type=float, default=0.02)
    parser.add_argument("--commission", type=float, default=0.0)
    parser.add_argument("--starting-equity", type=float, default=1000.0)
    parser.add_argument("--capital-per-trade", type=float, default=1000.0)
    parser.add_argument("--out-csv", default=str(OUT_DIR / "fair_value_regime_ab.csv"))
    parser.add_argument("--out-md",  default=str(OUT_DIR / "fair_value_regime_ab.md"))
    args = parser.parse_args()

    results = evaluate(
        candidate=args.candidate,
        window_months=args.window_months,
        step_months=args.step_months,
        full_only=args.full_only,
        spread_pct=args.spread_pct,
        slippage_pct=args.slippage_pct,
        commission=args.commission,
        starting_equity=args.starting_equity,
        capital_per_trade=args.capital_per_trade,
        out_csv=Path(args.out_csv),
        out_md=Path(args.out_md),
    )

    # Compact JSON dump to stdout for easy programmatic consumption.
    payload = {
        "candidate": args.candidate,
        "windows": [
            {
                "label": r.label,
                "start": r.start.strftime("%Y-%m-%d"),
                "end":   r.end.strftime("%Y-%m-%d"),
                "bars":  r.bars,
                "off":   r.off.__dict__,
                "on":    r.on.__dict__,
                "delta": r.delta(),
            }
            for r in results
        ],
    }
    print("\n" + json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

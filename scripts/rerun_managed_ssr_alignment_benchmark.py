from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.regenerate_current_portfolio_benchmark import (
    ARTIFACT_DIR,
    SLEEVE_CAPITAL,
    SYMBOLS,
    WINDOW_END,
    WINDOW_START,
    _managed_engine,
    _prepare_symbol,
    _sum_trade_pnl_by_symbol,
)
from reporting.managed_portfolio_backtest import ManagedPortfolioSymbolInput
from strategies import get_strategy, list_strategies


BASELINE_STEM = f"current_repo_state_portfolio_benchmark_{WINDOW_START.date()}_to_{WINDOW_END.date()}"
BASELINE_CSV = ARTIFACT_DIR / f"{BASELINE_STEM}.csv"
BASELINE_JSON = ARTIFACT_DIR / f"{BASELINE_STEM}_summary.json"


def _load_baseline_rows() -> list[dict[str, str]]:
    with BASELINE_CSV.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _load_baseline_summary() -> dict:
    return json.loads(BASELINE_JSON.read_text(encoding="utf-8"))


def main() -> None:
    if not BASELINE_CSV.exists():
        raise FileNotFoundError(f"Baseline CSV not found: {BASELINE_CSV}")
    if not BASELINE_JSON.exists():
        raise FileNotFoundError(f"Baseline summary not found: {BASELINE_JSON}")

    baseline_rows = _load_baseline_rows()
    baseline_by_ticker = {
        str(row["Ticker"]).strip().upper(): row for row in baseline_rows
    }
    baseline_summary = _load_baseline_summary()
    total_starting_equity = float(baseline_summary["total_initial_equity"])
    strategy_name_map = {str(item["id"]): str(item["name"]) for item in list_strategies()}

    print(
        f"Re-running managed-only benchmark with SSR alignment for {len(SYMBOLS)} symbols "
        f"({WINDOW_START.date()} -> {WINDOW_END.date()})",
        flush=True,
    )
    prep_started = time.perf_counter()
    prepared_inputs = []
    for idx, symbol in enumerate(SYMBOLS, start=1):
        print(f"[prep {idx}/{len(SYMBOLS)}] {symbol}", flush=True)
        prepared = _prepare_symbol(symbol, strategy_name_map)
        prepared_inputs.append(
            ManagedPortfolioSymbolInput(
                symbol=prepared.symbol,
                strategy_id=prepared.strategy_id,
                strategy_name=prepared.strategy_name,
                strategy=get_strategy(prepared.strategy_id)(params={}),
                data=prepared.data,
            )
        )
    prep_elapsed = time.perf_counter() - prep_started
    print(f"Prepared {len(prepared_inputs)} tickers in {prep_elapsed:.2f}s", flush=True)

    managed_started = time.perf_counter()
    managed_result = _managed_engine(max_open_positions=len(SYMBOLS)).run(
        prepared_inputs,
        leverage=1.0,
        capital_per_trade=SLEEVE_CAPITAL,
        starting_equity=total_starting_equity,
    )
    managed_elapsed = time.perf_counter() - managed_started
    print(
        f"Managed SSR-aligned run finished in {managed_elapsed:.2f}s: "
        f"{managed_result.total_return_pct:.2f}% ({managed_result.total_trades} trades)",
        flush=True,
    )

    managed_pnl_by_symbol = _sum_trade_pnl_by_symbol(managed_result.trades)
    rows: list[dict[str, object]] = []
    changed_symbols: list[dict[str, object]] = []
    for symbol in SYMBOLS:
        base = baseline_by_ticker[symbol]
        independent_pnl = float(base["Independent PnL"])
        previous_managed_pnl = float(base["Managed Contribution PnL"])
        new_managed_pnl = float(managed_pnl_by_symbol.get(symbol, 0.0))
        old_diff = previous_managed_pnl - independent_pnl
        new_diff = new_managed_pnl - independent_pnl
        drift_improvement = abs(old_diff) - abs(new_diff)
        row = {
            "Ticker": symbol,
            "Strategy": base["Strategy"],
            "Independent PnL": round(independent_pnl, 6),
            "Independent Return %": round(float(base["Independent Return %"]), 6),
            "Previous Managed PnL": round(previous_managed_pnl, 6),
            "Previous Managed Return %": round(float(base["Managed Normalized Return %"]), 6),
            "SSR-Aligned Managed PnL": round(new_managed_pnl, 6),
            "SSR-Aligned Managed Return %": round((new_managed_pnl / SLEEVE_CAPITAL) * 100.0, 6),
            "Previous Managed Diff vs Independent": round(old_diff, 6),
            "SSR-Aligned Diff vs Independent": round(new_diff, 6),
            "Drift Improvement": round(drift_improvement, 6),
        }
        rows.append(row)
        if abs(new_diff - old_diff) > 1e-9:
            changed_symbols.append(row)

    new_total_pnl = float(sum(managed_pnl_by_symbol.values()))
    previous_total_pnl = float(baseline_summary["managed_total_pnl"])
    independent_total_pnl = float(baseline_summary["independent_total_pnl"])
    new_total_return = (new_total_pnl / total_starting_equity) * 100.0
    previous_total_return = float(baseline_summary["managed_total_return_pct"])

    summary = {
        "window_start": str(WINDOW_START),
        "window_end": str(WINDOW_END),
        "ticker_count": len(SYMBOLS),
        "sleeve_capital": SLEEVE_CAPITAL,
        "total_initial_equity": total_starting_equity,
        "independent_total_pnl": independent_total_pnl,
        "previous_managed_total_pnl": previous_total_pnl,
        "previous_managed_total_return_pct": previous_total_return,
        "ssr_aligned_managed_total_pnl": new_total_pnl,
        "ssr_aligned_managed_total_return_pct": new_total_return,
        "previous_capture_vs_independent_pct": (previous_total_pnl / independent_total_pnl) * 100.0 if abs(independent_total_pnl) > 1e-9 else None,
        "ssr_aligned_capture_vs_independent_pct": (new_total_pnl / independent_total_pnl) * 100.0 if abs(independent_total_pnl) > 1e-9 else None,
        "managed_total_trades": int(managed_result.total_trades),
        "managed_candidate_entries": int(managed_result.candidate_entries),
        "managed_skipped_entries": int(managed_result.skipped_entries),
        "managed_replaced_positions": int(managed_result.replaced_positions),
        "managed_peak_open_positions": int(managed_result.max_concurrent_positions_seen),
        "symbols_changed_after_ssr_alignment": changed_symbols,
        "timing_sec": {
            "prepare": prep_elapsed,
            "managed": managed_elapsed,
        },
    }

    stem = f"current_repo_state_portfolio_benchmark_ssr_aligned_{WINDOW_START.date()}_to_{WINDOW_END.date()}"
    csv_path = ARTIFACT_DIR / f"{stem}.csv"
    json_path = ARTIFACT_DIR / f"{stem}_summary.json"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote {csv_path}", flush=True)
    print(f"Wrote {json_path}", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

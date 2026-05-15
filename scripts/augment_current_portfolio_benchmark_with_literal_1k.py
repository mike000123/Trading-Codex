from __future__ import annotations

import csv
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.regenerate_current_portfolio_benchmark import (  # noqa: E402
    ARTIFACT_DIR,
    SLEEVE_CAPITAL,
    SYMBOLS,
    WINDOW_END,
    WINDOW_START,
    _prepare_symbol,
    _risk_manager,
)
from reporting.backtest import BacktestEngine  # noqa: E402
from strategies import get_strategy, list_strategies  # noqa: E402


BASELINE_STEM = f"current_repo_state_portfolio_benchmark_{WINDOW_START.date()}_to_{WINDOW_END.date()}"
BASELINE_CSV = ARTIFACT_DIR / f"{BASELINE_STEM}.csv"
BASELINE_JSON = ARTIFACT_DIR / f"{BASELINE_STEM}_summary.json"


def _load_baseline_rows() -> list[dict[str, str]]:
    with BASELINE_CSV.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _load_baseline_summary() -> dict:
    return json.loads(BASELINE_JSON.read_text(encoding="utf-8"))


def _run_literal_standalone_1k(prepared) -> dict[str, float | int | str]:
    strategy = get_strategy(prepared.strategy_id)(params={})
    engine = BacktestEngine(
        strategy,
        risk_manager=_risk_manager(max_open_positions=999),
        spread_pct=0.06,
        slippage_pct=0.02,
        commission_per_trade=0.0,
        enforce_rth=True,
        extended_hours=False,
        enforce_pdt=True,
        enforce_ssr=True,
        enforce_fractional=True,
        fill_diagnostic=True,
        enforce_monday_open_delay=False,
    )
    result = engine.run(
        prepared.data,
        prepared.symbol,
        leverage=1.0,
        capital_per_trade=SLEEVE_CAPITAL,
        starting_equity=SLEEVE_CAPITAL,
    )
    final_equity = float(result.equity_curve["equity"].iloc[-1]) if not result.equity_curve.empty else float(SLEEVE_CAPITAL)
    pnl = final_equity - float(SLEEVE_CAPITAL)
    return {
        "Ticker": prepared.symbol,
        "Literal 1k Trades": int(result.total_trades),
        "Literal 1k PnL": float(pnl),
        "Literal 1k End Equity": float(final_equity),
        "Literal 1k Return %": float(result.total_return_pct),
        "Literal 1k Win Rate %": float(result.win_rate_pct),
        "Literal 1k Max DD %": float(result.max_drawdown_pct),
        "Literal 1k Sharpe": float(result.sharpe_ratio),
    }


def main() -> None:
    if not BASELINE_CSV.exists():
        raise FileNotFoundError(f"Baseline benchmark CSV not found: {BASELINE_CSV}")
    if not BASELINE_JSON.exists():
        raise FileNotFoundError(f"Baseline benchmark summary not found: {BASELINE_JSON}")

    baseline_rows = _load_baseline_rows()
    baseline_summary = _load_baseline_summary()
    baseline_by_ticker = {str(row["Ticker"]).strip().upper(): row for row in baseline_rows}
    strategy_name_map = {str(item["id"]): str(item["name"]) for item in list_strategies()}

    print(
        f"Augmenting benchmark with literal $1k standalone runs for {len(SYMBOLS)} symbols "
        f"({WINDOW_START.date()} -> {WINDOW_END.date()})",
        flush=True,
    )
    prep_started = time.perf_counter()
    prepared_inputs = []
    for idx, symbol in enumerate(SYMBOLS, start=1):
        print(f"[prep {idx}/{len(SYMBOLS)}] {symbol}", flush=True)
        prepared_inputs.append(_prepare_symbol(symbol, strategy_name_map))
    prep_elapsed = time.perf_counter() - prep_started
    print(f"Prepared {len(prepared_inputs)} tickers in {prep_elapsed:.2f}s", flush=True)

    literal_started = time.perf_counter()
    literal_results: dict[str, dict[str, float | int | str]] = {}
    max_workers = min(6, len(prepared_inputs))
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="literal-1k") as executor:
        futures = {
            executor.submit(_run_literal_standalone_1k, prepared): prepared.symbol
            for prepared in prepared_inputs
        }
        completed = 0
        for future in as_completed(futures):
            symbol = futures[future]
            result = future.result()
            literal_results[symbol] = result
            completed += 1
            print(
                f"[literal {completed}/{len(prepared_inputs)}] {symbol}: "
                f"{float(result['Literal 1k Return %']):.2f}% ({float(result['Literal 1k PnL']):+.2f})",
                flush=True,
            )
    literal_elapsed = time.perf_counter() - literal_started
    print(f"Literal $1k runs finished in {literal_elapsed:.2f}s", flush=True)

    augmented_rows: list[dict[str, object]] = []
    literal_total_pnl = 0.0
    for symbol in SYMBOLS:
        base = dict(baseline_by_ticker[symbol])
        lit = literal_results[symbol]
        literal_total_pnl += float(lit["Literal 1k PnL"])
        managed_return = float(base["Managed Normalized Return %"])
        literal_return = float(lit["Literal 1k Return %"])
        base.update(
            {
                "Literal 1k Account Equity Used": round(SLEEVE_CAPITAL, 6),
                "Literal 1k Trades": int(lit["Literal 1k Trades"]),
                "Literal 1k PnL": round(float(lit["Literal 1k PnL"]), 6),
                "Literal 1k End Equity": round(float(lit["Literal 1k End Equity"]), 6),
                "Literal 1k Return %": round(literal_return, 6),
                "Literal 1k Win Rate %": round(float(lit["Literal 1k Win Rate %"]), 6),
                "Literal 1k Max DD %": round(float(lit["Literal 1k Max DD %"]), 6),
                "Literal 1k Sharpe": round(float(lit["Literal 1k Sharpe"]), 6),
                "Managed Minus Literal 1k PnL": round(float(base["Managed Contribution PnL"]) - float(lit["Literal 1k PnL"]), 6),
                "Managed Minus Literal 1k Return Pts": round(managed_return - literal_return, 6),
            }
        )
        augmented_rows.append(base)

    literal_total_initial = float(len(SYMBOLS) * SLEEVE_CAPITAL)
    literal_total_final = literal_total_initial + literal_total_pnl
    augmented_summary = dict(baseline_summary)
    augmented_summary.update(
        {
            "literal_1k_total_initial_equity": literal_total_initial,
            "literal_1k_total_pnl": literal_total_pnl,
            "literal_1k_total_final_equity": literal_total_final,
            "literal_1k_total_return_pct": (literal_total_pnl / literal_total_initial) * 100.0 if literal_total_initial else 0.0,
            "literal_1k_notes": [
                "Each ticker is run independently with starting_equity=1000 and capital_per_trade=1000 under the current exact repo state.",
                "These are the closest per-ticker figures to the way strategies were originally calibrated one by one.",
                "Compare `Literal 1k Return %` against `Managed Normalized Return %` to see how the multi-ticker allocator captures each ticker's independent small-account outcome.",
            ],
            "timing_sec": {
                **dict(baseline_summary.get("timing_sec") or {}),
                "literal_1k_independent": literal_elapsed,
                "literal_1k_prepare": prep_elapsed,
            },
        }
    )

    stem = f"current_repo_state_portfolio_benchmark_with_literal_1k_{WINDOW_START.date()}_to_{WINDOW_END.date()}"
    csv_path = ARTIFACT_DIR / f"{stem}.csv"
    json_path = ARTIFACT_DIR / f"{stem}_summary.json"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(augmented_rows[0].keys()))
        writer.writeheader()
        writer.writerows(augmented_rows)
    json_path.write_text(json.dumps(augmented_summary, indent=2), encoding="utf-8")

    print(f"Wrote {csv_path}", flush=True)
    print(f"Wrote {json_path}", flush=True)
    print(
        json.dumps(
            {
                "literal_1k_total_initial_equity": literal_total_initial,
                "literal_1k_total_pnl": literal_total_pnl,
                "literal_1k_total_final_equity": literal_total_final,
                "literal_1k_total_return_pct": augmented_summary["literal_1k_total_return_pct"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

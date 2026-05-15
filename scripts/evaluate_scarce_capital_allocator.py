from __future__ import annotations

import argparse
import csv
import json
import sys
import time
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
    _managed_engine,
    _prepare_symbol,
)
from reporting.managed_portfolio_backtest import ManagedPortfolioSymbolInput  # noqa: E402
from strategies import get_strategy, list_strategies  # noqa: E402


BASELINE_SUMMARY_PATH = (
    ARTIFACT_DIR
    / f"current_repo_state_portfolio_benchmark_{WINDOW_START.date()}_to_{WINDOW_END.date()}_summary.json"
)
LITERAL_1K_SUMMARY_PATH = (
    ARTIFACT_DIR
    / f"current_repo_state_portfolio_benchmark_with_literal_1k_{WINDOW_START.date()}_to_{WINDOW_END.date()}_summary.json"
)
STARTING_EQUITIES = [5000.0, 10000.0]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _prepare_inputs() -> list[ManagedPortfolioSymbolInput]:
    strategy_name_map = {str(item["id"]): str(item["name"]) for item in list_strategies()}
    inputs: list[ManagedPortfolioSymbolInput] = []
    for idx, symbol in enumerate(SYMBOLS, start=1):
        print(f"[prep {idx}/{len(SYMBOLS)}] {symbol}", flush=True)
        prepared = _prepare_symbol(symbol, strategy_name_map)
        inputs.append(
            ManagedPortfolioSymbolInput(
                symbol=prepared.symbol,
                strategy_id=prepared.strategy_id,
                strategy_name=prepared.strategy_name,
                strategy=get_strategy(prepared.strategy_id)(params={}),
                data=prepared.data,
            )
        )
    return inputs


def _sum_trade_pnl_by_symbol(trades) -> dict[str, float]:
    out: dict[str, float] = {}
    for trade in trades:
        sym = str(trade.symbol).strip().upper()
        out[sym] = out.get(sym, 0.0) + float(trade.pnl or 0.0)
    return out


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the managed allocator under scarce starting equity levels."
    )
    parser.add_argument(
        "--starting-equities",
        nargs="+",
        type=float,
        default=STARTING_EQUITIES,
        help="One or more starting equity values to test, e.g. --starting-equities 2000 3000",
    )
    parser.add_argument(
        "--dynamic-sizing",
        action="store_true",
        help="Enable the managed engine's bounded dynamic sizing overlay.",
    )
    parser.add_argument(
        "--allow-replacement",
        action="store_true",
        help="Allow stronger later setups to replace weaker open positions.",
    )
    parser.add_argument(
        "--replacement-edge",
        type=float,
        default=10.0,
        help="Minimum score edge percentage required before a replacement is allowed.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    starting_equities = [float(v) for v in args.starting_equities if float(v) > 0]
    if not starting_equities:
        raise ValueError("Provide at least one positive starting equity.")
    if not BASELINE_SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Missing full benchmark summary: {BASELINE_SUMMARY_PATH}")
    if not LITERAL_1K_SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Missing literal 1k summary: {LITERAL_1K_SUMMARY_PATH}")

    baseline_summary = _load_json(BASELINE_SUMMARY_PATH)
    literal_summary = _load_json(LITERAL_1K_SUMMARY_PATH)

    print(
        f"Preparing {len(SYMBOLS)} managed inputs for scarce-capital checks "
        f"({WINDOW_START.date()} -> {WINDOW_END.date()})",
        flush=True,
    )
    prep_started = time.perf_counter()
    managed_inputs = _prepare_inputs()
    prep_elapsed = time.perf_counter() - prep_started
    print(f"Prepared {len(managed_inputs)} inputs in {prep_elapsed:.2f}s", flush=True)

    rows: list[dict[str, object]] = []
    summary_runs: list[dict[str, object]] = []

    for starting_equity in starting_equities:
        max_open_positions = max(1, int(starting_equity // SLEEVE_CAPITAL))
        print(
            f"Running scarce-capital managed check: equity={starting_equity:.0f}, "
            f"cap_per_trade={SLEEVE_CAPITAL:.0f}, max_open={max_open_positions}, "
            f"dynamic_sizing={'on' if args.dynamic_sizing else 'off'}",
            flush=True,
        )
        started = time.perf_counter()
        result = _managed_engine(
            max_open_positions=max_open_positions,
            dynamic_sizing=bool(args.dynamic_sizing),
            allow_replacement=bool(args.allow_replacement),
            replacement_score_edge_pct=float(args.replacement_edge),
        ).run(
            managed_inputs,
            leverage=1.0,
            capital_per_trade=SLEEVE_CAPITAL,
            starting_equity=starting_equity,
        )
        elapsed = time.perf_counter() - started
        managed_pnl_by_symbol = _sum_trade_pnl_by_symbol(result.trades)
        total_pnl = float(sum(managed_pnl_by_symbol.values()))
        total_return_pct = float((total_pnl / starting_equity) * 100.0 if starting_equity else 0.0)
        capture_vs_full = (
            float(total_pnl / float(baseline_summary["managed_total_pnl"]) * 100.0)
            if float(baseline_summary["managed_total_pnl"]) != 0
            else None
        )
        capture_vs_literal = (
            float(total_pnl / float(literal_summary["literal_1k_total_pnl"]) * 100.0)
            if float(literal_summary["literal_1k_total_pnl"]) != 0
            else None
        )
        top_symbols = sorted(managed_pnl_by_symbol.items(), key=lambda item: item[1], reverse=True)[:10]
        print(
            f"Finished equity={starting_equity:.0f} in {elapsed:.2f}s: "
            f"{total_return_pct:.2f}% ({total_pnl:+.2f}), trades={result.total_trades}, "
            f"skipped={result.skipped_entries}, peak_open={result.max_concurrent_positions_seen}",
            flush=True,
        )

        summary_runs.append(
            {
                "starting_equity": starting_equity,
                "capital_per_trade": SLEEVE_CAPITAL,
                "max_open_positions": max_open_positions,
                "managed_total_pnl": total_pnl,
                "managed_total_return_pct": total_return_pct,
                "managed_total_trades": int(result.total_trades),
                "managed_candidate_entries": int(result.candidate_entries),
                "managed_skipped_entries": int(result.skipped_entries),
                "managed_replaced_positions": int(result.replaced_positions),
                "managed_peak_open_positions": int(result.max_concurrent_positions_seen),
                "dynamic_sizing": bool(args.dynamic_sizing),
                "allow_replacement": bool(args.allow_replacement),
                "replacement_score_edge_pct": float(args.replacement_edge),
                "capture_vs_full_36k_benchmark_pct": capture_vs_full,
                "capture_vs_literal_1k_sum_pct": capture_vs_literal,
                "top_symbol_contributors": [
                    {"ticker": sym, "pnl": pnl} for sym, pnl in top_symbols
                ],
                "elapsed_sec": elapsed,
            }
        )

        for symbol in SYMBOLS:
            sym_pnl = float(managed_pnl_by_symbol.get(symbol, 0.0))
            rows.append(
                {
                    "Starting Equity": round(starting_equity, 2),
                    "Capital per Trade": round(SLEEVE_CAPITAL, 2),
                    "Max Open Positions": int(max_open_positions),
                    "Dynamic Sizing": bool(args.dynamic_sizing),
                    "Allow Replacement": bool(args.allow_replacement),
                    "Replacement Edge %": float(args.replacement_edge),
                    "Ticker": symbol,
                    "Managed Contribution PnL": round(sym_pnl, 6),
                    "Managed Normalized Return %": round((sym_pnl / SLEEVE_CAPITAL) * 100.0, 6),
                }
            )

    output_summary = {
        "window_start": str(WINDOW_START),
        "window_end": str(WINDOW_END),
        "ticker_count": len(SYMBOLS),
        "sleeve_capital": SLEEVE_CAPITAL,
        "baseline_full_benchmark_total_pnl": float(baseline_summary["managed_total_pnl"]),
        "baseline_full_benchmark_total_return_pct": float(baseline_summary["managed_total_return_pct"]),
        "baseline_literal_1k_total_pnl": float(literal_summary["literal_1k_total_pnl"]),
        "baseline_literal_1k_total_return_pct": float(literal_summary["literal_1k_total_return_pct"]),
        "dynamic_sizing": bool(args.dynamic_sizing),
        "allow_replacement": bool(args.allow_replacement),
        "replacement_score_edge_pct": float(args.replacement_edge),
        "runs": summary_runs,
        "notes": [
            (
                "These scarce-capital checks use the current bucket-aware allocator "
                + ("with replacement ON." if args.allow_replacement else "with replacement OFF.")
            ),
            (
                "When dynamic_sizing is true, the engine may use 0.5x, 1.0x, 1.5x, or 2.0x of the base trade capital, "
                "and may scale higher when cash is abundant and only a few standout setups are active."
            ),
            "They are intended to isolate allocator quality under scarce capital with the current sizing mode.",
            "Managed Normalized Return % is still expressed per 1000-dollar base sleeve for cross-ticker comparison, while managed_total_return_pct is the actual portfolio return on the chosen starting equity.",
        ],
        "timing_sec": {
            "prepare": prep_elapsed,
            "total_managed": sum(float(item["elapsed_sec"]) for item in summary_runs),
        },
    }

    equity_suffix = "_".join(str(int(v)) for v in starting_equities)
    mode_suffix = "dynamic" if args.dynamic_sizing else "fixed"
    if args.allow_replacement:
        mode_suffix += f"_replace_on_edge_{str(int(args.replacement_edge)) if float(args.replacement_edge).is_integer() else str(args.replacement_edge).replace('.', '_')}"
    stem = f"scarce_capital_allocator_eval_{mode_suffix}_{equity_suffix}_{WINDOW_START.date()}_to_{WINDOW_END.date()}"
    csv_path = ARTIFACT_DIR / f"{stem}.csv"
    json_path = ARTIFACT_DIR / f"{stem}_summary.json"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(output_summary, indent=2), encoding="utf-8")

    print(f"Wrote {csv_path}", flush=True)
    print(f"Wrote {json_path}", flush=True)
    print(json.dumps(output_summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

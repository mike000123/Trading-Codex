from __future__ import annotations

import csv
import json
import sys
import time
import types
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _install_dummy_logger() -> None:
    if "core.logger" in sys.modules:
        return
    logger_mod = types.ModuleType("core.logger")

    class _Dummy:
        def info(self, *args, **kwargs):
            pass

        def warning(self, *args, **kwargs):
            pass

        def error(self, *args, **kwargs):
            pass

        def debug(self, *args, **kwargs):
            pass

    logger_mod.log = _Dummy()
    sys.modules["core.logger"] = logger_mod


_install_dummy_logger()

from config.settings import RiskConfig
from data.ingestion import load_forward_blended_data, prepare_strategy_data
from reporting.backtest import BacktestEngine
from reporting.managed_portfolio_backtest import (
    ManagedPortfolioBacktestEngine,
    ManagedPortfolioSymbolInput,
)
from risk.manager import RiskManager
from strategies import get_strategy, list_strategies
from ui.components import recommended_primary_strategy_id, strategy_display_name_from_id


ARTIFACT_DIR = ROOT / "artifacts" / "reports"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_START = pd.Timestamp("2024-04-04")
WINDOW_END = pd.Timestamp("2026-04-23 23:59:00")
SLEEVE_CAPITAL = 1000.0
SYMBOLS = [
    "ABNB",
    "APLD",
    "ARM",
    "AVGO",
    "AXON",
    "CHTR",
    "CPRT",
    "DASH",
    "DDOG",
    "FTNT",
    "IDXX",
    "INTC",
    "MNST",
    "MRVL",
    "MSTR",
    "NXPI",
    "PDD",
    "PYPL",
    "QCOM",
    "RBLX",
    "SNPS",
    "STX",
    "TEAM",
    "TSLA",
    "TXN",
    "WBD",
    "ZS",
    "GLD",
    "UVXY",
    "USO",
    "SPY",
    "QQQ",
    "IWM",
    "VXX",
    "VXZ",
    "XLF",
]


@dataclass
class PreparedTicker:
    symbol: str
    strategy_id: str
    strategy_name: str
    data: pd.DataFrame


@dataclass
class IndependentTickerResult:
    ticker: str
    strategy_id: str
    strategy_name: str
    independent_account_starting_equity: float
    initial_invested_amount: float
    independent_trades: int
    independent_pnl: float
    independent_end_value: float
    independent_return_pct: float
    engine_return_pct_on_total_equity: float
    win_rate_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float


def _risk_manager(max_open_positions: int = 999) -> RiskManager:
    return RiskManager(
        RiskConfig(
            max_capital_per_trade_pct=100.0,
            max_daily_loss_pct=100.0,
            max_open_positions=max_open_positions,
            default_max_loss_pct_of_capital=50.0,
        )
    )


def _independent_engine() -> BacktestEngine:
    return BacktestEngine(
        strategy=None,  # replaced in use-site
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


def _managed_engine(max_open_positions: int, **engine_overrides) -> ManagedPortfolioBacktestEngine:
    kwargs = dict(
        risk_manager=_risk_manager(max_open_positions=max_open_positions),
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
        allow_replacement=False,
        replacement_score_edge_pct=10.0,
        max_open_positions=max_open_positions,
    )
    kwargs.update(engine_overrides)
    return ManagedPortfolioBacktestEngine(**kwargs)


def _prepare_symbol(symbol: str, strategy_name_map: dict[str, str]) -> PreparedTicker:
    symbol = symbol.strip().upper()
    strategy_id = recommended_primary_strategy_id(symbol, start=WINDOW_START, end=WINDOW_END)
    strategy_name = strategy_display_name_from_id(strategy_id, symbol=symbol, strategies=list_strategies())
    strategy_cls = get_strategy(strategy_id)
    strategy = strategy_cls(params={})
    raw = load_forward_blended_data(symbol, "1m", WINDOW_START, WINDOW_END)
    if raw is None or raw.empty:
        raise RuntimeError(f"No data available for {symbol}.")
    prepared = prepare_strategy_data(
        raw,
        strategy,
        primary_symbol=symbol,
        source="forward_blend",
        interval="1m",
        start=WINDOW_START,
        end=WINDOW_END,
    )
    if prepared is None or prepared.empty:
        raise RuntimeError(f"Prepared frame is empty for {symbol}.")
    return PreparedTicker(
        symbol=symbol,
        strategy_id=strategy_id,
        strategy_name=strategy_name or strategy_name_map.get(strategy_id, strategy_id),
        data=prepared,
    )


def _run_independent(prepared: PreparedTicker, total_starting_equity: float) -> IndependentTickerResult:
    strategy_cls = get_strategy(prepared.strategy_id)
    strategy = strategy_cls(params={})
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
        starting_equity=total_starting_equity,
    )
    final_equity = float(result.equity_curve["equity"].iloc[-1]) if not result.equity_curve.empty else float(total_starting_equity)
    pnl = final_equity - float(total_starting_equity)
    return IndependentTickerResult(
        ticker=prepared.symbol,
        strategy_id=prepared.strategy_id,
        strategy_name=prepared.strategy_name,
        independent_account_starting_equity=float(total_starting_equity),
        initial_invested_amount=float(SLEEVE_CAPITAL),
        independent_trades=int(result.total_trades),
        independent_pnl=float(pnl),
        independent_end_value=float(SLEEVE_CAPITAL + pnl),
        independent_return_pct=float((pnl / SLEEVE_CAPITAL) * 100.0),
        engine_return_pct_on_total_equity=float(result.total_return_pct),
        win_rate_pct=float(result.win_rate_pct),
        max_drawdown_pct=float(result.max_drawdown_pct),
        sharpe_ratio=float(result.sharpe_ratio),
    )


def _sum_trade_pnl_by_symbol(trades) -> dict[str, float]:
    out: dict[str, float] = {}
    for trade in trades:
        sym = str(trade.symbol).strip().upper()
        out[sym] = out.get(sym, 0.0) + float(trade.pnl or 0.0)
    return out


def main() -> None:
    strategy_name_map = {str(item["id"]): str(item["name"]) for item in list_strategies()}
    total_starting_equity = float(len(SYMBOLS) * SLEEVE_CAPITAL)

    print(
        f"Preparing {len(SYMBOLS)} calibrated tickers for {WINDOW_START.date()} -> {WINDOW_END.date()} "
        f"with sleeve={SLEEVE_CAPITAL:.0f} and portfolio base={total_starting_equity:.0f}",
        flush=True,
    )
    prep_started = time.perf_counter()
    prepared_inputs: list[PreparedTicker] = []
    for idx, symbol in enumerate(SYMBOLS, start=1):
        print(f"[prep {idx}/{len(SYMBOLS)}] {symbol}", flush=True)
        prepared_inputs.append(_prepare_symbol(symbol, strategy_name_map))
    prep_elapsed = time.perf_counter() - prep_started
    print(f"Prepared {len(prepared_inputs)} tickers in {prep_elapsed:.2f}s", flush=True)

    print("Running independent sleeve backtests...", flush=True)
    ind_started = time.perf_counter()
    independent_results: list[IndependentTickerResult] = []
    max_workers = min(6, len(prepared_inputs))
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="ind-bt") as executor:
        futures = {
            executor.submit(_run_independent, prepared, total_starting_equity): prepared.symbol
            for prepared in prepared_inputs
        }
        for future in as_completed(futures):
            symbol = futures[future]
            res = future.result()
            independent_results.append(res)
            print(
                f"[ind {len(independent_results)}/{len(prepared_inputs)}] {symbol}: "
                f"{res.independent_return_pct:.2f}% ({res.independent_pnl:+.2f})",
                flush=True,
            )
    independent_results.sort(key=lambda r: SYMBOLS.index(r.ticker))
    ind_elapsed = time.perf_counter() - ind_started
    print(f"Independent runs finished in {ind_elapsed:.2f}s", flush=True)

    print("Running simultaneous managed portfolio backtest...", flush=True)
    managed_inputs = [
        ManagedPortfolioSymbolInput(
            symbol=prepared.symbol,
            strategy_id=prepared.strategy_id,
            strategy_name=prepared.strategy_name,
            strategy=get_strategy(prepared.strategy_id)(params={}),
            data=prepared.data,
        )
        for prepared in prepared_inputs
    ]
    managed_started = time.perf_counter()
    managed_result = _managed_engine(max_open_positions=len(SYMBOLS)).run(
        managed_inputs,
        leverage=1.0,
        capital_per_trade=SLEEVE_CAPITAL,
        starting_equity=total_starting_equity,
    )
    managed_elapsed = time.perf_counter() - managed_started
    print(
        f"Managed run finished in {managed_elapsed:.2f}s: "
        f"{managed_result.total_return_pct:.2f}% ({managed_result.total_trades} trades)",
        flush=True,
    )

    managed_pnl_by_symbol = _sum_trade_pnl_by_symbol(managed_result.trades)
    rows: list[dict[str, Any]] = []
    for res in independent_results:
        managed_pnl = float(managed_pnl_by_symbol.get(res.ticker, 0.0))
        rows.append(
            {
                "Ticker": res.ticker,
                "Strategy": res.strategy_name,
                "Initial Invested Amount": round(SLEEVE_CAPITAL, 2),
                "Independent Account Equity Used": round(total_starting_equity, 2),
                "Independent Trades": res.independent_trades,
                "Independent PnL": round(res.independent_pnl, 6),
                "Independent End Equity": round(res.independent_end_value, 6),
                "Independent Return %": round(res.independent_return_pct, 6),
                "Managed Contribution PnL": round(managed_pnl, 6),
                "Managed Normalized End Equity": round(SLEEVE_CAPITAL + managed_pnl, 6),
                "Managed Normalized Return %": round((managed_pnl / SLEEVE_CAPITAL) * 100.0, 6),
                "Managed Capture vs Independent %": round((managed_pnl / res.independent_pnl) * 100.0, 6) if abs(res.independent_pnl) > 1e-9 else None,
                "Independent Win Rate %": round(res.win_rate_pct, 6),
                "Independent Max DD %": round(res.max_drawdown_pct, 6),
                "Independent Sharpe": round(res.sharpe_ratio, 6),
            }
        )

    independent_total_pnl = float(sum(r.independent_pnl for r in independent_results))
    independent_total_final = float(total_starting_equity + independent_total_pnl)
    managed_total_pnl = float(sum(managed_pnl_by_symbol.values()))
    managed_total_final = float(total_starting_equity + managed_total_pnl)

    summary = {
        "window_start": str(WINDOW_START),
        "window_end": str(WINDOW_END),
        "symbols": SYMBOLS,
        "ticker_count": len(SYMBOLS),
        "sleeve_capital": SLEEVE_CAPITAL,
        "total_initial_equity": total_starting_equity,
        "independent_total_pnl": independent_total_pnl,
        "independent_total_final_equity": independent_total_final,
        "independent_total_return_pct": (independent_total_pnl / total_starting_equity) * 100.0,
        "managed_total_pnl": managed_total_pnl,
        "managed_total_final_equity": managed_total_final,
        "managed_total_return_pct": (managed_total_pnl / total_starting_equity) * 100.0,
        "managed_total_trades": int(managed_result.total_trades),
        "managed_candidate_entries": int(managed_result.candidate_entries),
        "managed_skipped_entries": int(managed_result.skipped_entries),
        "managed_replaced_positions": int(managed_result.replaced_positions),
        "managed_peak_open_positions": int(managed_result.max_concurrent_positions_seen),
        "managed_capture_vs_independent_pct": (managed_total_pnl / independent_total_pnl) * 100.0 if abs(independent_total_pnl) > 1e-9 else None,
        "notes": [
            "Independent runs use the current exact repo state and current strategy auto-assignment.",
            "Each ticker uses a 1000-dollar sleeve, but account-level constraints are evaluated on the fully funded total portfolio base so PDT does not artificially choke the independent benchmark.",
            "Simultaneous managed run uses the same total starting equity, 1000-dollar capital-per-trade, replacement OFF, and max open positions equal to ticker count.",
            "Managed and standalone runs both enforce the same SSR heuristic.",
            "Standalone and managed runs both realize any still-open position at the final available close so ending equity is aligned across modes.",
        ],
        "timing_sec": {
            "prepare": prep_elapsed,
            "independent": ind_elapsed,
            "managed": managed_elapsed,
        },
    }

    stem = f"current_repo_state_portfolio_benchmark_{WINDOW_START.date()}_to_{WINDOW_END.date()}"
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

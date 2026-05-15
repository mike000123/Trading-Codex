from __future__ import annotations

import argparse
import csv
import json
import math
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

    dummy = _Dummy()
    logger_mod.log = dummy
    logger_mod.logger = dummy
    sys.modules["core.logger"] = logger_mod


_install_dummy_logger()

from config.settings import RiskConfig  # noqa: E402
from data.ingestion import load_forward_blended_data, prepare_strategy_data  # noqa: E402
from reporting.backtest import BacktestEngine  # noqa: E402
from risk.manager import RiskManager  # noqa: E402
from strategies import get_strategy  # noqa: E402


ARTIFACT_DIR = ROOT / "artifacts" / "reports"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_START = pd.Timestamp("2024-04-04")
WINDOW_END = pd.Timestamp("2026-04-23 23:59:00")
SLEEVE_CAPITAL = 1000.0
TOTAL_EQUITY = 36000.0

CURRENT_SYMBOLS = [
    "ABNB", "APLD", "ARM", "AVGO", "AXON", "CHTR", "CPRT", "DASH", "DDOG",
    "FTNT", "IDXX", "INTC", "MNST", "MRVL", "MSTR", "NXPI", "PDD", "PYPL",
    "QCOM", "RBLX", "SNPS", "STX", "TEAM", "TSLA", "TXN", "WBD", "ZS", "GLD",
    "UVXY", "USO", "SPY", "QQQ", "IWM", "VXX", "VXZ", "XLF", "SLV", "UUP", "XLE",
]

CANDIDATES = ["IEF", "UUP", "SLV", "TLT", "GDX"]


@dataclass
class ConfigEval:
    symbol: str
    strategy_id: str
    label: str
    total_return_pct: float
    engine_return_pct: float
    pnl: float
    end_equity: float
    max_drawdown_pct: float
    sharpe_ratio: float
    win_rate_pct: float
    total_trades: int
    active_days: int
    outside_current_days: int
    outside_current_ratio_pct: float
    fit_score: float
    params: dict[str, Any]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate potential new independent sleeves against the current active universe."
    )
    parser.add_argument(
        "--candidates",
        nargs="+",
        default=CANDIDATES,
        help="Symbols to evaluate sequentially, e.g. --candidates XLE XLV XLU",
    )
    return parser.parse_args()


def _risk_manager(max_open_positions: int = 999) -> RiskManager:
    return RiskManager(
        RiskConfig(
            max_capital_per_trade_pct=100.0,
            max_daily_loss_pct=100.0,
            max_open_positions=max_open_positions,
            default_max_loss_pct_of_capital=50.0,
        )
    )


def _safe_sharpe(value: float) -> float:
    return 0.0 if math.isnan(float(value)) else float(value)


def _trade_active_days(trades) -> set[pd.Timestamp]:
    active: set[pd.Timestamp] = set()
    for trade in trades:
        entry = pd.Timestamp(getattr(trade, "entry_time", None))
        if pd.isna(entry):
            continue
        exit_time = getattr(trade, "exit_time", None)
        exit_ts = pd.Timestamp(exit_time) if exit_time is not None else entry
        start_day = entry.normalize()
        end_day = exit_ts.normalize()
        if end_day < start_day:
            end_day = start_day
        for day in pd.date_range(start_day, end_day, freq="D"):
            active.add(pd.Timestamp(day).normalize())
    return active


def _prepare_data(symbol: str, strategy_id: str, params: dict[str, Any]) -> pd.DataFrame:
    strategy_cls = get_strategy(strategy_id)
    strategy = strategy_cls(params=dict(params))
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
        raise RuntimeError(f"Prepared frame is empty for {symbol} using {strategy_id}.")
    return prepared


def _run_eval(
    symbol: str,
    strategy_id: str,
    label: str,
    params: dict[str, Any],
    baseline_active_days: set[pd.Timestamp],
) -> ConfigEval:
    strategy_cls = get_strategy(strategy_id)
    strategy = strategy_cls(params=dict(params))
    prepared = _prepare_data(symbol, strategy_id, params)
    result = BacktestEngine(
        strategy,
        risk_manager=_risk_manager(),
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
    ).run(
        prepared,
        symbol,
        leverage=1.0,
        capital_per_trade=SLEEVE_CAPITAL,
        starting_equity=TOTAL_EQUITY,
    )
    end_equity = float(result.equity_curve["equity"].iloc[-1]) if not result.equity_curve.empty else TOTAL_EQUITY
    pnl = end_equity - TOTAL_EQUITY
    active_days = _trade_active_days(result.trades)
    outside_current = active_days - baseline_active_days
    outside_ratio = (len(outside_current) / len(active_days) * 100.0) if active_days else 0.0
    sharpe = _safe_sharpe(float(result.sharpe_ratio))
    fit_score = (
        float(pnl / SLEEVE_CAPITAL * 100.0)
        + sharpe * 8.0
        - abs(float(result.max_drawdown_pct)) * 2.0
        + min(float(result.total_trades), 40.0) * 0.05
        + min(len(outside_current), 40) * 0.6
        + outside_ratio * 0.35
    )
    return ConfigEval(
        symbol=symbol,
        strategy_id=strategy_id,
        label=label,
        total_return_pct=float((pnl / SLEEVE_CAPITAL) * 100.0),
        engine_return_pct=float(result.total_return_pct),
        pnl=float(pnl),
        end_equity=float(SLEEVE_CAPITAL + pnl),
        max_drawdown_pct=float(result.max_drawdown_pct),
        sharpe_ratio=sharpe,
        win_rate_pct=float(result.win_rate_pct),
        total_trades=int(result.total_trades),
        active_days=len(active_days),
        outside_current_days=len(outside_current),
        outside_current_ratio_pct=float(outside_ratio),
        fit_score=float(fit_score),
        params=dict(params),
    )


def _default_params(strategy_id: str, symbol: str | None = None) -> dict[str, Any]:
    strategy_cls = get_strategy(strategy_id)
    strategy = strategy_cls(params={})
    if symbol and hasattr(strategy, "effective_default_params"):
        return dict(strategy.effective_default_params(symbol=symbol))
    return dict(strategy.default_params())


def _baseline_active_days_for_current_universe() -> set[pd.Timestamp]:
    baseline: set[pd.Timestamp] = set()

    def _worker(symbol: str) -> set[pd.Timestamp]:
        strategy_id = "bollinger_rsi" if symbol in {"GLD", "UVXY", "USO", "SPY", "QQQ", "IWM", "VXX", "VXZ", "XLF", "APLD"} else "earnings_event_hybrid"
        if symbol == "APLD":
            strategy_id = "earnings_event_hybrid"
        params = _default_params(strategy_id, symbol if strategy_id == "bollinger_rsi" else None)
        strategy_cls = get_strategy(strategy_id)
        strategy = strategy_cls(params=dict(params))
        prepared = _prepare_data(symbol, strategy_id, params)
        result = BacktestEngine(
            strategy,
            risk_manager=_risk_manager(),
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
        ).run(
            prepared,
            symbol,
            leverage=1.0,
            capital_per_trade=SLEEVE_CAPITAL,
            starting_equity=TOTAL_EQUITY,
        )
        return _trade_active_days(result.trades)

    with ThreadPoolExecutor(max_workers=min(6, len(CURRENT_SYMBOLS)), thread_name_prefix="baseline-days") as executor:
        futures = {executor.submit(_worker, symbol): symbol for symbol in CURRENT_SYMBOLS}
        for future in as_completed(futures):
            symbol = futures[future]
            baseline.update(future.result())
            print(f"[baseline days] {symbol}", flush=True)
    return baseline


def _candidate_configs(symbol: str) -> list[tuple[str, str, dict[str, Any]]]:
    configs: list[tuple[str, str, dict[str, Any]]] = []
    for sid in ("atr_rsi", "ma_crossover", "macd_crossover"):
        configs.append((sid, f"{sid}_default", _default_params(sid)))
    if symbol in {"IEF", "UUP"}:
        configs.append(("rsi_threshold", "rsi_threshold_default", _default_params("rsi_threshold")))
    configs.append(("bollinger_rsi", "bollinger_generic", _default_params("bollinger_rsi")))
    for family_symbol in ("GLD", "USO", "SPY", "QQQ", "IWM", "UVXY", "VXX", "VXZ", "XLF", "XLE"):
        configs.append(("bollinger_rsi", f"bollinger_family_{family_symbol.lower()}", _default_params("bollinger_rsi", family_symbol)))
    return configs


def _decision(best: ConfigEval) -> str:
    if best.total_return_pct >= 8.0 and best.outside_current_days >= 20 and best.outside_current_ratio_pct >= 15.0:
        return "keep"
    if best.total_return_pct >= 4.0 and best.outside_current_days >= 10 and best.outside_current_ratio_pct >= 10.0:
        return "watchlist"
    return "discard"


def main() -> None:
    args = _parse_args()
    candidates = [str(sym).strip().upper() for sym in args.candidates if str(sym).strip()]
    if not candidates:
        raise ValueError("Provide at least one candidate symbol.")
    started = time.perf_counter()
    print(
        f"Building current-universe activity baseline for {WINDOW_START.date()} -> {WINDOW_END.date()}...",
        flush=True,
    )
    baseline_days = _baseline_active_days_for_current_universe()
    print(f"Baseline active days across current 36 sleeves: {len(baseline_days)}", flush=True)

    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "window_start": str(WINDOW_START),
        "window_end": str(WINDOW_END),
        "current_universe_count": len(CURRENT_SYMBOLS),
        "candidate_order": list(candidates),
        "baseline_active_day_count": len(baseline_days),
        "candidates": {},
    }

    for idx, symbol in enumerate(candidates, start=1):
        print(f"[candidate {idx}/{len(candidates)}] {symbol}", flush=True)
        evals: list[ConfigEval] = []
        for strategy_id, label, params in _candidate_configs(symbol):
            try:
                result = _run_eval(symbol, strategy_id, label, params, baseline_days)
                evals.append(result)
                print(
                    f"  {label}: return={result.total_return_pct:.2f}% trades={result.total_trades} "
                    f"outside_days={result.outside_current_days} outside_ratio={result.outside_current_ratio_pct:.1f}%",
                    flush=True,
                )
            except Exception as exc:
                print(f"  {label}: skipped ({exc})", flush=True)
        if not evals:
            summary["candidates"][symbol] = {"decision": "discard", "reason": "No valid evaluations."}
            continue
        ranked = sorted(evals, key=lambda item: item.fit_score, reverse=True)
        best = ranked[0]
        summary["candidates"][symbol] = {
            "decision": _decision(best),
            "best": asdict(best),
            "top_configs": [asdict(item) for item in ranked[:10]],
        }
        for item in ranked:
            rows.append(
                {
                    "Symbol": item.symbol,
                    "Decision": _decision(best),
                    "Strategy": item.strategy_id,
                    "Config": item.label,
                    "Literal Return % (36k-context sleeve)": round(item.total_return_pct, 6),
                    "Engine Return % on 36k": round(item.engine_return_pct, 6),
                    "PnL": round(item.pnl, 6),
                    "End Equity": round(item.end_equity, 6),
                    "Trades": item.total_trades,
                    "Win Rate %": round(item.win_rate_pct, 6),
                    "Max DD %": round(item.max_drawdown_pct, 6),
                    "Sharpe": round(item.sharpe_ratio, 6),
                    "Active Days": item.active_days,
                    "Outside Current 36 Days": item.outside_current_days,
                    "Outside Current 36 Ratio %": round(item.outside_current_ratio_pct, 6),
                    "Fit Score": round(item.fit_score, 6),
                }
            )

    summary["elapsed_sec"] = time.perf_counter() - started
    stem_suffix = "_".join(candidates)
    csv_path = ARTIFACT_DIR / f"independent_sleeve_candidate_eval_{stem_suffix}_{WINDOW_START.date()}_to_{WINDOW_END.date()}.csv"
    json_path = ARTIFACT_DIR / f"independent_sleeve_candidate_eval_{stem_suffix}_{WINDOW_START.date()}_to_{WINDOW_END.date()}_summary.json"
    if rows:
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

from __future__ import annotations

import argparse
import copy
import json
import sys
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

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
from core.models import Signal, SignalAction
from data.ingestion import prepare_strategy_data
from reporting.backtest import BacktestEngine
from risk.manager import RiskManager
from strategies import get_strategy
from strategies.base import BaseStrategy


ARTIFACT_DIR = ROOT / "artifacts" / "optimization"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
APLD_CACHE = ROOT / "data_cache" / "alpaca" / "APLD" / "1Min.csv"


@dataclass
class FilterResult:
    variant: str
    total_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    total_trades: int
    blocked_longs: int
    blocked_tuesday_longs: int
    remaining_tuesday_long_trades: int
    remaining_tuesday_long_mean_ret: float | None


class _PrecomputedFilterStrategy(BaseStrategy):
    strategy_id = "precomputed_apld_long_context"
    name = "Precomputed APLD Long Context Filter"
    description = "Research wrapper around the APLD overlay with extra long-side context gates."

    def __init__(self, base_strategy: BaseStrategy, actions: list[SignalAction], meta: list[dict[str, Any]]) -> None:
        super().__init__(params=dict(base_strategy.params))
        self.base_strategy = base_strategy
        self._actions = list(actions)
        self._meta = [copy.deepcopy(item) for item in meta]

    def generate_signal(self, data: pd.DataFrame, symbol: str) -> Signal:
        idx = len(data) - 1
        meta = self._meta[idx]
        return Signal(
            strategy_id=self.base_strategy.strategy_id,
            symbol=symbol,
            action=self._actions[idx],
            confidence=1.0,
            suggested_tp=meta.get("suggested_tp"),
            suggested_sl=meta.get("suggested_sl"),
            metadata=copy.deepcopy(meta.get("metadata") or {}),
        )

    def generate_signals_bulk(self, data: pd.DataFrame, symbol: str) -> tuple[list, list]:
        return list(self._actions), [copy.deepcopy(item) for item in self._meta]


def _load_apld_prices(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    prices = pd.read_csv(APLD_CACHE)
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
    prices = prices.dropna(subset=["date", "open", "high", "low", "close"])
    prices = prices[(prices["date"] >= start) & (prices["date"] <= end)].sort_values("date").reset_index(drop=True)
    if prices.empty:
        raise SystemExit("No cached APLD rows were found for the requested window.")
    return prices


def _remaining_tuesday_longs(result) -> tuple[int, float | None]:
    tuesday_longs: list[float] = []
    for trade in result.trades:
        ts = pd.Timestamp(trade.entry_time)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        entry_et = ts.tz_convert("America/New_York")
        if entry_et.day_name() != "Tuesday":
            continue
        if "regime=apld_btc_confirm_long" not in (trade.notes or ""):
            continue
        tuesday_longs.append(float(trade.leveraged_return_pct or 0.0))
    if not tuesday_longs:
        return 0, None
    return len(tuesday_longs), float(sum(tuesday_longs) / len(tuesday_longs))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate extra long-side context filters for the APLD BTC-opening overlay.")
    parser.add_argument("--start", default="2024-04-04", help="Backtest start date (default: 2024-04-04).")
    parser.add_argument("--end", default="2026-05-01", help="Backtest end date (default: 2026-05-01).")
    args = parser.parse_args()

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    prices = _load_apld_prices(start, end)
    cls = get_strategy("bollinger_rsi")
    base_params = dict(cls(params={}).effective_default_params(symbol="APLD"))
    base_strategy = cls(params=base_params)
    prepared = prepare_strategy_data(
        prices,
        base_strategy,
        primary_symbol="APLD",
        source="alpaca",
        interval="1Min",
        start=prices["date"].min(),
        end=prices["date"].max(),
    )
    actions, meta = base_strategy.generate_signals_bulk(prepared, "APLD")
    prepared_dates = pd.to_datetime(prepared["date"], utc=True, errors="coerce").dt.tz_convert("America/New_York")

    qqq_close_from_open: list[float | None] = []
    for _, row in prepared.iterrows():
        benchmark_open = row.get("benchmark_open")
        benchmark_close = row.get("benchmark_close")
        if pd.notna(benchmark_open) and pd.notna(benchmark_close) and float(benchmark_open) != 0.0:
            qqq_close_from_open.append((float(benchmark_close) / float(benchmark_open) - 1.0) * 100.0)
        else:
            qqq_close_from_open.append(None)

    filter_defs: list[tuple[str, Callable[[dict[str, Any], float | None], bool]]] = [
        ("baseline", lambda md, qqq: True),
        ("qqq_nonneg", lambda md, qqq: qqq is not None and qqq >= 0.0),
        ("qqq_005", lambda md, qqq: qqq is not None and qqq >= 0.05),
        (
            "qqqweak_confirm05",
            lambda md, qqq: (
                qqq is None
                or qqq >= 0.0
                or float(md.get("confirm_close_from_open_pct") or -999.0) >= 0.5
            ),
        ),
        (
            "qqqweak_rebound15",
            lambda md, qqq: (
                qqq is None
                or qqq >= 0.0
                or float(md.get("rebound_from_trough_pct") or -999.0) >= 1.5
            ),
        ),
        (
            "qqqweak_confirm05_rebound15",
            lambda md, qqq: (
                qqq is None
                or qqq >= 0.0
                or (
                    float(md.get("confirm_close_from_open_pct") or -999.0) >= 0.5
                    and float(md.get("rebound_from_trough_pct") or -999.0) >= 1.5
                )
            ),
        ),
        (
            "qqqweak_confirm025_rebound15",
            lambda md, qqq: (
                qqq is None
                or qqq >= 0.0
                or (
                    float(md.get("confirm_close_from_open_pct") or -999.0) >= 0.25
                    and float(md.get("rebound_from_trough_pct") or -999.0) >= 1.5
                )
            ),
        ),
        (
            "qqqweak_confirm05_rebound10",
            lambda md, qqq: (
                qqq is None
                or qqq >= 0.0
                or (
                    float(md.get("confirm_close_from_open_pct") or -999.0) >= 0.5
                    and float(md.get("rebound_from_trough_pct") or -999.0) >= 1.0
                )
            ),
        ),
        (
            "qqq_nonneg_confirm025",
            lambda md, qqq: (
                qqq is not None
                and qqq >= 0.0
                and float(md.get("confirm_close_from_open_pct") or -999.0) >= 0.25
            ),
        ),
        (
            "qqq_nonneg_rebound15",
            lambda md, qqq: (
                qqq is not None
                and qqq >= 0.0
                and float(md.get("rebound_from_trough_pct") or -999.0) >= 1.5
            ),
        ),
        (
            "qqq_nonneg_confirm025_rebound15",
            lambda md, qqq: (
                qqq is not None
                and qqq >= 0.0
                and float(md.get("confirm_close_from_open_pct") or -999.0) >= 0.25
                and float(md.get("rebound_from_trough_pct") or -999.0) >= 1.5
            ),
        ),
    ]

    risk = RiskManager(
        RiskConfig(
            max_capital_per_trade_pct=100.0,
            max_daily_loss_pct=100.0,
            max_open_positions=999,
            default_max_loss_pct_of_capital=50.0,
        )
    )

    results: list[FilterResult] = []
    for name, keep_fn in filter_defs:
        filt_actions = list(actions)
        filt_meta = [copy.deepcopy(item) for item in meta]
        blocked_longs = 0
        blocked_tuesday_longs = 0

        for idx, action in enumerate(filt_actions):
            if getattr(action, "value", str(action)) != "BUY":
                continue
            metadata = dict(filt_meta[idx].get("metadata") or {})
            if metadata.get("regime") != "apld_btc_confirm_long":
                continue
            if keep_fn(metadata, qqq_close_from_open[idx]):
                continue
            blocked_longs += 1
            if prepared_dates.iloc[idx].day_name() == "Tuesday":
                blocked_tuesday_longs += 1
            filt_actions[idx] = SignalAction.HOLD
            metadata["blocked_by_extra_filter"] = name
            filt_meta[idx]["metadata"] = metadata
            filt_meta[idx]["suggested_tp"] = None
            filt_meta[idx]["suggested_sl"] = None

        strategy = _PrecomputedFilterStrategy(base_strategy, filt_actions, filt_meta)
        result = BacktestEngine(
            strategy,
            risk_manager=risk,
            spread_pct=0.06,
            slippage_pct=0.02,
            commission_per_trade=0.0,
            enforce_monday_open_delay=False,
        ).run(
            prepared,
            "APLD",
            leverage=1.0,
            capital_per_trade=1000.0,
            starting_equity=1000.0,
        )

        remaining_tuesday_long_trades, remaining_tuesday_long_mean_ret = _remaining_tuesday_longs(result)
        results.append(
            FilterResult(
                variant=name,
                total_return_pct=float(result.total_return_pct),
                max_drawdown_pct=float(result.max_drawdown_pct),
                sharpe_ratio=float(result.sharpe_ratio),
                total_trades=int(result.total_trades),
                blocked_longs=blocked_longs,
                blocked_tuesday_longs=blocked_tuesday_longs,
                remaining_tuesday_long_trades=remaining_tuesday_long_trades,
                remaining_tuesday_long_mean_ret=remaining_tuesday_long_mean_ret,
            )
        )

    ranked = sorted(results, key=lambda row: (row.total_return_pct, row.sharpe_ratio), reverse=True)
    payload = {
        "symbol": "APLD",
        "window": {
            "start": pd.Timestamp(start).isoformat(),
            "end": pd.Timestamp(end).isoformat(),
        },
        "prepared_rows": int(len(prepared)),
        "variants_ranked": [asdict(item) for item in ranked],
        "notes": [
            "This pass focuses only on the long side of the APLD BTC-opening overlay, because Tuesday drag was concentrated in apld_btc_confirm_long trades.",
            "The new context feature tested here is early QQQ tape confirmation, optionally combined with stricter APLD reclaim-quality thresholds.",
            "These are research-only filters; they do not change the live strategy unless explicitly promoted later.",
        ],
    }

    out_json = ARTIFACT_DIR / "apld_long_context_filter_analysis.json"
    out_csv = ARTIFACT_DIR / "apld_long_context_filter_variants.csv"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pd.DataFrame([asdict(item) for item in ranked]).to_csv(out_csv, index=False)

    print(f"Prepared rows: {len(prepared)}")
    for item in ranked:
        print(
            f"{item.variant}: return={item.total_return_pct:.3f}% "
            f"dd={item.max_drawdown_pct:.3f}% sharpe={item.sharpe_ratio:.4f} "
            f"trades={item.total_trades} blocked_longs={item.blocked_longs}"
        )
    print(f"Wrote results to {out_json}")


if __name__ == "__main__":
    main()

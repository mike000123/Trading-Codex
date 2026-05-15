from __future__ import annotations

import argparse
import copy
import itertools
import json
import sys
import types
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
from core.models import Signal, SignalAction
from reporting.backtest import BacktestEngine
from risk.manager import RiskManager
from strategies import get_strategy
from strategies.base import BaseStrategy
from data.ingestion import prepare_strategy_data


ARTIFACT_DIR = ROOT / "artifacts" / "optimization"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

APLD_CACHE = ROOT / "data_cache" / "alpaca" / "APLD" / "1Min.csv"
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


@dataclass
class VariantResult:
    label: str
    blocked_weekdays: list[str]
    total_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    win_rate_pct: float
    total_trades: int
    avg_win_pct: float
    avg_loss_pct: float
    score: float


class PrecomputedWeekdayGateStrategy(BaseStrategy):
    strategy_id = "precomputed_weekday_gate"
    name = "Precomputed Weekday Gate"
    description = "Research wrapper that mutes precomputed entry signals on selected weekdays."

    def __init__(
        self,
        *,
        base_strategy: BaseStrategy,
        dates: pd.Series,
        actions: list[SignalAction],
        meta: list[dict[str, Any]],
        blocked_weekdays: set[str],
    ) -> None:
        super().__init__(params=dict(base_strategy.params))
        self.base_strategy = base_strategy
        self._dates = pd.to_datetime(dates, utc=True, errors="coerce")
        self._actions = list(actions)
        self._meta = [copy.deepcopy(item) for item in meta]
        self._blocked_weekdays = set(blocked_weekdays)
        self._weekday_names = self._dates.dt.tz_convert("America/New_York").dt.day_name().fillna("")

    def _is_blocked_index(self, idx: int) -> bool:
        if idx < 0 or idx >= len(self._weekday_names):
            return False
        return self._weekday_names.iloc[idx] in self._blocked_weekdays

    def _masked_meta(self, idx: int) -> dict[str, Any]:
        base_meta = copy.deepcopy(self._meta[idx]) if 0 <= idx < len(self._meta) else {}
        metadata = dict(base_meta.get("metadata") or {})
        metadata["weekday_gate_blocked"] = True
        metadata["blocked_weekday"] = self._weekday_names.iloc[idx] if 0 <= idx < len(self._weekday_names) else None
        base_meta["metadata"] = metadata
        base_meta["suggested_tp"] = None
        base_meta["suggested_sl"] = None
        return base_meta

    def generate_signal(self, data: pd.DataFrame, symbol: str) -> Signal:
        idx = len(data) - 1
        action = self._actions[idx]
        meta = self._meta[idx]
        if action != SignalAction.HOLD and self._is_blocked_index(idx):
            muted = self._masked_meta(idx)
            return Signal(
                strategy_id=self.base_strategy.strategy_id,
                symbol=symbol,
                action=SignalAction.HOLD,
                confidence=1.0,
                suggested_tp=muted.get("suggested_tp"),
                suggested_sl=muted.get("suggested_sl"),
                metadata=muted.get("metadata") or {},
            )
        return Signal(
            strategy_id=self.base_strategy.strategy_id,
            symbol=symbol,
            action=action,
            confidence=1.0,
            suggested_tp=meta.get("suggested_tp"),
            suggested_sl=meta.get("suggested_sl"),
            metadata=copy.deepcopy(meta.get("metadata") or {}),
        )

    def generate_signals_bulk(self, data: pd.DataFrame, symbol: str) -> tuple[list, list]:
        actions = list(self._actions)
        meta = [copy.deepcopy(item) for item in self._meta]
        if not self._blocked_weekdays:
            return actions, meta
        for idx, action in enumerate(actions):
            if action == SignalAction.HOLD:
                continue
            if not self._is_blocked_index(idx):
                continue
            actions[idx] = SignalAction.HOLD
            meta[idx] = self._masked_meta(idx)
        return actions, meta

    def default_params(self) -> dict[str, Any]:
        return self.base_strategy.default_params()

    def symbol_param_overrides(
        self,
        symbol: str,
        source: str | None = None,
        interval: str | None = None,
    ) -> dict[str, Any]:
        return self.base_strategy.symbol_param_overrides(symbol, source=source, interval=interval)

    def effective_default_params(
        self,
        symbol: str | None = None,
        source: str | None = None,
        interval: str | None = None,
    ) -> dict[str, Any]:
        return self.base_strategy.effective_default_params(symbol=symbol, source=source, interval=interval)

    def resolve_params(
        self,
        symbol: str | None = None,
        source: str | None = None,
        interval: str | None = None,
    ) -> dict[str, Any]:
        return self.base_strategy.resolve_params(symbol=symbol, source=source, interval=interval)

    def validate_params(self) -> list[str]:
        return self.base_strategy.validate_params()

    def min_warmup_bars(
        self,
        symbol: str | None = None,
        source: str | None = None,
        interval: str | None = None,
    ) -> int:
        return self.base_strategy.min_warmup_bars(symbol=symbol, source=source, interval=interval)

    def companion_symbols(
        self,
        symbol: str,
        source: str | None = None,
        interval: str | None = None,
    ) -> list[str]:
        return self.base_strategy.companion_symbols(symbol, source=source, interval=interval)

    def companion_contexts(
        self,
        symbol: str,
        source: str | None = None,
        interval: str | None = None,
    ) -> list[str]:
        return self.base_strategy.companion_contexts(symbol, source=source, interval=interval)

    def derived_contexts(
        self,
        symbol: str,
        source: str | None = None,
        interval: str | None = None,
    ) -> list[str]:
        return self.base_strategy.derived_contexts(symbol, source=source, interval=interval)


def _score_result(result) -> float:
    trades_bonus = min(float(result.total_trades), 100.0) * 0.03
    sharpe_bonus = float(result.sharpe_ratio) * 8.0
    dd_penalty = abs(float(result.max_drawdown_pct)) * 2.25
    return float(result.total_return_pct) + sharpe_bonus + trades_bonus - dd_penalty


def _load_cached_apld_prices(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if not APLD_CACHE.exists():
        raise SystemExit(f"Missing cached APLD data at {APLD_CACHE}")
    prices = pd.read_csv(APLD_CACHE)
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
    prices = prices.dropna(subset=["date", "open", "high", "low", "close"])
    prices = prices[(prices["date"] >= start) & (prices["date"] <= end)].sort_values("date").reset_index(drop=True)
    if prices.empty:
        raise SystemExit("No cached APLD rows were found for the requested window.")
    return prices


def _prepare_apld_data(prices: pd.DataFrame) -> pd.DataFrame:
    cls = get_strategy("bollinger_rsi")
    strategy = cls(params={})
    return prepare_strategy_data(
        prices,
        strategy,
        primary_symbol="APLD",
        source="alpaca",
        interval="1Min",
        start=prices["date"].min(),
        end=prices["date"].max(),
    )


def _evaluate_variant(
    prepared: pd.DataFrame,
    *,
    base_strategy: BaseStrategy,
    base_actions: list[SignalAction],
    base_meta: list[dict[str, Any]],
    blocked_weekdays: set[str],
):
    strategy = PrecomputedWeekdayGateStrategy(
        base_strategy=base_strategy,
        dates=prepared["date"],
        actions=base_actions,
        meta=base_meta,
        blocked_weekdays=blocked_weekdays,
    )
    result = BacktestEngine(
        strategy,
        risk_manager=RiskManager(
            RiskConfig(
                max_capital_per_trade_pct=100.0,
                max_daily_loss_pct=100.0,
                max_open_positions=999,
                default_max_loss_pct_of_capital=50.0,
            )
        ),
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
    label = "baseline" if not blocked_weekdays else "block_" + "_".join(sorted(blocked_weekdays))
    summary = VariantResult(
        label=label,
        blocked_weekdays=sorted(blocked_weekdays),
        total_return_pct=float(result.total_return_pct),
        max_drawdown_pct=float(result.max_drawdown_pct),
        sharpe_ratio=float(result.sharpe_ratio),
        win_rate_pct=float(result.win_rate_pct),
        total_trades=int(result.total_trades),
        avg_win_pct=float(result.avg_win_pct),
        avg_loss_pct=float(result.avg_loss_pct),
        score=_score_result(result),
    )
    return summary, result


def _trade_frame(result) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for trade in result.trades:
        if trade.entry_time is None or trade.leveraged_return_pct is None:
            continue
        entry_ts = pd.to_datetime(trade.entry_time, utc=True)
        entry_et = entry_ts.tz_convert("America/New_York")
        rows.append(
            {
                "regime": (trade.notes or "").strip(),
                "entry_time_et": entry_et,
                "entry_weekday": entry_et.day_name(),
                "leveraged_return_pct": float(trade.leveraged_return_pct or 0.0),
                "pnl": float(trade.pnl or 0.0),
                "win": float((trade.pnl or 0.0) > 0.0),
            }
        )
    return pd.DataFrame(rows)


def _weekday_summary(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    grouped = (
        frame.groupby("entry_weekday", dropna=False)
        .agg(
            trades=("pnl", "size"),
            total_pnl=("pnl", "sum"),
            avg_pnl=("pnl", "mean"),
            mean_return_pct=("leveraged_return_pct", "mean"),
            win_rate_pct=("win", lambda s: float(s.mean() * 100.0) if len(s) else 0.0),
        )
        .reset_index()
    )
    compounded_rows: list[float] = []
    for weekday in grouped["entry_weekday"]:
        subset = frame.loc[frame["entry_weekday"] == weekday, "leveraged_return_pct"].astype(float)
        compounded_rows.append(float(((1.0 + subset / 100.0).prod() - 1.0) * 100.0) if not subset.empty else 0.0)
    grouped["compounded_return_pct"] = compounded_rows
    order = {name: idx for idx, name in enumerate(WEEKDAYS)}
    grouped["sort_key"] = grouped["entry_weekday"].map(order).fillna(999)
    grouped = grouped.sort_values(["sort_key"]).drop(columns=["sort_key"])
    return grouped.to_dict(orient="records")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate weekday filters for the APLD BTC-opening overlay.")
    parser.add_argument("--start", default="2024-04-04", help="Backtest start date (default: 2024-04-04).")
    parser.add_argument("--end", default="2026-05-01", help="Backtest end date (default: 2026-05-01).")
    parser.add_argument("--max-combo-size", type=int, default=2, help="Largest weekday-block combination to test (default: 2).")
    args = parser.parse_args()

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    prices = _load_cached_apld_prices(start, end)
    prepared = _prepare_apld_data(prices)

    cls = get_strategy("bollinger_rsi")
    overlay_params = dict(cls(params={}).effective_default_params(symbol="APLD"))
    base_strategy = cls(params=overlay_params)
    base_actions, base_meta = base_strategy.generate_signals_bulk(prepared, "APLD")

    baseline_summary, baseline_result = _evaluate_variant(
        prepared,
        base_strategy=base_strategy,
        base_actions=base_actions,
        base_meta=base_meta,
        blocked_weekdays=set(),
    )
    baseline_trades = _trade_frame(baseline_result)
    weekday_summary = _weekday_summary(baseline_trades)

    variants: list[VariantResult] = [baseline_summary]
    for weekday in WEEKDAYS:
        summary, _ = _evaluate_variant(
            prepared,
            base_strategy=base_strategy,
            base_actions=base_actions,
            base_meta=base_meta,
            blocked_weekdays={weekday},
        )
        variants.append(summary)

    max_combo_size = max(int(args.max_combo_size), 1)
    if max_combo_size >= 2:
        for combo_size in range(2, max_combo_size + 1):
            for combo in itertools.combinations(WEEKDAYS, combo_size):
                summary, _ = _evaluate_variant(
                    prepared,
                    base_strategy=base_strategy,
                    base_actions=base_actions,
                    base_meta=base_meta,
                    blocked_weekdays=set(combo),
                )
                variants.append(summary)

    ranked = sorted(variants, key=lambda item: item.score, reverse=True)
    best_single = sorted(
        [item for item in variants if len(item.blocked_weekdays) == 1],
        key=lambda item: item.score,
        reverse=True,
    )
    best_combo = sorted(
        [item for item in variants if len(item.blocked_weekdays) >= 2],
        key=lambda item: item.score,
        reverse=True,
    )

    payload = {
        "symbol": "APLD",
        "window": {
            "start": pd.Timestamp(start).isoformat(),
            "end": pd.Timestamp(end).isoformat(),
        },
        "prepared_rows": int(len(prepared)),
        "baseline": asdict(baseline_summary),
        "weekday_trade_summary": weekday_summary,
        "variants_ranked": [asdict(item) for item in ranked],
        "best_single_weekday_blocks": [asdict(item) for item in best_single[:5]],
        "best_multi_weekday_blocks": [asdict(item) for item in best_combo[:5]],
        "notes": [
            "This pass keeps the APLD BTC-opening overlay unchanged and only blocks fresh entries on selected weekdays.",
            "Because the overlay closes by session end, muting entries by weekday is a fair way to test whether certain days are structurally dragging the edge down.",
            "Single-weekday blocks are easier to trust; multi-weekday combinations are more exploratory and should be treated as overfit-sensitive.",
        ],
    }

    out_json = ARTIFACT_DIR / "apld_weekday_filter_analysis.json"
    out_csv = ARTIFACT_DIR / "apld_weekday_filter_variants.csv"
    out_weekdays_csv = ARTIFACT_DIR / "apld_weekday_filter_trade_summary.csv"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pd.DataFrame([asdict(item) for item in ranked]).to_csv(out_csv, index=False)
    pd.DataFrame(weekday_summary).to_csv(out_weekdays_csv, index=False)

    print(f"Prepared rows: {len(prepared)}")
    print(
        f"Baseline: return={baseline_summary.total_return_pct:.3f}% "
        f"dd={baseline_summary.max_drawdown_pct:.3f}% "
        f"sharpe={baseline_summary.sharpe_ratio:.4f} trades={baseline_summary.total_trades}"
    )
    if best_single:
        top = best_single[0]
        print(
            f"Best single weekday block: {','.join(top.blocked_weekdays)} | "
            f"return={top.total_return_pct:.3f}% dd={top.max_drawdown_pct:.3f}% "
            f"sharpe={top.sharpe_ratio:.4f} trades={top.total_trades}"
        )
    if best_combo:
        top = best_combo[0]
        print(
            f"Best multi-weekday block: {','.join(top.blocked_weekdays)} | "
            f"return={top.total_return_pct:.3f}% dd={top.max_drawdown_pct:.3f}% "
            f"sharpe={top.sharpe_ratio:.4f} trades={top.total_trades}"
        )
    print(f"Wrote results to {out_json}")


if __name__ == "__main__":
    main()

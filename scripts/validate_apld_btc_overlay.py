from __future__ import annotations

import argparse
import json
import sys
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
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

from config.settings import settings
from core.models import Signal, SignalAction
from data.ingestion import load_from_alpaca_history, prepare_strategy_data
from strategies.base import BaseStrategy


ARTIFACT_DIR = ROOT / "artifacts" / "optimization"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

MARKET_TZ = "America/New_York"
ALPACA_SAFE_BUFFER_DAYS = 3


class _APLDBTCProbeStrategy(BaseStrategy):
    strategy_id = "apld_btc_probe"
    name = "APLD BTC Context Probe"
    description = "Companion-data validation scaffold for APLD vs BTC-USD."

    def generate_signal(self, data: pd.DataFrame, symbol: str) -> Signal:
        return Signal(
            strategy_id=self.strategy_id,
            symbol=symbol,
            action=SignalAction.HOLD,
            confidence=0.0,
            suggested_tp=None,
            suggested_sl=None,
            metadata={"probe": True},
        )

    def companion_contexts(
        self,
        symbol: str,
        source: str | None = None,
        interval: str | None = None,
    ) -> list[str]:
        return ["equity_benchmark", "crypto_benchmark"]


@dataclass(frozen=True)
class OverlayRule:
    name: str
    side: str
    gap_metric: str
    btc_threshold: float
    gap_threshold: float
    entry_offset_min: int
    peak_threshold: float | None = None
    trough_threshold: float | None = None
    pullback_threshold: float | None = None
    rebound_threshold: float | None = None
    confirm_close_threshold: float | None = None


@dataclass
class OverlayResult:
    rule_name: str
    family: str
    side: str
    stop_loss_pct: float | None
    trail_pct: float | None
    trail_activation_pct: float | None
    signals: int
    win_rate_pct: float
    mean_return_pct: float
    median_return_pct: float
    compounded_return_pct: float
    score: float


BEST_SHORT_RULE = OverlayRule(
    name="apld_btc_confirm_short",
    side="short",
    gap_metric="relative_gap_pct",
    btc_threshold=1.0,
    gap_threshold=0.5,
    entry_offset_min=15,
    peak_threshold=0.5,
    pullback_threshold=0.25,
    confirm_close_threshold=-0.25,
)

BEST_LONG_RULE = OverlayRule(
    name="apld_btc_confirm_long",
    side="long",
    gap_metric="apld_gap_pct",
    btc_threshold=1.0,
    gap_threshold=0.5,
    entry_offset_min=5,
    trough_threshold=0.5,
    rebound_threshold=0.25,
    confirm_close_threshold=-1.0,
)


def _safe_pct(current, base) -> float:
    try:
        current_f = float(current)
        base_f = float(base)
    except Exception:
        return np.nan
    if not np.isfinite(current_f) or not np.isfinite(base_f) or base_f == 0:
        return np.nan
    return (current_f / base_f - 1.0) * 100.0


def _compounded_return_pct(returns_pct: pd.Series) -> float:
    clean = pd.to_numeric(returns_pct, errors="coerce").dropna()
    if clean.empty:
        return np.nan
    return float((np.prod(1.0 + clean / 100.0) - 1.0) * 100.0)


def _score_result(compounded_return_pct: float, mean_return_pct: float, win_rate_pct: float, signals: int) -> float:
    signal_bonus = min(float(signals), 60.0) * 0.15
    return float(compounded_return_pct) + float(mean_return_pct) * 1.5 + float(win_rate_pct) * 0.03 + signal_bonus


def _load_prepared_context(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if not settings.alpaca.has_paper_credentials():
        raise SystemExit("Alpaca paper credentials are required for the APLD overlay validation pass.")
    primary = load_from_alpaca_history(
        "APLD",
        "1Min",
        start,
        end,
        settings.alpaca.paper_api_key,
        settings.alpaca.paper_secret_key,
        paper=True,
        use_cache=True,
    )
    if primary is None or primary.empty:
        raise SystemExit("Could not load APLD Alpaca history for the overlay validation pass.")
    primary["date"] = pd.to_datetime(primary["date"], errors="coerce")
    primary = primary.dropna(subset=["date", "open", "high", "low", "close"]).sort_values("date").reset_index(drop=True)
    strategy = _APLDBTCProbeStrategy(params={})
    return prepare_strategy_data(
        primary,
        strategy,
        primary_symbol="APLD",
        source="alpaca",
        interval="1Min",
        start=primary["date"].min(),
        end=primary["date"].max(),
    )


def _prepare_sessions(prepared: pd.DataFrame) -> list[dict[str, Any]]:
    work = prepared.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce", utc=True)
    work = work.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    work["date_ny"] = work["date"].dt.tz_convert(MARKET_TZ)
    work["minutes_ny"] = work["date_ny"].dt.hour * 60 + work["date_ny"].dt.minute
    work["session_date"] = work["date_ny"].dt.date
    work = work.loc[(work["minutes_ny"] >= 570) & (work["minutes_ny"] < 960)].reset_index(drop=True)
    if work.empty:
        raise SystemExit("No regular-session rows were available for APLD after preprocessing.")

    grouped = [
        grp.reset_index(drop=True)
        for _, grp in work.groupby("session_date", sort=True)
        if len(grp) >= 60
    ]
    out: list[dict[str, Any]] = []
    for prev_session, session in zip(grouped, grouped[1:]):
        prev_last = prev_session.iloc[-1]
        open_row = session.iloc[0]
        open_price = float(open_row["open"])
        benchmark_open = open_row.get("benchmark_open")
        feature: dict[str, Any] = {
            "session_date": pd.Timestamp(open_row["date_ny"]).date().isoformat(),
            "btc_overnight_pct": _safe_pct(open_row.get("crypto_close"), prev_last.get("crypto_close")),
            "apld_gap_pct": _safe_pct(open_price, prev_last["close"]),
            "qqq_gap_pct": _safe_pct(benchmark_open, prev_last.get("benchmark_close")),
        }
        feature["relative_gap_pct"] = (
            feature["apld_gap_pct"] - feature["qqq_gap_pct"]
            if np.isfinite(feature["apld_gap_pct"]) and np.isfinite(feature["qqq_gap_pct"])
            else np.nan
        )

        for minute in [5, 10, 15]:
            prefix = session.iloc[:minute]
            if prefix.empty:
                continue
            confirm_row = session.iloc[minute - 1]
            confirm_close = float(confirm_row["close"])
            feature[f"confirm_{minute}m_from_open_pct"] = _safe_pct(confirm_close, open_price)
            feature[f"peak_{minute}m_from_open_pct"] = _safe_pct(float(prefix["high"].max()), open_price)
            feature[f"trough_{minute}m_from_open_pct"] = _safe_pct(float(prefix["low"].min()), open_price)
            feature[f"confirm_{minute}m_pullback_from_peak_pct"] = _safe_pct(float(prefix["high"].max()), confirm_close)
            feature[f"confirm_{minute}m_rebound_from_trough_pct"] = _safe_pct(confirm_close, float(prefix["low"].min()))

        out.append({"session": session, "feature": feature})
    return out


def _rule_matches(feature: dict[str, Any], rule: OverlayRule) -> bool:
    if rule.side == "short":
        return bool(
            pd.notna(feature.get(rule.gap_metric))
            and float(feature["btc_overnight_pct"]) >= rule.btc_threshold
            and float(feature[rule.gap_metric]) >= rule.gap_threshold
            and float(feature[f"peak_{rule.entry_offset_min}m_from_open_pct"]) >= float(rule.peak_threshold)
            and float(feature[f"confirm_{rule.entry_offset_min}m_pullback_from_peak_pct"]) >= float(rule.pullback_threshold)
            and float(feature[f"confirm_{rule.entry_offset_min}m_from_open_pct"]) <= float(rule.confirm_close_threshold)
        )
    return bool(
        pd.notna(feature.get(rule.gap_metric))
        and float(feature["btc_overnight_pct"]) <= -rule.btc_threshold
        and float(feature[rule.gap_metric]) <= -rule.gap_threshold
        and float(feature[f"trough_{rule.entry_offset_min}m_from_open_pct"]) <= -float(rule.trough_threshold)
        and float(feature[f"confirm_{rule.entry_offset_min}m_rebound_from_trough_pct"]) >= float(rule.rebound_threshold)
        and float(feature[f"confirm_{rule.entry_offset_min}m_from_open_pct"]) >= float(rule.confirm_close_threshold)
    )


def _exit_trade(
    session: pd.DataFrame,
    *,
    side: str,
    entry_index: int,
    stop_loss_pct: float | None,
    trail_pct: float | None,
    trail_activation_pct: float | None,
) -> tuple[float, str]:
    entry_row = session.iloc[entry_index]
    entry_price = float(entry_row["close"])
    best_high = entry_price
    best_low = entry_price
    trail_active = trail_pct is not None and trail_activation_pct is not None and trail_activation_pct <= 0

    for i in range(entry_index + 1, len(session)):
        row = session.iloc[i]
        high = float(row["high"])
        low = float(row["low"])
        close = float(row["close"])

        if side == "long":
            if stop_loss_pct is not None:
                hard_stop = entry_price * (1.0 - stop_loss_pct / 100.0)
                if low <= hard_stop:
                    return hard_stop, "stop_loss"

            if trail_pct is not None and trail_activation_pct is not None:
                best_high = max(best_high, high)
                activation_price = entry_price * (1.0 + trail_activation_pct / 100.0)
                if best_high >= activation_price:
                    trail_active = True
                if trail_active:
                    trail_stop = best_high * (1.0 - trail_pct / 100.0)
                    if low <= trail_stop:
                        return trail_stop, "trailing_stop"
        else:
            if stop_loss_pct is not None:
                hard_stop = entry_price * (1.0 + stop_loss_pct / 100.0)
                if high >= hard_stop:
                    return hard_stop, "stop_loss"

            if trail_pct is not None and trail_activation_pct is not None:
                best_low = min(best_low, low)
                activation_price = entry_price * (1.0 - trail_activation_pct / 100.0)
                if best_low <= activation_price:
                    trail_active = True
                if trail_active:
                    trail_stop = best_low * (1.0 + trail_pct / 100.0)
                    if high >= trail_stop:
                        return trail_stop, "trailing_stop"

    return float(session.iloc[-1]["close"]), "session_close"


def _trade_return_pct(side: str, entry_price: float, exit_price: float) -> float:
    if side == "long":
        return _safe_pct(exit_price, entry_price)
    return -_safe_pct(exit_price, entry_price)


def _evaluate_overlay(
    session_blobs: list[dict[str, Any]],
    *,
    rule: OverlayRule,
    family: str,
    stop_loss_pct: float | None,
    trail_pct: float | None,
    trail_activation_pct: float | None,
) -> tuple[OverlayResult | None, pd.DataFrame]:
    trades: list[dict[str, Any]] = []
    entry_idx = max(rule.entry_offset_min - 1, 0)

    for blob in session_blobs:
        session = blob["session"]
        feature = blob["feature"]
        if entry_idx >= len(session):
            continue
        if not _rule_matches(feature, rule):
            continue
        entry_price = float(session.iloc[entry_idx]["close"])
        exit_price, exit_reason = _exit_trade(
            session,
            side=rule.side,
            entry_index=entry_idx,
            stop_loss_pct=stop_loss_pct,
            trail_pct=trail_pct,
            trail_activation_pct=trail_activation_pct,
        )
        ret_pct = _trade_return_pct(rule.side, entry_price, exit_price)
        trades.append(
            {
                "session_date": feature["session_date"],
                "entry_price": entry_price,
                "exit_price": exit_price,
                "exit_reason": exit_reason,
                "ret_pct": ret_pct,
            }
        )

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        return None, trades_df

    returns = pd.to_numeric(trades_df["ret_pct"], errors="coerce").dropna()
    if returns.empty:
        return None, trades_df

    compounded = _compounded_return_pct(returns)
    mean_ret = float(returns.mean())
    median_ret = float(returns.median())
    win_rate = float((returns > 0).mean() * 100.0)
    score = _score_result(compounded, mean_ret, win_rate, int(len(returns)))
    result = OverlayResult(
        rule_name=rule.name,
        family=family,
        side=rule.side,
        stop_loss_pct=stop_loss_pct,
        trail_pct=trail_pct,
        trail_activation_pct=trail_activation_pct,
        signals=int(len(returns)),
        win_rate_pct=win_rate,
        mean_return_pct=mean_ret,
        median_return_pct=median_ret,
        compounded_return_pct=float(compounded),
        score=float(score),
    )
    return result, trades_df


def _search_rule_variants(session_blobs: list[dict[str, Any]], rule: OverlayRule) -> tuple[list[OverlayResult], dict[str, pd.DataFrame]]:
    families: list[tuple[str, float | None, float | None, float | None]] = []

    for stop in [0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]:
        families.append(("stop_eod", stop, None, None))
    for trail in [0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]:
        for activation in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
            families.append(("trail_eod", None, trail, activation))
    for stop in [0.75, 1.0, 1.25, 1.5, 2.0, 2.5]:
        for trail in [0.75, 1.0, 1.25, 1.5, 2.0, 2.5]:
            for activation in [0.25, 0.5, 0.75, 1.0, 1.5]:
                families.append(("stop_trail_eod", stop, trail, activation))
    families.append(("close_only", None, None, None))

    results: list[OverlayResult] = []
    trades_lookup: dict[str, pd.DataFrame] = {}

    for family, stop, trail, activation in families:
        result, trades_df = _evaluate_overlay(
            session_blobs,
            rule=rule,
            family=family,
            stop_loss_pct=stop,
            trail_pct=trail,
            trail_activation_pct=activation,
        )
        if result is not None:
            key = f"{family}|{stop}|{trail}|{activation}"
            results.append(result)
            trades_lookup[key] = trades_df

    results.sort(key=lambda r: (r.score, r.compounded_return_pct, r.win_rate_pct), reverse=True)
    return results, trades_lookup


def _year_breakdown(trades_df: pd.DataFrame) -> dict[str, dict[str, float | int | None]]:
    if trades_df.empty:
        return {}
    work = trades_df.copy()
    work["session_date"] = pd.to_datetime(work["session_date"])

    def _stats(frame: pd.DataFrame) -> dict[str, float | int | None]:
        if frame.empty:
            return {"signals": 0, "win_rate_pct": None, "mean_return_pct": None, "compounded_return_pct": None}
        vals = pd.to_numeric(frame["ret_pct"], errors="coerce").dropna()
        if vals.empty:
            return {"signals": 0, "win_rate_pct": None, "mean_return_pct": None, "compounded_return_pct": None}
        return {
            "signals": int(len(vals)),
            "win_rate_pct": float((vals > 0).mean() * 100.0),
            "mean_return_pct": float(vals.mean()),
            "compounded_return_pct": _compounded_return_pct(vals),
        }

    return {
        "2024": _stats(work.loc[work["session_date"].dt.year == 2024]),
        "2025": _stats(work.loc[work["session_date"].dt.year == 2025]),
        "2026": _stats(work.loc[work["session_date"].dt.year == 2026]),
        "first_half": _stats(work.loc[work["session_date"] < pd.Timestamp("2025-04-01")]),
        "second_half": _stats(work.loc[work["session_date"] >= pd.Timestamp("2025-04-01")]),
    }


def _key_for_result(result: OverlayResult) -> str:
    return f"{result.family}|{result.stop_loss_pct}|{result.trail_pct}|{result.trail_activation_pct}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate research-only APLD BTC overlays with explicit SL/trail/EOD exits.")
    parser.add_argument("--start", default="2024-04-01", help="Start date for the validation window (default: 2024-04-01).")
    parser.add_argument("--end", default="", help="Optional end date; defaults to now - safe Alpaca buffer.")
    args = parser.parse_args()

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end) if str(args.end).strip() else pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=ALPACA_SAFE_BUFFER_DAYS)

    prepared = _load_prepared_context(start, end)
    session_blobs = _prepare_sessions(prepared)

    short_results, short_trades = _search_rule_variants(session_blobs, BEST_SHORT_RULE)
    long_results, long_trades = _search_rule_variants(session_blobs, BEST_LONG_RULE)

    best_short = short_results[0] if short_results else None
    best_long = long_results[0] if long_results else None
    best_short_trades = short_trades.get(_key_for_result(best_short), pd.DataFrame()) if best_short else pd.DataFrame()
    best_long_trades = long_trades.get(_key_for_result(best_long), pd.DataFrame()) if best_long else pd.DataFrame()

    payload = {
        "symbol": "APLD",
        "source": "alpaca",
        "interval": "1Min",
        "window": {
            "start": pd.Timestamp(start).isoformat(),
            "end": pd.Timestamp(end).isoformat(),
        },
        "sessions": int(len(session_blobs)),
        "prepared_rows": int(len(prepared)),
        "short_rule": asdict(BEST_SHORT_RULE),
        "long_rule": asdict(BEST_LONG_RULE),
        "best_short_overlay": asdict(best_short) if best_short else None,
        "best_long_overlay": asdict(best_long) if best_long else None,
        "best_short_breakdown": _year_breakdown(best_short_trades),
        "best_long_breakdown": _year_breakdown(best_long_trades),
        "notes": [
            "This pass turns the best short and long BTC-confirmation rules into research-only overlays with explicit hard-stop, trailing-stop, and session-close exits.",
            "The goal is not immediate promotion; it is to check whether the BTC-open edge survives once exits become more realistic.",
            "Any successful overlay from this pass should still be validated for sensitivity before being integrated into the live strategy stack.",
        ],
    }

    (ARTIFACT_DIR / "apld_btc_overlay_validation.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame([asdict(r) for r in short_results[:40]]).to_csv(
        ARTIFACT_DIR / "apld_btc_overlay_top_shorts.csv",
        index=False,
    )
    pd.DataFrame([asdict(r) for r in long_results[:40]]).to_csv(
        ARTIFACT_DIR / "apld_btc_overlay_top_longs.csv",
        index=False,
    )
    if not best_short_trades.empty:
        best_short_trades.to_csv(ARTIFACT_DIR / "apld_btc_overlay_best_short_trades.csv", index=False)
    if not best_long_trades.empty:
        best_long_trades.to_csv(ARTIFACT_DIR / "apld_btc_overlay_best_long_trades.csv", index=False)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

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
    description = "Companion-data optimization scaffold for APLD vs BTC-USD."

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


@dataclass
class RuleResult:
    label: str
    family: str
    side: str
    gap_metric: str
    btc_threshold: float
    gap_threshold: float
    entry_offset_min: int
    exit_mode: str
    min_signals: int
    stretch_threshold: float | None
    peak_threshold: float | None
    trough_threshold: float | None
    pullback_threshold: float | None
    rebound_threshold: float | None
    confirm_close_threshold: float | None
    signals: int
    win_rate_pct: float
    mean_return_pct: float
    median_return_pct: float
    compounded_return_pct: float
    score: float


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
    signal_bonus = min(float(signals), 40.0) * 0.25
    return float(compounded_return_pct) + float(mean_return_pct) * 1.75 + float(win_rate_pct) * 0.04 + signal_bonus


def _load_prepared_context(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if not settings.alpaca.has_paper_credentials():
        raise SystemExit("Alpaca paper credentials are required for the APLD confirmation pass.")
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
        raise SystemExit("Could not load APLD Alpaca history for the confirmation pass.")
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


def _row_at_minute(session: pd.DataFrame, minute_offset: int) -> pd.Series | None:
    if session.empty:
        return None
    idx = min(max(int(minute_offset) - 1, 0), len(session) - 1)
    return session.iloc[idx]


def _build_session_rows(prepared: pd.DataFrame) -> pd.DataFrame:
    work = prepared.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce", utc=True)
    work = work.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    work["date_ny"] = work["date"].dt.tz_convert(MARKET_TZ)
    work["minutes_ny"] = work["date_ny"].dt.hour * 60 + work["date_ny"].dt.minute
    work["session_date"] = work["date_ny"].dt.date
    work = work.loc[(work["minutes_ny"] >= 570) & (work["minutes_ny"] < 960)].reset_index(drop=True)
    if work.empty:
        raise SystemExit("No regular-session rows were available for APLD after preprocessing.")

    sessions = [
        grp.reset_index(drop=True)
        for _, grp in work.groupby("session_date", sort=True)
        if len(grp) >= 60
    ]
    rows: list[dict[str, Any]] = []
    confirm_minutes = [5, 10, 15]
    windows = [5, 10, 15]
    exits = {"30m": 30, "60m": 60, "close": None}

    for prev_session, curr_session in zip(sessions, sessions[1:]):
        prev_last = prev_session.iloc[-1]
        open_row = curr_session.iloc[0]
        open_price = float(open_row["open"])
        benchmark_open = open_row.get("benchmark_open")
        session_base: dict[str, Any] = {
            "session_date": pd.Timestamp(open_row["date_ny"]).date().isoformat(),
            "open_bar_utc": pd.Timestamp(open_row["date"]).isoformat(),
            "btc_overnight_pct": _safe_pct(open_row.get("crypto_close"), prev_last.get("crypto_close")),
            "apld_gap_pct": _safe_pct(open_price, prev_last["close"]),
            "qqq_gap_pct": _safe_pct(benchmark_open, prev_last.get("benchmark_close")),
        }
        session_base["relative_gap_pct"] = (
            session_base["apld_gap_pct"] - session_base["qqq_gap_pct"]
            if np.isfinite(session_base["apld_gap_pct"]) and np.isfinite(session_base["qqq_gap_pct"])
            else np.nan
        )

        for minute in windows:
            prefix = curr_session.iloc[:minute]
            if prefix.empty:
                continue
            high_w = float(prefix["high"].max())
            low_w = float(prefix["low"].min())
            session_base[f"peak_{minute}m_from_open_pct"] = _safe_pct(high_w, open_price)
            session_base[f"trough_{minute}m_from_open_pct"] = _safe_pct(low_w, open_price)

        for minute in confirm_minutes:
            confirm_row = _row_at_minute(curr_session, minute)
            if confirm_row is None:
                continue
            confirm_close = float(confirm_row["close"])
            confirm_benchmark = confirm_row.get("benchmark_close")
            session_base[f"confirm_{minute}m_close"] = confirm_close
            session_base[f"confirm_{minute}m_from_open_pct"] = _safe_pct(confirm_close, open_price)
            session_base[f"confirm_{minute}m_benchmark_from_open_pct"] = _safe_pct(confirm_benchmark, benchmark_open)
            session_base[f"confirm_{minute}m_relative_from_open_pct"] = (
                session_base[f"confirm_{minute}m_from_open_pct"] - session_base[f"confirm_{minute}m_benchmark_from_open_pct"]
                if np.isfinite(session_base[f"confirm_{minute}m_from_open_pct"])
                and np.isfinite(session_base[f"confirm_{minute}m_benchmark_from_open_pct"])
                else np.nan
            )

            prefix = curr_session.iloc[:minute]
            high_w = float(prefix["high"].max())
            low_w = float(prefix["low"].min())
            session_base[f"confirm_{minute}m_pullback_from_peak_pct"] = _safe_pct(high_w, confirm_close)
            session_base[f"confirm_{minute}m_rebound_from_trough_pct"] = _safe_pct(confirm_close, low_w)

            for exit_label, exit_minute in exits.items():
                exit_row = curr_session.iloc[-1] if exit_minute is None else _row_at_minute(curr_session, exit_minute)
                if exit_row is None:
                    continue
                apld_ret = _safe_pct(exit_row["close"], confirm_close)
                qqq_ret = _safe_pct(exit_row.get("benchmark_close"), confirm_benchmark)
                session_base[f"confirm_{minute}m_to_{exit_label}_long_pct"] = apld_ret
                session_base[f"confirm_{minute}m_to_{exit_label}_short_pct"] = -apld_ret if np.isfinite(apld_ret) else np.nan
                session_base[f"confirm_{minute}m_to_{exit_label}_relative_long_pct"] = (
                    apld_ret - qqq_ret if np.isfinite(apld_ret) and np.isfinite(qqq_ret) else np.nan
                )
                session_base[f"confirm_{minute}m_to_{exit_label}_relative_short_pct"] = (
                    -(apld_ret - qqq_ret) if np.isfinite(apld_ret) and np.isfinite(qqq_ret) else np.nan
                )

        rows.append(session_base)

    out = pd.DataFrame(rows)
    if out.empty:
        raise SystemExit("No session rows were built for the APLD confirmation pass.")
    return out


def _evaluate_raw_rule(
    sessions: pd.DataFrame,
    *,
    side: str,
    gap_metric: str,
    btc_threshold: float,
    gap_threshold: float,
    entry_minute: int,
    stretch_threshold: float,
    exit_mode: str,
    min_signals: int,
) -> RuleResult | None:
    confirm_col = f"confirm_{entry_minute}m_from_open_pct"
    ret_col = f"confirm_{entry_minute}m_to_{exit_mode}_{'short' if side == 'short' else 'long'}_pct"
    if confirm_col not in sessions.columns or ret_col not in sessions.columns:
        return None

    if side == "short":
        mask = (
            (pd.to_numeric(sessions["btc_overnight_pct"], errors="coerce") >= btc_threshold)
            & (pd.to_numeric(sessions[gap_metric], errors="coerce") >= gap_threshold)
            & (pd.to_numeric(sessions[confirm_col], errors="coerce") >= stretch_threshold)
        )
    else:
        mask = (
            (pd.to_numeric(sessions["btc_overnight_pct"], errors="coerce") <= -btc_threshold)
            & (pd.to_numeric(sessions[gap_metric], errors="coerce") <= -gap_threshold)
            & (pd.to_numeric(sessions[confirm_col], errors="coerce") <= -stretch_threshold)
        )

    subset = sessions.loc[mask].copy()
    returns = pd.to_numeric(subset[ret_col], errors="coerce").dropna()
    if len(returns) < min_signals:
        return None
    compounded = _compounded_return_pct(returns)
    mean_ret = float(returns.mean())
    median_ret = float(returns.median())
    win_rate = float((returns > 0).mean() * 100.0)
    score = _score_result(compounded, mean_ret, win_rate, int(len(returns)))
    return RuleResult(
        label=f"raw_{side}_{gap_metric}_entry_{entry_minute}m_exit_{exit_mode}",
        family="raw_gap_context",
        side=side,
        gap_metric=gap_metric,
        btc_threshold=float(btc_threshold),
        gap_threshold=float(gap_threshold),
        entry_offset_min=int(entry_minute),
        exit_mode=exit_mode,
        min_signals=int(min_signals),
        stretch_threshold=float(stretch_threshold),
        peak_threshold=None,
        trough_threshold=None,
        pullback_threshold=None,
        rebound_threshold=None,
        confirm_close_threshold=None,
        signals=int(len(returns)),
        win_rate_pct=win_rate,
        mean_return_pct=mean_ret,
        median_return_pct=median_ret,
        compounded_return_pct=float(compounded),
        score=float(score),
    )


def _evaluate_confirmation_rule(
    sessions: pd.DataFrame,
    *,
    side: str,
    gap_metric: str,
    btc_threshold: float,
    gap_threshold: float,
    confirm_minute: int,
    peak_or_trough_threshold: float,
    reversal_threshold: float,
    confirm_close_threshold: float,
    exit_mode: str,
    min_signals: int,
) -> RuleResult | None:
    confirm_col = f"confirm_{confirm_minute}m_from_open_pct"
    ret_col = f"confirm_{confirm_minute}m_to_{exit_mode}_{'short' if side == 'short' else 'long'}_pct"
    if confirm_col not in sessions.columns or ret_col not in sessions.columns:
        return None

    if side == "short":
        peak_col = f"peak_{confirm_minute}m_from_open_pct"
        reversal_col = f"confirm_{confirm_minute}m_pullback_from_peak_pct"
        mask = (
            (pd.to_numeric(sessions["btc_overnight_pct"], errors="coerce") >= btc_threshold)
            & (pd.to_numeric(sessions[gap_metric], errors="coerce") >= gap_threshold)
            & (pd.to_numeric(sessions[peak_col], errors="coerce") >= peak_or_trough_threshold)
            & (pd.to_numeric(sessions[reversal_col], errors="coerce") >= reversal_threshold)
            & (pd.to_numeric(sessions[confirm_col], errors="coerce") <= confirm_close_threshold)
        )
    else:
        trough_col = f"trough_{confirm_minute}m_from_open_pct"
        reversal_col = f"confirm_{confirm_minute}m_rebound_from_trough_pct"
        mask = (
            (pd.to_numeric(sessions["btc_overnight_pct"], errors="coerce") <= -btc_threshold)
            & (pd.to_numeric(sessions[gap_metric], errors="coerce") <= -gap_threshold)
            & (pd.to_numeric(sessions[trough_col], errors="coerce") <= -peak_or_trough_threshold)
            & (pd.to_numeric(sessions[reversal_col], errors="coerce") >= reversal_threshold)
            & (pd.to_numeric(sessions[confirm_col], errors="coerce") >= confirm_close_threshold)
        )

    subset = sessions.loc[mask].copy()
    returns = pd.to_numeric(subset[ret_col], errors="coerce").dropna()
    if len(returns) < min_signals:
        return None
    compounded = _compounded_return_pct(returns)
    mean_ret = float(returns.mean())
    median_ret = float(returns.median())
    win_rate = float((returns > 0).mean() * 100.0)
    score = _score_result(compounded, mean_ret, win_rate, int(len(returns)))
    return RuleResult(
        label=f"confirm_{side}_{gap_metric}_entry_{confirm_minute}m_exit_{exit_mode}",
        family="confirmation_overlay",
        side=side,
        gap_metric=gap_metric,
        btc_threshold=float(btc_threshold),
        gap_threshold=float(gap_threshold),
        entry_offset_min=int(confirm_minute),
        exit_mode=exit_mode,
        min_signals=int(min_signals),
        stretch_threshold=None,
        peak_threshold=float(peak_or_trough_threshold) if side == "short" else None,
        trough_threshold=float(peak_or_trough_threshold) if side == "long" else None,
        pullback_threshold=float(reversal_threshold) if side == "short" else None,
        rebound_threshold=float(reversal_threshold) if side == "long" else None,
        confirm_close_threshold=float(confirm_close_threshold),
        signals=int(len(returns)),
        win_rate_pct=win_rate,
        mean_return_pct=mean_ret,
        median_return_pct=median_ret,
        compounded_return_pct=float(compounded),
        score=float(score),
    )


def _search_raw(sessions: pd.DataFrame, side: str, min_signals: int) -> list[RuleResult]:
    results: list[RuleResult] = []
    for gap_metric in ["apld_gap_pct", "relative_gap_pct"]:
        for btc_th in [1.0, 1.5, 2.0, 3.0]:
            for gap_th in [0.5, 1.0, 1.5, 2.0]:
                for entry_minute in [1, 5, 10, 15]:
                    for stretch_th in [0.0, 0.5, 1.0, 1.5, 2.0]:
                        for exit_mode in ["30m", "60m", "close"]:
                            res = _evaluate_raw_rule(
                                sessions,
                                side=side,
                                gap_metric=gap_metric,
                                btc_threshold=btc_th,
                                gap_threshold=gap_th,
                                entry_minute=entry_minute,
                                stretch_threshold=stretch_th,
                                exit_mode=exit_mode,
                                min_signals=min_signals,
                            )
                            if res is not None:
                                results.append(res)
    results.sort(key=lambda r: (r.score, r.compounded_return_pct, r.signals), reverse=True)
    return results


def _search_confirmation(sessions: pd.DataFrame, side: str, min_signals: int) -> list[RuleResult]:
    results: list[RuleResult] = []
    for gap_metric in ["apld_gap_pct", "relative_gap_pct"]:
        for btc_th in [1.0, 1.5, 2.0, 3.0]:
            for gap_th in [0.5, 1.0, 1.5, 2.0]:
                for confirm_minute in [5, 10, 15]:
                    for peak_or_trough_th in [0.5, 1.0, 1.5, 2.0, 3.0]:
                        for reversal_th in [0.25, 0.5, 1.0, 1.5]:
                            close_thresholds = [1.5, 1.0, 0.5, 0.0, -0.25] if side == "short" else [-1.5, -1.0, -0.5, 0.0, 0.25]
                            for confirm_close_th in close_thresholds:
                                for exit_mode in ["30m", "60m", "close"]:
                                    res = _evaluate_confirmation_rule(
                                        sessions,
                                        side=side,
                                        gap_metric=gap_metric,
                                        btc_threshold=btc_th,
                                        gap_threshold=gap_th,
                                        confirm_minute=confirm_minute,
                                        peak_or_trough_threshold=peak_or_trough_th,
                                        reversal_threshold=reversal_th,
                                        confirm_close_threshold=confirm_close_th,
                                        exit_mode=exit_mode,
                                        min_signals=min_signals,
                                    )
                                    if res is not None:
                                        results.append(res)
    results.sort(key=lambda r: (r.score, r.compounded_return_pct, r.signals), reverse=True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize APLD BTC open-context rules with confirmation overlays on 1-minute Alpaca data.")
    parser.add_argument("--start", default="2024-04-01", help="Start date for the validation window (default: 2024-04-01).")
    parser.add_argument("--end", default="", help="Optional end date; defaults to now - safe Alpaca buffer.")
    parser.add_argument("--min-signals", type=int, default=8, help="Minimum number of signals required to keep a candidate rule (default: 8).")
    args = parser.parse_args()

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end) if str(args.end).strip() else pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=ALPACA_SAFE_BUFFER_DAYS)
    min_signals = max(int(args.min_signals), 3)

    prepared = _load_prepared_context(start, end)
    sessions = _build_session_rows(prepared)

    raw_shorts = _search_raw(sessions, "short", min_signals)
    raw_longs = _search_raw(sessions, "long", min_signals)
    confirm_shorts = _search_confirmation(sessions, "short", min_signals)
    confirm_longs = _search_confirmation(sessions, "long", min_signals)

    payload = {
        "symbol": "APLD",
        "source": "alpaca",
        "interval": "1Min",
        "window": {
            "start": pd.Timestamp(start).isoformat(),
            "end": pd.Timestamp(end).isoformat(),
        },
        "sessions": int(len(sessions)),
        "prepared_rows": int(len(prepared)),
        "min_signals": int(min_signals),
        "best_raw_short": asdict(raw_shorts[0]) if raw_shorts else None,
        "best_raw_long": asdict(raw_longs[0]) if raw_longs else None,
        "best_confirmation_short": asdict(confirm_shorts[0]) if confirm_shorts else None,
        "best_confirmation_long": asdict(confirm_longs[0]) if confirm_longs else None,
        "notes": [
            "This pass reuses the existing companion-data framework and tests whether BTC-driven opening dislocations become tradable only after early APLD confirmation.",
            "The raw family measures simple BTC+gap context rules; the confirmation family adds an early failure/reclaim requirement before entry.",
            "If confirmation wins decisively on the larger Alpaca-backed sample, the next step is a research overlay with explicit stops and session-close handling rather than immediate always-on integration.",
        ],
    }

    (ARTIFACT_DIR / "apld_btc_confirmation_context_results.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame([asdict(r) for r in raw_shorts[:40]]).to_csv(
        ARTIFACT_DIR / "apld_btc_confirmation_top_raw_shorts.csv",
        index=False,
    )
    pd.DataFrame([asdict(r) for r in raw_longs[:40]]).to_csv(
        ARTIFACT_DIR / "apld_btc_confirmation_top_raw_longs.csv",
        index=False,
    )
    pd.DataFrame([asdict(r) for r in confirm_shorts[:40]]).to_csv(
        ARTIFACT_DIR / "apld_btc_confirmation_top_confirm_shorts.csv",
        index=False,
    )
    pd.DataFrame([asdict(r) for r in confirm_longs[:40]]).to_csv(
        ARTIFACT_DIR / "apld_btc_confirmation_top_confirm_longs.csv",
        index=False,
    )
    sessions.to_csv(
        ARTIFACT_DIR / "apld_btc_confirmation_sessions.csv",
        index=False,
    )

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

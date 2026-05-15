"""
risk/runtime_controls.py
────────────────────────
Runtime-editable risk overrides persisted in the app DB.

These are intentionally narrow: the goal is to let the operator tune the
global portfolio allocation cap from the UI without editing `.env` / secrets
or restarting the app. All other risk controls still default to the values
loaded in config.settings unless we later promote them to editable runtime
controls too.
"""
from __future__ import annotations

from config.settings import RiskConfig, settings
from core.logger import log
from db.database import Database


_KEY = "runtime_risk_controls_v1"


def _db() -> Database:
    return Database(settings.db_path)


def _default_overrides() -> dict:
    # Runtime default requested by the user: allow the full per-run allocation
    # unless the operator deliberately tightens it.
    return {"max_capital_per_trade_pct": 100.0}


def load_runtime_risk_overrides() -> dict:
    payload = _default_overrides()
    try:
        raw = _db().load_config(_KEY) or {}
    except Exception as exc:
        log.warning(f"runtime risk override load failed — using defaults: {exc}")
        return payload
    if not isinstance(raw, dict):
        return payload
    if "max_capital_per_trade_pct" in raw:
        try:
            payload["max_capital_per_trade_pct"] = float(raw["max_capital_per_trade_pct"])
        except Exception:
            pass
    return payload


def save_runtime_risk_overrides(*, max_capital_per_trade_pct: float) -> dict:
    value = float(min(max(max_capital_per_trade_pct, 0.1), 100.0))
    payload = {"max_capital_per_trade_pct": value}
    _db().save_config(_KEY, payload)
    log.info(f"Runtime risk override saved: max_capital_per_trade_pct={value:.1f}%")
    return payload


def get_effective_risk_config() -> RiskConfig:
    merged = settings.risk.model_dump()
    merged.update(_default_overrides())
    merged.update(load_runtime_risk_overrides())
    return RiskConfig(**merged)

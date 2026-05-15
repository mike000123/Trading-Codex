"""
core/kill_switch.py
───────────────────
Global trading kill switch — a single boolean that, when tripped, blocks all
NEW entries from being routed (local sim and Alpaca paper/live alike). Exits,
trailing-stop updates, and reconciliation continue to run so open positions
can still be closed and reconciled safely.

Persistence: stored in the `configs` table of the SQLite DB under the key
"kill_switch", so it survives app restarts. Session-state is NOT the source of
truth — we re-read from DB on every render to avoid stale caches when the user
flips the switch on a different tab/page.

Shape of the stored value:
    {
        "tripped":   bool,          # True = halt new entries
        "reason":    str,           # freeform human-readable note
        "tripped_at":  str|None,    # ISO-8601 UTC when tripped was last set True
        "untripped_at": str|None,   # ISO-8601 UTC when tripped was last cleared
        "actor":     str|None,      # whoever flipped it (email / "system")
    }

Any module can call `is_tripped(db)` before routing an order to enforce the
switch. This file has NO streamlit dependency so it can be used by CLI tools
or a websocket watcher.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from core.logger import log


_KEY = "kill_switch"


def _default() -> dict:
    return {
        "tripped":      False,
        "reason":       "",
        "tripped_at":   None,
        "untripped_at": None,
        "actor":        None,
    }


def load(db) -> dict:
    """Return the current kill-switch record (never None — defaults if absent)."""
    try:
        raw = db.load_config(_KEY)
    except Exception as exc:
        log.warning(f"kill_switch.load failed — defaulting to safe-open: {exc}")
        return _default()
    if not raw:
        return _default()
    # Forward-compat: merge with defaults so new fields never break old rows.
    out = _default()
    out.update({k: v for k, v in raw.items() if k in out})
    return out


def is_tripped(db) -> bool:
    return bool(load(db).get("tripped"))


def trip(db, *, reason: str = "manual", actor: Optional[str] = None) -> dict:
    """Arm the kill switch. Idempotent — re-arming updates reason + timestamp."""
    state = load(db)
    state["tripped"]    = True
    state["reason"]     = (reason or "manual").strip()[:500]
    state["tripped_at"] = datetime.utcnow().isoformat()
    state["actor"]      = (actor or "user").strip()[:100]
    db.save_config(_KEY, state)
    log.warning(
        f"KILL SWITCH TRIPPED by {state['actor']}: {state['reason']}"
    )
    return state


def untrip(db, *, actor: Optional[str] = None) -> dict:
    """Clear the kill switch. Keeps the previous reason in history fields."""
    state = load(db)
    state["tripped"]      = False
    state["untripped_at"] = datetime.utcnow().isoformat()
    state["actor"]        = (actor or "user").strip()[:100]
    db.save_config(_KEY, state)
    log.info(f"Kill switch cleared by {state['actor']}.")
    return state

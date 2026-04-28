"""
execution/paper_worker.py
─────────────────────────
Server-side paper-trading tick worker.

Runs as a daemon thread launched at Streamlit app boot. Ticks every 60s
independent of any browser session, so phone-locked users or refreshed
pages keep getting bars processed.

Architecture
────────────
The Streamlit page (`pages.page_paper_trading`) does its work against
`st.session_state` for four mutable keys: `_RUNS`, `_CACHE`, `_SIGNALS`,
`_TRAIL`. Those reads/writes are now routed through small accessor
helpers (`_runs_state` / `_signals_state` / `_trail_state_dict` /
`_cache_state` and `_state_set`) which fall through to a thread-local
dict whenever one is bound — and to `st.session_state` otherwise.

The worker simply:

  1. Loads the four state dicts from DB into a thread-local container
     (`bind_worker_state(container)`).
  2. Reuses the page's own `_run_tick(symbol, run)` for each active run.
     Every persist-to-DB call inside that path now reads from the
     thread-local container, so the page picks up fresh state on its
     next render automatically.
  3. Unbinds the container (`bind_worker_state(None)`) so other threads
     in the process never see the worker's state.

Race protection: each run has a `_last_processed_bar` cursor in its
config. The page and worker both honour it via `_bar_positions_to_process`
so the same bar is never handled twice — whichever ticks first advances
the cursor, the other tick finds no new bars and exits cheaply.

Limitations (intentional, documented):

  * Trailing stops update via the worker just like the page; what the
    worker does NOT do is push every trail tightening to Alpaca.
    Hard SL / TP brackets are submitted at entry and broker-managed,
    so worst case the position exits at the original SL.
  * Counter-signal exits work as before; the shadow-Alpaca side is
    flattened via `OrderRouter.close_position_alpaca_paper`.
  * If the user has the Streamlit page open AND the worker is running,
    bar de-dup is the only line of defence against double-execution —
    but `_last_processed_bar` is updated atomically inside `_run_tick`
    via `_persist_runs_config`, so the second tick to arrive reads the
    advanced cursor and skips the bar.
"""
from __future__ import annotations

import threading
import time
from datetime import datetime
from typing import Optional

from config.settings import settings
from core.logger import log
from db.database import Database


_WORKER_HEARTBEAT_KEY = "paper_worker_heartbeat_v1"
_DEFAULT_TICK_SECONDS = 60


# ── Singleton state ──────────────────────────────────────────────────────────
_LOCK = threading.Lock()
_THREAD: Optional[threading.Thread] = None
_STOP_EVENT: Optional[threading.Event] = None
_LAST_TICK_AT: Optional[datetime] = None
_LAST_TICK_DURATION_S: Optional[float] = None
_LAST_ERROR: Optional[str] = None
_ENABLED = True   # gate to allow disabling without killing the thread


def _db() -> Database:
    return Database(settings.db_path)


def _persist_heartbeat(payload: dict) -> None:
    try:
        _db().save_config(_WORKER_HEARTBEAT_KEY, payload)
    except Exception as exc:  # noqa: BLE001
        log.warning(f"paper_worker: heartbeat save failed — {exc}")


def get_heartbeat() -> dict:
    """Return the latest heartbeat dict (Streamlit page reads this for UI)."""
    try:
        return _db().load_config(_WORKER_HEARTBEAT_KEY) or {}
    except Exception:
        return {}


def get_status() -> dict:
    """In-process status snapshot (no DB hit). Used by the page if the
    worker is in the same process; otherwise call `get_heartbeat()`."""
    return {
        "running": bool(_THREAD and _THREAD.is_alive()),
        "enabled": _ENABLED,
        "last_tick_at": _LAST_TICK_AT.isoformat() if _LAST_TICK_AT else None,
        "last_tick_duration_s": _LAST_TICK_DURATION_S,
        "last_error": _LAST_ERROR,
    }


def _load_state_from_db() -> dict:
    """Build a worker-local session container from the DB-persisted keys
    written by the Streamlit page. Mirrors the four mutable session keys.

    Fail-soft: any DB hiccup yields an empty container so the loop survives
    a transient outage and tries again on the next tick.
    """
    try:
        db = _db()
    except Exception as exc:  # noqa: BLE001
        log.warning(f"paper_worker: DB construction failed — {exc}")
        return {
            "pt_active_runs":   {},
            "pt_all_signals":   [],
            "pt_trail_state":   {},
            "pt_prices_cache":  {},
        }
    try:
        runs_payload = db.load_config("paper_trading_runs_v1") or {}
    except Exception:
        runs_payload = {}
    try:
        signals_payload = db.load_config("paper_trading_signals_v1") or {}
    except Exception:
        signals_payload = {}
    try:
        trail_payload = db.load_config("paper_trading_trail_v1") or {}
    except Exception:
        trail_payload = {}

    runs = runs_payload.get("runs") or {}
    if not isinstance(runs, dict):
        runs = {}
    # Sanitize: only keep dict-shaped runs with a string symbol.
    clean_runs: dict[str, dict] = {}
    for sym, cfg in runs.items():
        if not isinstance(cfg, dict):
            continue
        s = str(cfg.get("symbol") or sym or "").upper().strip()
        if not s:
            continue
        c = dict(cfg)
        c["symbol"] = s
        c.setdefault("active", True)
        clean_runs[s] = c

    signals = signals_payload.get("signals") or []
    if not isinstance(signals, list):
        signals = []

    trail = trail_payload.get("trail") or {}
    if not isinstance(trail, dict):
        trail = {}

    # Use the same key strings as page_paper_trading._RUNS / _SIGNALS / _TRAIL /
    # _CACHE so the accessors find them.
    return {
        "pt_active_runs":   clean_runs,
        "pt_all_signals":   list(signals),
        "pt_trail_state":   {str(k): dict(v) for k, v in trail.items() if isinstance(v, dict)},
        "pt_prices_cache":  {},
    }


def _tick_once() -> None:
    """One worker pass: load state from DB, tick every active run, unbind.

    The page's `_run_tick` writes back to DB at the end (runs / signals /
    trail). We import it lazily so the module is importable without a
    Streamlit context.
    """
    global _LAST_TICK_AT, _LAST_TICK_DURATION_S, _LAST_ERROR

    try:
        # Lazy import — `pages.page_paper_trading` pulls in altair / streamlit
        # which are heavy and only needed when we actually tick.
        from pages.page_paper_trading import (
            _run_tick,
            bind_worker_state,
        )
    except Exception as exc:  # noqa: BLE001
        _LAST_ERROR = f"import _run_tick failed: {exc}"
        log.warning(f"paper_worker: {_LAST_ERROR}")
        _persist_heartbeat({
            "at": datetime.utcnow().isoformat(),
            "ok": False,
            "error": _LAST_ERROR,
        })
        return

    try:
        state = _load_state_from_db()
    except Exception as exc:  # noqa: BLE001
        _LAST_ERROR = f"_load_state_from_db: {exc}"
        log.warning(f"paper_worker: {_LAST_ERROR}")
        _persist_heartbeat({
            "at":    datetime.utcnow().isoformat(),
            "ok":    False,
            "error": _LAST_ERROR,
        })
        return
    started = time.monotonic()
    bind_worker_state(state)
    processed: list[str] = []
    errors: list[dict] = []
    try:
        runs = state.get("pt_active_runs") or {}
        for symbol, run in list(runs.items()):
            if not isinstance(run, dict):
                continue
            if not run.get("active", True):
                continue
            try:
                _run_tick(symbol, run)
                processed.append(symbol)
            except Exception as exc:  # noqa: BLE001
                err = {
                    "symbol": symbol,
                    "error":  f"{exc.__class__.__name__}: {exc}",
                    "at":     datetime.utcnow().isoformat(),
                }
                errors.append(err)
                log.warning(f"paper_worker: {symbol} tick failed — {err['error']}")
    finally:
        bind_worker_state(None)

    duration = time.monotonic() - started
    _LAST_TICK_AT = datetime.utcnow()
    _LAST_TICK_DURATION_S = round(duration, 3)
    _LAST_ERROR = errors[0]["error"] if errors else None
    _persist_heartbeat({
        "at":          _LAST_TICK_AT.isoformat(),
        "duration_s":  _LAST_TICK_DURATION_S,
        "ok":          not errors,
        "processed":   processed,
        "errors":      errors[-5:],
    })


def _worker_loop(stop_event: threading.Event, tick_seconds: int) -> None:
    """Daemon loop body. Sleeps in 1s slices so `stop()` is responsive."""
    log.info(
        f"paper_worker: started (tick_seconds={tick_seconds}, "
        f"db={settings.db_path})"
    )
    while not stop_event.is_set():
        if _ENABLED:
            try:
                _tick_once()
            except Exception as exc:  # noqa: BLE001
                global _LAST_ERROR
                _LAST_ERROR = f"unhandled: {exc.__class__.__name__}: {exc}"
                log.warning(f"paper_worker: {_LAST_ERROR}")
        # Responsive sleep: wake every second to check the stop flag.
        for _ in range(int(tick_seconds)):
            if stop_event.is_set():
                break
            time.sleep(1.0)
    log.info("paper_worker: stopped")


def start(tick_seconds: int = _DEFAULT_TICK_SECONDS) -> bool:
    """Launch the worker thread once per process. Idempotent.

    Returns True if a new thread was started, False if one was already
    running (or if Alpaca paper credentials are missing — there's no
    point ticking without a way to act on signals).
    """
    global _THREAD, _STOP_EVENT
    with _LOCK:
        if _THREAD and _THREAD.is_alive():
            return False
        # Don't bother running if we can't actually submit orders.
        # Backtest-only deployments shouldn't burn CPU on this loop.
        if not settings.alpaca.has_paper_credentials():
            log.info(
                "paper_worker: skipping start — no Alpaca paper credentials. "
                "Sim-only runs still tick from the Streamlit page."
            )
            return False
        _STOP_EVENT = threading.Event()
        t = threading.Thread(
            target=_worker_loop,
            args=(_STOP_EVENT, int(tick_seconds)),
            name="paper-trading-worker",
            daemon=True,
        )
        t.start()
        _THREAD = t
        return True


def stop(timeout: float = 5.0) -> None:
    """Signal the worker loop to stop and join the thread."""
    global _THREAD, _STOP_EVENT
    with _LOCK:
        if _STOP_EVENT is not None:
            _STOP_EVENT.set()
        t = _THREAD
    if t is not None:
        t.join(timeout=timeout)
    with _LOCK:
        _THREAD = None
        _STOP_EVENT = None


def set_enabled(enabled: bool) -> None:
    """Pause / resume ticking without tearing down the thread.
    Useful for a UI toggle."""
    global _ENABLED
    _ENABLED = bool(enabled)

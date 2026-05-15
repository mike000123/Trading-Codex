"""
execution/trading_stream.py
───────────────────────────
Background WebSocket listener for Alpaca's `trade_updates` stream — delivers
real-time fill / partial-fill / cancel / reject events so our DB no longer has
to wait for the next UI refresh to learn about order state changes.

Why a WebSocket when Step 4 already has a REST poll?
  • REST only fires on a manual refresh or auto-refresh tick (often 10-30s).
  • Stream fills the DB within ~100ms of the broker reporting the event.
  • The poll remains as a safety net — any events missed while the socket was
    down get picked up on the next refresh (the DB update is idempotent).

Design constraints (important — Streamlit-specific):
  • We can't hold a reference across st.rerun() cheaply, so the stream runs
    in a **daemon background thread** anchored to the Python process, not
    the Streamlit session. A module-level singleton (`_STREAM`) prevents the
    thread from being started twice.
  • The handler does its DB work synchronously via the same `Database` class
    the rest of the app uses (SQLAlchemy + check_same_thread=False is safe
    for multithread writes on our small SQLite footprint).
  • The handler never raises — exceptions are caught + logged so a single
    malformed event can't kill the listener.

Lifecycle:
  start_trading_stream(db)       → idempotent; starts the thread if needed.
  stop_trading_stream()          → requests a clean shutdown.
  get_stream_status() -> dict    → {running, connected, events_received,
                                    last_event_at, last_event_summary,
                                    last_error, last_error_at, paper}

Public DB effect:
  For every event whose `order.id` matches an existing `broker_order_id` in
  our trades table, we update:
      broker_status, filled_qty, filled_avg_price, filled_at, last_synced_at
  ONLY those six fields — we never touch outcome/exit_price/pnl from the
  stream because the reconciliation path (Step 5) owns position-close logic.
"""
from __future__ import annotations

import asyncio
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from config.settings import settings
from core.logger import log


# ── Module-level singleton ──────────────────────────────────────────────────

_LOCK   = threading.Lock()
_STREAM: Optional["_StreamWorker"] = None


class _StreamWorker:
    """Owns the asyncio event loop + Alpaca TradingStream on a daemon thread."""

    def __init__(self, *, paper: bool, db_path: Path) -> None:
        self.paper    = paper
        self.db_path  = db_path
        self.thread:  Optional[threading.Thread] = None
        self.stream:  Any = None   # alpaca.trading.stream.TradingStream
        self._stop_event = threading.Event()

        # Diagnostics exposed via get_stream_status()
        self.running          = False
        self.connected        = False
        self.events_received  = 0
        self.last_event_at:       Optional[datetime] = None
        self.last_event_summary:  str = ""
        self.last_error:          Optional[str] = None
        self.last_error_at:       Optional[datetime] = None
        self.last_update_by_id:   dict[str, datetime] = {}   # broker_order_id → ts

    # ── Handler ─────────────────────────────────────────────────────────────

    async def _on_trade_update(self, event) -> None:
        """
        TradeUpdate schema (pydantic): event.event (str), event.order (Order),
        event.timestamp (datetime), event.price (Decimal|None),
        event.qty (Decimal|None), event.position_qty (Decimal|None).

        We never let this raise — a bad event must not tear down the socket.
        """
        try:
            self.events_received += 1
            self.last_event_at    = datetime.utcnow()

            order_obj = getattr(event, "order", None)
            order_id  = str(getattr(order_obj, "id", "") or "")
            event_str = str(getattr(event, "event", "") or "")
            symbol    = str(getattr(order_obj, "symbol", "") or "")
            self.last_event_summary = f"{event_str} {symbol} [{order_id[:8]}]"
            log.info(f"TRADING-STREAM {self.last_event_summary}")

            if not order_id:
                return  # nothing we can key off — drop

            # Lazy-import Database so an import failure in the main thread
            # doesn't poison the whole module at start-up.
            from db.database import Database

            db = Database(self.db_path)
            self._upsert_broker_fields(db, order_obj, event)
            self.last_update_by_id[order_id] = self.last_event_at
        except Exception as exc:
            self.last_error    = f"handler: {exc}"
            self.last_error_at = datetime.utcnow()
            log.error(f"TRADING-STREAM handler failed: {exc}")

    @staticmethod
    def _upsert_broker_fields(db, order_obj, event) -> None:
        """Mirror the broker's view onto the matching trades row, if any."""
        from sqlalchemy import text
        order_id = str(getattr(order_obj, "id", "") or "")
        if not order_id:
            return

        def _f(v):
            try: return float(v) if v is not None else None
            except Exception: return None

        status_val = getattr(order_obj, "status", None)
        status_str = (status_val.value if hasattr(status_val, "value")
                      else str(status_val) if status_val is not None else None)
        filled_qty = _f(getattr(order_obj, "filled_qty", None))
        filled_avg = _f(getattr(order_obj, "filled_avg_price", None))

        filled_at  = getattr(order_obj, "filled_at", None)
        if filled_at is None:
            filled_at = getattr(event, "timestamp", None)
        if isinstance(filled_at, str):
            try:
                filled_at = datetime.fromisoformat(filled_at.replace("Z", "+00:00"))
            except Exception:
                filled_at = None

        now_iso = datetime.utcnow().isoformat()
        params = {
            "order_id":       order_id,
            "broker_status":  status_str,
            "filled_qty":     filled_qty,
            "filled_avg":     filled_avg,
            "filled_at":      filled_at.isoformat() if isinstance(filled_at, datetime) else None,
            "last_synced_at": now_iso,
        }

        with db._engine.begin() as conn:
            # COALESCE keeps existing values when the stream event doesn't
            # carry a new one (e.g. 'new' events arrive before the fill).
            conn.execute(text("""
                UPDATE trades
                   SET broker_status    = COALESCE(:broker_status, broker_status),
                       filled_qty       = COALESCE(:filled_qty, filled_qty),
                       filled_avg_price = COALESCE(:filled_avg, filled_avg_price),
                       filled_at        = COALESCE(:filled_at, filled_at),
                       last_synced_at   = :last_synced_at
                 WHERE broker_order_id  = :order_id
            """), params)

    # ── Thread body ─────────────────────────────────────────────────────────

    def _run(self) -> None:
        """Daemon-thread entry point. Owns its asyncio loop + TradingStream."""
        try:
            from alpaca.trading.stream import TradingStream
            api_key    = settings.alpaca.paper_api_key  if self.paper else settings.alpaca.live_api_key
            secret_key = settings.alpaca.paper_secret_key if self.paper else settings.alpaca.live_secret_key
            self.stream = TradingStream(
                api_key=api_key,
                secret_key=secret_key,
                paper=self.paper,
            )
            self.stream.subscribe_trade_updates(self._on_trade_update)

            self.running   = True
            self.connected = True
            log.info(
                f"TRADING-STREAM started (paper={self.paper}). "
                f"Listening for trade_updates…"
            )
            # .run() is blocking — returns when .stop() is called.
            self.stream.run()
        except Exception as exc:
            self.last_error    = f"thread: {exc}"
            self.last_error_at = datetime.utcnow()
            log.error(f"TRADING-STREAM thread died: {exc}")
        finally:
            self.connected = False
            self.running   = False
            log.info("TRADING-STREAM thread exited.")

    # ── Control ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        if self.thread and self.thread.is_alive():
            return
        self._stop_event.clear()
        self.thread = threading.Thread(
            target=self._run,
            name="alpaca-trading-stream",
            daemon=True,
        )
        self.thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self.stream is not None:
            try:
                self.stream.stop()
            except Exception as exc:
                log.warning(f"TRADING-STREAM stop error: {exc}")
        self.connected = False
        self.running   = False


# ── Public API ──────────────────────────────────────────────────────────────

def start_trading_stream(*, paper: bool = True, db_path: Optional[Path] = None) -> dict:
    """
    Idempotent. Returns a small status dict describing the state after the call.

    Safe to call every render — subsequent calls are a no-op if the thread is
    already alive.
    """
    global _STREAM
    if not settings.alpaca.has_paper_credentials() and paper:
        return {"started": False, "reason": "no paper credentials"}

    with _LOCK:
        # Idempotent: thread.is_alive() is the source of truth — the `running`
        # flag flips ON slightly later (inside the thread body) so checking
        # both would race on a fast re-call. Alive thread ⇒ still connected
        # or connecting, either way we skip restart.
        if _STREAM and _STREAM.thread and _STREAM.thread.is_alive():
            return {"started": False, "reason": "already running",
                    "running": True, "paper": _STREAM.paper}
        _STREAM = _StreamWorker(
            paper=paper,
            db_path=db_path or settings.db_path,
        )
        _STREAM.start()
        return {"started": True, "running": True, "paper": paper}


def stop_trading_stream() -> dict:
    global _STREAM
    with _LOCK:
        if _STREAM is None:
            return {"stopped": False, "reason": "no stream"}
        _STREAM.stop()
        return {"stopped": True}


def get_stream_status() -> dict:
    """
    Snapshot of the stream's diagnostic fields for the UI banner. Cheap —
    reads in-memory counters only. Returns {running: False} when no stream
    has ever been started.
    """
    with _LOCK:
        s = _STREAM
        if s is None:
            return {"running": False, "connected": False, "events_received": 0}
        return {
            "running":            bool(s.running),
            "connected":          bool(s.connected),
            "paper":              bool(s.paper),
            "events_received":    int(s.events_received),
            "last_event_at":      s.last_event_at,
            "last_event_summary": s.last_event_summary,
            "last_error":         s.last_error,
            "last_error_at":      s.last_error_at,
            "tracked_orders":     dict(s.last_update_by_id),
        }

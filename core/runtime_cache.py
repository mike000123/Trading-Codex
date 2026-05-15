"""
Process-local runtime cache for live, non-persisted objects.

This is intentionally in-memory only. It lets background workers publish
fresh prepared frames that Streamlit page reruns can reuse immediately
without re-fetching the same heavy datasets on the UI thread.
"""
from __future__ import annotations

from datetime import datetime, timezone
from threading import Lock
from typing import Any


_LOCK = Lock()
_STORE: dict[str, dict[str, dict[str, Any]]] = {}


def put(namespace: str, key: str, value: Any) -> None:
    ns = str(namespace or "").strip()
    item_key = str(key or "").strip()
    if not ns or not item_key:
        return
    with _LOCK:
        bucket = _STORE.setdefault(ns, {})
        bucket[item_key] = {
            "value": value,
            "updated_at": datetime.now(timezone.utc),
        }


def get(namespace: str, key: str, default: Any = None) -> Any:
    ns = str(namespace or "").strip()
    item_key = str(key or "").strip()
    if not ns or not item_key:
        return default
    with _LOCK:
        entry = (_STORE.get(ns) or {}).get(item_key)
        return entry["value"] if entry is not None else default


def snapshot(namespace: str) -> dict[str, Any]:
    ns = str(namespace or "").strip()
    if not ns:
        return {}
    with _LOCK:
        bucket = _STORE.get(ns) or {}
        return {key: entry["value"] for key, entry in bucket.items()}


def updated_at(namespace: str, key: str):
    ns = str(namespace or "").strip()
    item_key = str(key or "").strip()
    if not ns or not item_key:
        return None
    with _LOCK:
        entry = (_STORE.get(ns) or {}).get(item_key)
        return None if entry is None else entry.get("updated_at")

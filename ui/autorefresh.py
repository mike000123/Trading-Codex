from __future__ import annotations

from datetime import timedelta
from time import monotonic

import streamlit as st


def render_autorefresh_timer(enabled: bool, interval_seconds: int, *, key: str) -> None:
    """
    Schedule a normal Streamlit app rerun without forcing a full browser reload.

    This keeps the UI much steadier than `window.location.reload()`, while still
    letting stateful pages like Forward Test and Paper Trading process their next
    tick on a timer.
    """
    timer_key = str(key)
    deadline_key = f"__autorefresh_deadline_{timer_key}"

    if not enabled or interval_seconds <= 0:
        st.session_state.pop(deadline_key, None)

        @st.fragment
        def _disabled_fragment() -> None:
            st.empty()

        _disabled_fragment()
        return

    interval_seconds = int(interval_seconds)

    @st.fragment(run_every=timedelta(seconds=interval_seconds))
    def _autorefresh_fragment() -> None:
        now = monotonic()
        next_due = st.session_state.get(deadline_key)
        if next_due is None:
            st.session_state[deadline_key] = now + interval_seconds
            st.empty()
            return

        if now >= float(next_due):
            st.session_state[deadline_key] = now + interval_seconds
            st.rerun(scope="app")

        st.empty()

    _autorefresh_fragment()

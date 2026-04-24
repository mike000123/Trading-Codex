from __future__ import annotations

import json

import streamlit.components.v1 as components


def render_autorefresh_timer(enabled: bool, interval_seconds: int, *, key: str) -> None:
    timer_key = json.dumps(str(key))
    if enabled and interval_seconds > 0:
        interval_ms = int(interval_seconds * 1000)
        html = f"""
<script>
const root = window.parent || window;
root.__algoRefreshTimers = root.__algoRefreshTimers || {{}};
if (root.__algoRefreshTimers[{timer_key}]) {{
  clearTimeout(root.__algoRefreshTimers[{timer_key}]);
}}
root.__algoRefreshTimers[{timer_key}] = setTimeout(() => {{
  try {{
    root.location.reload();
  }} catch (e) {{
    window.location.reload();
  }}
}}, {interval_ms});
</script>
"""
    else:
        html = f"""
<script>
const root = window.parent || window;
if (root.__algoRefreshTimers && root.__algoRefreshTimers[{timer_key}]) {{
  clearTimeout(root.__algoRefreshTimers[{timer_key}]);
  delete root.__algoRefreshTimers[{timer_key}];
}}
</script>
"""
    components.html(html, height=0)

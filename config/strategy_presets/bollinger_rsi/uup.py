"""UUP preset for the Bollinger + RSI spike-aware strategy.

UUP did not earn a deep bespoke calibration pass, but it did clear the bar as
an independent low-frequency macro sleeve. The best current-engine result was
the same selective profile we already trust for GLD's macro ETF path, so we
promote UUP conservatively by inheriting that family rather than inventing a
new rule set prematurely.
"""
from __future__ import annotations

from .gld import PRESET as GLD_PRESET


PRESET: dict[str, object] = dict(GLD_PRESET)

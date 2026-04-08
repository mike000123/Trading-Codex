"""Symbol presets for the Bollinger + RSI spike-aware strategy."""
from __future__ import annotations

from .uvxy import PRESET as UVXY_PRESET
from .gld import PRESET as GLD_PRESET
from .vxx import PRESET as VXX_PRESET
from .vxz import PRESET as VXZ_PRESET


_SYMBOL_PRESETS: dict[str, dict[str, object]] = {
    "UVXY": UVXY_PRESET,
    "GLD": GLD_PRESET,
    "VXX": VXX_PRESET,
    "VXZ": VXZ_PRESET,
}


def get_symbol_preset(symbol: str) -> dict[str, object]:
    """Return a copy of the symbol-specific overrides for this strategy."""
    return dict(_SYMBOL_PRESETS.get(symbol.strip().upper(), {}))

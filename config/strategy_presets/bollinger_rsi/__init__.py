"""Symbol presets for the Bollinger + RSI spike-aware strategy."""
from __future__ import annotations

from .apld import PRESET as APLD_PRESET
from .uvxy import PRESET as UVXY_PRESET
from .gld import PRESET as GLD_PRESET
from .iwm import PRESET as IWM_PRESET
from .qqq import PRESET as QQQ_PRESET
from .slv import PRESET as SLV_PRESET
from .uup import PRESET as UUP_PRESET
from .uso import PRESET as USO_PRESET
from .spy import PRESET as SPY_PRESET
from .xle import PRESET as XLE_PRESET
from .vxx import PRESET as VXX_PRESET
from .vxz import PRESET as VXZ_PRESET
from .xlf import PRESET as XLF_PRESET


_SYMBOL_PRESETS: dict[str, dict[str, object]] = {
    "APLD": APLD_PRESET,
    "UVXY": UVXY_PRESET,
    "GLD": GLD_PRESET,
    "IWM": IWM_PRESET,
    "QQQ": QQQ_PRESET,
    "SLV": SLV_PRESET,
    "UUP": UUP_PRESET,
    "USO": USO_PRESET,
    "SPY": SPY_PRESET,
    "XLE": XLE_PRESET,
    "VXX": VXX_PRESET,
    "VXZ": VXZ_PRESET,
    "XLF": XLF_PRESET,
}


def get_symbol_preset(symbol: str) -> dict[str, object]:
    """Return a copy of the symbol-specific overrides for this strategy."""
    return dict(_SYMBOL_PRESETS.get(symbol.strip().upper(), {}))

"""
core/models.py
──────────────
Shared domain types used across all modules.
Keep this import-free of internal packages so any module can import safely.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
import uuid


class Direction(str, Enum):
    LONG = "Long"
    SHORT = "Short"


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(str, Enum):
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    PARTIAL = "partial"


class TradeOutcome(str, Enum):
    TAKE_PROFIT  = "Take-profit hit"
    STOP_LOSS    = "Stop-loss hit"
    SIGNAL_EXIT  = "Counter-signal exit"   # closed by opposing RSI/strategy signal
    AMBIGUOUS    = "Ambiguous candle"
    OPEN         = "Open"
    NO_DATA      = "No data"


class SignalAction(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class OHLCV:
    """Single price bar."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


@dataclass
class Signal:
    """Output from a strategy."""
    strategy_id: str
    symbol: str
    action: SignalAction
    confidence: float = 1.0                    # 0–1 scale
    suggested_tp: Optional[float] = None
    suggested_sl: Optional[float] = None
    metadata: dict = field(default_factory=dict)
    generated_at: datetime = field(default_factory=datetime.utcnow)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class TradeRecord:
    """Persisted record of an executed or simulated trade."""
    id: str
    symbol: str
    direction: Direction
    entry_price: float
    take_profit: Optional[float]
    stop_loss: Optional[float]
    leverage: float
    capital_allocated: float
    entry_time: datetime
    mode: str                                  # "paper" | "live" | "backtest"
    strategy_id: str
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    outcome: Optional[TradeOutcome] = None
    leveraged_return_pct: Optional[float] = None
    pnl: Optional[float] = None               # absolute $ P&L
    notes: str = ""


@dataclass
class PortfolioSnapshot:
    """Point-in-time portfolio state."""
    timestamp: datetime
    total_equity: float
    cash: float
    open_positions_count: int
    daily_pnl: float
    total_pnl: float
    mode: str

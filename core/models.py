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
    TAKE_PROFIT      = "TP hit"              # price reached take-profit level
    STOP_LOSS        = "SL hit"              # price reached stop-loss level
    TRAIL_STOP       = "Trail stop"          # trailing stop closed the trade
    SIGNAL_RSI_OB    = "RSI overbought exit" # RSI crossed above sell threshold → close Long
    SIGNAL_RSI_OS    = "RSI oversold exit"   # RSI crossed below buy threshold → close Short
    SIGNAL_EXIT      = "Counter-signal exit" # generic counter-signal (non-RSI strategies)
    AMBIGUOUS        = "Ambiguous candle"    # TP and SL both touched same bar
    OPEN             = "Open"
    NO_DATA          = "No data"


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
    mode: str                                  # "paper" | "alpaca_paper" | "live" | "backtest"
    strategy_id: str
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    outcome: Optional[TradeOutcome] = None
    leveraged_return_pct: Optional[float] = None
    pnl: Optional[float] = None               # absolute $ P&L
    notes: str = ""

    # Counter-signal-exit suppression threshold (None = always exit on opposing
    # signal; float = only exit if current profit < this %). Set per-trade by
    # the strategy via signal metadata so a regime can opt into "ride past the
    # regime flip if I'm already winning enough." Used by reporting.backtest
    # and pages.page_paper_trading.
    counter_signal_min_profit_pct: Optional[float] = None

    # ── Broker-order lifecycle (populated only for alpaca_paper / live) ───
    # These mirror what Alpaca's Order object returns so we can reconcile
    # our internal trade_id with the broker's view.
    origin:              Optional[str]      = None  # app instance / channel that created the broker order
    broker_order_id:     Optional[str]      = None  # Alpaca order UUID
    broker_status:       Optional[str]      = None  # new / accepted / filled / canceled / …
    broker_submitted_at: Optional[datetime] = None  # when we submitted to the broker
    filled_qty:          Optional[float]    = None  # actual filled qty (may differ from requested)
    filled_avg_price:    Optional[float]    = None  # avg fill price (reality)
    filled_at:           Optional[datetime] = None  # last fill timestamp
    last_synced_at:      Optional[datetime] = None  # when we last polled the broker


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

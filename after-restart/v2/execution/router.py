"""
execution/router.py
────────────────────
Order routing layer.

Paper mode  → simulates fills instantly at current price, persists to DB.
Live mode   → sends real orders via Alpaca API with explicit safeguard checks.

SAFEGUARDS (live mode):
  - Requires TRADING_MODE=live in environment (not just settings object)
  - Requires explicit live credential validation
  - Emits WARNING log on every live order
  - Requires `confirm_live=True` flag on every execute() call
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime
from typing import Optional

from config.settings import TradingMode, settings
from core.models import Direction, OrderSide, OrderStatus, TradeRecord, TradeOutcome
from core.logger import log
from risk.manager import RiskManager, RiskCheckResult


class OrderRouter:
    """
    Routes orders to paper simulation or live Alpaca execution.
    Instantiate once and reuse across the session.
    """

    def __init__(self, risk_manager: RiskManager) -> None:
        self.risk = risk_manager
        self._alpaca_client = None  # lazy-initialised

    # ─── Public entry point ─────────────────────────────────────────────────

    def execute(
        self,
        *,
        symbol: str,
        direction: Direction,
        entry_price: float,
        take_profit: Optional[float],
        stop_loss: Optional[float],
        leverage: float,
        capital: float,
        strategy_id: str,
        confirm_live: bool = False,   # must be True to allow live orders
    ) -> TradeRecord:
        """
        Run risk checks → route to paper or live execution.
        Returns a TradeRecord (may have status REJECTED if checks fail).
        """
        check: RiskCheckResult = self.risk.check(
            direction=direction,
            entry_price=entry_price,
            take_profit=take_profit,
            stop_loss=stop_loss,
            leverage=leverage,
            capital_requested=capital,
        )

        trade_id = str(uuid.uuid4())

        if not check.approved:
            log.warning(f"TRADE REJECTED [{trade_id[:8]}] {symbol}: {check.reason}")
            return TradeRecord(
                id=trade_id,
                symbol=symbol,
                direction=direction,
                entry_price=entry_price,
                take_profit=take_profit,
                stop_loss=stop_loss,
                leverage=leverage,
                capital_allocated=0.0,
                entry_time=datetime.utcnow(),
                mode=settings.trading_mode.value,
                strategy_id=strategy_id,
                outcome=TradeOutcome.NO_DATA,
                notes=f"REJECTED: {check.reason}",
            )

        effective_sl = check.adjusted_sl or stop_loss
        effective_capital = check.adjusted_size or capital

        if settings.trading_mode == TradingMode.LIVE:
            return self._execute_live(
                trade_id=trade_id,
                symbol=symbol,
                direction=direction,
                entry_price=entry_price,
                take_profit=take_profit,
                stop_loss=effective_sl,
                leverage=leverage,
                capital=effective_capital,
                strategy_id=strategy_id,
                confirm_live=confirm_live,
            )
        else:
            return self._execute_paper(
                trade_id=trade_id,
                symbol=symbol,
                direction=direction,
                entry_price=entry_price,
                take_profit=take_profit,
                stop_loss=effective_sl,
                leverage=leverage,
                capital=effective_capital,
                strategy_id=strategy_id,
            )

    # ─── Paper execution ────────────────────────────────────────────────────

    def _execute_paper(
        self,
        trade_id: str,
        symbol: str,
        direction: Direction,
        entry_price: float,
        take_profit: Optional[float],
        stop_loss: Optional[float],
        leverage: float,
        capital: float,
        strategy_id: str,
    ) -> TradeRecord:
        log.info(
            f"TRADE PAPER [{trade_id[:8]}] {direction.value} {symbol} "
            f"entry={entry_price:.4f} TP={take_profit} SL={stop_loss} "
            f"lev={leverage}x capital={capital:.2f}"
        )
        return TradeRecord(
            id=trade_id,
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            take_profit=take_profit,
            stop_loss=stop_loss,
            leverage=leverage,
            capital_allocated=capital,
            entry_time=datetime.utcnow(),
            mode="paper",
            strategy_id=strategy_id,
            outcome=TradeOutcome.OPEN,
            notes="Paper order filled at entry price.",
        )

    # ─── Live execution ─────────────────────────────────────────────────────

    def _execute_live(
        self,
        trade_id: str,
        symbol: str,
        direction: Direction,
        entry_price: float,
        take_profit: Optional[float],
        stop_loss: Optional[float],
        leverage: float,
        capital: float,
        strategy_id: str,
        confirm_live: bool,
    ) -> TradeRecord:

        # ── Hard safeguard 1: env var must literally say "live" ──────────────
        if os.getenv("TRADING_MODE", "").lower() != "live":
            msg = (
                "LIVE ORDER BLOCKED: TRADING_MODE env var is not 'live'. "
                "Set TRADING_MODE=live in your .env to enable live trading."
            )
            log.error(msg)
            return self._rejected_record(trade_id, symbol, direction, entry_price,
                                         take_profit, stop_loss, leverage, capital,
                                         strategy_id, msg)

        # ── Hard safeguard 2: caller must explicitly pass confirm_live=True ──
        if not confirm_live:
            msg = (
                "LIVE ORDER BLOCKED: confirm_live=True not passed. "
                "This is a required explicit confirmation for every live order."
            )
            log.error(msg)
            return self._rejected_record(trade_id, symbol, direction, entry_price,
                                         take_profit, stop_loss, leverage, capital,
                                         strategy_id, msg)

        # ── Hard safeguard 3: validate live credentials exist ────────────────
        if not settings.alpaca.has_live_credentials():
            msg = "LIVE ORDER BLOCKED: Live Alpaca credentials not configured."
            log.error(msg)
            return self._rejected_record(trade_id, symbol, direction, entry_price,
                                         take_profit, stop_loss, leverage, capital,
                                         strategy_id, msg)

        log.warning(
            f"⚠️  LIVE TRADE [{trade_id[:8]}] {direction.value} {symbol} "
            f"entry≈{entry_price:.4f} TP={take_profit} SL={stop_loss} "
            f"lev={leverage}x capital={capital:.2f}  ← REAL MONEY"
        )

        try:
            client = self._get_alpaca_client(paper=False)
            side = OrderSide.BUY if direction == Direction.LONG else OrderSide.SELL
            qty = self._calc_qty(capital, entry_price, leverage)

            from alpaca.trading.requests import MarketOrderRequest, TakeProfitRequest, StopLossRequest
            from alpaca.trading.enums import OrderSide as AlpacaSide, TimeInForce

            order_data = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=AlpacaSide.BUY if side == OrderSide.BUY else AlpacaSide.SELL,
                time_in_force=TimeInForce.DAY,
                take_profit=TakeProfitRequest(limit_price=round(take_profit, 2)) if take_profit else None,
                stop_loss=StopLossRequest(stop_price=round(stop_loss, 2)) if stop_loss else None,
            )
            order = client.submit_order(order_data)
            log.info(f"LIVE ORDER SUBMITTED: alpaca_id={order.id} status={order.status}")

            return TradeRecord(
                id=trade_id,
                symbol=symbol,
                direction=direction,
                entry_price=entry_price,
                take_profit=take_profit,
                stop_loss=stop_loss,
                leverage=leverage,
                capital_allocated=capital,
                entry_time=datetime.utcnow(),
                mode="live",
                strategy_id=strategy_id,
                outcome=TradeOutcome.OPEN,
                notes=f"Live order submitted. Alpaca ID: {order.id}",
            )

        except Exception as exc:
            msg = f"Live order submission failed: {exc}"
            log.error(f"LIVE ORDER ERROR [{trade_id[:8]}]: {exc}")
            return self._rejected_record(trade_id, symbol, direction, entry_price,
                                         take_profit, stop_loss, leverage, capital,
                                         strategy_id, msg)

    # ─── Alpaca client (lazy) ───────────────────────────────────────────────

    def _get_alpaca_client(self, paper: bool = True):
        if self._alpaca_client is None:
            from alpaca.trading.client import TradingClient
            if paper:
                self._alpaca_client = TradingClient(
                    api_key=settings.alpaca.paper_api_key,
                    secret_key=settings.alpaca.paper_secret_key,
                    paper=True,
                )
            else:
                self._alpaca_client = TradingClient(
                    api_key=settings.alpaca.live_api_key,
                    secret_key=settings.alpaca.live_secret_key,
                    paper=False,
                )
        return self._alpaca_client

    # ─── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _calc_qty(capital: float, price: float, leverage: float) -> float:
        """Notional qty = (capital * leverage) / price."""
        if price <= 0:
            return 0.0
        return round((capital * leverage) / price, 6)

    @staticmethod
    def _rejected_record(
        trade_id, symbol, direction, entry_price, take_profit,
        stop_loss, leverage, capital, strategy_id, reason
    ) -> TradeRecord:
        return TradeRecord(
            id=trade_id,
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            take_profit=take_profit,
            stop_loss=stop_loss,
            leverage=leverage,
            capital_allocated=0.0,
            entry_time=datetime.utcnow(),
            mode=settings.trading_mode.value,
            strategy_id=strategy_id,
            outcome=TradeOutcome.NO_DATA,
            notes=f"REJECTED: {reason}",
        )

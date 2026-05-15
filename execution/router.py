"""
execution/router.py
────────────────────
Order routing layer — THREE possible destinations:

  route="sim"           → Pure local simulation. Instant fill at entry_price.
                          Used by the local paper-trading page and backtest.
  route="alpaca_paper"  → Real API call to Alpaca's PAPER endpoint.
                          No real money, but goes through Alpaca's matching
                          engine (RTH gating, real fills, real slippage).
  route="alpaca_live"   → Real-money Alpaca live endpoint. Gated by several
                          hard safeguards (see _execute_live).

If `route` is not passed, the destination is derived from
`settings.trading_mode`:

  TradingMode.PAPER        → route="sim"
  TradingMode.ALPACA_PAPER → route="alpaca_paper"
  TradingMode.LIVE         → route="alpaca_live"
  TradingMode.BACKTEST     → route="sim"

SHADOW MODE:
  Callers that want to dual-write (one local sim record + one real Alpaca
  paper order for comparison) simply call execute() twice with different
  route= values. The router is stateless between calls.

SAFEGUARDS (alpaca_live only):
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


# Route string constants — use these instead of bare strings.
ROUTE_SIM          = "sim"
ROUTE_ALPACA_PAPER = "alpaca_paper"
ROUTE_ALPACA_LIVE  = "alpaca_live"
VALID_ROUTES       = {ROUTE_SIM, ROUTE_ALPACA_PAPER, ROUTE_ALPACA_LIVE}


def _route_from_mode(mode: TradingMode) -> str:
    if mode == TradingMode.LIVE:
        return ROUTE_ALPACA_LIVE
    if mode == TradingMode.ALPACA_PAPER:
        return ROUTE_ALPACA_PAPER
    # PAPER and BACKTEST both map to local sim.
    return ROUTE_SIM


class OrderRouter:
    """
    Routes orders to local sim, Alpaca paper, or Alpaca live execution.
    Instantiate once and reuse across the session.
    """

    def __init__(self, risk_manager: RiskManager) -> None:
        self.risk = risk_manager
        # Separate caches per endpoint — a single TradingClient is bound to
        # exactly one endpoint (paper or live) at construction time.
        self._alpaca_paper_client = None
        self._alpaca_live_client  = None

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
        route: Optional[str] = None,  # "sim" | "alpaca_paper" | "alpaca_live"
    ) -> TradeRecord:
        """
        Run risk checks → route to the chosen execution path.
        Returns a TradeRecord (may have status REJECTED if checks fail).
        """
        # Resolve route (explicit arg wins; otherwise derive from settings).
        if route is None:
            route = _route_from_mode(settings.trading_mode)
        if route not in VALID_ROUTES:
            raise ValueError(
                f"Invalid route {route!r}. Must be one of {sorted(VALID_ROUTES)}."
            )

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
            log.warning(f"TRADE REJECTED [{trade_id[:8]}] {symbol} ({route}): {check.reason}")
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
                mode=self._mode_label(route),
                strategy_id=strategy_id,
                outcome=TradeOutcome.NO_DATA,
                notes=f"REJECTED: {check.reason}",
            )

        effective_sl      = check.adjusted_sl   or stop_loss
        effective_capital = check.adjusted_size or capital

        common_kwargs = dict(
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

        if route == ROUTE_ALPACA_LIVE:
            return self._execute_live(**common_kwargs, confirm_live=confirm_live)
        if route == ROUTE_ALPACA_PAPER:
            return self._execute_alpaca_paper(**common_kwargs)
        # Default: local sim
        return self._execute_paper(**common_kwargs)

    # ─── Local sim execution ────────────────────────────────────────────────

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
            f"TRADE SIM [{trade_id[:8]}] {direction.value} {symbol} "
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
            notes="Local sim order filled at entry price.",
        )

    # ─── Alpaca paper execution ─────────────────────────────────────────────

    def _execute_alpaca_paper(
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
        """
        Submit a real order to Alpaca's paper endpoint (no real money).
        Mirrors _execute_live but:
          - paper=True on the TradingClient
          - does NOT require TRADING_MODE=live env var
          - does NOT require confirm_live
          - uses paper credentials
        """
        if not settings.alpaca.has_paper_credentials():
            msg = "ALPACA PAPER BLOCKED: paper credentials not configured in .env."
            log.error(msg)
            return self._rejected_record(
                trade_id, symbol, direction, entry_price, take_profit,
                stop_loss, leverage, capital, strategy_id, msg,
                mode_label="alpaca_paper",
            )

        log.info(
            f"TRADE ALPACA-PAPER [{trade_id[:8]}] {direction.value} {symbol} "
            f"entry≈{entry_price:.4f} TP={take_profit} SL={stop_loss} "
            f"lev={leverage}x capital={capital:.2f}"
        )

        try:
            client = self._get_alpaca_client(paper=True)
            side = OrderSide.BUY if direction == Direction.LONG else OrderSide.SELL
            qty = self._calc_qty(capital, entry_price, leverage)

            from alpaca.trading.requests import (
                MarketOrderRequest, TakeProfitRequest, StopLossRequest
            )
            from alpaca.trading.enums import OrderSide as AlpacaSide, TimeInForce

            # Bracket orders (TP + SL attached) require integer qty AND not being
            # fractional. Keep bracket only when we have both TP and SL and qty
            # is a whole number; otherwise send a plain market order.
            is_whole = abs(qty - round(qty)) < 1e-9
            want_bracket = (take_profit is not None) and (stop_loss is not None) and is_whole

            order_kwargs = dict(
                symbol=symbol,
                qty=qty,
                side=AlpacaSide.BUY if side == OrderSide.BUY else AlpacaSide.SELL,
                time_in_force=TimeInForce.DAY,
            )
            if want_bracket:
                order_kwargs["take_profit"] = TakeProfitRequest(
                    limit_price=round(take_profit, 2)
                )
                order_kwargs["stop_loss"] = StopLossRequest(
                    stop_price=round(stop_loss, 2)
                )

            order_data = MarketOrderRequest(**order_kwargs)
            order = client.submit_order(order_data)

            log.info(
                f"ALPACA-PAPER ORDER SUBMITTED [{trade_id[:8]}]: "
                f"alpaca_id={order.id} status={order.status} "
                f"qty={qty} bracket={want_bracket}"
            )

            bracket_note = "bracket (TP+SL attached)" if want_bracket else "plain market"
            now = datetime.utcnow()
            # Alpaca's Order object exposes .submitted_at, .filled_at,
            # .filled_qty, .filled_avg_price — may be None immediately after
            # market-order submission (filled_* filled in later via polling).
            def _to_dt(v):
                if v is None:
                    return None
                if isinstance(v, datetime):
                    return v
                try:
                    return datetime.fromisoformat(str(v).replace("Z", "+00:00"))
                except Exception:
                    return None
            def _to_float(v):
                try:
                    return float(v) if v is not None else None
                except Exception:
                    return None

            return TradeRecord(
                id=trade_id,
                symbol=symbol,
                direction=direction,
                entry_price=entry_price,
                take_profit=take_profit,
                stop_loss=stop_loss,
                leverage=leverage,
                capital_allocated=capital,
                entry_time=now,
                mode="alpaca_paper",
                strategy_id=strategy_id,
                outcome=TradeOutcome.OPEN,
                notes=(
                    f"Alpaca-paper {bracket_note} order submitted. "
                    f"Alpaca ID: {order.id}"
                ),
                broker_order_id=str(order.id),
                broker_status=(order.status.value if hasattr(order.status, "value")
                               else str(order.status) if order.status else None),
                broker_submitted_at=_to_dt(getattr(order, "submitted_at", None)) or now,
                filled_qty=_to_float(getattr(order, "filled_qty", None)),
                filled_avg_price=_to_float(getattr(order, "filled_avg_price", None)),
                filled_at=_to_dt(getattr(order, "filled_at", None)),
                last_synced_at=now,
            )

        except Exception as exc:
            msg = f"Alpaca-paper order submission failed: {exc}"
            log.error(f"ALPACA-PAPER ERROR [{trade_id[:8]}]: {exc}")
            return self._rejected_record(
                trade_id, symbol, direction, entry_price, take_profit,
                stop_loss, leverage, capital, strategy_id, msg,
                mode_label="alpaca_paper",
            )

    # ─── Alpaca live execution ──────────────────────────────────────────────

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
            return self._rejected_record(
                trade_id, symbol, direction, entry_price, take_profit,
                stop_loss, leverage, capital, strategy_id, msg,
                mode_label="live",
            )

        # ── Hard safeguard 2: caller must explicitly pass confirm_live=True ──
        if not confirm_live:
            msg = (
                "LIVE ORDER BLOCKED: confirm_live=True not passed. "
                "This is a required explicit confirmation for every live order."
            )
            log.error(msg)
            return self._rejected_record(
                trade_id, symbol, direction, entry_price, take_profit,
                stop_loss, leverage, capital, strategy_id, msg,
                mode_label="live",
            )

        # ── Hard safeguard 3: validate live credentials exist ────────────────
        if not settings.alpaca.has_live_credentials():
            msg = "LIVE ORDER BLOCKED: Live Alpaca credentials not configured."
            log.error(msg)
            return self._rejected_record(
                trade_id, symbol, direction, entry_price, take_profit,
                stop_loss, leverage, capital, strategy_id, msg,
                mode_label="live",
            )

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

            now = datetime.utcnow()
            def _to_dt(v):
                if v is None:
                    return None
                if isinstance(v, datetime):
                    return v
                try:
                    return datetime.fromisoformat(str(v).replace("Z", "+00:00"))
                except Exception:
                    return None
            def _to_float(v):
                try:
                    return float(v) if v is not None else None
                except Exception:
                    return None

            return TradeRecord(
                id=trade_id,
                symbol=symbol,
                direction=direction,
                entry_price=entry_price,
                take_profit=take_profit,
                stop_loss=stop_loss,
                leverage=leverage,
                capital_allocated=capital,
                entry_time=now,
                mode="live",
                strategy_id=strategy_id,
                outcome=TradeOutcome.OPEN,
                notes=f"Live order submitted. Alpaca ID: {order.id}",
                broker_order_id=str(order.id),
                broker_status=(order.status.value if hasattr(order.status, "value")
                               else str(order.status) if order.status else None),
                broker_submitted_at=_to_dt(getattr(order, "submitted_at", None)) or now,
                filled_qty=_to_float(getattr(order, "filled_qty", None)),
                filled_avg_price=_to_float(getattr(order, "filled_avg_price", None)),
                filled_at=_to_dt(getattr(order, "filled_at", None)),
                last_synced_at=now,
            )

        except Exception as exc:
            msg = f"Live order submission failed: {exc}"
            log.error(f"LIVE ORDER ERROR [{trade_id[:8]}]: {exc}")
            return self._rejected_record(
                trade_id, symbol, direction, entry_price, take_profit,
                stop_loss, leverage, capital, strategy_id, msg,
                mode_label="live",
            )

    # ─── Broker-order status polling ────────────────────────────────────────

    def fetch_alpaca_order(self, order_id: str, *, paper: bool = True) -> dict:
        """
        Poll Alpaca for a single order's current state.

        Returns a dict with keys:
          ok (bool), order_id (str), status (str or None),
          filled_qty (float or None), filled_avg_price (float or None),
          filled_at (datetime or None), updated_at (datetime or None),
          error (str, only if ok=False).

        Swallows connectivity errors — callers can decide whether to log or
        retry. A 404 is returned as ok=False with error="not found".
        """
        from datetime import datetime as _dt
        def _to_dt(v):
            if v is None:               return None
            if isinstance(v, _dt):      return v
            try:
                return _dt.fromisoformat(str(v).replace("Z", "+00:00"))
            except Exception:
                return None
        def _to_float(v):
            try:
                return float(v) if v is not None else None
            except Exception:
                return None

        try:
            client = self._get_alpaca_client(paper=paper)
        except Exception as exc:
            return {"ok": False, "order_id": order_id, "error": f"client init: {exc}"}

        try:
            # alpaca-py's TradingClient.get_order_by_id(order_id) returns Order
            order = client.get_order_by_id(order_id)
        except Exception as exc:
            status_code = getattr(exc, "status_code", None)
            note = "not found" if status_code == 404 else str(exc)
            return {"ok": False, "order_id": order_id, "error": note}

        status_val = getattr(order, "status", None)
        status_str = (status_val.value if hasattr(status_val, "value")
                      else str(status_val) if status_val else None)

        return {
            "ok":               True,
            "order_id":         str(order.id),
            "status":           status_str,
            "filled_qty":       _to_float(getattr(order, "filled_qty", None)),
            "filled_avg_price": _to_float(getattr(order, "filled_avg_price", None)),
            "filled_at":        _to_dt(getattr(order, "filled_at", None)),
            "updated_at":       _to_dt(getattr(order, "updated_at", None)),
        }

    # Terminal states — once an order reaches one of these it will not change.
    ALPACA_TERMINAL_STATES = frozenset({
        "filled", "canceled", "expired", "rejected", "done_for_day",
        "suspended", "stopped", "replaced",
    })

    # ─── Account + positions snapshot (Step 5) ──────────────────────────────

    def fetch_alpaca_account(self, *, paper: bool = True) -> dict:
        """
        Read-only account snapshot. Returns a dict with the fields Cowork
        actually needs to display + gate decisions:

          ok, id, status, equity, last_equity, cash, buying_power,
          daytrade_count, pattern_day_trader, trading_blocked,
          account_blocked, currency, fetched_at, (error on failure)

        Equity values are floats; booleans are plain Python booleans so the
        result can be JSON-serialised for session state.
        """
        from datetime import datetime as _dt
        def _f(v):
            try: return float(v) if v is not None else None
            except Exception: return None

        try:
            client = self._get_alpaca_client(paper=paper)
        except Exception as exc:
            return {"ok": False, "error": f"client init: {exc}"}

        try:
            acct = client.get_account()
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

        status_val = getattr(acct, "status", None)
        status_str = (status_val.value if hasattr(status_val, "value")
                      else str(status_val) if status_val else None)

        return {
            "ok":                 True,
            "id":                 str(getattr(acct, "id", "") or ""),
            "status":             status_str,
            "equity":             _f(getattr(acct, "equity", None)),
            "last_equity":        _f(getattr(acct, "last_equity", None)),
            "cash":               _f(getattr(acct, "cash", None)),
            "buying_power":       _f(getattr(acct, "buying_power", None)),
            "daytrade_count":     int(getattr(acct, "daytrade_count", 0) or 0),
            "pattern_day_trader": bool(getattr(acct, "pattern_day_trader", False)),
            "trading_blocked":    bool(getattr(acct, "trading_blocked", False)),
            "account_blocked":    bool(getattr(acct, "account_blocked", False)),
            "currency":           str(getattr(acct, "currency", "USD") or "USD"),
            "fetched_at":         _dt.utcnow(),
        }

    def fetch_alpaca_positions(self, *, paper: bool = True) -> dict:
        """
        Read-only list of current broker-side positions.

        Returns {ok, positions: [dict, ...], fetched_at} on success. Each
        position dict has: symbol, qty (signed for short), side, avg_entry_price,
        market_value, cost_basis, current_price, unrealized_pl,
        unrealized_plpc, lastday_price.
        """
        from datetime import datetime as _dt
        def _f(v):
            try: return float(v) if v is not None else None
            except Exception: return None

        try:
            client = self._get_alpaca_client(paper=paper)
        except Exception as exc:
            return {"ok": False, "error": f"client init: {exc}", "positions": []}

        try:
            raw = client.get_all_positions()
        except Exception as exc:
            return {"ok": False, "error": str(exc), "positions": []}

        out = []
        for p in raw:
            side_val = getattr(p, "side", None)
            side_str = (side_val.value if hasattr(side_val, "value")
                        else str(side_val) if side_val else None)
            qty = _f(getattr(p, "qty", None))
            # Alpaca returns short qty as negative already, but some SDK
            # versions expose "qty_available"/"side" separately. Normalise.
            if qty is not None and side_str == "short" and qty > 0:
                qty = -qty
            out.append({
                "symbol":           getattr(p, "symbol", None),
                "qty":              qty,
                "side":             side_str,
                "avg_entry_price":  _f(getattr(p, "avg_entry_price", None)),
                "market_value":     _f(getattr(p, "market_value", None)),
                "cost_basis":       _f(getattr(p, "cost_basis", None)),
                "current_price":    _f(getattr(p, "current_price", None)),
                "unrealized_pl":    _f(getattr(p, "unrealized_pl", None)),
                "unrealized_plpc":  _f(getattr(p, "unrealized_plpc", None)),
                "lastday_price":    _f(getattr(p, "lastday_price", None)),
            })
        return {"ok": True, "positions": out, "fetched_at": _dt.utcnow()}

    def fetch_alpaca_clock(self, *, paper: bool = True) -> dict:
        """
        Read the Alpaca market-clock endpoint. Returns:
          {ok, is_open, timestamp, next_open, next_close, fetched_at}
        on success, or {ok:False, error} on failure.

        The clock is the same for paper and live endpoints, but we use the
        selected endpoint so a single broken credential set is surfaced here
        instead of masked by the other.
        """
        from datetime import datetime as _dt
        def _to_dt(v):
            if v is None: return None
            if isinstance(v, datetime): return v
            try:
                return datetime.fromisoformat(str(v).replace("Z", "+00:00"))
            except Exception:
                return None

        try:
            client = self._get_alpaca_client(paper=paper)
        except Exception as exc:
            return {"ok": False, "error": f"client init: {exc}"}

        try:
            clk = client.get_clock()
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

        return {
            "ok":         True,
            "is_open":    bool(getattr(clk, "is_open", False)),
            "timestamp":  _to_dt(getattr(clk, "timestamp",  None)),
            "next_open":  _to_dt(getattr(clk, "next_open",  None)),
            "next_close": _to_dt(getattr(clk, "next_close", None)),
            "fetched_at": _dt.utcnow(),
        }

    def find_closing_order(
        self,
        symbol: str,
        *,
        after: Optional[datetime] = None,
        opposite_side: Optional[str] = None,
        paper: bool = True,
        limit: int = 20,
    ) -> Optional[dict]:
        """
        Look up the most recent FILLED order on `symbol` that would have
        closed our position. Used when a tracked position disappears from
        /positions so we can stamp an accurate exit price + time on the DB
        row instead of approximating.

        Returns {filled_avg_price, filled_qty, filled_at, order_id, side}
        or None if no matching order is found.
        """
        def _f(v):
            try: return float(v) if v is not None else None
            except Exception: return None
        def _to_dt(v):
            if v is None: return None
            if isinstance(v, datetime): return v
            try:
                return datetime.fromisoformat(str(v).replace("Z", "+00:00"))
            except Exception:
                return None

        try:
            client = self._get_alpaca_client(paper=paper)
        except Exception:
            return None

        try:
            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums import QueryOrderStatus
            req = GetOrdersRequest(
                status=QueryOrderStatus.CLOSED,
                symbols=[symbol],
                after=after,
                limit=limit,
                direction="desc",
            )
            orders = client.get_orders(filter=req)
        except Exception:
            return None

        # Walk newest → oldest, pick the first FILLED order matching opposite_side
        for o in orders or []:
            status_val = getattr(o, "status", None)
            status_str = (status_val.value if hasattr(status_val, "value")
                          else str(status_val) if status_val else None)
            if status_str != "filled":
                continue
            side_val = getattr(o, "side", None)
            side_str = (side_val.value if hasattr(side_val, "value")
                        else str(side_val) if side_val else None)
            if opposite_side and side_str != opposite_side:
                continue
            px = _f(getattr(o, "filled_avg_price", None))
            if px is None:
                continue
            return {
                "order_id":         str(getattr(o, "id", "") or ""),
                "filled_avg_price": px,
                "filled_qty":       _f(getattr(o, "filled_qty", None)),
                "filled_at":        _to_dt(getattr(o, "filled_at", None)),
                "side":             side_str,
            }
        return None

    # ─── Counter-signal / manual flatten helpers ────────────────────────────

    def close_position_alpaca_paper(self, symbol: str) -> dict:
        """
        Flatten any open Alpaca-paper position in `symbol`. Safe no-op if
        no position exists. Returns a small dict the caller can log.
        """
        try:
            client = self._get_alpaca_client(paper=True)
            from alpaca.common.exceptions import APIError
            try:
                order = client.close_position(symbol)
                log.info(
                    f"ALPACA-PAPER CLOSE {symbol}: alpaca_id={order.id} "
                    f"status={order.status}"
                )
                return {"ok": True, "symbol": symbol, "alpaca_id": str(order.id)}
            except APIError as exc:
                # 404 = no open position for this symbol — treat as success.
                code = getattr(exc, "status_code", None)
                if code in (404, 422):
                    log.info(f"ALPACA-PAPER CLOSE {symbol}: no open position (HTTP {code})")
                    return {"ok": True, "symbol": symbol, "note": "no open position"}
                raise
        except Exception as exc:
            log.error(f"ALPACA-PAPER CLOSE {symbol} failed: {exc}")
            return {"ok": False, "symbol": symbol, "error": str(exc)}

    # ─── Alpaca client (lazy, per endpoint) ─────────────────────────────────

    def _get_alpaca_client(self, paper: bool = True):
        """
        Return a cached TradingClient bound to the requested endpoint.
        Paper and live each get their own cached client, so mixing shadow-
        mode (paper) and live within the same session never cross-wires.
        """
        from alpaca.trading.client import TradingClient
        if paper:
            if self._alpaca_paper_client is None:
                self._alpaca_paper_client = TradingClient(
                    api_key=settings.alpaca.paper_api_key,
                    secret_key=settings.alpaca.paper_secret_key,
                    paper=True,
                )
            return self._alpaca_paper_client
        else:
            if self._alpaca_live_client is None:
                self._alpaca_live_client = TradingClient(
                    api_key=settings.alpaca.live_api_key,
                    secret_key=settings.alpaca.live_secret_key,
                    paper=False,
                )
            return self._alpaca_live_client

    # ─── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _mode_label(route: str) -> str:
        """Map a route to the TradeRecord.mode string."""
        if route == ROUTE_ALPACA_LIVE:
            return "live"
        if route == ROUTE_ALPACA_PAPER:
            return "alpaca_paper"
        return "paper"

    @staticmethod
    def _calc_qty(capital: float, price: float, leverage: float) -> float:
        """Notional qty = (capital * leverage) / price."""
        if price <= 0:
            return 0.0
        return round((capital * leverage) / price, 6)

    @staticmethod
    def _rejected_record(
        trade_id, symbol, direction, entry_price, take_profit,
        stop_loss, leverage, capital, strategy_id, reason,
        mode_label: Optional[str] = None,
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
            mode=(mode_label or settings.trading_mode.value),
            strategy_id=strategy_id,
            outcome=TradeOutcome.NO_DATA,
            notes=f"REJECTED: {reason}",
        )

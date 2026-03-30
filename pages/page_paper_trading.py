"""
pages/page_paper_trading.py
────────────────────────────
Paper (simulated) live trading:
  - Generate signal from any strategy on freshly fetched data
  - Run through risk manager
  - Execute paper order (stored in DB, never sent to broker)
  - View open paper positions and close them manually
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

from config.settings import settings, TradingMode
from core.models import Direction, TradeOutcome
from db.database import Database
from execution.router import OrderRouter
from risk.manager import RiskManager
from strategies import list_strategies, get_strategy
from ui.components import (
    render_mode_banner, render_data_source_selector,
    render_strategy_params, live_trade_confirm_dialog,
)
from ui.charts import price_chart


def _db() -> Database:
    return Database(settings.db_path)


def render() -> None:
    render_mode_banner()

    mode = settings.trading_mode

    if mode == TradingMode.LIVE:
        st.warning("🔴 **Live mode detected.** Switch to Paper mode in Settings to use this page safely.")

    st.title("📝 Paper Trading")
    st.caption("Submit paper orders via strategy signals. Stored locally — no real orders sent.")

    prices = render_data_source_selector()
    if prices is None:
        st.info("← Select a data source in the sidebar to begin.")
        return

    symbol = st.session_state.get("loaded_symbol", "AAPL")
    st.success(f"**{symbol}** — {len(prices)} bars · last close: **{prices['close'].iloc[-1]:.4f}**")

    st.divider()
    st.subheader("📡 Signal Generator")

    strategies = list_strategies()
    strat_names = {s["name"]: s["id"] for s in strategies}

    col_strat, col_order = st.columns([0.5, 0.5])

    with col_strat:
        selected_name = st.selectbox("Strategy", list(strat_names.keys()), key="pt_strategy")
        selected_id   = strat_names[selected_name]
        params        = render_strategy_params(selected_id, leverage=leverage if "leverage" in dir() else 1.0)

    with col_order:
        direction_override = st.selectbox("Direction", ["Follow signal", "Force Long", "Force Short"], key="pt_dir")
        leverage  = st.number_input("Leverage", 1.0, 100.0, 1.0, 0.5, key="pt_lev")
        capital   = st.number_input("Capital ($)", 10.0, value=500.0, key="pt_capital")
        manual_tp = st.number_input("Override TP (0 = use strategy)", 0.0, value=0.0, key="pt_tp")
        manual_sl = st.number_input("Override SL (0 = use strategy)", 0.0, value=0.0, key="pt_sl")

    if st.button("🔍 Generate Signal + Submit Paper Order", type="primary", key="pt_submit"):
        cls      = get_strategy(selected_id)
        strategy = cls(params=params)
        signal   = strategy.generate_signal(prices, symbol)

        st.info(
            f"Signal: **{signal.action.value}** | "
            f"Confidence: **{signal.confidence:.0%}** | "
            f"TP: {signal.suggested_tp:.4f if signal.suggested_tp else '—'} | "
            f"SL: {signal.suggested_sl:.4f if signal.suggested_sl else '—'}"
        )

        from core.models import SignalAction
        if signal.action == SignalAction.HOLD and direction_override == "Follow signal":
            st.warning("Strategy says HOLD — no order placed. Use a direction override to force.")
            return

        # Determine direction
        if direction_override == "Force Long":
            direction = Direction.LONG
        elif direction_override == "Force Short":
            direction = Direction.SHORT
        else:
            direction = Direction.LONG if signal.action == SignalAction.BUY else Direction.SHORT

        entry_price = float(prices["close"].iloc[-1])
        tp = manual_tp if manual_tp > 0 else signal.suggested_tp
        sl = manual_sl if manual_sl > 0 else signal.suggested_sl

        if sl is None:
            st.error("No stop-loss provided. Set one manually or choose a strategy that generates SL levels.")
            return

        risk = RiskManager(settings.risk)
        risk.update_portfolio_state(daily_pnl=0, open_positions=0, total_equity=capital * 10)
        router = OrderRouter(risk_manager=risk)

        trade = router.execute(
            symbol=symbol, direction=direction,
            entry_price=entry_price, take_profit=tp, stop_loss=sl,
            leverage=leverage, capital=capital,
            strategy_id=selected_id,
            confirm_live=False,  # paper only
        )

        # Save signal + trade
        db = _db()
        db.save_signal(signal)
        db.save_trade(trade)

        if "REJECTED" in trade.notes:
            st.error(f"Order REJECTED: {trade.notes}")
        else:
            st.success(
                f"✅ Paper order submitted: **{direction.value} {symbol}** "
                f"@ {entry_price:.4f} | TP={tp:.4f if tp else '—'} | SL={sl:.4f} "
                f"| Capital=${capital:.2f}"
            )

        fig = price_chart(
            prices,
            take_profit=tp,
            stop_loss=sl,
            title=f"{symbol} – Paper Order",
        )
        st.altair_chart(fig, use_container_width=True)

    # ── Open paper positions ─────────────────────────────────────────────────
    st.divider()
    st.subheader("📂 Open Paper Positions")

    db = _db()
    all_trades = db.get_trades(mode="paper")
    open_trades = [t for t in all_trades if t.get("outcome") == "Open"]

    if not open_trades:
        st.info("No open paper positions.")
    else:
        df = pd.DataFrame(open_trades)[
            ["id", "symbol", "direction", "entry_price", "take_profit", "stop_loss",
             "leverage", "capital_allocated", "entry_time", "strategy_id"]
        ]
        st.dataframe(df, use_container_width=True)

        # Manual close
        close_id = st.selectbox("Close position (by ID prefix)", [t["id"][:8] for t in open_trades], key="pt_close")
        close_price = st.number_input("Close price", 0.0, value=float(prices["close"].iloc[-1]), key="pt_close_price")

        if st.button("❌ Close Position", key="pt_close_btn"):
            full_id = next(t["id"] for t in open_trades if t["id"].startswith(close_id))
            trade_row = next(t for t in open_trades if t["id"] == full_id)
            ep = trade_row["entry_price"]
            lev = trade_row["leverage"]
            cap = trade_row["capital_allocated"]
            d = Direction(trade_row["direction"])
            raw = (close_price - ep) / ep * (1 if d == Direction.LONG else -1)
            ret_pct = raw * lev * 100
            pnl = cap * ret_pct / 100

            # Update trade in DB by re-saving with exit info
            from core.models import TradeRecord
            from dataclasses import replace
            # Minimal patch: just update outcome fields via DB upsert
            import uuid
            updated = {
                "id": full_id,
                "symbol": trade_row["symbol"],
                "direction": d,
                "entry_price": ep,
                "take_profit": trade_row["take_profit"],
                "stop_loss": trade_row["stop_loss"],
                "leverage": lev,
                "capital_allocated": cap,
                "entry_time": datetime.fromisoformat(trade_row["entry_time"]) if trade_row["entry_time"] else datetime.utcnow(),
                "mode": "paper",
                "strategy_id": trade_row["strategy_id"],
                "exit_price": close_price,
                "exit_time": datetime.utcnow(),
                "outcome": TradeOutcome.TAKE_PROFIT if ret_pct > 0 else TradeOutcome.STOP_LOSS,
                "leveraged_return_pct": ret_pct,
                "pnl": pnl,
                "notes": "Manually closed.",
            }

            class _TR:  # minimal duck-type for db.save_trade
                def __init__(self, d):
                    for k, v in d.items(): setattr(self, k, v)

            db.save_trade(_TR(updated))
            st.success(f"Position closed. P&L: ${pnl:+,.2f} ({ret_pct:+.2f}%)")
            st.rerun()

    # ── Closed paper history ─────────────────────────────────────────────────
    with st.expander("📋 Paper Trade History", expanded=False):
        closed = [t for t in all_trades if t.get("outcome") != "Open"]
        if closed:
            st.dataframe(pd.DataFrame(closed), use_container_width=True)
        else:
            st.info("No closed paper trades yet.")

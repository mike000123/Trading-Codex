"""
pages/page_backtest.py
───────────────────────
Full walk-forward backtester page.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from config.settings import settings
from reporting.backtest import BacktestEngine
from risk.manager import RiskManager
from strategies import list_strategies, get_strategy
from ui.components import (
    render_mode_banner, render_data_source_selector,
    render_strategy_params, render_metrics_row,
)
from ui.charts import equity_curve_chart, pnl_distribution


def render() -> None:
    render_mode_banner()
    st.title("⏪ Backtester")
    st.caption("Walk-forward simulation. Signals are generated bar-by-bar on historical data.")

    prices = render_data_source_selector()
    if prices is None:
        st.info("← Select a data source in the sidebar to begin.")
        return

    symbol = st.session_state.get("loaded_symbol", "DATA")
    st.success(f"**{symbol}** — {len(prices)} bars")

    st.divider()

    strategies = list_strategies()
    strat_names = {s["name"]: s["id"] for s in strategies}

    col_cfg, col_risk = st.columns(2)

    with col_cfg:
        selected_name  = st.selectbox("Strategy", list(strat_names.keys()), key="bt_strategy")
        selected_id    = strat_names[selected_name]
        leverage       = st.number_input("Leverage", 1.0, 100.0, 1.0, 0.5, key="bt_lev")
        capital_per_trade = st.number_input("Capital per trade ($)", 100.0, value=1000.0, key="bt_cap")
        starting_equity   = st.number_input("Starting equity ($)", 1000.0, value=10_000.0, key="bt_equity")
        direction_filter  = st.selectbox("Direction filter", ["Both", "Long only", "Short only"], key="bt_dir")

    with col_risk:
        st.markdown("**Risk Controls**")
        use_risk = st.checkbox("Apply risk manager", value=True, key="bt_risk")
        max_loss = st.slider("Max loss per trade (% of capital)", 5, 100, 50, key="bt_maxloss")

    st.divider()
    params = render_strategy_params(selected_id)

    if st.button("▶ Run Backtest", type="primary", key="bt_run"):
        cls      = get_strategy(selected_id)
        strategy = cls(params=params)

        errors = strategy.validate_params()
        if errors:
            for e in errors: st.error(e)
            return

        from config.settings import RiskConfig
        risk_cfg = RiskConfig(
            max_capital_per_trade_pct=100.0,   # sizing handled by capital_per_trade
            max_daily_loss_pct=100.0,
            max_open_positions=999,
            default_max_loss_pct_of_capital=float(max_loss),
        )
        rm = RiskManager(risk_cfg) if use_risk else None

        from core.models import Direction as Dir
        dir_filter = None
        if direction_filter == "Long only":  dir_filter = Dir.LONG
        if direction_filter == "Short only": dir_filter = Dir.SHORT

        engine = BacktestEngine(strategy, risk_manager=rm, direction_filter=dir_filter)

        with st.spinner("Running backtest…"):
            result = engine.run(
                data=prices, symbol=symbol,
                leverage=leverage,
                capital_per_trade=capital_per_trade,
                starting_equity=starting_equity,
            )

        # ── Summary metrics ──────────────────────────────────────────────────
        st.subheader("📊 Results")
        s = result.summary()
        render_metrics_row({
            "Total Trades":  s["Total Trades"],
            "Win Rate":      s["Win Rate"],
            "Total Return":  s["Total Return"],
            "Max Drawdown":  s["Max Drawdown"],
            "Sharpe Ratio":  s["Sharpe Ratio"],
            "Avg Win":       s["Avg Win"],
            "Avg Loss":      s["Avg Loss"],
        })

        # ── Equity curve ─────────────────────────────────────────────────────
        if not result.equity_curve.empty:
            st.plotly_chart(equity_curve_chart(result.equity_curve, f"{symbol} Equity Curve"), use_container_width=True)

        # ── P&L distribution ─────────────────────────────────────────────────
        if result.trades:
            trades_df = pd.DataFrame([
                {
                    "symbol": t.symbol, "direction": t.direction.value,
                    "entry_price": t.entry_price, "exit_price": t.exit_price,
                    "outcome": t.outcome.value if t.outcome else None,
                    "leveraged_return_pct": t.leveraged_return_pct,
                    "pnl": t.pnl,
                    "entry_time": t.entry_time, "exit_time": t.exit_time,
                }
                for t in result.trades
            ])
            st.plotly_chart(pnl_distribution(trades_df), use_container_width=True)

            with st.expander("📋 Trade Log", expanded=False):
                st.dataframe(trades_df.sort_values("entry_time", ascending=False), use_container_width=True)

        # ── Save to DB ──────────────────────────────────────────────────────
        try:
            from db.database import Database
            db = Database(settings.db_path)
            for t in result.trades:
                db.save_trade(t)
            st.caption(f"✓ {len(result.trades)} trades saved to database.")
        except Exception as e:
            st.caption(f"DB save skipped: {e}")

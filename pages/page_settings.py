"""
pages/page_settings.py
───────────────────────
App settings, credential status, risk config display,
and live-trading mode switch with explicit safeguards.
"""
from __future__ import annotations

import os

import streamlit as st

from config.settings import settings, TradingMode
from ui.components import render_mode_banner


def render() -> None:
    render_mode_banner()
    st.title("Settings")

    tab_mode, tab_risk, tab_creds, tab_about = st.tabs(
        ["🔄 Trading Mode", "🛡️ Risk Controls", "🔑 Credentials", "ℹ️ About"]
    )

    # ── Trading Mode ─────────────────────────────────────────────────────────
    with tab_mode:
        st.subheader("Trading Mode")
        current = settings.trading_mode.value.upper()
        st.markdown(
            f"Current mode from environment: **`{current}`**  \n"
            "To change: edit `TRADING_MODE` in your `.env` file and restart the app."
        )

        st.warning(
            "⚠️ **Live mode sends real orders to Alpaca with real money.**  \n"
            "Only set `TRADING_MODE=live` after thorough paper trading validation."
        )

        st.markdown("#### Live Mode Checklist")
        checks = [
            "Strategy backtested over 6+ months of data with positive Sharpe ratio",
            "Paper trading tested for at least 2 weeks with expected signal frequency",
            "Risk controls reviewed: max daily loss, position sizing, SL/TP validated",
            "Alpaca live credentials set and account funded",
            "Live orders manually confirmed by operator (confirm_live=True enforced in code)",
        ]
        for c in checks:
            st.checkbox(c, value=False, key=f"live_check_{c[:20]}")

        st.info("These checkboxes are for your own review only — they do not unlock live mode.")

    # ── Risk Controls ─────────────────────────────────────────────────────────
    with tab_risk:
        st.subheader("Risk Controls (read from .env)")
        st.caption("Modify these values in your `.env` file and restart.")

        r = settings.risk
        col1, col2 = st.columns(2)
        col1.metric("Max Capital per Trade", f"{r.max_capital_per_trade_pct:.1f}%",
                    help="Max % of total portfolio allocated to a single trade.")
        col2.metric("Max Daily Loss", f"{r.max_daily_loss_pct:.1f}%",
                    help="Circuit-breaker: halts trading if daily loss exceeds this.")
        col1.metric("Max Open Positions", str(r.max_open_positions))
        col2.metric("Max Loss per Trade", f"{r.default_max_loss_pct_of_capital:.1f}%",
                    help="SL is clamped so loss cannot exceed this % of trade capital.")

        st.markdown("#### .env Variables")
        st.code(
            f"""MAX_CAPITAL_PER_TRADE_PCT={r.max_capital_per_trade_pct}
MAX_DAILY_LOSS_PCT={r.max_daily_loss_pct}
MAX_OPEN_POSITIONS={r.max_open_positions}
DEFAULT_MAX_LOSS_PCT_OF_CAPITAL={r.default_max_loss_pct_of_capital}""",
            language="bash",
        )

    # ── Credentials ───────────────────────────────────────────────────────────
    with tab_creds:
        st.subheader("Alpaca API Credentials")
        st.caption("Never enter credentials here – set them in your `.env` file.")

        paper_ok = settings.alpaca.has_paper_credentials()
        live_ok  = settings.alpaca.has_live_credentials()

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Paper Trading**")
            if paper_ok:
                st.success("✅ Paper credentials detected")
                st.caption(f"Key: `...{settings.alpaca.paper_api_key[-6:]}`")
            else:
                st.error("❌ Paper credentials missing")
                st.caption("Set `ALPACA_PAPER_API_KEY` and `ALPACA_PAPER_SECRET_KEY` in `.env`")

        with col2:
            st.markdown("**Live Trading**")
            if live_ok:
                st.warning("⚠️ Live credentials detected — ensure mode=live only when intentional")
                st.caption(f"Key: `...{settings.alpaca.live_api_key[-6:]}`")
            else:
                st.info("ℹ️ Live credentials not set (recommended until ready)")

        st.markdown("---")
        st.markdown("#### .env Template")
        st.code(
            """TRADING_MODE=paper
ALPACA_PAPER_API_KEY=your_paper_key
ALPACA_PAPER_SECRET_KEY=your_paper_secret
ALPACA_LIVE_API_KEY=
ALPACA_LIVE_SECRET_KEY=""",
            language="bash",
        )

    # ── About ─────────────────────────────────────────────────────────────────
    with tab_about:
        st.subheader("MRMI Platform")
        st.markdown("""
**Architecture Overview**

```
app.py                  ← Streamlit entry point, theme + navigation
├── pages/              ← One file per page, each exposes render()
│   ├── page_simulator      Historical TP/SL backtester (original app)
│   ├── page_strategy_lab   Live signal inspection on any dataset
│   ├── page_backtest       Walk-forward automated backtest
│   ├── page_paper_trading  Paper order submission via strategy signals
│   ├── page_portfolio      Aggregate P&L, equity curves, trade history
│   └── page_settings       Mode, risk config, credentials
├── config/             ← Pydantic settings loaded from .env
├── core/               ← Domain models (Signal, TradeRecord, …) + logger
├── data/               ← CSV / yfinance / Alpaca ingestion
├── strategies/         ← BaseStrategy + RSI, MA Crossover, MACD
├── risk/               ← RiskManager (circuit-breakers, SL cap, sizing)
├── execution/          ← OrderRouter (paper sim / live Alpaca with safeguards)
├── reporting/          ← BacktestEngine, metrics, equity curve
├── db/                 ← SQLite via SQLAlchemy (trades, signals, portfolio)
└── ui/                 ← Theme CSS, Plotly charts, shared components
```

**Data Flow**
```
Data Ingestion → Strategy Signal → Risk Check → Order Router
      ↓                                              ↓
  OHLCV df        BUY/SELL/HOLD            Paper DB  /  Alpaca API
                  + TP/SL levels
```

**Adding a Strategy**
1. Create `strategies/my_strategy.py`
2. Subclass `BaseStrategy`, set `strategy_id`, `name`, `description`
3. Implement `generate_signal(data, symbol) → Signal`
4. Add `from . import my_strategy` to `strategies/__init__.py`

That's it — it appears in all dropdowns automatically.
        """)

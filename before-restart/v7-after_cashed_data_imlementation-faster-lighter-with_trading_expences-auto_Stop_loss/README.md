# AlgoTrader Pro

A modular Streamlit-based algorithmic trading platform extending the original Historical Trade Outcome Simulator.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Copy and configure environment
cp .env.example .env
# Edit .env with your Alpaca paper keys (live keys optional)

# 3. Run the app
streamlit run app.py
```

---

## Project Structure

```
trading_platform/
├── app.py                      ← Entry point (theme + page router)
├── requirements.txt
├── .env.example                ← Copy to .env and fill in credentials
│
├── config/
│   └── settings.py             ← Pydantic settings loaded from .env
│
├── core/
│   ├── models.py               ← Shared domain types (Signal, TradeRecord, …)
│   └── logger.py               ← Loguru structured logging
│
├── data/
│   └── ingestion.py            ← CSV / Yahoo Finance / Alpaca data loaders
│
├── strategies/
│   ├── base.py                 ← BaseStrategy + auto-registration system
│   ├── rsi_strategy.py         ← RSI threshold strategy
│   ├── ma_crossover.py         ← EMA/SMA golden/death cross
│   └── macd_strategy.py        ← MACD signal-line crossover
│
├── risk/
│   └── manager.py              ← RiskManager: circuit-breakers, SL cap, sizing
│
├── execution/
│   └── router.py               ← OrderRouter: paper sim / live Alpaca (with safeguards)
│
├── reporting/
│   └── backtest.py             ← Walk-forward BacktestEngine + metrics
│
├── db/
│   └── database.py             ← SQLite via SQLAlchemy (trades, signals, portfolio)
│
├── ui/
│   ├── themes.py               ← 4 CSS themes: Dark, Light, Terminal, Midnight Blue
│   ├── charts.py               ← Plotly chart builders
│   └── components.py           ← Reusable Streamlit widgets
│
└── pages/
    ├── page_simulator.py       ← Historical TP/SL simulator (original app, enhanced)
    ├── page_strategy_lab.py    ← Live signal inspection on any dataset
    ├── page_backtest.py        ← Automated walk-forward backtest
    ├── page_paper_trading.py   ← Paper order submission via strategy signals
    ├── page_portfolio.py       ← Aggregate P&L, equity curves, trade history
    └── page_settings.py        ← Mode, risk config, credentials, architecture docs
```

---

## Pages

| Page | Description |
|---|---|
| 📊 Historical Simulator | Original app: given TP/SL find outcome date, or suggest levels for desired profit |
| 🔬 Strategy Lab | Walk all bars, plot BUY/SELL signals, inspect indicators |
| ⏪ Backtester | Automated walk-forward test with metrics: Sharpe, drawdown, win rate |
| 📝 Paper Trading | Submit paper orders via strategy signal; manage open positions |
| 💼 Portfolio | Aggregate P&L by mode, equity curves, per-strategy breakdown, CSV export |
| ⚙️ Settings | Mode switch docs, risk config, credential status, architecture overview |

---

## Adding a New Strategy

1. Create `strategies/my_strategy.py`:

```python
from strategies.base import BaseStrategy, register_strategy
from core.models import Signal, SignalAction

@register_strategy
class MyStrategy(BaseStrategy):
    strategy_id = "my_strategy"
    name = "My Strategy"
    description = "One-line explanation."

    def default_params(self):
        return {"period": 20, "threshold": 0.5}

    def generate_signal(self, data, symbol):
        # Your logic here
        return Signal(
            strategy_id=self.strategy_id,
            symbol=symbol,
            action=SignalAction.BUY,
            suggested_tp=data["close"].iloc[-1] * 1.05,
            suggested_sl=data["close"].iloc[-1] * 0.97,
        )
```

2. Add one line to `strategies/__init__.py`:
```python
from . import my_strategy  # noqa: F401
```

Your strategy now appears in every dropdown automatically.

---

## Execution Flow

```
Data Ingestion (CSV / Yahoo / Alpaca)
        ↓
   OHLCV DataFrame
        ↓
Strategy.generate_signal()  →  Signal (BUY/SELL/HOLD + TP/SL)
        ↓
RiskManager.check()         →  Approved / Rejected + adjusted SL + sized capital
        ↓
OrderRouter.execute()
    ├── paper  → TradeRecord stored in SQLite
    └── live   → Alpaca API (3 hard safeguards: env var + confirm_live flag + credentials)
        ↓
Database.save_trade()       →  db/trading.db
        ↓
Portfolio page              →  Aggregate view, equity curve, CSV export
```

---

## Trading Modes

| Mode | Env var | Orders sent? | Safeguards |
|---|---|---|---|
| `backtest` | `TRADING_MODE=backtest` | No | N/A |
| `paper` | `TRADING_MODE=paper` | No | Default |
| `live` | `TRADING_MODE=live` | ✅ Yes — real money | 3 hard gates (see below) |

### Live Trading Safeguards (all 3 must pass)

1. `TRADING_MODE=live` must be set in the **environment** (not just the settings object)
2. `confirm_live=True` must be explicitly passed to `router.execute()` on every call
3. Live Alpaca credentials must be present and non-empty

---

## Risk Controls (.env)

| Variable | Default | Description |
|---|---|---|
| `MAX_CAPITAL_PER_TRADE_PCT` | 5.0 | Max % of portfolio per trade |
| `MAX_DAILY_LOSS_PCT` | 10.0 | Circuit-breaker: halt if daily loss exceeds this |
| `MAX_OPEN_POSITIONS` | 10 | Hard cap on simultaneous positions |
| `DEFAULT_MAX_LOSS_PCT_OF_CAPITAL` | 50.0 | SL is clamped to prevent larger losses |

---

## Themes

Select at runtime from the sidebar:
- **Dark** – default dark grey + purple accent
- **Light** – clean white with blue accent
- **Terminal** – green-on-black retro terminal
- **Midnight Blue** – deep navy with cyan accents

---

## Logs

All logs written to `logs/` directory:
- `app_YYYY-MM-DD.log` – debug/info rotating daily, 30-day retention
- `trades.log` – trade audit trail, never auto-deleted

---

## Database

SQLite at `db/trading.db` (auto-created on first run).

Tables: `trades`, `signals`, `portfolio`, `configs`

View with any SQLite browser, or export via the Portfolio page CSV button.

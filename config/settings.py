"""
config/settings.py
──────────────────
Central settings — works in both environments:

  LOCAL (PC):
    Reads from .env file via python-dotenv.
    Copy .env.example → .env and fill in your Alpaca keys.

  STREAMLIT CLOUD:
    No .env file. Instead, go to your app's Settings → Secrets and add:

      [alpaca]
      paper_api_key    = "PK..."
      paper_secret_key = "..."
      live_api_key     = ""
      live_secret_key  = ""

      [risk]
      max_capital_per_trade_pct      = 5.0
      max_daily_loss_pct             = 10.0
      max_open_positions             = 10
      default_max_loss_pct_of_capital = 50.0

      [app]
      trading_mode = "paper"
      db_path      = "db/trading.db"
      log_dir      = "logs"

    Settings are read from st.secrets when .env is absent or empty.

All other modules import `settings` from here — never read os.environ directly.
"""
from __future__ import annotations

import os
from enum import Enum
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

# ── Load .env (local only — silently ignored on Streamlit Cloud) ──────────────
_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env", override=False)


# ── Streamlit secrets helper ──────────────────────────────────────────────────
def _st_secret(section: str, key: str, default: str = "") -> str:
    """
    Read from st.secrets[section][key] if available (Streamlit Cloud),
    otherwise return default. Silently returns default if streamlit is
    not installed or secrets are not configured.
    """
    try:
        import streamlit as st
        return str(st.secrets.get(section, {}).get(key, default))
    except Exception:
        return default


def _get(env_key: str, st_section: str, st_key: str, default: str = "") -> str:
    """
    Priority: environment variable > Streamlit secret > default.
    This means .env always wins over st.secrets if both are set,
    which is the safe behaviour (explicit local override).
    """
    env_val = os.getenv(env_key, "").strip()
    if env_val:
        return env_val
    st_val = _st_secret(st_section, st_key, "")
    if st_val:
        return st_val
    return default


# ── Config models ─────────────────────────────────────────────────────────────

class TradingMode(str, Enum):
    PAPER        = "paper"          # local simulation, no API calls
    ALPACA_PAPER = "alpaca_paper"   # real Alpaca paper-account orders
    LIVE         = "live"           # real-money Alpaca live orders
    BACKTEST     = "backtest"


class RiskConfig(BaseModel):
    max_capital_per_trade_pct:       float = Field(default=5.0,  ge=0.1, le=100.0)
    max_daily_loss_pct:              float = Field(default=10.0, ge=0.1, le=100.0)
    max_open_positions:              int   = Field(default=10,   ge=1)
    default_max_loss_pct_of_capital: float = Field(default=50.0, ge=1.0, le=100.0)


class AlpacaConfig(BaseModel):
    paper_api_key:    str = ""
    paper_secret_key: str = ""
    live_api_key:     str = ""
    live_secret_key:  str = ""

    @field_validator("live_api_key", "live_secret_key",
                     "paper_api_key", "paper_secret_key", mode="before")
    @classmethod
    def _strip(cls, v: str) -> str:
        return (v or "").strip()

    def has_paper_credentials(self) -> bool:
        return bool(self.paper_api_key and self.paper_secret_key)

    def has_live_credentials(self) -> bool:
        return bool(self.live_api_key and self.live_secret_key)


class AppSettings(BaseModel):
    trading_mode: TradingMode  = TradingMode.PAPER
    alpaca:       AlpacaConfig = AlpacaConfig()
    risk:         RiskConfig   = RiskConfig()
    db_path:      Path         = Path("db/trading.db")
    log_dir:      Path         = Path("logs")

    @classmethod
    def from_env(cls) -> "AppSettings":
        mode_str = _get("TRADING_MODE", "app", "trading_mode", "paper").lower()
        return cls(
            trading_mode = TradingMode(mode_str),
            alpaca = AlpacaConfig(
                paper_api_key    = _get("ALPACA_PAPER_API_KEY",    "alpaca", "paper_api_key"),
                paper_secret_key = _get("ALPACA_PAPER_SECRET_KEY", "alpaca", "paper_secret_key"),
                live_api_key     = _get("ALPACA_LIVE_API_KEY",     "alpaca", "live_api_key"),
                live_secret_key  = _get("ALPACA_LIVE_SECRET_KEY",  "alpaca", "live_secret_key"),
            ),
            risk = RiskConfig(
                max_capital_per_trade_pct       = float(_get("MAX_CAPITAL_PER_TRADE_PCT",       "risk", "max_capital_per_trade_pct",       "5.0")),
                max_daily_loss_pct              = float(_get("MAX_DAILY_LOSS_PCT",              "risk", "max_daily_loss_pct",              "10.0")),
                max_open_positions              = int(  _get("MAX_OPEN_POSITIONS",              "risk", "max_open_positions",              "10")),
                default_max_loss_pct_of_capital = float(_get("DEFAULT_MAX_LOSS_PCT_OF_CAPITAL", "risk", "default_max_loss_pct_of_capital", "50.0")),
            ),
            db_path = Path(_get("DB_PATH",  "app", "db_path",  "db/trading.db")),
            log_dir = Path(_get("LOG_DIR",  "app", "log_dir",  "logs")),
        )

    def is_live(self)          -> bool: return self.trading_mode == TradingMode.LIVE
    def is_paper(self)         -> bool: return self.trading_mode == TradingMode.PAPER
    def is_alpaca_paper(self)  -> bool: return self.trading_mode == TradingMode.ALPACA_PAPER
    def is_backtest(self)      -> bool: return self.trading_mode == TradingMode.BACKTEST
    def uses_real_alpaca(self) -> bool:
        """True when the selected mode hits Alpaca's servers (paper or live)."""
        return self.trading_mode in (TradingMode.ALPACA_PAPER, TradingMode.LIVE)


# Singleton — import this everywhere
settings = AppSettings.from_env()

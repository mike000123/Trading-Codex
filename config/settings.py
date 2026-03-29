"""
config/settings.py
──────────────────
Central settings loaded from environment / .env file.
All other modules import from here – never read os.environ directly.
"""
from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

# Load .env from project root (works whether app is run from root or subdirs)
_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env", override=False)


class TradingMode(str, Enum):
    PAPER = "paper"
    LIVE = "live"
    BACKTEST = "backtest"


class RiskConfig(BaseModel):
    max_capital_per_trade_pct: float = Field(default=5.0, ge=0.1, le=100.0)
    max_daily_loss_pct: float = Field(default=10.0, ge=0.1, le=100.0)
    max_open_positions: int = Field(default=10, ge=1)
    default_max_loss_pct_of_capital: float = Field(default=50.0, ge=1.0, le=100.0)


class AlpacaConfig(BaseModel):
    paper_api_key: str = ""
    paper_secret_key: str = ""
    live_api_key: str = ""
    live_secret_key: str = ""

    @field_validator("live_api_key", "live_secret_key", mode="before")
    @classmethod
    def _strip(cls, v: str) -> str:
        return (v or "").strip()

    def has_paper_credentials(self) -> bool:
        return bool(self.paper_api_key and self.paper_secret_key)

    def has_live_credentials(self) -> bool:
        return bool(self.live_api_key and self.live_secret_key)


class AppSettings(BaseModel):
    trading_mode: TradingMode = TradingMode.PAPER
    alpaca: AlpacaConfig = AlpacaConfig()
    risk: RiskConfig = RiskConfig()
    db_path: Path = Path("db/trading.db")
    log_dir: Path = Path("logs")

    @classmethod
    def from_env(cls) -> "AppSettings":
        mode_str = os.getenv("TRADING_MODE", "paper").lower()
        return cls(
            trading_mode=TradingMode(mode_str),
            alpaca=AlpacaConfig(
                paper_api_key=os.getenv("ALPACA_PAPER_API_KEY", ""),
                paper_secret_key=os.getenv("ALPACA_PAPER_SECRET_KEY", ""),
                live_api_key=os.getenv("ALPACA_LIVE_API_KEY", ""),
                live_secret_key=os.getenv("ALPACA_LIVE_SECRET_KEY", ""),
            ),
            risk=RiskConfig(
                max_capital_per_trade_pct=float(os.getenv("MAX_CAPITAL_PER_TRADE_PCT", 5.0)),
                max_daily_loss_pct=float(os.getenv("MAX_DAILY_LOSS_PCT", 10.0)),
                max_open_positions=int(os.getenv("MAX_OPEN_POSITIONS", 10)),
                default_max_loss_pct_of_capital=float(
                    os.getenv("DEFAULT_MAX_LOSS_PCT_OF_CAPITAL", 50.0)
                ),
            ),
            db_path=Path(os.getenv("DB_PATH", "db/trading.db")),
            log_dir=Path(os.getenv("LOG_DIR", "logs")),
        )

    def is_live(self) -> bool:
        return self.trading_mode == TradingMode.LIVE

    def is_paper(self) -> bool:
        return self.trading_mode == TradingMode.PAPER


# Singleton – import this everywhere
settings = AppSettings.from_env()

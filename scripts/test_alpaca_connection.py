"""
scripts/test_alpaca_connection.py
─────────────────────────────────
One-shot smoke test for Alpaca paper-account connectivity.

Run from repo root:

    python scripts/test_alpaca_connection.py

What it does:
  1. Loads credentials from `.env` via config.settings.
  2. Instantiates a PAPER TradingClient (never touches the live endpoint).
  3. Fetches account info, open positions, recent orders, and clock status.
  4. Prints a clear pass/fail report.

Does NOT place any orders. Read-only.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Make repo imports work when run as `python scripts/test_alpaca_connection.py`
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _bold(s: str) -> str:
    return f"\033[1m{s}\033[0m"


def _ok(s: str) -> str:
    return f"\033[92m✓\033[0m {s}"


def _fail(s: str) -> str:
    return f"\033[91m✗\033[0m {s}"


def _info(s: str) -> str:
    return f"  {s}"


def main() -> int:
    print(_bold("Alpaca Connectivity Test — PAPER endpoint"))
    print("=" * 56)

    # 1. Credentials sanity
    try:
        from config.settings import settings
    except Exception as exc:  # pragma: no cover
        print(_fail(f"Failed to import settings: {exc}"))
        return 1

    api_key = settings.alpaca.paper_api_key or ""
    secret = settings.alpaca.paper_secret_key or ""

    if not api_key or not secret:
        print(_fail("Missing ALPACA_PAPER_API_KEY / ALPACA_PAPER_SECRET_KEY in .env"))
        return 2

    print(_ok(f"Credentials loaded (key ends with …{api_key[-4:]}, "
              f"secret length = {len(secret)} chars)"))
    if len(secret) < 40:
        print(_info(
            f"Note: Alpaca secret keys are typically 40 chars; yours is "
            f"{len(secret)}. If the API calls below return 401, paste the "
            f"full secret from the Alpaca UI into `.env` directly."
        ))

    # 2. Import SDK
    try:
        from alpaca.trading.client import TradingClient
        from alpaca.common.exceptions import APIError
    except ImportError:
        print(_fail("alpaca-py not installed. Run: pip install alpaca-py"))
        return 3

    # 3. Instantiate client (paper endpoint)
    try:
        client = TradingClient(api_key, secret, paper=True)
    except Exception as exc:
        print(_fail(f"Failed to instantiate TradingClient: {exc}"))
        return 4

    # 4. Account
    try:
        account = client.get_account()
    except APIError as exc:
        print(_fail(f"get_account() failed: HTTP {getattr(exc, 'status_code', '?')} — {exc}"))
        print(_info("Likely causes: wrong key, wrong secret, or paper account not enabled."))
        return 5
    except Exception as exc:
        print(_fail(f"get_account() unexpected error: {exc}"))
        return 6

    print()
    print(_bold("Account"))
    print(_info(f"ID:              {account.id}"))
    print(_info(f"Status:          {account.status}"))
    print(_info(f"Equity:          ${float(account.equity):,.2f}"))
    print(_info(f"Cash:            ${float(account.cash):,.2f}"))
    print(_info(f"Buying power:    ${float(account.buying_power):,.2f}"))
    print(_info(f"Pattern day trader: {bool(account.pattern_day_trader)}"))
    print(_info(f"Trading blocked: {bool(account.trading_blocked)}"))
    print(_info(f"Account blocked: {bool(account.account_blocked)}"))

    # 5. Clock (market open?)
    try:
        clock = client.get_clock()
        print()
        print(_bold("Market clock"))
        print(_info(f"Now (ET):     {clock.timestamp}"))
        print(_info(f"Market open:  {clock.is_open}"))
        print(_info(f"Next open:    {clock.next_open}"))
        print(_info(f"Next close:   {clock.next_close}"))
    except Exception as exc:
        print(_fail(f"get_clock() failed: {exc}"))

    # 6. Positions (read-only)
    try:
        positions = client.get_all_positions()
        print()
        print(_bold("Open positions"))
        if not positions:
            print(_info("(none)"))
        else:
            for p in positions:
                print(_info(
                    f"{p.symbol:8s}  qty={p.qty:>10s}  "
                    f"avg_entry=${float(p.avg_entry_price):,.4f}  "
                    f"mkt=${float(p.market_value):,.2f}  "
                    f"upl=${float(p.unrealized_pl):+,.2f}"
                ))
    except Exception as exc:
        print(_fail(f"get_all_positions() failed: {exc}"))

    # 7. Recent orders (last 7 days, read-only)
    try:
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus
        req = GetOrdersRequest(
            status=QueryOrderStatus.ALL,
            after=datetime.now(timezone.utc) - timedelta(days=7),
            limit=10,
        )
        orders = client.get_orders(filter=req)
        print()
        print(_bold("Recent orders (last 7 days, up to 10)"))
        if not orders:
            print(_info("(none)"))
        else:
            for o in orders:
                fp = f"${float(o.filled_avg_price):.4f}" if o.filled_avg_price else "—"
                print(_info(
                    f"{str(o.submitted_at)[:19]}  {o.symbol:6s}  "
                    f"{o.side.value:5s}  qty={o.qty:>8s}  "
                    f"status={o.status.value:<10s}  fill={fp}"
                ))
    except Exception as exc:
        print(_fail(f"get_orders() failed: {exc}"))

    print()
    print(_bold("Result: ") + _ok("PAPER connectivity OK — you can safely submit orders."))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

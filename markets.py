"""
markets.py

Polymarket BTC 15-minute slug/time helpers.

Single responsibility:
- Compute current/next market slug strings
- Compute candle open/close timestamps
- Compute seconds until close/open

No API calls, no streaming, no trading decisions.
"""

from __future__ import annotations

import time
import datetime as dt
from zoneinfo import ZoneInfo


INTERVAL_SECONDS = 15 * 60
SLUG_PREFIX = "btc-updown-15m-"
BTC_TICKER_PREFIX = "KXBTC15M-"
ETH_TICKER_PREFIX = "KXETH15M-"
EDT_TZ = ZoneInfo("America/New_York")


def _now_ts() -> int:
    """Current UNIX timestamp in whole seconds."""
    return int(time.time())


def candle_open_ts(ts: int | float | None = None) -> int:
    """
    UNIX timestamp of the current 15-minute candle open.

    If ``ts`` is provided, computes the candle open for that timestamp.
    """
    t = _now_ts() if ts is None else int(ts)
    return t - (t % INTERVAL_SECONDS)


def candle_close_ts(ts: int | float | None = None) -> int:
    """
    UNIX timestamp of the current 15-minute candle close.

    If ``ts`` is provided, computes the candle close for that timestamp.
    """
    return candle_open_ts(ts) + INTERVAL_SECONDS


def slug_from_ts(ts: int | float) -> str:
    """
    Compute BTC 15m market slug for a given timestamp.

    Slug timestamp corresponds to the candle close boundary for the
    15-minute interval containing ``ts``.
    """
    return f"{SLUG_PREFIX}{candle_close_ts(ts)}"


def ticker_from_ts(ts: int | float, symbol: str = "BTCUSDT") -> str:
    """
    Compute BTC 15m market ticker for a given timestamp.
    """

    if symbol == "BTCUSDT":
        prefix = BTC_TICKER_PREFIX
    elif symbol == "ETHUSDT":
        prefix = ETH_TICKER_PREFIX

    time = dt.datetime.fromtimestamp(candle_close_ts(ts), tz=EDT_TZ)
    return f"{prefix}{time.strftime('%y%b%d%H%M')}-{time.strftime('%M')}".upper()

def current_ticker(symbol: str = "BTCUSDT") -> str:
    """Compute active market ticker from system clock."""
    return ticker_from_ts(_now_ts(), symbol)

def next_ticker(symbol: str = "BTCUSDT") -> str:
    """Compute next market ticker from system clock (for pre-warming)."""
    return ticker_from_ts(_now_ts() + INTERVAL_SECONDS, symbol)

def current_slug() -> str:
    """Compute active market slug from system clock."""
    return slug_from_ts(_now_ts())


def next_slug() -> str:
    """Compute next market slug from system clock (for pre-warming)."""
    return slug_from_ts(_now_ts() + INTERVAL_SECONDS)


def seconds_until_close(ts: int | float | None = None) -> int:
    """Seconds until the current candle closes."""
    t = _now_ts() if ts is None else int(ts)
    return max(0, candle_close_ts(t) - t)


def seconds_until_open(ts: int | float | None = None) -> int:
    """Seconds until the next candle opens."""
    return seconds_until_close(ts)

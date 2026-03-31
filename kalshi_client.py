"""
kalshi_client.py

Public Kalshi API communication layer (no authentication).

Responsibilities
----------------
- Fetch market metadata from Kalshi public REST endpoints
- Poll orderbook snapshots and expose YES/NO best bid/ask prices
- Keep a state dict interface compatible with the existing client pattern
"""

from __future__ import annotations

import asyncio
from typing import Dict, Optional

import requests


KALSHI_API_BASE = "https://api.elections.kalshi.com/trade-api/v2"


def _to_float(value: object) -> Optional[float]:
    """Best-effort conversion to float, returning None on failure."""
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _best_bid_from_side(side_levels: object) -> Optional[float]:
    """
    Parse the best bid from Kalshi orderbook side levels.

    Expected shape (from orderbook_fp):
      [[price_dollars, quantity], ...]
    """
    if not isinstance(side_levels, list) or not side_levels:
        return None

    best: Optional[float] = None
    for level in side_levels:
        if not isinstance(level, (list, tuple)) or not level:
            continue
        price = _to_float(level[0])
        if price is None:
            continue
        if best is None or price > best:
            best = price
    return best


def _compute_best_quotes(orderbook: dict) -> tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    Compute yes_bid, yes_ask, no_bid, no_ask from Kalshi orderbook snapshot.

    Kalshi orderbook responses provide YES and NO bid ladders. Asks are implied:
      yes_ask ~= 1 - no_bid
      no_ask  ~= 1 - yes_bid
    """
    orderbook_fp = orderbook.get("orderbook_fp", {})
    yes_levels = orderbook_fp.get("yes_dollars", [])
    no_levels = orderbook_fp.get("no_dollars", [])

    yes_bid = _best_bid_from_side(yes_levels)
    no_bid = _best_bid_from_side(no_levels)

    yes_ask = (1.0 - no_bid) if no_bid is not None else None
    no_ask = (1.0 - yes_bid) if yes_bid is not None else None

    return yes_bid, yes_ask, no_bid, no_ask


def fetch_market_by_ticker(ticker: str) -> dict:
    """
    Fetch market metadata for a Kalshi market ticker.

    Parameters
    ----------
    ticker : Kalshi market ticker (e.g., 'KXBTC-26MAR15-B95000')

    Returns
    -------
    dict — raw market object from Kalshi API under 'market'
    """
    r = requests.get(f"{KALSHI_API_BASE}/markets/{ticker}", timeout=30)
    r.raise_for_status()
    data = r.json()

    market = data.get("market")
    if not isinstance(market, dict):
        raise ValueError(f"No market payload found for ticker='{ticker}'")
    return market


def fetch_market_by_slug(slug: str) -> dict:
    """
    Compatibility alias with the existing Polymarket client interface.

    In Kalshi, this argument should be the market ticker.
    """
    return fetch_market_by_ticker(slug)


def fetch_orderbook_by_ticker(ticker: str, depth: int = 0) -> dict:
    """
    Fetch public orderbook snapshot for a Kalshi market ticker.

    Parameters
    ----------
    ticker : Kalshi market ticker
    depth  : orderbook depth (0 returns full depth per Kalshi docs)
    """
    r = requests.get(
        f"{KALSHI_API_BASE}/markets/{ticker}/orderbook",
        params={"depth": depth},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


async def stream_prices(
    slug: str,
    state: Dict[str, Dict[str, Optional[str]]],
    *,
    ping_interval: int = 10,  # kept for signature compatibility
    poll_interval: float = 1.0,
    depth: int = 0,
) -> None:
    """
    Poll Kalshi public orderbook and keep YES/NO best bid/ask in shared state.

    State structure
    ---------------
    {
        "<ticker>:YES": {"bid": "0.54", "ask": "0.56"},
        "<ticker>:NO":  {"bid": "0.44", "ask": "0.46"},
        "yes_id": "<ticker>:YES",
        "no_id": "<ticker>:NO",
        "ticker": "<ticker>",
    }
    """
    del ping_interval  # not used in polling mode

    ticker = slug
    yes_id = f"{ticker}:YES"
    no_id = f"{ticker}:NO"

    state[yes_id] = {"bid": None, "ask": None}
    state[no_id] = {"bid": None, "ask": None}
    state["yes_id"] = yes_id
    state["no_id"] = no_id
    state["ticker"] = ticker

    while True:
        try:
            orderbook = fetch_orderbook_by_ticker(ticker, depth=depth)
            yes_bid, yes_ask, no_bid, no_ask = _compute_best_quotes(orderbook)

            state[yes_id]["bid"] = str(yes_bid) if yes_bid is not None else None
            state[yes_id]["ask"] = str(yes_ask) if yes_ask is not None else None
            state[no_id]["bid"] = str(no_bid) if no_bid is not None else None
            state[no_id]["ask"] = str(no_ask) if no_ask is not None else None
        except requests.RequestException as e:
            raise ConnectionError(f"Kalshi orderbook polling failed: {e}")

        await asyncio.sleep(max(0.05, poll_interval))


def get_yes_ask(state: Dict) -> Optional[float]:
    """Read current YES ask from state; return None if unavailable."""
    yes_id = state.get("yes_id")
    if not yes_id:
        return None
    ask = state.get(yes_id, {}).get("ask")
    return float(ask) if ask is not None else None


def get_no_ask(state: Dict) -> Optional[float]:
    """Read current NO ask from state; return None if unavailable."""
    no_id = state.get("no_id")
    if not no_id:
        return None
    ask = state.get(no_id, {}).get("ask")
    return float(ask) if ask is not None else None

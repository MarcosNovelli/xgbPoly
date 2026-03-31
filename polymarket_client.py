"""
client.py

Raw Polymarket API communication layer.

Responsibilities
----------------
- Fetch market metadata from the Gamma API
- Stream live YES/NO prices via the CLOB WebSocket
- No business logic, no slug generation, no trading decisions

APIs used
---------
- Gamma API : https://gamma-api.polymarket.com  (market metadata, no auth)
- CLOB WS   : wss://ws-subscriptions-clob.polymarket.com/ws/market (live prices)
"""

import asyncio
import json
from typing import Dict, Optional, Tuple

import requests
import websockets


GAMMA_URL = "https://gamma-api.polymarket.com"
CLOB_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

def _extract_yes_no_token_ids(market: dict) -> Tuple[str, str]:
    """
    Extract YES/NO token IDs from a Gamma market object.

    Supports:
      - market["tokens"] as list of dicts
      - market["clobTokenIds"] as list or JSON string
    """
    tokens = market.get("tokens")
    if isinstance(tokens, list) and tokens:
        yes_id, no_id = None, None
        for t in tokens:
            outcome = str(t.get("outcome", "")).strip().lower()
            tid = t.get("token_id") or t.get("tokenId") or t.get("clobTokenId")
            if not tid:
                continue
            if outcome in {"yes", "up", "true", "1"}:
                yes_id = str(tid)
            elif outcome in {"no", "down", "false", "0"}:
                no_id = str(tid)
        if yes_id and no_id:
            return yes_id, no_id
        tids = [str(t.get("token_id")) for t in tokens if t.get("token_id")]
        if len(tids) >= 2:
            return tids[0], tids[1]

    token_ids = market.get("clobTokenIds")
    if isinstance(token_ids, str):
        token_ids = token_ids.strip()
        if token_ids.startswith("["):
            token_ids = json.loads(token_ids)
        else:
            token_ids = [x.strip() for x in token_ids.split(",") if x.strip()]
    if isinstance(token_ids, list) and len(token_ids) >= 2:
        return str(token_ids[0]), str(token_ids[1])

    raise ValueError(
        "Could not extract YES/NO token IDs from market object."
    )

def fetch_market_by_slug(slug: str) -> dict:
    """
    Fetch market metadata from the Gamma API for a given slug.

    Parameters
    ----------
    slug : Polymarket market slug e.g. 'btc-updown-15m-1773245700'

    Returns
    -------
    dict — raw market object from Gamma API

    Raises
    ------
    ValueError if no market is found for the slug
    requests.HTTPError on API failure
    """
    r = requests.get(
        f"{GAMMA_URL}/markets",
        params={"slug": slug},
        timeout=30,
    )
    r.raise_for_status()
    markets = r.json()
    if not markets:
        raise ValueError(f"No market found for slug='{slug}'")
    return markets[0]

async def stream_prices(
    slug: str,
    state: Dict[str, Dict[str, Optional[str]]],
    *,
    ping_interval: int = 10,
) -> None:
    """
    Connect to Polymarket CLOB WebSocket and stream live YES/NO prices.

    Updates state dict in place continuously. Caller reads from state
    whenever it needs the current price — no data is returned directly.

    State structure
    ---------------
    {
        yes_id: {"bid": "0.54", "ask": "0.56"},
        no_id:  {"bid": "0.44", "ask": "0.46"},
    }

    Parameters
    ----------
    slug           : active Polymarket market slug
    state          : shared dict updated in place with latest prices
    ping_interval  : WebSocket keepalive ping interval in seconds

    Raises
    ------
    ConnectionError on unrecoverable WebSocket failure
    """
    market = fetch_market_by_slug(slug)
    yes_id, no_id = _extract_yes_no_token_ids(market)

    # Initialise state keys
    state[yes_id] = {"bid": None, "ask": None}
    state[no_id]  = {"bid": None, "ask": None}

    # Store token IDs on state so caller can look up yes_id/no_id
    state["yes_id"] = yes_id
    state["no_id"]  = no_id

    async with websockets.connect(CLOB_WS_URL, ping_interval=ping_interval) as ws:
        await ws.send(json.dumps({
            "assets_ids"           : [yes_id, no_id],
            "type"                 : "market",
            "custom_feature_enabled": True,
        }))

        while True:
            try:
                raw = await ws.recv()
            except (
                websockets.ConnectionClosedError,
                websockets.ConnectionClosed,
                asyncio.TimeoutError,
                OSError,
            ) as e:
                raise ConnectionError(f"WebSocket connection lost: {e}")

            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue

            messages = (
                [data] if isinstance(data, dict)
                else data if isinstance(data, list)
                else []
            )

            for msg in messages:
                if not isinstance(msg, dict):
                    continue
                _handle_message(msg, state)

def _handle_message(
    msg: dict,
    state: Dict[str, Dict[str, Optional[str]]],
) -> None:
    """
    Parse a single WebSocket message and update state in place.

    Handles three event types:
      book            — full orderbook snapshot
      best_bid_ask    — top of book update
      price_change    — incremental price update
    """
    event_type = msg.get("event_type")

    if event_type == "book":
        asset_id = msg.get("asset_id")
        if asset_id not in state:
            return
        bids = msg.get("bids") or []
        asks = msg.get("asks") or []
        state[asset_id]["bid"] = (
            max(bids, key=lambda x: float(x["price"]))["price"]
            if bids else None
        )
        state[asset_id]["ask"] = (
            min(asks, key=lambda x: float(x["price"]))["price"]
            if asks else None
        )

    elif event_type == "best_bid_ask":
        asset_id = msg.get("asset_id")
        if asset_id not in state:
            return
        state[asset_id]["bid"] = msg.get("best_bid")
        state[asset_id]["ask"] = msg.get("best_ask")

    elif event_type == "price_change":
        for pc in msg.get("price_changes", []):
            asset_id = pc.get("asset_id")
            if asset_id not in state:
                continue
            state[asset_id]["bid"] = pc.get("best_bid")
            state[asset_id]["ask"] = pc.get("best_ask")

def get_yes_ask(state: Dict) -> Optional[float]:
    """
    Read current YES ask price from state.
    Returns None if price not yet available.
    """
    yes_id = state.get("yes_id")
    if not yes_id:
        return None
    ask = state.get(yes_id, {}).get("ask")
    return float(ask) if ask is not None else None


def get_no_ask(state: Dict) -> Optional[float]:
    """
    Read current NO ask price from state.
    Returns None if price not yet available.
    """
    no_id = state.get("no_id")
    if not no_id:
        return None
    ask = state.get(no_id, {}).get("ask")
    return float(ask) if ask is not None else None
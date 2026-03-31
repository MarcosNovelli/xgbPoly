"""
bot.py

Starter runtime loop for a paper Polymarket BTC 15m bot.

What this file does:
- Tracks active Polymarket slug and rolls over every 15 minutes
- Streams YES/NO asks for active market via client.py
- Fetches recent BTCUSDT 15m candles from Binance public REST
- Builds latest feature row and runs model prediction
- Computes YES/NO edge and outputs a paper decision
- Appends structured JSONL logs (no trade execution)
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import datetime as dt

from features import build_features
from markets import current_slug, seconds_until_close, current_ticker
from predict import Predictor
from kalshi_api import market_order_kalshi


BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"


@dataclass
class BotConfig:
    model_path: str
    provider: str = "kalshi"
    market_ticker: str | None = None
    min_edge: float = 0.01
    poll_seconds: float = 2.0
    klines_limit: int = 300
    logs_path: str = "logs/bot_events.jsonl"
    symbol: str = "BTCUSDT"
    interval: str = "15m"
    min_seconds_left: int = 900
    heartbeat_seconds: int = 30
    log_level: str = "INFO"
    no_color: bool = False


@dataclass
class Decision:
    action: str
    p_up: float
    edge_yes: float
    edge_no: float
    chosen_edge: float
    reason: str


def _setup_logger(level: str, no_color: bool) -> logging.Logger:
    """Create terminal logger with rich colors when available."""
    logger = logging.getLogger("xgbpoly.bot")
    logger.propagate = False
    logger.handlers.clear()

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    log_fmt = "%(asctime)s | %(levelname)-7s | %(message)s"
    date_fmt = "%H:%M:%S"

    if not no_color:
        try:
            from rich.logging import RichHandler

            handler = RichHandler(
                show_time=False,
                rich_tracebacks=True,
                markup=False,
            )
            handler.setFormatter(logging.Formatter(log_fmt, datefmt=date_fmt))
            logger.addHandler(handler)
            return logger
        except Exception:
            # Fall back to standard logger if rich is unavailable.
            pass

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(log_fmt, datefmt=date_fmt))
    logger.addHandler(handler)
    return logger


def _format_decision_line(
    provider: str,
    market: str,
    decision: Decision,
    yes_ask: float,
    no_ask: float,
    close_in_s: int,
) -> str:
    """Return aligned, scan-friendly decision line for terminal output."""
    return (
        f"{provider:<10} | {market:<18} | {decision.action:<7} | "
        f"p_up={decision.p_up:>6.3f} | yes={yes_ask:>6.3f} | no={no_ask:>6.3f} | "
        f"e_yes={decision.edge_yes:>+7.4f} | e_no={decision.edge_no:>+7.4f} | "
        f"close_in={close_in_s:>4d}s | reason={decision.reason}"
    )


def fetch_recent_klines(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    """Fetch recent klines and return standardized OHLCV DataFrame."""
    print(f"Fetching {limit} {interval} klines for {symbol}")
    r = requests.get(
        BINANCE_KLINES_URL,
        params={"symbol": symbol, "interval": interval, "limit": limit},
        timeout=20,
    )
    r.raise_for_status()
    rows = r.json()
    if not rows:
        raise ValueError("No klines returned from Binance.")

    out = []
    for k in rows:
        # Binance kline schema:
        # [open_time, open, high, low, close, volume, close_time, ...]
        out.append(
            {
                "timestamp": int(k[0]),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
                "close_time_ms": int(k[6]),
            }
        )

    return pd.DataFrame(out)


def latest_confirmed_row(klines: pd.DataFrame) -> pd.Series:
    """
    Return the latest fully closed candle row.

    Binance may include an in-progress final candle, so we filter by close time.
    """
    now_ms = int(time.time() * 1000)
    closed = klines[klines["close_time_ms"] <= now_ms].copy()
    if closed.empty:
        raise ValueError("No closed candles available yet.")
    return closed.iloc[-1]


def compute_decision(p_up: float, yes_ask: float, no_ask: float, min_edge: float, min_seconds_left: int) -> Decision:
    """Compute paper decision from predicted probability and current asks."""
    edge_yes = p_up - yes_ask
    edge_no = (1.0 - p_up) - no_ask

    if seconds_until_close() < min_seconds_left:
        return Decision(
            action="HOLD",
            p_up=p_up,
            edge_yes=edge_yes,
            edge_no=edge_no,
            chosen_edge=max(edge_yes, edge_no),
            reason="seconds_until_close_below_threshold",
        )

    if max(edge_yes, edge_no) <= min_edge:
        return Decision(
            action="HOLD",
            p_up=p_up,
            edge_yes=edge_yes,
            edge_no=edge_no,
            chosen_edge=max(edge_yes, edge_no),
            reason="edge_below_threshold",
        )

    if edge_yes >= edge_no:
        return Decision(
            action="BUY_YES",
            p_up=p_up,
            edge_yes=edge_yes,
            edge_no=edge_no,
            chosen_edge=edge_yes,
            reason="yes_edge_greater",
        )

    return Decision(
        action="BUY_NO",
        p_up=p_up,
        edge_yes=edge_yes,
        edge_no=edge_no,
        chosen_edge=edge_no,
        reason="no_edge_greater",
    )

def compute_decision_v2(p_up: float, yes_ask: float, no_ask: float, min_edge: float, min_seconds_left: int) -> Decision:
    """Enter a trade when predicted prob is > .5 + min_edge """
    
    if seconds_until_close() < min_seconds_left:
        return Decision(
            action="HOLD",
            p_up=p_up,
            edge_yes=0,
            edge_no=0,
            chosen_edge=0,
            reason="seconds_until_close_below_threshold",
        )

    if p_up >= .5 + min_edge:
        return Decision(
            action="BUY_YES",
            p_up=p_up,
            edge_yes=p_up - yes_ask,
            edge_no=(1.0 - p_up) - no_ask,
            chosen_edge=p_up - yes_ask,
            reason="p_up_greater_than_threshold",
        )

    if p_up <= .5 - min_edge:
        return Decision(
            action="BUY_NO",
            p_up=p_up,
            edge_yes=p_up - yes_ask,
            edge_no=(1.0 - p_up) - no_ask,
            chosen_edge=(1.0 - p_up) - no_ask,
            reason="p_up_less_than_threshold",
        )

    return Decision(
        action="HOLD",
        p_up=p_up,
        edge_yes=0,
        edge_no=0,
        chosen_edge=0,
        reason="p_up_between_threshold",
    )


def append_event(log_path: Path, event: dict[str, Any]) -> None:
    """Append one JSON line to the event log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=True) + "\n")


def _load_provider_client(provider: str):
    """Return provider-specific client module functions."""
    if provider == "kalshi":
        from kalshi_client import get_no_ask, get_yes_ask, stream_prices
    else:
        from polymarket_client import get_no_ask, get_yes_ask, stream_prices
    return stream_prices, get_yes_ask, get_no_ask


async def stream_task_runner(
    market_id: str,
    state: dict[str, Any],
    stream_prices_fn,
) -> None:
    """Run the market data stream task for the selected provider."""
    await stream_prices_fn(market_id, state)


async def run_bot(cfg: BotConfig) -> None:
    """Main paper-bot loop."""
    logger = _setup_logger(cfg.log_level, cfg.no_color)
    predictor = Predictor.from_file(cfg.model_path)
    log_path = Path(cfg.logs_path)
    stream_prices_fn, get_yes_ask_fn, get_no_ask_fn = _load_provider_client(cfg.provider)

    if cfg.provider == "kalshi":
        active_market = current_ticker(symbol=cfg.symbol)
    else:
        active_market = current_slug()

    state: dict[str, Any] = {}
    ws_task = asyncio.create_task(stream_task_runner(active_market, state, stream_prices_fn))
    logger.info("STARTUP provider=%s market=%s", cfg.provider, active_market)

    last_candle_ts: int | None = None
    last_heartbeat_ts = 0.0

    try:
        while True:
            loop_started = time.time()

            if cfg.provider == "polymarket":
                new_market = current_slug()
            else:
                new_market = current_ticker(cfg.symbol)

            if new_market != active_market:
                pre_roll_yes_ask = get_yes_ask_fn(state)
                pre_roll_no_ask = get_no_ask_fn(state)
                ws_task.cancel()
                try:
                    await ws_task
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    logger.warning("Old websocket task ended with error: %s", e)

                active_market = new_market
                state = {}
                ws_task = asyncio.create_task(
                    stream_task_runner(active_market, state, stream_prices_fn)
                )
                logger.info(
                    "ROLLOVER provider=%s market=%s prev_yes=%s prev_no=%s",
                    cfg.provider,
                    active_market,
                    pre_roll_yes_ask,
                    pre_roll_no_ask,
                )
                append_event(
                    log_path,
                    {
                        "event_type": "market_roll",
                        "ts": int(loop_started),
                        "provider": cfg.provider,
                        "active_market": active_market,
                        "prev_yes_ask": pre_roll_yes_ask,
                        "prev_no_ask": pre_roll_no_ask,
                    },
                )

            yes_ask = get_yes_ask_fn(state)
            no_ask = get_no_ask_fn(state)
            now = time.time()
            if now - last_heartbeat_ts >= cfg.heartbeat_seconds:
                status = "waiting_for_quotes" if (yes_ask is None or no_ask is None) else "running"
                logger.debug(
                    "Heartbeat status=%s provider=%s market=%s yes=%s no=%s last_candle_ts=%s",
                    status,
                    cfg.provider,
                    active_market,
                    yes_ask,
                    no_ask,
                    last_candle_ts,
                )
                last_heartbeat_ts = now
            if yes_ask is None or no_ask is None:
                await asyncio.sleep(cfg.poll_seconds)
                continue

            try:
                if (dt.datetime.now().minute % 15 == 0  and dt.datetime.now().second < 7) or last_candle_ts is None:

                    klines = fetch_recent_klines(cfg.symbol, cfg.interval, cfg.klines_limit)
                    latest_closed = latest_confirmed_row(klines)

                # Run prediction once per closed candle (keeps logs cleaner).
                candle_ts = int(latest_closed["timestamp"])
                if last_candle_ts is not None and candle_ts == last_candle_ts:
                    await asyncio.sleep(cfg.poll_seconds)
                    continue

                last_candle_ts = candle_ts
                featured = build_features(klines[["timestamp", "open", "high", "low", "close", "volume"]])
                latest_features = featured.iloc[-1]
                p_up = predictor.predict(latest_features)
                decision = compute_decision_v2(p_up, yes_ask, no_ask, cfg.min_edge, cfg.min_seconds_left)

                if decision.action == "BUY_YES":
                    order_response = market_order_kalshi(active_market, "yes", "buy", 1)
                    logger.info("BUY_YES order response: %s", order_response)
                elif decision.action == "BUY_NO":
                    order_response = market_order_kalshi(active_market, "no", "buy", 1)
                    logger.info("BUY_NO order response: %s", order_response)

                event = {
                    "event_type": "decision",
                    "ts": int(loop_started),
                    "provider": cfg.provider,
                    "active_market": active_market,
                    "seconds_until_close": (
                        seconds_until_close()
                    ),
                    "candle_ts_ms": candle_ts,
                    "candle_close": float(latest_closed["close"]),
                    "yes_ask": yes_ask,
                    "no_ask": no_ask,
                    **asdict(decision),
                }
                append_event(log_path, event)

                close_in = int(event["seconds_until_close"])
                decision_line = _format_decision_line(
                    provider=cfg.provider,
                    market=active_market,
                    decision=decision,
                    yes_ask=yes_ask,
                    no_ask=no_ask,
                    close_in_s=close_in,
                )
                if decision.action == "HOLD":
                    logger.info("%s", decision_line)
                else:
                    logger.info("%s", decision_line)
            except Exception as e:
                err_event = {
                    "event_type": "error",
                    "ts": int(loop_started),
                    "provider": cfg.provider,
                    "active_market": active_market,
                    "error": str(e),
                }
                append_event(log_path, err_event)
                logger.exception(
                    "Loop error provider=%s market=%s last_candle_ts=%s",
                    cfg.provider,
                    active_market,
                    last_candle_ts,
                )

            elapsed = time.time() - loop_started
            sleep_for = max(0.0, cfg.poll_seconds - elapsed)
            await asyncio.sleep(sleep_for)
    finally:
        ws_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await ws_task


def parse_args() -> BotConfig:
    parser = argparse.ArgumentParser(description="Run starter paper bot loop.")
    parser.add_argument("--model-path", required=True, help="Path to xgb_calibrated.pkl")
    parser.add_argument(
        "--provider",
        choices=["polymarket", "kalshi"],
        default="polymarket",
        help="Market data provider to use",
    )

    parser.add_argument("--min-edge", type=float, default=0.05, help="Min EV edge to trigger action")
    parser.add_argument("--poll-seconds", type=float, default=2.0, help="Loop polling interval")
    parser.add_argument("--klines-limit", type=int, default=300, help="Binance 15m candles to fetch")
    parser.add_argument("--logs-path", default="logs/bot_events.jsonl", help="JSONL event log path")
    parser.add_argument("--symbol", default="BTCUSDT", help="Binance symbol")
    parser.add_argument("--interval", default="15m", help="Binance interval")
    parser.add_argument("--min-seconds-left", type=int, default=800, help="Min seconds left to trigger action")
    parser.add_argument(
        "--heartbeat-seconds",
        type=int,
        default=30,
        help="Emit debug heartbeat every N seconds while waiting for quotes",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Terminal log verbosity",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable rich/color terminal logging",
    )

    args = parser.parse_args()

    return BotConfig(
        model_path=args.model_path,
        provider=args.provider,
        min_edge=args.min_edge,
        poll_seconds=args.poll_seconds,
        klines_limit=args.klines_limit,
        logs_path=args.logs_path,
        symbol=args.symbol,
        interval=args.interval,
        min_seconds_left=args.min_seconds_left,
        heartbeat_seconds=args.heartbeat_seconds,
        log_level=args.log_level,
        no_color=args.no_color,
    )


if __name__ == "__main__":
    config = parse_args()
    asyncio.run(run_bot(config))

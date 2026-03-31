"""
Fetch crypto candlestick (OHLCV) data from Binance API and save to Parquet.
Usage: specify start date, end date, symbol (e.g. BTCUSDT), and timeframe.
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
MAX_KLINES_PER_REQUEST = 1000

# --- Set these to run with: python binance.py (CLI args override these) ---
SYMBOL = "ETHUSDT"
INTERVAL = "15m"  # 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
START_DATE = "2020-01-01"
END_DATE = "2026-03-27"
OUTPUT_FILE = None  # None = auto-generated name from symbol, interval, dates


def fetch_klines(symbol: str, interval: str, start_time_ms: int, end_time_ms: int) -> list:
    """Fetch klines from Binance, paginating if needed (max 1000 per request)."""
    all_klines = []
    current_start = start_time_ms

    while current_start < end_time_ms:
        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "startTime": current_start,
            "endTime": end_time_ms,
            "limit": MAX_KLINES_PER_REQUEST,
        }
        resp = requests.get(BINANCE_KLINES_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            break
        all_klines.extend(data)
        current_start = data[-1][0] + 1
        if len(data) < MAX_KLINES_PER_REQUEST:
            break

    return all_klines


def klines_to_dataframe(klines: list) -> pd.DataFrame:
    """Build a DataFrame from raw Binance kline rows (arrays of 12 fields)."""
    if not klines:
        return pd.DataFrame()

    columns = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time_ms",
        "quote_volume",
        "trades",
        "taker_buy_base",
        "taker_buy_quote",
        "_ignore",
    ]
    df = pd.DataFrame(klines, columns=columns)

    for col in ("open", "high", "low", "close", "volume", "quote_volume", "taker_buy_base", "taker_buy_quote"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["trades"] = pd.to_numeric(df["trades"], errors="coerce").astype("Int64")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").astype("Int64")

    df = df.drop(columns=["close_time_ms", "_ignore", "trades", "quote_volume", "taker_buy_base", "taker_buy_quote"])

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Fetch Binance OHLCV candlestick data and save to Parquet."
    )
    parser.add_argument(
        "--symbol", "-s",
        default=SYMBOL,
        help="Trading pair symbol (e.g. BTCUSDT, ETHUSDT).",
    )
    parser.add_argument(
        "--interval", "-i",
        default=INTERVAL,
        choices=[
            "1m", "3m", "5m", "15m", "30m",
            "1h", "2h", "4h", "6h", "8h", "12h",
            "1d", "3d", "1w", "1M",
        ],
        help="Candlestick interval (timeframe).",
    )
    parser.add_argument(
        "--start", "-S",
        default=START_DATE,
        help="Start date/time (e.g. 2024-01-01 or 2024-01-01 00:00).",
    )
    parser.add_argument(
        "--end", "-E",
        default=END_DATE,
        help="End date/time (e.g. 2024-02-01 or 2024-02-01 23:59).",
    )
    parser.add_argument(
        "--output", "-o",
        default=OUTPUT_FILE,
        help="Output Parquet path. Default: auto-generated from symbol, interval, dates.",
    )
   
    args = parser.parse_args()

    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d", "%m/%d/%Y %H:%M", "%m/%d/%Y"):
        try:
            start_dt = datetime.strptime(args.start.strip(), fmt)
            break
        except ValueError:
            start_dt = None
    if start_dt is None:
        raise SystemExit(f"Invalid --start format: {args.start}")

    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d", "%m/%d/%Y %H:%M", "%m/%d/%Y"):
        try:
            end_dt = datetime.strptime(args.end.strip(), fmt)
            break
        except ValueError:
            end_dt = None
    if end_dt is None:
        raise SystemExit(f"Invalid --end format: {args.end}")

    start_dt = start_dt.replace(hour=0, minute=0, second=0, microsecond=0)

    
    start_dt = start_dt.replace(tzinfo=timezone.utc)
    end_dt = end_dt.replace(tzinfo=timezone.utc)
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    if start_ms >= end_ms:
        raise SystemExit("--start must be before --end.")

    klines = fetch_klines(args.symbol, args.interval, start_ms, end_ms)
    if not klines:
        print("No data returned for the given range.")
        return

    df = klines_to_dataframe(klines)

    out_path = args.output
    if not out_path:
        safe_start = args.start.replace(" ", "_").replace("/", "-")
        safe_end = args.end.replace(" ", "_").replace("/", "-")
        out_path = f"{args.symbol}_{args.interval}_{safe_start}_{safe_end}.csv"

    out_path = Path(out_path)
    # df.to_parquet(out_path, index=False, engine="pyarrow")
    df.to_csv(out_path, index=False)

    print(f"Wrote {len(df)} candlesticks to {out_path}")


if __name__ == "__main__":
    main()

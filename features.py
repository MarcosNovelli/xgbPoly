"""
Feature engineering pipeline for BTC 15-min OHLCV data.

All features are computed from prior candles only. The entire feature matrix
is shifted by 1 at the end of build_features() to prevent any lookahead.
The label column uses the current candle and is never shifted.

Input contract:
    DataFrame with columns: timestamp, open, high, low, close, volume.
    ``timestamp`` is Binance-style epoch milliseconds (or datetime); sorted
    ascending by timestamp. Do not load CSV with parse_dates on ``timestamp``.

Output contract:
    DataFrame with all original OHLCV columns + feature columns + 'label'.
    Rows with NaN features (burn-in period) are dropped.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Columns that are NOT features (excluded from shift and from feature list)
OHLCV_COLS = {"timestamp", "open", "high", "low", "close", "volume"}

# Minimum number of rows required to produce any valid features
MIN_ROWS = 60


def _timestamp_to_utc_series(s: pd.Series) -> pd.Series:
    """
    Normalize a timestamp column to UTC datetime64.

    Accepts Binance-style epoch milliseconds (int/float/string) or
    already-parsed datetimes. Do not use read_csv(parse_dates=...) for
    epoch-ms integers — pandas may mis-parse large numbers as calendar dates
    and raise OutOfBoundsDatetime.
    """
    if pd.api.types.is_datetime64_any_dtype(s):
        return pd.to_datetime(s, utc=True)
    num = pd.to_numeric(s, errors="coerce")
    return pd.to_datetime(num, unit="ms", utc=True)

def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    """Elementwise division with zero-denominator protection."""
    return num / den.replace(0, np.nan)

# ---------------------------------------------------------------------------
# Internal indicator helpers
# ---------------------------------------------------------------------------

def _rsi(series: pd.Series, period: int) -> pd.Series:
    """Compute RSI without lookahead using exponential smoothing (Wilder's method)."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average."""
    return series.ewm(span=period, adjust=False).mean()


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    """Average True Range."""
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift(1)).abs()
    lc = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _bollinger(series: pd.Series, period: int = 20, std: float = 2.0):
    """Returns (mid, upper, lower) Bollinger Bands."""
    mid = series.rolling(period).mean()
    sigma = series.rolling(period).std()
    return mid, mid + std * sigma, mid - std * sigma


def _session(hour: pd.Series) -> pd.Series:
    """
    Encode trading session as integer:
        0 = Asia       00:00 - 07:59 UTC
        1 = London     08:00 - 12:59 UTC
        2 = NY overlap 13:00 - 16:59 UTC
        3 = NY close   17:00 - 23:59 UTC
    """
    conditions = [
        hour < 8,
        (hour >= 8) & (hour < 13),
        (hour >= 13) & (hour < 17),
    ]
    return np.select(conditions, [0, 1, 2], default=3).astype(int)


def _consecutive_streak(green: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    Returns (green_streak, red_streak) — number of consecutive
    green or red candles immediately preceding each candle.
    """
    green_streak = pd.Series(0, index=green.index, dtype=int)
    red_streak = pd.Series(0, index=green.index, dtype=int)

    g_count = np.zeros(len(green), dtype=int)
    r_count = np.zeros(len(green), dtype=int)

    green_arr = green.values
    for i in range(1, len(green_arr)):
        if green_arr[i - 1] == 1:
            g_count[i] = g_count[i - 1] + 1
            r_count[i] = 0
        else:
            r_count[i] = r_count[i - 1] + 1
            g_count[i] = 0

    green_streak[:] = g_count
    red_streak[:] = r_count
    return green_streak, red_streak


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the full feature matrix from raw OHLCV data.

    Parameters
    ----------
    df : pd.DataFrame
        Raw OHLCV data with columns [timestamp, open, high, low, close, volume].
        ``timestamp`` must be epoch milliseconds (or datetime); sorted ascending
        with no duplicate timestamps.

    Returns
    -------
    pd.DataFrame
        Feature matrix with label column. NaN rows (burn-in) are dropped.
        All feature columns are shifted by 1 — no lookahead.

    Raises
    ------
    ValueError
        If required columns are missing or df is too short.
    """
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if len(df) < MIN_ROWS:
        raise ValueError(f"DataFrame too short: need at least {MIN_ROWS} rows, got {len(df)}")

    df = df.copy()
    df["timestamp"] = _timestamp_to_utc_series(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # -----------------------------------------------------------------------
    # 1. Label (current candle — never shifted)
    # -----------------------------------------------------------------------
    df["label"] = (df["close"] > df["open"]).astype(int)

    # -----------------------------------------------------------------------
    # 2. Price returns
    # -----------------------------------------------------------------------
    df["ret_1"] = df["close"].pct_change(1)
    df["ret_3"] = df["close"].pct_change(3)
    df["ret_6"] = df["close"].pct_change(6)
    df["ret_12"] = df["close"].pct_change(12)
    df["ret_24"] = df["close"].pct_change(24)
    df["ret_48"] = df["close"].pct_change(48)
    
    # --- BATCH A ---
    # Extra short-horizon returns
    df["ret_2"] = df["close"].pct_change(2)
    df["ret_4"] = df["close"].pct_change(4)
    df["ret_5"] = df["close"].pct_change(5)
    df["ret_8"] = df["close"].pct_change(8)

    # Short-term acceleration
    df["ret_accel_1_3"] = df["ret_1"] - df["ret_3"]
    df["ret_accel_3_6"] = df["ret_3"] - df["ret_6"]

    # Weighted recent return
    df["ret_weighted_1_3"] = 0.5 * df["ret_1"] + 0.3 * df["ret_2"] + 0.2 * df["ret_3"]

    # -----------------------------------------------------------------------
    # 3. Candle structure
    # -----------------------------------------------------------------------
    candle_range = (df["high"] - df["low"]).replace(0, np.nan)

    df["body_ratio"] = (df["close"] - df["open"]).abs() / candle_range
    df["upper_wick"] = (df["high"] - df[["open", "close"]].max(axis=1)) / candle_range
    df["lower_wick"] = (df[["open", "close"]].min(axis=1) - df["low"]) / candle_range
    df["body_size"] = (df["close"] - df["open"]).abs() / df["open"]
    df["prev_green"] = (df["close"] > df["open"]).astype(int)

    # --- BATCH A ---
    # Candle geometry
    df["clv"] = _safe_div(df["close"] - df["low"], candle_range)  # close location in range [0,1]
    df["wick_imbalance"] = _safe_div(df["upper_wick"] - df["lower_wick"], pd.Series(1.0, index=df.index))

    # Inside / outside bars
    prev_high = df["high"].shift(1)
    prev_low = df["low"].shift(1)
    df["inside_bar"] = ((df["high"] <= prev_high) & (df["low"] >= prev_low)).astype(int)
    df["outside_bar"] = ((df["high"] >= prev_high) & (df["low"] <= prev_low)).astype(int)

    # Short rolling pattern counts
    df["inside_bar_count_5"] = df["inside_bar"].rolling(5).sum()
    df["outside_bar_count_5"] = df["outside_bar"].rolling(5).sum()

    # -----------------------------------------------------------------------
    # 4. Volatility
    # -----------------------------------------------------------------------
    df["atr_14"] = _atr(df, 14)
    df["atr_6"] = _atr(df, 6)
    df["atr_ratio"] = df["atr_6"] / df["atr_14"].replace(0, np.nan)

    df["realized_vol_12"] = df["ret_1"].rolling(12).std()
    df["realized_vol_48"] = df["ret_1"].rolling(48).std()
    df["vol_ratio"] = df["realized_vol_12"] / df["realized_vol_48"].replace(0, np.nan)


    # --- BATCH B ---
    # True range for short regime features
    # tr = pd.concat(
    #     [
    #         df["high"] - df["low"],
    #         (df["high"] - df["close"].shift(1)).abs(),
    #         (df["low"] - df["close"].shift(1)).abs(),
    #     ],
    #     axis=1,
    # ).max(axis=1)

    # # Short realized vol windows
    # df["realized_vol_3"] = df["ret_1"].rolling(3).std()
    # df["realized_vol_5"] = df["ret_1"].rolling(5).std()
    # df["realized_vol_8"] = df["ret_1"].rolling(8).std()

    # # Vol-of-vol (short)
    # df["vol_of_vol_5"] = df["realized_vol_5"].rolling(5).std()

    # # Range compression / expansion
    # range_1 = (df["high"] - df["low"])
    # range_5_mean = range_1.rolling(5).mean()
    # df["range_compression_1_5"] = _safe_div(range_1, range_5_mean)

    # # TR spike vs short baseline
    # tr_med_10 = tr.rolling(10).median()
    # df["tr_spike_10"] = _safe_div(tr, tr_med_10)

    # # Return z-score (mean reversion / exhaustion proxy)
    # ret_mean_20 = df["ret_1"].rolling(20).mean()
    # ret_std_20 = df["ret_1"].rolling(20).std()
    # df["ret_z_20"] = _safe_div(df["ret_1"] - ret_mean_20, ret_std_20)

    # -----------------------------------------------------------------------
    # 5. Volume
    # -----------------------------------------------------------------------
    vol_ma_6 = df["volume"].rolling(6).mean()
    vol_ma_24 = df["volume"].rolling(24).mean()
    vol_ma_48 = df["volume"].rolling(48).mean()

    df["vol_ratio_6"] = df["volume"] / vol_ma_24.replace(0, np.nan)
    df["vol_ratio_24"] = vol_ma_6 / vol_ma_48.replace(0, np.nan)
    df["vol_spike"] = (df["vol_ratio_6"] > 2.0).astype(int)
    df["price_vol_corr"] = (
        df["ret_1"]
        .rolling(12)
        .corr(df["volume"].pct_change())
    )

    # -----------------------------------------------------------------------
    # 6. Momentum indicators
    # -----------------------------------------------------------------------
    df["rsi_14"] = _rsi(df["close"], 14)
    df["rsi_6"] = _rsi(df["close"], 6)

    ema_12 = _ema(df["close"], 12)
    ema_26 = _ema(df["close"], 26)
    ema_8 = _ema(df["close"], 8)
    ema_21 = _ema(df["close"], 21)
    ema_55 = _ema(df["close"], 55)

    df["macd_line"] = ema_12 - ema_26
    df["macd_signal"] = _ema(df["macd_line"], 9)
    df["macd_hist"] = df["macd_line"] - df["macd_signal"]
    df["ema_ratio_8_21"] = ema_8 / ema_21.replace(0, np.nan) - 1
    df["ema_ratio_21_55"] = ema_21 / ema_55.replace(0, np.nan) - 1

    # -----------------------------------------------------------------------
    # 7. Mean reversion / Bollinger Bands
    # -----------------------------------------------------------------------
    bb_mid, bb_upper, bb_lower = _bollinger(df["close"], period=20)
    bb_width = (bb_upper - bb_lower).replace(0, np.nan)
    df["bb_position"] = (df["close"] - bb_mid) / bb_width

    df["rolling_high_24"] = df["high"].rolling(24).max()
    df["rolling_low_24"] = df["low"].rolling(24).min()
    df["dist_from_high"] = (df["rolling_high_24"] - df["close"]) / df["close"]
    df["dist_from_low"] = (df["close"] - df["rolling_low_24"]) / df["close"]

    # -----------------------------------------------------------------------
    # 8. Time features
    # -----------------------------------------------------------------------
    ts = df["timestamp"]
    hour = ts.dt.hour
    dow = ts.dt.dayofweek  # 0=Monday, 6=Sunday

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    df["is_weekend"] = (dow >= 5).astype(int)
    df["session"] = _session(hour)

    # -----------------------------------------------------------------------
    # 9. Streak / pattern features
    # -----------------------------------------------------------------------
    green_flag = (df["close"] > df["open"]).astype(int)
    df["green_streak"], df["red_streak"] = _consecutive_streak(green_flag)

    prev1 = green_flag.shift(1)
    prev2 = green_flag.shift(2)
    df["alternating"] = ((green_flag != prev1) & (prev1 != prev2)).astype(int)


    # Interaction terms (keep set small and intentional)
    # --- BATCH C ---
    # df["int_rsi6_ret1"] = df["rsi_6"] * df["ret_1"]
    # df["int_rsi6_volspike"] = df["rsi_6"] * df["vol_spike"]
    # df["int_greenstreak_upperwick"] = df["green_streak"] * df["upper_wick"]
    # df["int_bbpos_volratio"] = df["bb_position"] * df["vol_ratio"]
    # df["int_ret1_bodyratio"] = df["ret_1"] * df["body_ratio"]
    # df["int_ret3_wickimb"] = df["ret_3"] * df["wick_imbalance"]
    # df["int_session_volspike"] = df["session"] * df["vol_spike"]
    # df["int_rsi6_atr_ratio"] = df["rsi_6"] * df["atr_ratio"]


    # -----------------------------------------------------------------------
    # 10. Last candle dynamics
    # -----------------------------------------------------------------------
    # last candle directional strength
    # df["ret_close_to_high"] = (df["high"] - df["close"]) / df["close"]
    # df["ret_close_to_low"] = (df["close"] - df["low"]) / df["close"]

    # # gap behavior
    # df["gap_open_prev_close"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)

    # # micro momentum flip
    # # Cast to int early so dtype stays numeric after shift/dropna.
    # df["ret_sign_flip"] = (np.sign(df["ret_1"]) != np.sign(df["ret_2"])).astype(int)

    # -----------------------------------------------------------------------
    # 11. Volume-weighted direction
    # -----------------------------------------------------------------------
    # volume-weighted direction
    df["signed_volume"] = df["volume"] * np.sign(df["close"] - df["open"])

    df["signed_vol_3"] = df["signed_volume"].rolling(3).sum()
    df["signed_vol_6"] = df["signed_volume"].rolling(6).sum()

    # pressure imbalance
    df["vol_imbalance_3"] = df["signed_vol_3"] / df["volume"].rolling(3).sum()

    # -----------------------------------------------------------------------
    #  Shift all features by 1 (prevent lookahead)
    # -----------------------------------------------------------------------
    feature_cols = [c for c in df.columns if c not in OHLCV_COLS and c != "label"]
    df[feature_cols] = df[feature_cols].shift(1)

    # -----------------------------------------------------------------------
    #  Drop burn-in NaN rows
    # -----------------------------------------------------------------------
    df = df.dropna(subset=feature_cols).reset_index(drop=True)

    return df


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return the list of feature column names (excludes OHLCV and label)."""
    return [c for c in df.columns if c not in OHLCV_COLS and c != "label"]


# ---------------------------------------------------------------------------
# Script entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    input_path = "btc_15.csv"
    output_path = f"{input_path.split('.')[0]}_features.parquet"


    print(f"Loading raw OHLCV from {input_path}...")
    raw = pd.read_csv(input_path)

    print(f"Building features on {len(raw)} candles...")
    featured = build_features(raw)

    feature_cols = get_feature_cols(featured)
    print(f"  → {len(featured)} rows after burn-in drop")
    print(f"  → {len(feature_cols)} features: {feature_cols}")
    print(f"  → Label balance: {featured['label'].mean():.3f} (green rate)")

    # If features are correctly shifted, they should NOT correlate
    # with the raw current candle return
    current_return = (featured['close'] - featured['open']) / featured['open']

    for feature in feature_cols:
        corr = featured[feature].corr(current_return)
        # Should be low — not proof of no lookahead but a good signal
        if corr > 0.1:
            print(f"Warning: {feature} correlates with current return (corr={corr:.3f})")

    current_ret = (featured['close'] - featured['open']) / featured['open']
    assert not featured['ret_1'].equals(current_ret), "Lookahead detected in ret_1"

    # 4. Infinite values
    numeric_feature_cols = list(
        featured[feature_cols].select_dtypes(include=[np.number]).columns
    )
    inf_counts = np.isinf(featured[numeric_feature_cols].values).sum(axis=0)
    inf_series = pd.Series(inf_counts, index=numeric_feature_cols)
    print(f"Features with inf: {(inf_series > 0).sum()} / {len(numeric_feature_cols)}")

    # 5. Zero variance features
    zero_var = featured[numeric_feature_cols].std() == 0
    print(f"Zero variance features: {zero_var.sum()}")

    # 6. Timestamp continuity — flags gaps larger than 15 minutes
    ts = pd.to_datetime(featured['timestamp'])
    gaps = ts.diff().dt.total_seconds() / 60
    large_gaps = gaps[gaps > 15]
    print(f"Gaps > 15min: {len(large_gaps)}")
    print(large_gaps)

    featured.to_parquet(output_path, index=False)
    print(f"Saved to {output_path}")
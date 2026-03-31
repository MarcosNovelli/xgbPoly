import numpy as np
import pandas as pd


def simulate_trades(
    merged_df: pd.DataFrame,
    *,
    prediction_col: str = "prediction",
    yes_ask_col: str = "yes_ask",
    no_ask_col: str = "no_ask",
    result_col: str = "result",
    min_edge: float = 0.0,
    stake: float = 10.0,
) -> tuple[pd.DataFrame, dict]:
    """
    Simulate a simple EV-based YES/NO strategy on Polymarket.

    Assumptions (matches your pipeline):
    - `prediction` is P(green) where green means the *next* 15m candle closes up.
      So:
        p_up = prediction
        p_down = 1 - prediction
    - `result` is the realized outcome from the market:
        result == 1  -> YES (up) won
        result == 0  -> NO (down) won
    - Buying YES at price `yes_ask` costs `yes_ask` and pays out 1 if YES wins else 0.
      Therefore EV(YES) = p_up - yes_ask.
    - Buying NO at price `no_ask` costs `no_ask` and pays out 1 if NO wins else 0.
      Therefore EV(NO) = p_down - no_ask = (1 - p_up) - no_ask.

    Strategy:
    - Compute EV for YES and NO.
    - Take at most one trade per row:
        - pick YES if EV(YES) >= EV(NO)
        - pick NO otherwise
    - Only trade if max(EV(YES), EV(NO)) > min_edge.

    Parameters
    ----------
    merged_df:
        DataFrame containing at least prediction/yes_ask/no_ask/result columns.
    min_edge:
        Minimum EV required to enter (e.g. 0.01 to require a +1 cent expected edge).

    Returns
    -------
    (trades_df, summary)
        trades_df contains one row per executed trade with columns:
          - side: "YES" or "NO"
          - ev_yes, ev_no, chosen_ev
          - realized_pnl_per_share
          - correct
        summary is a dict with aggregate stats.
    """
    df = merged_df.copy()

    for col in (prediction_col, yes_ask_col, no_ask_col, result_col):
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    prediction = pd.to_numeric(df[prediction_col], errors="coerce")
    yes_ask = pd.to_numeric(df[yes_ask_col], errors="coerce")
    no_ask = pd.to_numeric(df[no_ask_col], errors="coerce")
    result = pd.to_numeric(df[result_col], errors="coerce")  # 1 (YES win), 0 (NO win)

    valid = prediction.notna() & yes_ask.notna() & no_ask.notna() & result.notna()

    if valid.sum() == 0:
        return pd.DataFrame(), {
            "n_rows": len(df),
            "n_valid": 0,
            "n_trades": 0,
            "win_rate": np.nan,
            "total_pnl_per_share": 0.0,
        }

    p_up = prediction[valid]
    ev_yes = p_up - yes_ask[valid]
    ev_no = (1.0 - p_up) - no_ask[valid]

    chosen_is_yes = ev_yes >= ev_no
    chosen_side = np.where(chosen_is_yes, "YES", "NO")
    chosen_ev = np.where(chosen_is_yes, ev_yes, ev_no)

    trade_mask = chosen_ev > float(min_edge)
    # `trade_mask` is a NumPy boolean mask (already aligned with the `valid` subset)
    trade_index = valid[valid].index[trade_mask]

    if len(trade_index) == 0:
        return pd.DataFrame(columns=[
            "timestamp" if "timestamp" in df.columns else "row_index",
            "side",
            "ev_yes",
            "ev_no",
            "chosen_ev",
            "realized_pnl_per_share",
            "correct",
        ]), {
            "n_rows": len(df),
            "n_valid": int(valid.sum()),
            "n_trades": 0,
            "win_rate": np.nan,
            "total_pnl_per_share": 0.0,
        }

    out = df.loc[trade_index].copy()
    out["p_up"] = p_up.loc[trade_index]
    out["ev_yes"] = ev_yes.loc[trade_index]
    out["ev_no"] = ev_no.loc[trade_index]
    out["side"] = chosen_side[trade_mask]
    out["chosen_ev"] = chosen_ev[trade_mask]

    outcome_up = result.loc[trade_index].astype(int)  # 1 => up/YES won, 0 => down/NO won
    out["correct"] = np.where(out["side"] == "YES", outcome_up == 1, outcome_up == 0)

    # Per-share realized PnL in "probability points":
    # - YES: payoff is 1 if up else 0 => pnl = outcome_up - yes_ask
    # - NO : payoff is 1 if down else 0 => pnl = (1-outcome_up) - no_ask
    out["realized_pnl_per_share"] = np.where(
        out["side"] == "YES",
        outcome_up - out[yes_ask_col],
        (1 - outcome_up) - out[no_ask_col],
    )

    entry_price = np.where(out["side"] == "YES", out[yes_ask_col], out[no_ask_col])

    out["pnl_dollars"] = np.where(
        out["correct"],
        stake * (1.0 / entry_price - 1) ,   # win: profit proportional to odds
        -stake,                              # loss: lose full stake
    )

    trades_df = out[
        [
            c
            for c in (
                "timestamp",
                prediction_col,
                yes_ask_col,
                no_ask_col,
                result_col,
                "side",
                "ev_yes",
                "ev_no",
                "chosen_ev",
                "realized_pnl_per_share",
                "correct",
                "pnl_dollars",
            )
            if c in out.columns
        ]
    ].copy()

    n_trades = len(trades_df)
    summary = {
        "n_rows": len(df),
        "n_valid": int(valid.sum()),
        "n_trades": int(n_trades),
        "win_rate": float(trades_df["correct"].mean()) if n_trades else np.nan,
        "avg_chosen_ev": float(trades_df["chosen_ev"].mean()) if n_trades else np.nan,
        "total_pnl_per_share": float(trades_df["realized_pnl_per_share"].sum()) if n_trades else 0.0,
        "avg_pnl_per_trade": float(trades_df["realized_pnl_per_share"].mean()) if n_trades else np.nan,
        "total_pnl_dollars" : float(out["pnl_dollars"].sum()),
        "avg_pnl_dollars"   : float(out["pnl_dollars"].mean()),
        "roi_pct"           : float(out["pnl_dollars"].sum() / (stake * n_trades) * 100),
    }

    return trades_df, summary


def plot_cumulative_pnl(
    trades_csv_path: str = "trades_df.csv",
    *,
    pnl_col: str = "pnl_dollars",
    time_col: str = "timestamp",
    sort_by_time: bool = True,
    ax=None,
    title: str = "Cumulative PnL",
):
    """
    Load a trades CSV and plot cumulative PnL.

    Parameters
    ----------
    trades_csv_path:
        Path to CSV containing per-trade PnL.
    pnl_col:
        Column with per-trade PnL values.
    time_col:
        Optional datetime column to use on x-axis when available.
    sort_by_time:
        If True and `time_col` exists, sort rows by time before cum-summing.
    ax:
        Optional matplotlib Axes. If None, a new figure/axes is created.
    title:
        Plot title.

    Returns
    -------
    (df, ax)
        df: DataFrame with a `cum_pnl` column added.
        ax: Matplotlib Axes used for the plot.
    """
    import matplotlib.pyplot as plt

    df = pd.read_csv(trades_csv_path)
    if pnl_col not in df.columns:
        raise ValueError(f"Missing required PnL column: {pnl_col}")

    df[pnl_col] = pd.to_numeric(df[pnl_col], errors="coerce")
    df = df[df[pnl_col].notna()].copy()
    if df.empty:
        raise ValueError("No valid PnL rows found after numeric conversion.")

    x = pd.RangeIndex(start=0, stop=len(df), step=1)
    x_label = "Trade #"

    if time_col in df.columns:
        parsed_time = pd.to_datetime(df[time_col], errors="coerce", utc=True)
        has_valid_time = parsed_time.notna().any()
        if has_valid_time:
            df[time_col] = parsed_time
            if sort_by_time:
                df = df.sort_values(time_col).reset_index(drop=True)
            x = df[time_col]
            x_label = "Time"
        elif sort_by_time:
            # Keep a deterministic order when timestamps exist but are invalid.
            df = df.reset_index(drop=True)

    df["cum_pnl"] = df[pnl_col].cumsum()

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 5))

    ax.plot(x, df["cum_pnl"], linewidth=2.0, label="Cumulative PnL")
    ax.axhline(0.0, linestyle="--", linewidth=1.0, alpha=0.7, color="gray")
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("PnL ($)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    if ax is not None and ax.figure is not None:
        ax.figure.tight_layout()

    return df, ax


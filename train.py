import json
import numpy as np
import pandas as pd
from pathlib import Path
from dateutil.relativedelta import relativedelta
from xgboost import XGBClassifier

from calibrate import CalibratedXGB
from evaluate import compute_metrics, metrics_summary, plot_full_report
from features import get_feature_cols

XGB_PARAMS = {
    "n_estimators"         : 500,
    "max_depth"            : 4,       # shallow — reduces overfitting
    "learning_rate"        : 0.05,
    "subsample"            : 0.8,
    "colsample_bytree"     : 0.8,
    "min_child_weight"     : 20,      # high — prevents fitting on noise
    "gamma"                : 1.0,
    "reg_alpha"            : 1.0,
    "reg_lambda"           : 5.0,
    "eval_metric"          : "logloss",
    "early_stopping_rounds": 30,
    "random_state"         : 42,
    "n_jobs"               : -1,
    "verbosity"            : 0,
}

HOLDOUT_MONTHS   = 6    # reserved at end, never touched during training
FOLD_SIZE_WEEKS  = 4    # each val fold covers x weeks
MIN_TRAIN_MONTHS = 12   # minimum history before first fold
GAP_CANDLES      = 96   # 1 day gap between train end and val start


def _split_holdout(
    df: pd.DataFrame,
    holdout_months: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split df into (walk_forward_set, holdout_set).
    Holdout is the last holdout_months of data — never touched during training.
    """
    ts      = pd.to_datetime(df["timestamp"])
    cutoff  = ts.max() - relativedelta(months=holdout_months)
    mask    = ts <= cutoff
    return df[mask].reset_index(drop=True), df[~mask].reset_index(drop=True)

def _walk_forward_folds(
    df: pd.DataFrame,
    min_train_months: int,
    fold_size_weeks: int,
    gap_candles: int,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Generate expanding-window walk-forward folds.
    Each fold is (train_df, val_df) with a gap between them.
    """
    ts    = pd.to_datetime(df["timestamp"])
    start = ts.min()
    end   = ts.max()

    folds     = []
    val_start = start + relativedelta(months=min_train_months)

    while val_start + relativedelta(weeks=fold_size_weeks) <= end:
        val_end = val_start + relativedelta(weeks=fold_size_weeks)

        train_df = df[ts < val_start].reset_index(drop=True)
        val_df   = df[(ts >= val_start) & (ts < val_end)].reset_index(drop=True)

        # Drop last gap_candles rows from train to avoid boundary leakage
        if len(train_df) > gap_candles:
            train_df = train_df.iloc[:-gap_candles].reset_index(drop=True)

        if len(train_df) > 0 and len(val_df) > 0:
            folds.append((train_df, val_df))

        val_start += relativedelta(weeks=fold_size_weeks)

    return folds

def _fit_xgb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> XGBClassifier:
    """Fit XGBClassifier with early stopping monitored on the val set."""
    model = XGBClassifier(**XGB_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return model



def train(
    features_path: Path | str,
    output_dir: Path | str,
) -> dict:
    """
    Run the full walk-forward training pipeline.

    Parameters
    ----------
    features_path : path to processed features parquet file
    output_dir    : root directory for all model outputs

    Returns
    -------
    dict with keys:
        fold_metrics       : list of per-fold metric dicts
        summary            : DataFrame of aggregated metrics
        holdout_path       : path to holdout parquet
        final_model_path   : path to final deployed model bundle directory
    """
    features_path = Path(features_path)
    output_dir    = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print(f"Loading features from {features_path}...")
    df           = pd.read_parquet(features_path)
    feature_cols = get_feature_cols(df)
    print(f"  {len(df):,} rows | {len(feature_cols)} features")
    print(f"  Label balance: {df['label'].mean():.3f}")

    # ------------------------------------------------------------------
    # 2. Split holdout
    # ------------------------------------------------------------------
    wf_df, holdout_df = _split_holdout(df, HOLDOUT_MONTHS)

    holdout_path = output_dir / "holdout.parquet"
    holdout_df.to_parquet(holdout_path, index=False)

    wf_ts = pd.to_datetime(wf_df["timestamp"])
    h_ts  = pd.to_datetime(holdout_df["timestamp"])
    print(f"\n  Walk-forward : {len(wf_df):,} rows [{wf_ts.min().date()} → {wf_ts.max().date()}]")
    print(f"  Holdout      : {len(holdout_df):,} rows [{h_ts.min().date()} → {h_ts.max().date()}]")

    # ------------------------------------------------------------------
    # 3. Generate folds
    # ------------------------------------------------------------------
    folds = _walk_forward_folds(
        wf_df,
        min_train_months=MIN_TRAIN_MONTHS,
        fold_size_weeks=FOLD_SIZE_WEEKS,
        gap_candles=GAP_CANDLES,
    )
    print(f"\nWalk-forward: {len(folds)} folds")

    fold_results = []
    fold_metrics = []

    # ------------------------------------------------------------------
    # 4. Train + calibrate + evaluate per fold
    # ------------------------------------------------------------------
    for i, (train_df, val_df) in enumerate(folds):
        fold_num = i + 1
        fold_dir = output_dir / "trained" / f"fold_{fold_num:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        t_start = pd.to_datetime(train_df["timestamp"]).min().date()
        t_end   = pd.to_datetime(train_df["timestamp"]).max().date()
        v_start = pd.to_datetime(val_df["timestamp"]).min().date()
        v_end   = pd.to_datetime(val_df["timestamp"]).max().date()

        print(f"\n── Fold {fold_num}/{len(folds)} "
              f"| train [{t_start} → {t_end}] ({len(train_df):,}) "
              f"| val [{v_start} → {v_end}] ({len(val_df):,}) ──")

        X_train = train_df[feature_cols].values
        y_train = train_df["label"].values
        X_val   = val_df[feature_cols].values
        y_val   = val_df["label"].values

        # Fit
        xgb = _fit_xgb(X_train, y_train, X_val, y_val)
        print(f"  Best iteration : {xgb.best_iteration}")

        # Calibrate
        cal_model = CalibratedXGB(xgb)
        cal_model.fit(X_val, y_val)
        cal_model.feature_cols = feature_cols

        # Evaluate
        p_green = cal_model.predict_green(X_val)
        metrics = compute_metrics(y_val, p_green)
        print(f"  Brier={metrics['brier']:.4f}  "
              f"AUC={metrics['auc']:.4f}  "
              f"Acc={metrics['accuracy']:.4f}")

        # Feature importance
        raw_imp   = xgb.get_booster().get_score(importance_type="gain")
        imp_array = np.array([
            raw_imp.get(f"f{j}", 0.0)
            for j in range(len(feature_cols))
        ])

        # Save
        cal_model.save(fold_dir / "xgb_calibrated")
        with open(fold_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        fold_result = {
            "y_true"      : y_val,
            "p_green"     : p_green,
            "metrics"     : metrics,
            "importances" : imp_array,
        }
        fold_results.append(fold_result)
        fold_metrics.append(metrics)

        plot_full_report(
            fold_results=[fold_result],
            feature_names=feature_cols,
            output_path=fold_dir / "report.png",
        )

    # ------------------------------------------------------------------
    # 5. Aggregate across all folds
    # ------------------------------------------------------------------
    trained_dir = output_dir / "trained"
    summary     = metrics_summary(fold_metrics)
    summary.to_csv(trained_dir / "metrics_summary.csv")

    print(f"\n{'='*60}")
    print("Aggregated metrics across all folds:")
    print(summary.to_string())

    plot_full_report(
        fold_results=fold_results,
        feature_names=feature_cols,
        output_path=trained_dir / "full_report.png",
    )

    # ------------------------------------------------------------------
    # 6. Final model — train on all walk-forward data
    # ------------------------------------------------------------------
    print("\nTraining final model on all walk-forward data...")

    split_idx   = int(len(wf_df) * 0.90)
    X_all_train = wf_df.iloc[:split_idx][feature_cols].values
    y_all_train = wf_df.iloc[:split_idx]["label"].values
    X_all_val   = wf_df.iloc[split_idx:][feature_cols].values
    y_all_val   = wf_df.iloc[split_idx:]["label"].values

    final_xgb = _fit_xgb(X_all_train, y_all_train, X_all_val, y_all_val)
    final_cal = CalibratedXGB(final_xgb)
    final_cal.fit(X_all_val, y_all_val)
    final_cal.feature_cols = feature_cols

    # ------------------------------------------------------------------
    # 7. Save final model
    # ------------------------------------------------------------------
    final_dir = trained_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    final_model_path = final_dir / "xgb_calibrated"
    final_cal.save(final_model_path)

    with open(final_dir / "feature_cols.txt", "w") as f:
        f.write("\n".join(feature_cols))

    print(f"\n{'='*60}")
    print("Training complete.")
    print(f"  Final model  : {final_model_path}")
    print(f"  Holdout      : {holdout_path}")
    print(f"  Full report  : {trained_dir / 'full_report.png'}")
    print(f"{'='*60}")

    return {
        "fold_metrics"     : fold_metrics,
        "summary"          : summary,
        "holdout_path"     : holdout_path,
        "final_model_path" : final_model_path,
    }

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python train.py <features_parquet> <output_dir>")
        sys.exit(1)

    train(
        features_path=sys.argv[1],
        output_dir=sys.argv[2],
    )
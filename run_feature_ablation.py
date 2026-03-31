import argparse
from pathlib import Path
import pandas as pd

from train import train


def _safe_get(summary: pd.DataFrame, metric: str, agg: str = "mean") -> float:
    # Handles either summary index style: metric->agg or agg->metric
    if metric in summary.index and agg in summary.columns:
        return float(summary.loc[metric, agg])
    if agg in summary.index and metric in summary.columns:
        return float(summary.loc[agg, metric])
    raise KeyError(f"Could not find ({metric}, {agg}) in summary table.")


def run_ablation(features_root: Path, output_root: Path) -> pd.DataFrame:
    """
    Expects prebuilt feature files:
      - baseline.parquet
      - batch_a.parquet
      - batch_ab.parquet
      - batch_abc.parquet
    """
    variants = [
        ("baseline",  features_root / "data_features_baseline.parquet"),
        ("batch_a",   features_root / "data_features_a.parquet"),
        # ("batch_ab",  features_root / "data_features_ab.parquet"),
        # ("batch_abc", features_root / "data_features_abc.parquet"),
    ]

    rows = []
    for name, fpath in variants:
        if not fpath.exists():
            print(f"[skip] {name}: missing {fpath}")
            continue

        out_dir = output_root / name
        print(f"\n=== Running {name} ===")
        result = train(features_path=fpath, output_dir=out_dir)

        summary = result["summary"]
        rows.append(
            {
                "variant": name,
                "features_path": str(fpath),
                "auc_mean": _safe_get(summary, "auc", "mean"),
                "auc_std": _safe_get(summary, "auc", "std"),
                "brier_mean": _safe_get(summary, "brier", "mean"),
                "brier_std": _safe_get(summary, "brier", "std"),
                "acc_mean": _safe_get(summary, "accuracy", "mean"),
                "acc_std": _safe_get(summary, "accuracy", "std"),
            }
        )

    if not rows:
        raise RuntimeError("No experiments ran. Check features_root and file names.")

    leaderboard = pd.DataFrame(rows).sort_values(
        ["auc_mean", "brier_mean"], ascending=[False, True]
    )
    leaderboard_path = output_root / "ablation_summary.csv"
    output_root.mkdir(parents=True, exist_ok=True)
    leaderboard.to_csv(leaderboard_path, index=False)

    print("\n=== Ablation summary ===")
    print(leaderboard.to_string(index=False))
    print(f"\nSaved: {leaderboard_path}")

    return leaderboard


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    run_ablation(args.features_root, args.output_root)
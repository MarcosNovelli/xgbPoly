"""
polybot/model/calibrate.py

Isotonic calibration wrapper for XGBoost probability outputs.

Takes a fitted XGBClassifier and a validation set, fits an isotonic
calibrator on top, and returns a calibrated model that produces
trustworthy probabilities via predict_green().
"""

import numpy as np
import json
import pickle
from pathlib import Path
import warnings
import joblib
import sklearn
import xgboost
from sklearn.isotonic import IsotonicRegression
from xgboost import XGBClassifier

class CalibratedXGB:
    def __init__(self, xgb_model: XGBClassifier):
        self.xgb = xgb_model
        self.calibrator = IsotonicRegression(out_of_bounds="clip")
        self._fitted = False
        self.feature_cols: list[str] | None = None

    def fit(self, X_val: np.ndarray, y_val: np.ndarray) -> "CalibratedXGB":
        """
        Fit isotonic calibrator on validation set predictions.

        Parameters
        ----------
        X_val : array-like of shape (n_samples, n_features)
        y_val : array-like of shape (n_samples,) — binary 0/1

        Returns
        -------
        self
        """
        raw_probs = self.xgb.predict_proba(X_val)[:, 1]
        self.calibrator.fit(raw_probs, y_val)
        self._fitted = True
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns calibrated probabilities as (n, 2) array — [P(red), P(green)]."""
        if not self._fitted:
            raise RuntimeError("Call fit() before predict_proba()")
        raw = self.xgb.predict_proba(X)[:, 1]
        p_green = self.calibrator.predict(raw)
        return np.column_stack([1 - p_green, p_green])

    def predict_green(self, X: np.ndarray) -> np.ndarray:
        """Returns P(green) as 1D array — primary interface for the bot."""
        return self.predict_proba(X)[:, 1]

    def save(self, path: Path | str) -> None:
        """
        Persist to disk in a version-stable format by default.

        - If `path` ends with `.pkl`, uses legacy pickle serialization.
        - Otherwise, writes a directory bundle:
          - `xgb.json` (XGBoost native model format)
          - `calibrator.joblib` (sklearn IsotonicRegression)
          - `meta.json` (library versions)
        """
        p = Path(path)

        if p.suffix.lower() == ".pkl":
            with open(p, "wb") as f:
                pickle.dump(self, f)
            return

        p.mkdir(parents=True, exist_ok=True)
        self.xgb.save_model(str(p / "xgb.json"))
        joblib.dump(self.calibrator, p / "calibrator.joblib")
        meta = {
            "format": "CalibratedXGB.bundle.v1",
            "xgboost_version": getattr(xgboost, "__version__", None),
            "sklearn_version": getattr(sklearn, "__version__", None),
            "feature_cols": self.feature_cols,
        }
        (p / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> "CalibratedXGB":
        """
        Load from disk.

        Supports both:
        - Legacy `.pkl` pickles (may warn on version mismatch)
        - The directory bundle produced by `save()`
        """
        p = Path(path)

        if p.is_file() and p.suffix.lower() == ".pkl":
            warnings.warn(
                "Loading a legacy pickled CalibratedXGB. "
                "This is sensitive to xgboost/scikit-learn version differences; "
                "re-save using the directory bundle format to eliminate warnings.",
                UserWarning,
                stacklevel=2,
            )
            with open(p, "rb") as f:
                obj = pickle.load(f)
            if not isinstance(obj, cls):
                raise TypeError(f"Expected CalibratedXGB, got {type(obj)}")
            return obj

        if not p.is_dir():
            raise FileNotFoundError(f"Expected a .pkl file or bundle directory, got: {p}")

        xgb_path = p / "xgb.json"
        calib_path = p / "calibrator.joblib"
        if not xgb_path.exists() or not calib_path.exists():
            raise FileNotFoundError(
                f"Missing bundle files in {p}. Expected `xgb.json` and `calibrator.joblib`."
            )

        xgb_model = XGBClassifier()
        xgb_model.load_model(str(xgb_path))

        obj = cls(xgb_model)
        obj.calibrator = joblib.load(calib_path)
        obj._fitted = True

        meta_path = p / "meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                feature_cols = meta.get("feature_cols")
                if isinstance(feature_cols, list) and all(isinstance(x, str) for x in feature_cols):
                    obj.feature_cols = feature_cols
            except Exception:
                pass
        return obj
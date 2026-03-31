"""
polybot/model/predict.py

Load the final trained + calibrated model and produce P(green)
for incoming candle data.

This is the only file the live bot imports from the model layer.
It knows nothing about training, calibration, or evaluation.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from calibrate import CalibratedXGB

class Predictor:
    """
    Loads a CalibratedXGB model from disk and produces P(green)
    for incoming candle data.
    """

    def __init__(self, model: CalibratedXGB, feature_cols: list[str]):
        self.model        = model
        self.feature_cols = feature_cols

    @classmethod
    def from_file(cls, model_path: Path | str) -> "Predictor":
        """
        Load predictor from a saved CalibratedXGB pickle.
        Feature columns are loaded from the model object directly.

        Parameters
        ----------
        model_path : path to CalibratedXGB pickle
        """
        model = CalibratedXGB.load(model_path)

        if not hasattr(model, "feature_cols"):
            raise ValueError(
                "feature_cols not found on model object. "
                "Re-train the model to attach feature_cols."
            )

        return cls(model=model, feature_cols=model.feature_cols)

    
    def predict(self, features: pd.Series | dict) -> float:
        """
        Predict P(green) for a single candle.

        Parameters
        ----------
        features : pd.Series or dict with feature values

        Returns
        -------
        float — calibrated P(green) in [0, 1]
        """
        if isinstance(features, dict):
            features = pd.Series(features)

        x = features[self.feature_cols].values.reshape(1, -1)
        return float(self.model.predict_green(x)[0])

    def predict_batch(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict P(green) for a batch of candles.

        Parameters
        ----------
        df : DataFrame containing feature columns

        Returns
        -------
        np.ndarray of shape (n,) — calibrated P(green) per row
        """
        missing = set(self.feature_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")

        X = df[self.feature_cols].values
        return self.model.predict_green(X)

    def edge(self, features: pd.Series | dict, yes_price: float) -> float:
        """
        Compute edge for a single candle given the Polymarket YES price.

        Parameters
        ----------
        features  : feature row for the current candle
        yes_price : current Polymarket YES price in [0, 1]

        Returns
        -------
        float — edge = P(green) - yes_price
                positive → model thinks YES is underpriced → buy YES
                negative → model thinks NO is underpriced → buy NO
        """
        p = self.predict(features)
        return round(p - yes_price, 6)



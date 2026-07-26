"""
Inference helpers for the Student Success Prediction model.
Used by both the Streamlit app (app.py) and automated tests.
"""

from __future__ import annotations

import json

import joblib
import pandas as pd

from src import config
from src.data_processing import engineer_features, get_feature_lists
from src.risk_scoring import compute_risk_score, risk_tier_from_score


class StudentSuccessPredictor:
    """Loads the trained pipeline once and exposes simple predict methods."""

    def __init__(self):
        self.model = joblib.load(config.MODEL_PATH)
        self.preprocessor = joblib.load(config.PREPROCESSOR_PATH)
        self.label_encoder = joblib.load(config.LABEL_ENCODER_PATH)
        self.categorical, self.numerical = get_feature_lists()
        self.feature_columns = self.categorical + self.numerical

        with open(config.METADATA_PATH) as f:
            self.metadata = json.load(f)

    def _prepare_input(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        df = engineer_features(raw_df)
        # Ensure every expected column is present (fill missing engineered/raw
        # columns with 0 so partial/manual input from the UI never crashes).
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0
        return df[self.feature_columns]

    def predict(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        raw_df: DataFrame with the original UCI column names (one row per student).
        Returns a DataFrame with predicted_status, risk_level, and per-class probabilities.
        """
        X = self._prepare_input(raw_df)
        X_t = self.preprocessor.transform(X)
        preds = self.model.predict(X_t)
        probs = self.model.predict_proba(X_t)

        labels = self.label_encoder.inverse_transform(preds)
        class_index = {cls: i for i, cls in enumerate(self.label_encoder.classes_)}
        prob_dropout = probs[:, class_index["Dropout"]]
        prob_enrolled = probs[:, class_index["Enrolled"]]

        risk_scores = [compute_risk_score(pd_, pe_) for pd_, pe_ in zip(prob_dropout, prob_enrolled)]
        risk_levels = [risk_tier_from_score(s) for s in risk_scores]

        out = pd.DataFrame(
            {
                "predicted_status": labels,
                "risk_score": risk_scores,
                "risk_level": risk_levels,
            }
        )
        for i, cls in enumerate(self.label_encoder.classes_):
            out[f"probability_{cls.lower()}"] = probs[:, i]

        out.index = raw_df.index
        return out

    def predict_one(self, student: dict) -> dict:
        """Convenience wrapper for a single student passed as a dict."""
        raw_df = pd.DataFrame([student])
        result = self.predict(raw_df)
        return result.iloc[0].to_dict()

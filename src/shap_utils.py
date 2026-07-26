"""
SHAP-based explainability for the Student Success model.

Uses shap.Explainer's automatic backend selection: TreeExplainer for
Random Forest / XGBoost / LightGBM, LinearExplainer for Logistic
Regression — whichever model training selected as the winner. This keeps
the app's "explain this prediction" feature working regardless of which
model is currently in models/student_success_model.joblib.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import shap

from src import config
from src.data_processing import engineer_features, get_feature_lists, get_transformed_feature_names

DROPOUT_CLASS_INDEX = 0  # LabelEncoder sorts alphabetically: Dropout, Enrolled, Graduate


class ShapExplainer:
    """Loads model/preprocessor/background once; computes SHAP values on demand."""

    def __init__(self):
        self.model = joblib.load(config.MODEL_PATH)
        self.preprocessor = joblib.load(config.PREPROCESSOR_PATH)
        self.background = joblib.load(config.SHAP_BACKGROUND_PATH)
        self.categorical, self.numerical = get_feature_lists()
        self.feature_columns = self.categorical + self.numerical
        self.feature_names = get_transformed_feature_names(self.preprocessor)

        bg_t = self.preprocessor.transform(self.background)
        # max_evals/background size kept modest (<=200 rows) so this stays fast
        self._explainer = shap.Explainer(self.model, bg_t)

    def _prepare(self, raw_df: pd.DataFrame) -> np.ndarray:
        df = engineer_features(raw_df)
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0
        X = df[self.feature_columns]
        return self.preprocessor.transform(X)

    def explain(self, raw_df: pd.DataFrame):
        """Return a shap.Explanation for the given raw student rows."""
        X_t = self._prepare(raw_df)
        return self._explainer(X_t)

    def dropout_contributions(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        Per-row SHAP contributions toward the *Dropout* class, as a DataFrame
        (rows = students, columns = features). Positive = pushes risk up.
        """
        explanation = self.explain(raw_df)
        values = explanation.values  # shape (n_rows, n_features, n_classes)
        dropout_values = values[:, :, DROPOUT_CLASS_INDEX]
        return pd.DataFrame(dropout_values, columns=self.feature_names, index=raw_df.index)

    def global_importance(self, sample_size: int = 200) -> pd.DataFrame:
        """Mean |SHAP value| for the Dropout class across a background-sized sample,
        for a global 'what drives dropout risk overall' view."""
        sample = self.background.head(sample_size)
        contrib = self.dropout_contributions(sample)
        mean_abs = contrib.abs().mean().sort_values(ascending=False)
        return mean_abs.reset_index().rename(columns={"index": "feature", 0: "mean_abs_shap"})

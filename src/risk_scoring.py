"""
Risk scoring: turns raw class probabilities into a single, intuitive
0-100 "dropout risk score" plus a discrete risk tier, and (given SHAP
values) a ranked list of the factors driving that score for a specific
student. This is the layer that makes model output actionable for
non-technical staff.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config


def compute_risk_score(prob_dropout: float, prob_enrolled: float) -> float:
    """
    Blend P(Dropout) and P(Enrolled) into a single 0-100 risk score.

    Rationale: "Enrolled" (still enrolled past the expected graduation date)
    is not a positive outcome either — it often precedes a later dropout —
    so it should push the score up, just less strongly than an outright
    predicted dropout.
    """
    score = 100 * (prob_dropout + 0.35 * prob_enrolled)
    return float(np.clip(score, 0, 100))


def risk_tier_from_score(score: float) -> str:
    if score >= config.RISK_SCORE_HIGH_THRESHOLD:
        return "High Risk"
    elif score >= config.RISK_SCORE_MODERATE_THRESHOLD:
        return "Moderate Risk"
    return "Low Risk"


def top_risk_factors(shap_row: pd.Series, direction: str = "increase", top_n: int = 5) -> pd.DataFrame:
    """
    Given a Series of SHAP values (feature -> contribution to the Dropout
    class, one row = one student), return the top_n features pushing risk
    up (direction='increase') or down (direction='decrease').
    """
    s = shap_row.sort_values(ascending=(direction == "decrease"))
    if direction == "increase":
        s = s[s > 0]
    else:
        s = s[s < 0]
    return s.head(top_n).reset_index().rename(columns={"index": "feature", 0: "shap_value"})


FEATURE_FRIENDLY_NAMES = {
    "approval_ratio": "Course-unit approval ratio",
    "average_grade": "Average grade (both semesters)",
    "grade_trend": "Grade trend (semester 2 vs 1)",
    "socioeconomic_risk_flags": "Socio-economic risk flags",
    "total_units_enrolled": "Total curricular units enrolled",
    "total_units_approved": "Total curricular units approved",
    "Curricular units 1st sem (approved)": "1st semester units approved",
    "Curricular units 2nd sem (approved)": "2nd semester units approved",
    "Curricular units 1st sem (grade)": "1st semester average grade",
    "Curricular units 2nd sem (grade)": "2nd semester average grade",
    "Tuition fees up to date": "Tuition fee status",
    "Debtor": "Debtor status",
    "Scholarship holder": "Scholarship status",
    "Age at enrollment": "Age at enrollment",
}


def friendly_feature_name(raw_name: str) -> str:
    """Map an (possibly one-hot-encoded) feature name to a human-friendly label."""
    if raw_name in FEATURE_FRIENDLY_NAMES:
        return FEATURE_FRIENDLY_NAMES[raw_name]
    # One-hot encoded columns look like "Gender_1" or "Course_9500" -> tidy them up
    if "_" in raw_name:
        base = raw_name.rsplit("_", 1)[0]
        return FEATURE_FRIENDLY_NAMES.get(base, base)
    return raw_name

"""
Basic smoke tests for the Student Success Prediction pipeline.

Run:
    pytest -v
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.data_processing import load_raw_data, engineer_features, prepare_dataset, get_feature_lists


def test_raw_data_loads():
    df = load_raw_data()
    assert len(df) > 4000
    assert config.TARGET_COLUMN in df.columns
    assert set(df[config.TARGET_COLUMN].unique()) == set(config.TARGET_CLASSES)


def test_no_missing_values():
    df = load_raw_data()
    assert df.isna().sum().sum() == 0


def test_feature_engineering_adds_columns():
    df = load_raw_data()
    df_fe = engineer_features(df)
    for col in ["approval_ratio", "average_grade", "grade_trend", "socioeconomic_risk_flags"]:
        assert col in df_fe.columns
    assert df_fe["approval_ratio"].between(0, 1.001).all()


def test_prepare_dataset_shapes():
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_dataset()
    total = len(X_train) + len(X_val) + len(X_test)
    assert total == 4424
    categorical, numerical = get_feature_lists()
    assert set(X_train.columns) == set(categorical + numerical)
    # stratification sanity check: all three classes present in every split
    for y in (y_train, y_val, y_test):
        assert set(y.unique()) == set(config.TARGET_CLASSES)


@pytest.mark.skipif(
    not config.MODEL_PATH.exists(), reason="Model not trained yet — run `python -m src.train` first"
)
def test_trained_model_predicts():
    from src.predict import StudentSuccessPredictor

    predictor = StudentSuccessPredictor()
    df = load_raw_data().drop(columns=[config.TARGET_COLUMN]).head(5)
    result = predictor.predict(df)
    assert len(result) == 5
    assert set(result["predicted_status"].unique()).issubset(set(config.TARGET_CLASSES))
    assert set(result["risk_level"].unique()).issubset({"Low Risk", "Moderate Risk", "High Risk"})
    assert result["risk_score"].between(0, 100).all()
    prob_cols = [c for c in result.columns if c.startswith("probability_")]
    assert len(prob_cols) == 3
    row_sums = result[prob_cols].sum(axis=1)
    assert (row_sums.round(3) == 1.0).all()


def test_risk_score_and_tier_are_consistent():
    from src.risk_scoring import compute_risk_score, risk_tier_from_score

    high = compute_risk_score(prob_dropout=0.9, prob_enrolled=0.05)
    low = compute_risk_score(prob_dropout=0.02, prob_enrolled=0.05)
    assert high > low
    assert risk_tier_from_score(high) == "High Risk"
    assert risk_tier_from_score(low) == "Low Risk"


@pytest.mark.skipif(
    not config.SHAP_BACKGROUND_PATH.exists(), reason="SHAP background not saved — run `python -m src.train` first"
)
def test_shap_explainer_produces_contributions():
    from src.shap_utils import ShapExplainer

    explainer = ShapExplainer()
    df = load_raw_data().drop(columns=[config.TARGET_COLUMN]).head(3)
    contrib = explainer.dropout_contributions(df)
    assert contrib.shape[0] == 3
    assert contrib.shape[1] > 0
    global_imp = explainer.global_importance(sample_size=50)
    assert "feature" in global_imp.columns
    assert len(global_imp) > 0


def test_recommendation_fallback_without_api_key(monkeypatch):
    from src.llm_recommendations import StudentContext, generate_recommendation

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    ctx = StudentContext(
        student_label="Test Student",
        predicted_status="Dropout",
        risk_level="High Risk",
        risk_score=85.0,
        top_risk_factors=[("Course-unit approval ratio", 1.1)],
        key_stats={"approval_ratio": 0.2},
    )
    text, source = generate_recommendation(ctx)
    assert source == "rule_based"
    assert len(text) > 0

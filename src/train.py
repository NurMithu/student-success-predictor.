"""
Train, evaluate, and persist the Student Success Prediction model.

Run:
    python -m src.train

Compares Logistic Regression (baseline), Random Forest, XGBoost, and
LightGBM on a validation split, picks the best by macro F1, refits it on
train+val, evaluates once on a held-out test set, and saves all artifacts
(model, preprocessor, label encoder, metrics, feature importance, and a
background sample for SHAP) under models/.
"""

from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from src import config
from src.data_processing import (
    build_preprocessor,
    get_feature_lists,
    get_transformed_feature_names,
    prepare_dataset,
)


def get_candidate_models(random_state: int):
    """Candidate models to compare. Random Forest / XGBoost / LightGBM are the
    primary tree-based ensemble candidates (fast, strong tabular performance,
    and natively supported by SHAP's TreeExplainer). Logistic Regression is
    kept as an interpretable baseline for comparison."""
    return {
        "logistic_regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=random_state
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            random_state=random_state,
            n_jobs=-1,
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=300,
            max_depth=-1,
            num_leaves=31,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1,
        ),
    }


def _needs_sample_weight(name: str) -> bool:
    # XGBoost/LightGBM don't take class_weight='balanced' directly -> use sample_weight instead.
    return name in ("xgboost", "lightgbm")


def evaluate(model, preprocessor, X, y, label_encoder):
    X_t = preprocessor.transform(X)
    y_enc = label_encoder.transform(y)
    preds = model.predict(X_t)
    probs = model.predict_proba(X_t)
    acc = accuracy_score(y_enc, preds)
    f1_macro = f1_score(y_enc, preds, average="macro")
    try:
        auc = roc_auc_score(y_enc, probs, multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")
    return {"accuracy": acc, "f1_macro": f1_macro, "roc_auc_macro": auc}, preds, probs


def main():
    print("=" * 60)
    print("Student Success Prediction — Training Pipeline")
    print("=" * 60)

    print("\n[1/6] Loading and preparing data...")
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_dataset()
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    label_encoder = LabelEncoder()
    label_encoder.fit(pd.concat([y_train, y_val, y_test]))

    print("\n[2/6] Fitting preprocessor...")
    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)
    X_train_t = preprocessor.transform(X_train)
    y_train_enc = label_encoder.transform(y_train)
    sample_weight_train = compute_sample_weight("balanced", y_train_enc)

    print("\n[3/6] Training candidate models and comparing on validation set...")
    candidates = get_candidate_models(config.RANDOM_STATE)
    results = {}
    for name, model in candidates.items():
        start = time.time()
        if _needs_sample_weight(name):
            model.fit(X_train_t, y_train_enc, sample_weight=sample_weight_train)
        else:
            model.fit(X_train_t, y_train_enc)
        metrics, _, _ = evaluate(model, preprocessor, X_val, y_val, label_encoder)
        elapsed = time.time() - start
        results[name] = metrics
        print(
            f"  {name:22s} | acc={metrics['accuracy']:.4f} "
            f"f1_macro={metrics['f1_macro']:.4f} "
            f"roc_auc={metrics['roc_auc_macro']:.4f} "
            f"({elapsed:.1f}s)"
        )

    best_name = max(results, key=lambda n: results[n]["f1_macro"])
    print(f"\n  Best model on validation set: {best_name}")

    print(f"\n[4/6] Refitting '{best_name}' on train+val and evaluating on test set...")
    X_train_val = pd.concat([X_train, X_val])
    y_train_val = pd.concat([y_train, y_val])

    final_preprocessor = build_preprocessor()
    final_preprocessor.fit(X_train_val)
    X_train_val_t = final_preprocessor.transform(X_train_val)
    y_train_val_enc = label_encoder.transform(y_train_val)
    sample_weight_full = compute_sample_weight("balanced", y_train_val_enc)

    final_model = get_candidate_models(config.RANDOM_STATE)[best_name]
    if _needs_sample_weight(best_name):
        final_model.fit(X_train_val_t, y_train_val_enc, sample_weight=sample_weight_full)
    else:
        final_model.fit(X_train_val_t, y_train_val_enc)

    test_metrics, test_preds, test_probs = evaluate(
        final_model, final_preprocessor, X_test, y_test, label_encoder
    )
    y_test_enc = label_encoder.transform(y_test)
    report = classification_report(
        y_test_enc, test_preds, target_names=label_encoder.classes_, output_dict=True
    )
    cm = confusion_matrix(y_test_enc, test_preds).tolist()

    print("\n  Held-out TEST set performance:")
    for k, v in test_metrics.items():
        print(f"    {k}: {v:.4f}")

    print("\n[5/6] Computing feature importance...")
    categorical, numerical = get_feature_lists()
    all_feature_names = get_transformed_feature_names(final_preprocessor)
    try:
        if hasattr(final_model, "feature_importances_"):
            importances = final_model.feature_importances_
        elif hasattr(final_model, "coef_"):
            importances = np.abs(final_model.coef_).mean(axis=0)
        else:
            importances = None

        if importances is not None:
            fi_df = pd.DataFrame(
                {"feature": all_feature_names, "importance": importances}
            ).sort_values("importance", ascending=False)
            fi_df.to_csv(config.FEATURE_IMPORTANCE_PATH, index=False)
    except Exception as e:  # pragma: no cover
        print(f"  (skipped feature importance export: {e})")

    print("\n[6/6] Saving artifacts to models/ ...")
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(final_model, config.MODEL_PATH)
    joblib.dump(final_preprocessor, config.PREPROCESSOR_PATH)
    joblib.dump(label_encoder, config.LABEL_ENCODER_PATH)

    # Small background sample (raw + transformed) for SHAP explainers in the app.
    # Keeping this small (200 rows) keeps SHAP fast for KernelExplainer fallbacks
    # while still being representative for TreeExplainer/LinearExplainer.
    bg_sample = X_train_val.sample(
        n=min(200, len(X_train_val)), random_state=config.RANDOM_STATE
    )
    joblib.dump(bg_sample, config.SHAP_BACKGROUND_PATH)

    metrics_out = {
        "best_model": best_name,
        "validation_comparison": results,
        "test_metrics": test_metrics,
        "test_classification_report": report,
        "test_confusion_matrix": cm,
        "class_order": list(label_encoder.classes_),
    }
    with open(config.METRICS_PATH, "w") as f:
        json.dump(metrics_out, f, indent=2)

    metadata = {
        "model_name": best_name,
        "n_train_val": len(X_train_val),
        "n_test": len(X_test),
        "categorical_features": categorical,
        "numerical_features": numerical,
        "target_classes": list(label_encoder.classes_),
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(config.METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Saved model      -> {config.MODEL_PATH}")
    print(f"  Saved preprocess -> {config.PREPROCESSOR_PATH}")
    print(f"  Saved metrics    -> {config.METRICS_PATH}")
    print(f"  Saved SHAP bg    -> {config.SHAP_BACKGROUND_PATH}")
    print("\nDone.")


if __name__ == "__main__":
    main()

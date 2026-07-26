"""
Data loading, cleaning, and feature engineering for the
Student Success Prediction pipeline.

Source dataset: UCI Machine Learning Repository —
"Predict Students' Dropout and Academic Success"
Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021)
https://doi.org/10.24432/C5MC89
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src import config


def load_raw_data(path=None) -> pd.DataFrame:
    """Load the raw UCI CSV file."""
    path = path or config.RAW_DATA_PATH
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a small number of interpretable derived features on top of the raw
    UCI columns. These are computed purely from information already known
    at (or shortly after) enrollment, so they don't leak future information.
    """
    df = df.copy()

    # Overall curricular performance across both semesters
    df["total_units_enrolled"] = (
        df["Curricular units 1st sem (enrolled)"] + df["Curricular units 2nd sem (enrolled)"]
    )
    df["total_units_approved"] = (
        df["Curricular units 1st sem (approved)"] + df["Curricular units 2nd sem (approved)"]
    )

    # Approval ratio = share of enrolled units the student actually passed.
    # This single ratio is one of the strongest early-warning signals.
    df["approval_ratio"] = df["total_units_approved"] / df["total_units_enrolled"].replace(0, pd.NA)
    df["approval_ratio"] = df["approval_ratio"].fillna(0)

    # Average grade across both semesters (0 where a student has no evaluations)
    grade_1 = df["Curricular units 1st sem (grade)"]
    grade_2 = df["Curricular units 2nd sem (grade)"]
    df["average_grade"] = (grade_1 + grade_2) / 2

    # Did the student's performance drop between semester 1 and semester 2?
    df["grade_trend"] = grade_2 - grade_1

    # Simple composite of socio-economic risk flags known at enrollment
    df["socioeconomic_risk_flags"] = (
        (df["Debtor"] == 1).astype(int)
        + (df["Tuition fees up to date"] == 0).astype(int)
        + (df["Scholarship holder"] == 0).astype(int)
    )

    return df


NEW_NUMERICAL_FEATURES = [
    "total_units_enrolled",
    "total_units_approved",
    "approval_ratio",
    "average_grade",
    "grade_trend",
    "socioeconomic_risk_flags",
]


def get_feature_lists():
    """Return the final categorical / numerical feature lists used for modeling."""
    categorical = list(config.CATEGORICAL_COLUMNS)
    numerical = list(config.NUMERICAL_COLUMNS) + NEW_NUMERICAL_FEATURES
    return categorical, numerical


def build_preprocessor() -> ColumnTransformer:
    """Build the sklearn ColumnTransformer used to encode/scale features."""
    categorical, numerical = get_feature_lists()

    numeric_pipeline = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_pipeline = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numerical),
            ("cat", categorical_pipeline, categorical),
        ]
    )
    return preprocessor


def get_transformed_feature_names(preprocessor: ColumnTransformer) -> list:
    """Return the flat list of feature names produced by `preprocessor.transform(...)`,
    in the same order (numerical columns first, then one-hot categorical columns)."""
    categorical, numerical = get_feature_lists()
    cat_names = list(
        preprocessor.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(categorical)
    )
    return numerical + cat_names


def prepare_dataset(path=None):
    """
    Full pipeline: load -> engineer features -> split into
    train / validation / test sets (stratified on the target).

    Returns X_train, X_val, X_test, y_train, y_val, y_test (all as DataFrames/Series).
    """
    df = load_raw_data(path)
    df = engineer_features(df)

    categorical, numerical = get_feature_lists()
    feature_columns = categorical + numerical

    X = df[feature_columns]
    y = df[config.TARGET_COLUMN]

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
    )
    val_fraction_of_remaining = config.VAL_SIZE / (1 - config.TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=val_fraction_of_remaining,
        random_state=config.RANDOM_STATE,
        stratify=y_train_full,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

"""
Central configuration for the Student Success Prediction project.
Keeping paths, column names, and constants in one place makes the
pipeline (data processing -> training -> app) easy to maintain.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"

RAW_DATA_PATH = DATA_DIR / "students_dropout_academic_success.csv"

MODEL_PATH = MODELS_DIR / "student_success_model.joblib"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"
FEATURE_IMPORTANCE_PATH = MODELS_DIR / "feature_importance.csv"
METADATA_PATH = MODELS_DIR / "metadata.json"
SHAP_BACKGROUND_PATH = MODELS_DIR / "shap_background.joblib"

# ---------------------------------------------------------------------------
# Target
# ---------------------------------------------------------------------------
TARGET_COLUMN = "Target"
TARGET_CLASSES = ["Dropout", "Enrolled", "Graduate"]

# Risk labels shown to end users (mapped from predicted class + probability)
RISK_LABELS = {
    "Dropout": "High Risk",
    "Enrolled": "Moderate Risk",
    "Graduate": "Low Risk",
}

# Continuous 0-100 risk score thresholds (score = calibrated P(Dropout) * 100,
# nudged by Enrolled probability — see src/risk_scoring.py for the exact formula).
RISK_SCORE_HIGH_THRESHOLD = 60
RISK_SCORE_MODERATE_THRESHOLD = 30

RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.1  # taken out of the remaining training data

# ---------------------------------------------------------------------------
# Columns that are naturally categorical/coded in the raw UCI dataset even
# though they are stored as integers (codes, not magnitudes).
# ---------------------------------------------------------------------------
CATEGORICAL_COLUMNS = [
    "Marital status",
    "Application mode",
    "Course",
    "Daytime/evening attendance",
    "Previous qualification",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "International",
]

NUMERICAL_COLUMNS = [
    "Application order",
    "Age at enrollment",
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)",
    "Unemployment rate",
    "Inflation rate",
    "GDP",
]

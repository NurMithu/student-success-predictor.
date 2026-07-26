# 🎓 AI-Powered Student Success Prediction & Intervention System

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/built%20with-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-pytest-0A9EDC.svg)](tests/)

A machine learning system that predicts whether a student will **Graduate**,
remain **Enrolled**, or **Dropout**, using enrollment demographics,
socio-economic factors, and 1st/2nd semester academic performance — with an
interactive dashboard for scoring individual students or entire cohorts,
**SHAP-based explanations** for every prediction, a continuous **0-100 risk
score**, and **AI-generated intervention plans** for teachers.

> Built on a **real, peer-reviewed dataset** of 4,424 students from a Portuguese
> higher-education institution (UCI Machine Learning Repository).

**Capabilities at a glance:**
- ✅ Data preprocessing & feature engineering (approval ratio, grade trend, socio-economic risk flags)
- ✅ Exploratory data analysis (outcome distribution, scholarship/tuition breakdowns, grade correlations)
- ✅ Multiple ML models compared head-to-head: **Logistic Regression, Random Forest, XGBoost, LightGBM**
- ✅ Explainable AI via **SHAP** (per-prediction + global feature importance)
- ✅ Continuous **dropout risk score (0-100)** with Low/Moderate/High tiers
- ✅ Interactive **Streamlit** dashboard (single-student form, batch CSV scoring, model dashboard)
- ✅ **LLM-generated intervention recommendations** for teachers (Claude API, with a rule-based fallback so the app works with zero configuration)

---

## Live Demo

🔗 **[Add your Streamlit Community Cloud URL here after deploying](#deployment)**

## Table of Contents

- [Problem & Motivation](#problem--motivation)
- [Dataset](#dataset)
- [System Architecture](#system-architecture)
- [Model Performance](#model-performance)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Deployment](#deployment)
- [Testing](#testing)
- [Limitations & Responsible Use](#limitations--responsible-use)
- [Roadmap](#roadmap)
- [License](#license)
- [Citation](#citation)

---

## Problem & Motivation

Academic dropout is costly for students, institutions, and society — but the
warning signs (declining grades, unpaid tuition, low course-completion ratio)
are often visible in institutional data long before a student formally
withdraws. This project turns that data into an **early-warning tool**:

- Score a **single student** interactively through a form
- Score an **entire cohort** by uploading a CSV
- Get a **risk tier** (Low / Moderate / High) instead of a raw score, so
  non-technical staff can act on it directly
- Inspect **model performance and feature importance** for transparency

## Dataset

**[Predict Students' Dropout and Academic Success](https://doi.org/10.24432/C5MC89)**
— UCI Machine Learning Repository, Dataset ID 697
(Realinho, Vieira Martins, Machado & Baptista, 2021, CC BY 4.0).

- 4,424 real, de-identified students
- 36 raw features (demographics, socio-economic status, academic path,
  1st/2nd semester performance, macroeconomic context)
- 3-class target: `Dropout` (32%) · `Enrolled` (18%) · `Graduate` (50%)

Full details and citation in [`data/README.md`](data/README.md).

## System Architecture

```
                 ┌─────────────────────┐
                 │  Raw UCI CSV data    │
                 └──────────┬──────────┘
                            ▼
                 ┌─────────────────────┐
                 │ Feature engineering  │  src/data_processing.py
                 │ (approval ratio,     │
                 │  grade trend, etc.)  │
                 └──────────┬──────────┘
                            ▼
                 ┌─────────────────────┐
                 │  Preprocessing       │  ColumnTransformer:
                 │  (scale + one-hot)   │  StandardScaler + OneHotEncoder
                 └──────────┬──────────┘
                            ▼
                 ┌─────────────────────┐
                 │ Model comparison &   │  src/train.py
                 │ selection (val set)  │  LogReg / RandomForest / XGBoost / LightGBM
                 └──────────┬──────────┘
                            ▼
                 ┌─────────────────────┐
                 │ Final fit + test-set │  models/*.joblib, metrics.json
                 │ evaluation           │
                 └──────────┬──────────┘
                            ▼
              ┌─────────────┴──────────────┐
              ▼                            ▼
   ┌─────────────────────┐      ┌───────────────────────┐
   │ Risk scoring          │      │ SHAP explainability    │
   │ (0-100 dropout score) │      │ (per-pred + global)    │  src/risk_scoring.py
   │ src/risk_scoring.py    │      │ src/shap_utils.py       │
   └──────────┬────────────┘      └───────────┬────────────┘
              └─────────────┬──────────────────┘
                            ▼
                 ┌─────────────────────┐
                 │ LLM intervention      │  src/llm_recommendations.py
                 │ recommendations        │  (Claude API, rule-based fallback)
                 └──────────┬──────────┘
                            ▼
                 ┌─────────────────────┐
                 │  Streamlit app        │  app.py
                 │  (single + batch      │
                 │   prediction, EDA,    │
                 │   model + SHAP        │
                 │   dashboard)          │
                 └─────────────────────┘
```

## Model Performance

Model selection is done on a validation split; the winner is refit on
train+validation and evaluated **once** on a held-out test set (20% of data,
stratified). Exact numbers are written to [`models/metrics.json`](models/metrics.json)
every time you run training — the table below is refreshed from the last run.

| Metric | Value |
|---|---|
| Model selected | Logistic Regression *(may vary by run/seed)* |
| Test accuracy | ~72% |
| Test macro F1 | ~0.68 |
| Test macro ROC-AUC | ~0.88 |

**Full validation-set comparison** (last training run):

| Model | Accuracy | Macro F1 | Macro ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0.772 | **0.737** | 0.892 |
| Random Forest | 0.754 | 0.694 | 0.885 |
| XGBoost | 0.754 | 0.713 | **0.897** |
| LightGBM | 0.768 | 0.710 | 0.893 |

Logistic Regression wins on macro F1 here mainly because class-balanced
weighting handles the minority "Enrolled" class better than the tree ensembles
did out-of-the-box in this run — the tree models are close competitors and
XGBoost edges ahead on ROC-AUC. This is a genuine, data-driven selection, not
a foregone conclusion: rerun `python -m src.train` after tuning tree
hyperparameters and a different model may well win. Regardless of which model
wins, SHAP explanations (below) work for either family automatically.

These numbers are in line with published benchmarks on this dataset (typically
75–80% accuracy for the 3-class task, which has a meaningful class imbalance).
The **Model Performance** tab in the app shows the full validation comparison,
confusion matrix, per-class precision/recall, and feature importance.

## Repository Structure

```
student-success-predictor/
├── app.py                      # Streamlit dashboard (entry point)
├── requirements.txt             # lean, deploy-time deps (no xgboost/lightgbm)
├── requirements-train.txt       # adds xgboost, lightgbm, pytest for training/testing
├── LICENSE
├── notebooks/
│   └── eda_and_modeling.ipynb   # exploratory analysis + model comparison walkthrough
├── data/
│   ├── students_dropout_academic_success.csv
│   └── README.md               # dataset documentation & citation
├── src/
│   ├── config.py                # paths & shared constants
│   ├── data_processing.py       # loading, feature engineering, splitting
│   ├── codebook.py              # human-readable labels for coded fields
│   ├── train.py                 # model training & evaluation pipeline (LogReg/RF/XGBoost/LightGBM)
│   ├── predict.py               # inference wrapper (predictions + risk score)
│   ├── risk_scoring.py          # 0-100 dropout risk score + risk-factor helpers
│   ├── shap_utils.py            # SHAP explainability wrapper (per-pred + global)
│   └── llm_recommendations.py   # Claude-generated intervention plans + rule-based fallback
├── models/                      # generated by `python -m src.train`
│   ├── student_success_model.joblib
│   ├── preprocessor.joblib
│   ├── label_encoder.joblib
│   ├── shap_background.joblib
│   ├── metrics.json
│   ├── metadata.json
│   └── feature_importance.csv
├── tests/
│   └── test_pipeline.py         # pytest smoke tests
└── .streamlit/
    └── config.toml              # app theme
```

## Installation

```bash
git clone https://github.com/<your-username>/student-success-predictor.git
cd student-success-predictor

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# To just run the app (fast install, no xgboost/lightgbm):
pip install -r requirements.txt

# To also retrain the model or run tests (adds xgboost, lightgbm, pytest):
pip install -r requirements-train.txt
```

`requirements.txt` is intentionally minimal so Streamlit Cloud deploys
quickly — see [Deployment](#deployment) for why.

## Usage

**1. Train the model** (writes artifacts to `models/`):

```bash
python -m src.train
```

Expected output ends with something like:

```
Held-out TEST set performance:
    accuracy: 0.72
    f1_macro: 0.68
    roc_auc_macro: 0.88
Saved model -> models/student_success_model.joblib
```

**2. Launch the app:**

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (default `http://localhost:8501`).

**3. (Optional) Enable AI-generated intervention recommendations:**

Without any setup, the app generates intervention plans using a rule-based
engine. To get Claude-generated plans instead, provide an Anthropic API key
either:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

or paste it into the "Anthropic API key (optional)" field in the app's sidebar
(session-only, never written to disk). Get a key at
[console.anthropic.com](https://console.anthropic.com/).

**4. (Optional) Use the model programmatically:**

```python
from src.predict import StudentSuccessPredictor

predictor = StudentSuccessPredictor()
result = predictor.predict_one({
    "Marital status": 1, "Application mode": 1, "Application order": 1,
    "Course": 9500, "Daytime/evening attendance": 1, "Previous qualification": 1,
    "Nacionality": 1, "Mother's qualification": 1, "Father's qualification": 1,
    "Mother's occupation": 5, "Father's occupation": 5, "Displaced": 1,
    "Educational special needs": 0, "Debtor": 0, "Tuition fees up to date": 1,
    "Gender": 0, "Scholarship holder": 1, "Age at enrollment": 19, "International": 0,
    "Curricular units 1st sem (credited)": 0, "Curricular units 1st sem (enrolled)": 6,
    "Curricular units 1st sem (evaluations)": 6, "Curricular units 1st sem (approved)": 6,
    "Curricular units 1st sem (grade)": 14.0, "Curricular units 1st sem (without evaluations)": 0,
    "Curricular units 2nd sem (credited)": 0, "Curricular units 2nd sem (enrolled)": 6,
    "Curricular units 2nd sem (evaluations)": 6, "Curricular units 2nd sem (approved)": 6,
    "Curricular units 2nd sem (grade)": 14.0, "Curricular units 2nd sem (without evaluations)": 0,
    "Unemployment rate": 11.0, "Inflation rate": 1.0, "GDP": 1.0,
})
print(result["predicted_status"], result["risk_level"])
```

## Deployment

This app is deployed via **Streamlit Community Cloud**.

**Before deploying:** make sure `models/` (the trained artifacts) is committed
to your repo — the app loads a pre-trained model rather than retraining on
startup, so deployment is fast and doesn't need `requirements-train.txt`.

1. Push this repo to GitHub (public repo, `models/` folder included).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Select your repo/branch and set **Main file path** to `app.py`.
4. **Click "Advanced settings" before deploying** and explicitly set **Python
   version to 3.11**. This step matters a lot: Streamlit Community Cloud has
   been defaulting new apps to Python 3.13/3.14, and older-but-well-supported
   packages this project depends on (`numpy`, `shap`'s `numba`/`llvmlite`,
   etc.) don't yet have installable wheels for those versions — the build
   fails silently and the app just never finishes loading. A `runtime.txt`
   file is *not* a reliable fix for this (there are multiple open Streamlit
   Cloud bug reports of it being ignored) — the Advanced settings dropdown is
   the mechanism that actually works. **If you already deployed and it's
   stuck**, Python version can't be changed on an existing app — delete it
   and redeploy, setting the version in Advanced settings this time.
5. (Optional, for Claude-generated recommendations) In the same **Advanced
   settings → Secrets** box, add:
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   ```
   Without this, the app automatically uses the rule-based recommendation
   fallback — fully functional either way.
6. Click **Deploy**. Streamlit installs `requirements.txt` automatically.
7. Copy the resulting `https://<app-name>.streamlit.app` URL into this README.

**If the app still fails to load after setting Python 3.11:** open **Manage
app → logs** from the Streamlit Cloud dashboard — the real error is always in
there (usually a `pip install` failure for a specific package). The app is
also written to degrade gracefully rather than crash outright: if `shap`
specifically fails to install, every other page (predictions, risk scores,
batch scoring, LLM recommendations) still works — only the SHAP explanation
panels show an informational message instead of a chart.

### Why the deploy should now be fast

Two dependency issues were fixed to keep the build quick:

- **`xgboost` no longer installs on deploy.** `requirements.txt` (used by the
  app) intentionally excludes `xgboost`/`lightgbm` — the currently-winning
  model is scikit-learn Logistic Regression, and unpickling a saved model
  only needs the library that created it. `xgboost` alone is a ~300MB wheel
  (it bundles CUDA support even for CPU-only use), so skipping it when it's
  not needed cuts real time off the build.
- **`xgboost` is pinned to `2.0.3` in `requirements-train.txt`** (used only
  for `python -m src.train` / `pytest`, run locally). Versions `2.1.0+`
  unconditionally pull in an extra `nvidia-nccl-cu12` package (~300MB) even
  on CPU-only machines — `2.0.3` avoids that entirely with identical
  training/inference behavior for this project.

If you retrain locally and a tree-based model (Random Forest/XGBoost/LightGBM)
wins instead of Logistic Regression, add `xgboost==2.0.3` and/or
`lightgbm==4.5.0` back to `requirements.txt` before redeploying — otherwise
the deployed app will fail to unpickle that model.

## Power BI Alternative

The task brief mentions Power BI or Streamlit — this repo ships the interactive
experience as Streamlit, but every output is plain CSV/JSON, so a Power BI
front end is a drop-in alternative if you ever need one:

1. Use the **Batch (CSV)** page in the app (or `StudentSuccessPredictor.predict()`
   directly) to export a `student_predictions.csv` with `predicted_status`,
   `risk_score`, `risk_level`, and per-class probabilities for a cohort.
2. In Power BI Desktop: **Get Data → Text/CSV**, point at that predictions
   file (and `models/feature_importance.csv` for a feature-importance chart),
   then build cards/tables/slicers on `risk_level` and `risk_score`.
3. Publish to the Power BI Service and schedule a refresh against a
   regenerated CSV if the underlying data changes.

## Testing

```bash
pip install -r requirements-train.txt   # adds pytest (not in the lean requirements.txt)
pytest -v
```

8 tests cover: data loading integrity, feature engineering correctness,
train/val/test split shapes and stratification, end-to-end prediction sanity
checks (including that predicted probabilities sum to 1 and risk scores fall
in [0, 100]), SHAP explanation output shape/validity, and the LLM
recommendation module's rule-based fallback path (so CI never needs a real
API key to pass).

## Limitations & Responsible Use

- Trained on data from **one institution in Portugal**; behavior on other
  institutions, countries, or student populations is not guaranteed and
  should be validated before use.
- The dataset reflects one point in time; enrollment policies, courses, and
  economic conditions change, so periodic retraining is recommended.
- Predictions are **decision support**, not a substitute for academic
  advisors — they should inform, not automate, decisions that affect a
  student's academic standing, funding, or enrollment.
- Class imbalance (Enrolled is the minority class) means recall for that
  class is lower than for Dropout/Graduate — see the confusion matrix in the
  **Model Performance** tab before relying on it for that group.
- **LLM-generated recommendations are drafting aids, not vetted advice.**
  They're grounded in the model's own SHAP factors (not invented data), but
  should be reviewed by the advisor before acting on them — treat them like a
  first draft from a knowledgeable colleague, not a final decision.
- Real student data is sensitive. If you deploy this with real (not sample)
  student records, review your institution's data-protection obligations
  (e.g. FERPA in the US, GDPR in the EU) before sending any student data to a
  third-party LLM API, and consider disabling the LLM feature entirely
  (leave `ANTHROPIC_API_KEY` unset) if that data cannot leave your systems.

## Roadmap

- [x] SHAP-based per-prediction and global explanations
- [x] Multiple ML models compared (Logistic Regression, Random Forest, XGBoost, LightGBM)
- [x] Continuous 0-100 risk score
- [x] LLM-generated intervention recommendations (with rule-based fallback)
- [x] CI workflow to re-run tests and re-train on every push
- [ ] Hyperparameter tuning (grid/Bayesian search) across all four model families
- [ ] Model versioning/experiment tracking (e.g. MLflow)
- [ ] Model calibration (e.g. `CalibratedClassifierCV`) so risk scores map more precisely to real-world dropout rates
- [ ] Multi-institution dataset support
- [ ] Optional Power BI dashboard as an alternative front end for institutions already standardized on Power BI

## License

Code is released under the [MIT License](LICENSE). The dataset is distributed
under CC BY 4.0 by its original authors — see [`data/README.md`](data/README.md).

## Citation

If you use this project, please cite the original dataset:

```bibtex
@misc{realinho2021predict,
  title        = {Predict Students' Dropout and Academic Success},
  author       = {Realinho, Valentim and Vieira Martins, M{\'o}nica and Machado, Jorge and Baptista, Lu{\'i}s},
  year         = {2021},
  howpublished = {UCI Machine Learning Repository},
  doi          = {10.24432/C5MC89}
}
```

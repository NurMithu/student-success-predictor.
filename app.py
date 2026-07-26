"""
AI-Powered Student Success Prediction & Intervention System
--------------------------------------------------------------
A Streamlit app that predicts a student's likely outcome
(Graduate / Enrolled / Dropout) from academic, demographic, and
socio-economic data, and surfaces early-warning risk signals so
academic staff can plan timely interventions.

Run locally:
    streamlit run app.py
"""

import json
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import config
from src.codebook import (
    MARITAL_STATUS, APPLICATION_MODE, COURSE, DAYTIME_EVENING,
    PREVIOUS_QUALIFICATION, PARENT_QUALIFICATION, PARENT_OCCUPATION,
    NATIONALITY, YES_NO, GENDER, options_for_select,
)
from src.predict import StudentSuccessPredictor
from src.shap_utils import ShapExplainer
from src.risk_scoring import friendly_feature_name
from src.llm_recommendations import StudentContext, generate_recommendation

st.set_page_config(
    page_title="Student Success Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------
@st.cache_resource
def load_predictor():
    return StudentSuccessPredictor()


@st.cache_resource
def load_shap_explainer():
    return ShapExplainer()


@st.cache_data
def load_metrics():
    with open(config.METRICS_PATH) as f:
        return json.load(f)


@st.cache_data
def load_dataset():
    return pd.read_csv(config.RAW_DATA_PATH)


@st.cache_data
def load_feature_importance():
    try:
        return pd.read_csv(config.FEATURE_IMPORTANCE_PATH)
    except FileNotFoundError:
        return None


RISK_COLORS = {"Low Risk": "#2E7D32", "Moderate Risk": "#F9A825", "High Risk": "#C62828"}


def risk_badge(risk_level: str) -> str:
    color = RISK_COLORS.get(risk_level, "#616161")
    return f"""<span style="background-color:{color}; color:white; padding:4px 12px;
                border-radius:14px; font-weight:600; font-size:0.85rem;">{risk_level}</span>"""


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("🎓 Student Success Predictor")
st.sidebar.caption("AI-powered early-warning system for academic dropout risk")
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Predict — Single Student", "Predict — Batch (CSV)", "Model Performance", "Data Insights", "About"],
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Dataset:** UCI *Predict Students' Dropout and Academic Success* "
    "(Realinho et al., 2021) — 4,424 real students, 3 outcome classes."
)
st.sidebar.markdown("[📄 Dataset source](https://doi.org/10.24432/C5MC89)")

st.sidebar.markdown("---")
st.sidebar.markdown("**🤖 LLM Recommendations**")
_api_key_input = st.sidebar.text_input(
    "Anthropic API key (optional)",
    type="password",
    help="Paste a key to enable Claude-generated intervention plans for this session only. "
    "Without a key, a rule-based recommendation engine is used instead — the app still fully works.",
)
if _api_key_input:
    os.environ["ANTHROPIC_API_KEY"] = _api_key_input
    st.sidebar.success("LLM recommendations enabled for this session.")
else:
    st.sidebar.caption("No key entered — using rule-based recommendations.")

try:
    predictor = load_predictor()
    metrics = load_metrics()
    model_ready = True
except FileNotFoundError:
    predictor = None
    metrics = None
    model_ready = False

# ---------------------------------------------------------------------------
# Page: Overview
# ---------------------------------------------------------------------------
if page == "Overview":
    st.title("🎓 AI-Powered Student Success Prediction & Intervention System")
    st.markdown(
        """
Higher-education institutions lose a substantial share of students to dropout every year —
often for reasons that are visible in the data *months* before a student actually leaves.
This system uses a machine learning model trained on **4,424 real student records** to flag
at-risk students early, so advisors and academic staff can step in with targeted support.
        """
    )

    col1, col2, col3, col4 = st.columns(4)
    if model_ready:
        col1.metric("Model", metrics["best_model"].replace("_", " ").title())
        col2.metric("Test Accuracy", f"{metrics['test_metrics']['accuracy']*100:.1f}%")
        col3.metric("Macro ROC-AUC", f"{metrics['test_metrics']['roc_auc_macro']:.3f}")
        col4.metric("Macro F1", f"{metrics['test_metrics']['f1_macro']:.3f}")
    else:
        st.warning("No trained model found. Run `python -m src.train` first.")

    st.markdown("### How it works")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("#### 1️⃣ Data")
        st.write(
            "Enrollment demographics, socio-economic factors, and 1st/2nd "
            "semester academic performance for each student."
        )
    with c2:
        st.markdown("#### 2️⃣ Model")
        st.write(
            "Logistic Regression, Random Forest, XGBoost, and LightGBM are "
            "compared on a validation set; the best is kept."
        )
    with c3:
        st.markdown("#### 3️⃣ Explain")
        st.write(
            "SHAP explains **why** — which specific factors pushed a "
            "student's risk score up or down."
        )
    with c4:
        st.markdown("#### 4️⃣ Act")
        st.write(
            "A 0-100 risk score plus an AI-generated (or rule-based) "
            "intervention plan gives staff a concrete next step."
        )

    st.markdown("### Try it now")
    st.write("Use the sidebar to jump to **Predict — Single Student** to score one student interactively, "
              "or **Predict — Batch (CSV)** to score an entire cohort at once.")

# ---------------------------------------------------------------------------
# Page: Single student prediction
# ---------------------------------------------------------------------------
elif page == "Predict — Single Student":
    st.title("Predict a Single Student's Outcome")

    if not model_ready:
        st.error("No trained model found. Run `python -m src.train` first, then restart the app.")
        st.stop()

    st.write("Fill in the student's enrollment details and academic record below.")

    with st.form("student_form"):
        st.subheader("Demographics & Enrollment")
        c1, c2, c3 = st.columns(3)
        with c1:
            gender_label = st.selectbox("Gender", list(GENDER.values()))
            gender = [k for k, v in GENDER.items() if v == gender_label][0]
            age = st.number_input("Age at enrollment", min_value=16, max_value=70, value=20)
            marital_label = st.selectbox("Marital status", [v for v in MARITAL_STATUS.values()])
            marital = [k for k, v in MARITAL_STATUS.items() if v == marital_label][0]
        with c2:
            course_label = st.selectbox("Course", [v for v in COURSE.values()])
            course = [k for k, v in COURSE.items() if v == course_label][0]
            attendance_label = st.selectbox("Attendance", list(DAYTIME_EVENING.values()))
            attendance = [k for k, v in DAYTIME_EVENING.items() if v == attendance_label][0]
            app_order = st.number_input("Application order (0=first choice)", min_value=0, max_value=9, value=1)
        with c3:
            app_mode_label = st.selectbox("Application mode", [v for v in APPLICATION_MODE.values()])
            app_mode = [k for k, v in APPLICATION_MODE.items() if v == app_mode_label][0]
            prev_qual_label = st.selectbox("Previous qualification", [v for v in PREVIOUS_QUALIFICATION.values()])
            prev_qual = [k for k, v in PREVIOUS_QUALIFICATION.items() if v == prev_qual_label][0]
            nationality_label = st.selectbox("Nationality", [v for v in NATIONALITY.values()], index=0)
            nationality = [k for k, v in NATIONALITY.items() if v == nationality_label][0]

        st.subheader("Socio-Economic Factors")
        c4, c5, c6 = st.columns(3)
        with c4:
            displaced_label = st.selectbox("Displaced student", list(YES_NO.values()))
            displaced = [k for k, v in YES_NO.items() if v == displaced_label][0]
            special_needs_label = st.selectbox("Educational special needs", list(YES_NO.values()), index=1)
            special_needs = [k for k, v in YES_NO.items() if v == special_needs_label][0]
        with c5:
            debtor_label = st.selectbox("Debtor", list(YES_NO.values()), index=1)
            debtor = [k for k, v in YES_NO.items() if v == debtor_label][0]
            tuition_label = st.selectbox("Tuition fees up to date", list(YES_NO.values()))
            tuition = [k for k, v in YES_NO.items() if v == tuition_label][0]
        with c6:
            scholarship_label = st.selectbox("Scholarship holder", list(YES_NO.values()), index=1)
            scholarship = [k for k, v in YES_NO.items() if v == scholarship_label][0]
            international_label = st.selectbox("International student", list(YES_NO.values()), index=1)
            international = [k for k, v in YES_NO.items() if v == international_label][0]

        st.subheader("Parents' Background")
        c7, c8 = st.columns(2)
        with c7:
            mother_qual_label = st.selectbox("Mother's qualification", [v for v in PARENT_QUALIFICATION.values()])
            mother_qual = [k for k, v in PARENT_QUALIFICATION.items() if v == mother_qual_label][0]
            mother_occ_label = st.selectbox("Mother's occupation", [v for v in PARENT_OCCUPATION.values()])
            mother_occ = [k for k, v in PARENT_OCCUPATION.items() if v == mother_occ_label][0]
        with c8:
            father_qual_label = st.selectbox("Father's qualification", [v for v in PARENT_QUALIFICATION.values()])
            father_qual = [k for k, v in PARENT_QUALIFICATION.items() if v == father_qual_label][0]
            father_occ_label = st.selectbox("Father's occupation", [v for v in PARENT_OCCUPATION.values()])
            father_occ = [k for k, v in PARENT_OCCUPATION.items() if v == father_occ_label][0]

        st.subheader("Semester 1 Academic Record")
        s1a, s1b, s1c, s1d, s1e = st.columns(5)
        s1_credited = s1a.number_input("Credited", 0, 30, 0, key="s1_cred")
        s1_enrolled = s1b.number_input("Enrolled", 0, 30, 6, key="s1_enr")
        s1_evals = s1c.number_input("Evaluations", 0, 30, 6, key="s1_eval")
        s1_approved = s1d.number_input("Approved", 0, 30, 5, key="s1_app")
        s1_grade = s1e.number_input("Avg grade (0-20)", 0.0, 20.0, 12.0, key="s1_grade")
        s1_no_eval = st.number_input("Units without evaluation (sem 1)", 0, 30, 0, key="s1_noeval")

        st.subheader("Semester 2 Academic Record")
        s2a, s2b, s2c, s2d, s2e = st.columns(5)
        s2_credited = s2a.number_input("Credited", 0, 30, 0, key="s2_cred")
        s2_enrolled = s2b.number_input("Enrolled", 0, 30, 6, key="s2_enr")
        s2_evals = s2c.number_input("Evaluations", 0, 30, 6, key="s2_eval")
        s2_approved = s2d.number_input("Approved", 0, 30, 5, key="s2_app")
        s2_grade = s2e.number_input("Avg grade (0-20)", 0.0, 20.0, 12.0, key="s2_grade")
        s2_no_eval = st.number_input("Units without evaluation (sem 2)", 0, 30, 0, key="s2_noeval")

        st.subheader("Macroeconomic Context (at enrollment year)")
        m1, m2, m3 = st.columns(3)
        unemployment = m1.number_input("Unemployment rate (%)", 0.0, 30.0, 11.0)
        inflation = m2.number_input("Inflation rate (%)", -5.0, 15.0, 1.0)
        gdp = m3.number_input("GDP growth", -10.0, 10.0, 1.0)

        submitted = st.form_submit_button("🔮 Predict Outcome", use_container_width=True)

    if submitted:
        student = {
            "Marital status": marital,
            "Application mode": app_mode,
            "Application order": app_order,
            "Course": course,
            "Daytime/evening attendance": attendance,
            "Previous qualification": prev_qual,
            "Nacionality": nationality,
            "Mother's qualification": mother_qual,
            "Father's qualification": father_qual,
            "Mother's occupation": mother_occ,
            "Father's occupation": father_occ,
            "Displaced": displaced,
            "Educational special needs": special_needs,
            "Debtor": debtor,
            "Tuition fees up to date": tuition,
            "Gender": gender,
            "Scholarship holder": scholarship,
            "Age at enrollment": age,
            "International": international,
            "Curricular units 1st sem (credited)": s1_credited,
            "Curricular units 1st sem (enrolled)": s1_enrolled,
            "Curricular units 1st sem (evaluations)": s1_evals,
            "Curricular units 1st sem (approved)": s1_approved,
            "Curricular units 1st sem (grade)": s1_grade,
            "Curricular units 1st sem (without evaluations)": s1_no_eval,
            "Curricular units 2nd sem (credited)": s2_credited,
            "Curricular units 2nd sem (enrolled)": s2_enrolled,
            "Curricular units 2nd sem (evaluations)": s2_evals,
            "Curricular units 2nd sem (approved)": s2_approved,
            "Curricular units 2nd sem (grade)": s2_grade,
            "Curricular units 2nd sem (without evaluations)": s2_no_eval,
            "Unemployment rate": unemployment,
            "Inflation rate": inflation,
            "GDP": gdp,
        }
        result = predictor.predict_one(student)

        st.markdown("---")
        st.subheader("Prediction Result")
        r1, r2 = st.columns([1, 2])
        with r1:
            st.markdown(f"**Predicted outcome:** {result['predicted_status']}")
            st.markdown(risk_badge(result["risk_level"]), unsafe_allow_html=True)
        with r2:
            probs = {
                cls: result[f"probability_{cls.lower()}"]
                for cls in config.TARGET_CLASSES
            }
            fig = go.Figure(
                go.Bar(
                    x=list(probs.values()),
                    y=list(probs.keys()),
                    orientation="h",
                    marker_color=["#C62828", "#F9A825", "#2E7D32"],
                    text=[f"{v*100:.1f}%" for v in probs.values()],
                    textposition="outside",
                )
            )
            fig.update_layout(
                xaxis_title="Probability", height=250, margin=dict(l=10, r=10, t=10, b=10)
            )
            st.plotly_chart(fig, use_container_width=True)

        st.metric("Dropout Risk Score", f"{result['risk_score']:.0f} / 100")

        # ---- SHAP explanation for this specific student -------------------
        st.markdown("---")
        st.subheader("🔍 Why this prediction? (SHAP explanation)")
        with st.spinner("Computing SHAP explanation..."):
            explainer = load_shap_explainer()
            student_df = pd.DataFrame([student])
            contrib = explainer.dropout_contributions(student_df).iloc[0]

        top_up = contrib.sort_values(ascending=False).head(6)
        top_down = contrib.sort_values(ascending=True).head(6)

        c_up, c_down = st.columns(2)
        with c_up:
            st.markdown("**⬆️ Factors increasing dropout risk**")
            up_df = pd.DataFrame(
                {"feature": [friendly_feature_name(f) for f in top_up.index], "impact": top_up.values}
            )
            up_df = up_df[up_df["impact"] > 0]
            if len(up_df) > 0:
                fig_up = px.bar(up_df.sort_values("impact"), x="impact", y="feature", orientation="h",
                                 color_discrete_sequence=["#C62828"])
                st.plotly_chart(fig_up, use_container_width=True)
            else:
                st.caption("No strong risk-increasing factors detected.")
        with c_down:
            st.markdown("**⬇️ Factors decreasing dropout risk**")
            down_df = pd.DataFrame(
                {"feature": [friendly_feature_name(f) for f in top_down.index], "impact": top_down.values}
            )
            down_df = down_df[down_df["impact"] < 0]
            if len(down_df) > 0:
                fig_down = px.bar(down_df.sort_values("impact", ascending=False), x="impact", y="feature",
                                   orientation="h", color_discrete_sequence=["#2E7D32"])
                st.plotly_chart(fig_down, use_container_width=True)
            else:
                st.caption("No strong risk-decreasing factors detected.")

        # ---- Rule-of-thumb intervention banner -----------------------------
        if result["risk_level"] == "High Risk":
            st.error(
                "⚠️ This student shows a high probability of dropout — see the AI-generated "
                "intervention plan below for specific next steps."
            )
        elif result["risk_level"] == "Moderate Risk":
            st.warning("This student's outcome is uncertain. See the intervention plan below.")
        else:
            st.success("This student is on track to graduate based on current indicators.")

        # ---- LLM (or rule-based) intervention recommendation --------------
        st.markdown("---")
        st.subheader("📋 Suggested Intervention Plan")
        top_factors_for_llm = [
            (friendly_feature_name(f), float(v)) for f, v in top_up.items() if v > 0
        ][:5]
        ctx = StudentContext(
            student_label=f"{gender_label}, age {age}, {course_label}",
            predicted_status=result["predicted_status"],
            risk_level=result["risk_level"],
            risk_score=result["risk_score"],
            top_risk_factors=top_factors_for_llm,
            key_stats={
                "1st semester approval ratio": f"{s1_approved}/{s1_enrolled} units",
                "2nd semester approval ratio": f"{s2_approved}/{s2_enrolled} units",
                "average grade": f"{(s1_grade + s2_grade) / 2:.1f} / 20",
                "tuition fees up to date": tuition_label,
                "scholarship holder": scholarship_label,
                "debtor": debtor_label,
            },
        )
        with st.spinner("Generating recommendation..."):
            recommendation_text, source = generate_recommendation(ctx)
        badge = "🤖 Claude-generated" if source == "llm" else "📐 Rule-based (no API key set)"
        st.caption(badge)
        st.markdown(recommendation_text)

# ---------------------------------------------------------------------------
# Page: Batch prediction
# ---------------------------------------------------------------------------
elif page == "Predict — Batch (CSV)":
    st.title("Batch Prediction from CSV")

    if not model_ready:
        st.error("No trained model found. Run `python -m src.train` first, then restart the app.")
        st.stop()

    st.write(
        "Upload a CSV with the same columns as the original UCI dataset "
        "(one row per student, no `Target` column needed). "
        "Don't have a file handy? Download a sample below."
    )

    sample_df = load_dataset().drop(columns=[config.TARGET_COLUMN]).head(10)
    st.download_button(
        "📥 Download sample CSV (10 rows)",
        sample_df.to_csv(index=False).encode("utf-8"),
        file_name="sample_students.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader("Upload student CSV", type=["csv"])
    if uploaded is not None:
        input_df = pd.read_csv(uploaded)
        st.write(f"Loaded {len(input_df)} students.")
        with st.spinner("Scoring students..."):
            preds = predictor.predict(input_df)
        combined = pd.concat([input_df.reset_index(drop=True), preds.reset_index(drop=True)], axis=1)

        st.subheader("Results Summary")
        risk_counts = combined["risk_level"].value_counts()
        fig = px.pie(
            names=risk_counts.index,
            values=risk_counts.values,
            color=risk_counts.index,
            color_discrete_map=RISK_COLORS,
            title="Risk Distribution",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Full Results")
        st.dataframe(combined, use_container_width=True)

        st.download_button(
            "📥 Download predictions as CSV",
            combined.to_csv(index=False).encode("utf-8"),
            file_name="student_predictions.csv",
            mime="text/csv",
        )

        high_risk = combined[combined["risk_level"] == "High Risk"].sort_values("risk_score", ascending=False)
        if len(high_risk) > 0:
            st.warning(f"⚠️ {len(high_risk)} student(s) flagged as High Risk — prioritize for advisor outreach.")

            st.subheader("📋 Generate Intervention Plans for Highest-Risk Students")
            max_n = min(10, len(high_risk))
            n_students = st.slider("Number of students to generate plans for", 1, max_n, min(3, max_n))
            if st.button("Generate intervention plans"):
                explainer = load_shap_explainer()
                subset = high_risk.head(n_students)
                subset_raw = input_df.loc[subset.index]
                contrib_df = explainer.dropout_contributions(subset_raw)

                for idx in subset.index:
                    row = combined.loc[idx]
                    contrib_row = contrib_df.loc[idx].sort_values(ascending=False)
                    top_factors = [
                        (friendly_feature_name(f), float(v)) for f, v in contrib_row.head(5).items() if v > 0
                    ]
                    ctx = StudentContext(
                        student_label=f"Student (row {idx})",
                        predicted_status=row["predicted_status"],
                        risk_level=row["risk_level"],
                        risk_score=row["risk_score"],
                        top_risk_factors=top_factors,
                        key_stats={"risk_score": f"{row['risk_score']:.0f}/100"},
                    )
                    with st.spinner(f"Generating plan for row {idx}..."):
                        text, source = generate_recommendation(ctx)
                    with st.expander(f"Row {idx} — risk score {row['risk_score']:.0f} "
                                      f"({'🤖 Claude' if source == 'llm' else '📐 rule-based'})"):
                        st.markdown(text)

# ---------------------------------------------------------------------------
# Page: Model performance
# ---------------------------------------------------------------------------
elif page == "Model Performance":
    st.title("Model Performance")

    if not model_ready:
        st.error("No trained model found. Run `python -m src.train` first.")
        st.stop()

    st.subheader("Validation Comparison Across Candidate Models")
    val_df = pd.DataFrame(metrics["validation_comparison"]).T
    st.dataframe(val_df.style.highlight_max(axis=0, color="#c6efce"), use_container_width=True)

    st.subheader(f"Held-out Test Set Performance — Final Model: `{metrics['best_model']}`")
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{metrics['test_metrics']['accuracy']*100:.2f}%")
    c2.metric("Macro F1", f"{metrics['test_metrics']['f1_macro']:.3f}")
    c3.metric("Macro ROC-AUC", f"{metrics['test_metrics']['roc_auc_macro']:.3f}")

    st.subheader("Confusion Matrix (Test Set)")
    cm = metrics["test_confusion_matrix"]
    classes = metrics["class_order"]
    fig_cm = px.imshow(
        cm, x=classes, y=classes, text_auto=True, color_continuous_scale="Blues",
        labels=dict(x="Predicted", y="Actual", color="Count"),
    )
    st.plotly_chart(fig_cm, use_container_width=True)

    st.subheader("Per-Class Metrics (Test Set)")
    report_df = pd.DataFrame(metrics["test_classification_report"]).T
    st.dataframe(report_df, use_container_width=True)

    fi_df = load_feature_importance()
    if fi_df is not None:
        st.subheader("Top 15 Most Important Features (model-native importance)")
        top15 = fi_df.head(15).sort_values("importance")
        fig_fi = px.bar(top15, x="importance", y="feature", orientation="h")
        st.plotly_chart(fig_fi, use_container_width=True)

    st.subheader("Global Feature Importance (mean |SHAP value| for Dropout class)")
    st.caption(
        "Computed on a 200-row background sample from the training data. SHAP values show each "
        "feature's actual contribution to model output, which can differ from model-native importance above."
    )
    with st.spinner("Computing SHAP global importance (first load only)..."):
        explainer = load_shap_explainer()
        global_imp = explainer.global_importance(sample_size=150).head(15)
    global_imp["feature"] = global_imp["feature"].apply(friendly_feature_name)
    global_imp = global_imp.groupby("feature", as_index=False)["mean_abs_shap"].sum().sort_values(
        "mean_abs_shap"
    ).tail(15)
    fig_shap = px.bar(global_imp, x="mean_abs_shap", y="feature", orientation="h",
                       color_discrete_sequence=["#5C6BC0"])
    st.plotly_chart(fig_shap, use_container_width=True)

# ---------------------------------------------------------------------------
# Page: Data insights
# ---------------------------------------------------------------------------
elif page == "Data Insights":
    st.title("Dataset Insights")
    df = load_dataset()

    st.subheader("Outcome Distribution")
    counts = df[config.TARGET_COLUMN].value_counts()
    fig1 = px.bar(x=counts.index, y=counts.values, labels={"x": "Outcome", "y": "Number of Students"},
                  color=counts.index, color_discrete_map={"Dropout": "#C62828", "Enrolled": "#F9A825", "Graduate": "#2E7D32"})
    st.plotly_chart(fig1, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Outcome by Scholarship Status")
        cross = pd.crosstab(df["Scholarship holder"].map({0: "No scholarship", 1: "Scholarship"}), df[config.TARGET_COLUMN], normalize="index") * 100
        fig2 = px.bar(cross, barmode="stack", labels={"value": "% of students"})
        st.plotly_chart(fig2, use_container_width=True)
    with col2:
        st.subheader("Outcome by Tuition Fee Status")
        cross2 = pd.crosstab(df["Tuition fees up to date"].map({0: "Fees overdue", 1: "Fees up to date"}), df[config.TARGET_COLUMN], normalize="index") * 100
        fig3 = px.bar(cross2, barmode="stack", labels={"value": "% of students"})
        st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Age at Enrollment by Outcome")
    fig4 = px.box(df, x=config.TARGET_COLUMN, y="Age at enrollment", color=config.TARGET_COLUMN,
                  color_discrete_map={"Dropout": "#C62828", "Enrolled": "#F9A825", "Graduate": "#2E7D32"})
    st.plotly_chart(fig4, use_container_width=True)

    st.subheader("1st Semester Grade vs 2nd Semester Grade")
    fig5 = px.scatter(
        df, x="Curricular units 1st sem (grade)", y="Curricular units 2nd sem (grade)",
        color=config.TARGET_COLUMN, opacity=0.5,
        color_discrete_map={"Dropout": "#C62828", "Enrolled": "#F9A825", "Graduate": "#2E7D32"},
    )
    st.plotly_chart(fig5, use_container_width=True)

    with st.expander("View raw data sample"):
        st.dataframe(df.head(50), use_container_width=True)

# ---------------------------------------------------------------------------
# Page: About
# ---------------------------------------------------------------------------
elif page == "About":
    st.title("About This Project")
    st.markdown(
        """
### Project
**AI-Powered Student Success Prediction & Intervention System** predicts whether a student
is likely to **Graduate**, remain **Enrolled**, or **Dropout**, using demographic,
socio-economic, and academic-performance data available at (or shortly after) enrollment.

### Dataset
- **Name:** *Predict Students' Dropout and Academic Success*
- **Source:** UCI Machine Learning Repository
- **Authors:** Valentim Realinho, Mónica Vieira Martins, Jorge Machado, Luís Baptista (2021)
- **DOI:** [10.24432/C5MC89](https://doi.org/10.24432/C5MC89)
- **Size:** 4,424 students · 36 raw features · 3-class target
- **Institution:** A Portuguese higher-education institution (multiple undergraduate degrees)

### Modeling Approach
1. Feature engineering (approval ratio, average grade, grade trend, socio-economic risk flags)
2. One-hot encoding of categorical fields + standard scaling of numeric fields
3. Comparison of **Logistic Regression, Random Forest, XGBoost, and LightGBM** on a validation split,
   with class-balanced sample weighting to handle the "Enrolled" minority class
4. Best model refit on train+validation, evaluated once on a held-out test set
5. Artifacts (model, preprocessor, label encoder, metrics, SHAP background sample) versioned under `models/`

### Explainability & Risk Scoring
- **SHAP** (`shap.Explainer`, auto-selecting TreeExplainer or LinearExplainer depending on the
  winning model) explains individual predictions and produces a global feature-importance view.
- Class probabilities are converted into a single **0-100 dropout risk score**
  (`score = 100 × (P(Dropout) + 0.35 × P(Enrolled))`), then bucketed into Low/Moderate/High risk tiers.

### AI-Generated Intervention Recommendations
Each flagged student's top SHAP risk factors are passed to Claude (via the Anthropic API) to draft
a short, specific intervention plan for the teacher/advisor. If no API key is configured, a
rule-based recommendation engine produces a comparable structured plan — the app is fully
functional either way.

### Intended Use & Limitations
This tool is designed to **support**, not replace, academic advisors. Predictions are based on
historical patterns in one institution's data and may not generalize perfectly to other schools,
countries, or student populations. It should be used as one input among many in advising decisions,
not as a sole determinant of any action taken toward a student.

### Repository
Built as an open, reproducible ML project: data pipeline, training script, evaluation, and this
interactive app all live in the same repository so results can be verified end-to-end.
        """
    )

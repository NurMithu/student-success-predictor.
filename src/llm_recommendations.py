"""
LLM-generated intervention recommendations for teachers/advisors.

Given a student's prediction, risk score, and top SHAP risk factors, this
module asks Claude to write a short, concrete, teacher-facing intervention
plan. If no API key is configured (e.g. running the public demo without
secrets set), it falls back to a rule-based recommendation engine so the
app's core functionality never depends on a paid API call.

To enable the LLM version, set an API key via either:
  - Streamlit secrets:  st.secrets["ANTHROPIC_API_KEY"]
  - Environment var:    export ANTHROPIC_API_KEY=sk-ant-...
"""

from __future__ import annotations

import os
from dataclasses import dataclass

MODEL_NAME = "claude-sonnet-4-6"


@dataclass
class StudentContext:
    student_label: str
    predicted_status: str
    risk_level: str
    risk_score: float
    top_risk_factors: list  # list of (friendly_feature_name, shap_value) tuples
    key_stats: dict  # e.g. {"approval_ratio": 0.4, "average_grade": 11.2, "tuition_up_to_date": "No"}


def _get_api_key() -> str | None:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    try:
        import streamlit as st

        return st.secrets.get("ANTHROPIC_API_KEY")
    except Exception:
        return None


def _build_prompt(ctx: StudentContext) -> str:
    factors_text = "\n".join(
        f"- {name}: SHAP contribution {value:+.3f} (higher = pushes dropout risk up)"
        for name, value in ctx.top_risk_factors
    )
    stats_text = "\n".join(f"- {k}: {v}" for k, v in ctx.key_stats.items())

    return f"""You are an academic advising assistant helping a teacher support a student.

Student: {ctx.student_label}
Model prediction: {ctx.predicted_status}
Risk tier: {ctx.risk_level} (dropout risk score: {ctx.risk_score:.0f}/100)

Top factors driving this student's risk score (from SHAP analysis):
{factors_text}

Key academic/financial stats:
{stats_text}

Write a short, concrete intervention plan for the teacher/advisor. Requirements:
- 3-5 bullet points, each one specific, actionable step (not generic advice like "monitor the student")
- Tie at least 2 recommendations directly to the specific risk factors listed above
- Keep a supportive, non-judgmental tone toward the student
- End with one suggested first conversation opener the advisor could use with the student
- Do not repeat the raw numbers back verbatim; translate them into plain language
- Keep the whole response under 180 words
"""


def rule_based_recommendation(ctx: StudentContext) -> str:
    """Deterministic fallback used when no LLM API key is configured."""
    lines = []
    factor_names = [name.lower() for name, _ in ctx.top_risk_factors]

    if ctx.risk_level == "Low Risk":
        return (
            "- This student's indicators look healthy — no urgent intervention needed.\n"
            "- A brief check-in at midterm can catch any late-emerging issues early.\n"
            "- Consider inviting them to a peer-mentoring program if available, to reinforce strong habits.\n\n"
            f"**Conversation opener:** \"Your first-semester results look solid — is there anything "
            f"you're finding challenging that I can help with before it becomes a bigger issue?\""
        )

    if any("approv" in f or "grade" in f for f in factor_names):
        lines.append(
            "- Schedule a one-on-one to review which specific course units are being missed or failed, "
            "and connect the student with tutoring or study-group resources for those courses."
        )
    if any("tuition" in f or "debtor" in f or "socioeconomic" in f for f in factor_names):
        lines.append(
            "- Refer the student to the financial aid office to check for available scholarships, "
            "payment plans, or emergency funding before financial stress compounds academic strain."
        )
    if any("enroll" in f or "unit" in f for f in factor_names):
        lines.append(
            "- Review the student's course load with them — an overload or a mismatch between "
            "enrolled and completed units often signals a need to adjust the schedule."
        )
    if any("age" in f for f in factor_names):
        lines.append(
            "- Check in on external commitments (work, family) that may be competing with study time, "
            "and flag flexible attendance or online resources if relevant."
        )
    if not lines:
        lines.append(
            "- Schedule a general advising check-in to understand what's driving the current trajectory "
            "and identify the single most useful next step together."
        )

    lines.append(
        "- Set a follow-up date (2-3 weeks out) to review whether the situation has improved."
    )

    opener = (
        "\"I noticed a few things in your record I wanted to check in on — how are things going for you "
        "this semester?\""
    )
    return "\n".join(lines) + f"\n\n**Conversation opener:** {opener}"


def generate_recommendation(ctx: StudentContext) -> tuple[str, str]:
    """
    Returns (recommendation_text, source) where source is 'llm' or 'rule_based'.
    Never raises — falls back to rule_based_recommendation on any error.
    """
    api_key = _get_api_key()
    if not api_key:
        return rule_based_recommendation(ctx), "rule_based"

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model=MODEL_NAME,
            max_tokens=400,
            messages=[{"role": "user", "content": _build_prompt(ctx)}],
        )
        text = "".join(block.text for block in message.content if block.type == "text")
        return text.strip(), "llm"
    except Exception as e:  # pragma: no cover - network/key errors, etc.
        fallback = rule_based_recommendation(ctx)
        return fallback + f"\n\n*(LLM recommendation unavailable: {e}. Showing rule-based fallback.)*", "rule_based"

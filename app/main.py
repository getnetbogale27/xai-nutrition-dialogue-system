"""Landing page for the XAI Nutrition Dialogue application."""

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ui_components import (
    initialize_state,
    inject_professional_theme,
    render_hero,
    render_wur_job_fit_portfolio,
)
from src.explainability.explanation_engine import (
    generate_counterfactual_explanation,
    generate_explanation,
)
from src.recommender.rules import UserProfile, generate_recommendation

st.set_page_config(page_title="Main", page_icon="🏠", layout="wide")
inject_professional_theme()
initialize_state()

render_hero(
    "XAI Nutrition Dialogue System",
    "Portfolio project tailored for the Wageningen DECIDE PhD on transparent, explainable AI for dietary behaviour change.",
)

render_wur_job_fit_portfolio()

st.info("Use the sidebar flow: **📋 Recommendation → 🔍 Explanations → 💬 Dialogue → 📘 User Guide**.")

st.sidebar.title("What-If Nutrition Simulator")

baseline_profile = st.session_state.profile or UserProfile(
    age=30,
    weight=70.0,
    activity_level="medium",
    dietary_preference="balanced",
    sugar_preference="low",
)

what_if_age = st.sidebar.slider("Age", min_value=10, max_value=80, value=int(baseline_profile.age))
what_if_weight = st.sidebar.slider(
    "Weight (kg)", min_value=30, max_value=150, value=int(round(baseline_profile.weight))
)
what_if_activity = st.sidebar.selectbox(
    "Activity Level", ["low", "medium", "high"], index=["low", "medium", "high"].index(baseline_profile.activity_level)
)
what_if_sugar = st.sidebar.selectbox(
    "Sugar Preference", ["low", "high"], index=["low", "high"].index(baseline_profile.sugar_preference)
)

simulate_clicked = st.sidebar.button("Simulate New Scenario", type="primary", use_container_width=True)

if simulate_clicked:
    original_profile = baseline_profile
    original_prediction = st.session_state.last_prediction or generate_recommendation(original_profile, mode="ml-based")
    original_explanation = st.session_state.last_explanation or generate_explanation(
        original_profile, original_prediction
    )

    counterfactual_profile = UserProfile(
        age=what_if_age,
        weight=float(what_if_weight),
        activity_level=what_if_activity,
        dietary_preference="balanced",
        sugar_preference=what_if_sugar,
    )
    new_prediction = generate_recommendation(counterfactual_profile, mode="ml-based")
    new_explanation = generate_explanation(counterfactual_profile, new_prediction)

    st.session_state.what_if_result = {
        "original_profile": original_profile,
        "original_prediction": original_prediction,
        "original_explanation": original_explanation,
        "new_profile": counterfactual_profile,
        "new_prediction": new_prediction,
        "new_explanation": new_explanation,
        "counterfactual_explanation": generate_counterfactual_explanation(original_profile, counterfactual_profile),
    }

if "what_if_result" in st.session_state:
    data = st.session_state.what_if_result
    st.divider()
    st.subheader("Current Prediction")
    st.success(data["new_prediction"].get("diet_label", "n/a").replace("_", " ").title())

    st.subheader("Counterfactual Comparison")
    before_col, after_col = st.columns(2)
    with before_col:
        st.markdown("#### Before (Original Input)")
        st.json(
            {
                "age": data["original_profile"].age,
                "weight": data["original_profile"].weight,
                "activity_level": data["original_profile"].activity_level,
                "sugar_preference": data["original_profile"].sugar_preference,
                "prediction": data["original_prediction"].get("diet_label"),
            }
        )
    with after_col:
        st.markdown("#### After (Modified Scenario)")
        st.json(
            {
                "age": data["new_profile"].age,
                "weight": data["new_profile"].weight,
                "activity_level": data["new_profile"].activity_level,
                "sugar_preference": data["new_profile"].sugar_preference,
                "prediction": data["new_prediction"].get("diet_label"),
            }
        )

    st.markdown("#### What changed")
    for change in data["counterfactual_explanation"]["key_differences"]:
        st.write(f"- {change}")

    st.markdown("#### How prediction changed")
    st.write(
        f"Recommendation: **{data['original_prediction'].get('diet_label', 'n/a')}** → "
        f"**{data['new_prediction'].get('diet_label', 'n/a')}**"
    )

    st.markdown("#### Why it changed")
    st.write(data["counterfactual_explanation"]["human_readable"])
    st.caption(
        "Updated SHAP rationale: "
        + data["new_explanation"].get("natural_language", "No SHAP-based explanation available.")
    )

    st.markdown("## Probabilistic Reasoning (Bayesian View)")
    bayesian_probs = data["new_explanation"].get("trace", {}).get("bayesian_probabilities", {})
    if bayesian_probs:
        bayesian_df = (
            pd.DataFrame(
                [{"Diet": diet.replace("_", " ").title(), "Probability": prob} for diet, prob in bayesian_probs.items()]
            )
            .sort_values("Probability", ascending=False)
            .reset_index(drop=True)
        )
        st.dataframe(
            bayesian_df.assign(Probability=bayesian_df["Probability"].map(lambda value: f"{value:.1%}")),
            use_container_width=True,
            hide_index=True,
        )
        most_likely = max(bayesian_probs.items(), key=lambda item: item[1])[0]
        st.success(f"Most likely Bayesian diet: **{most_likely.replace('_', ' ').title()}**")
        st.write(data["new_explanation"].get("trace", {}).get("bayesian_explanation", ""))
    else:
        st.info("Bayesian probabilities are not available for this scenario.")

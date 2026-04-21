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
)
from src.explainability import explanation_engine
from src.recommender.rules import UserProfile, generate_recommendation

st.set_page_config(page_title="Explainable Reasoning Engine", page_icon="🧠", layout="wide")
inject_professional_theme()
initialize_state()

baseline_profile = st.session_state.profile or UserProfile(
    age=30,
    weight=70.0,
    activity_level="medium",
    dietary_preference="balanced",
    sugar_preference="low",
)

if "input_age" not in st.session_state:
    st.session_state.input_age = int(baseline_profile.age)
if "input_weight" not in st.session_state:
    st.session_state.input_weight = int(round(baseline_profile.weight))
if "input_activity" not in st.session_state:
    st.session_state.input_activity = baseline_profile.activity_level
if "input_sugar" not in st.session_state:
    st.session_state.input_sugar = baseline_profile.sugar_preference

if "tab1_age" not in st.session_state:
    st.session_state.tab1_age = st.session_state.input_age
if "tab1_weight" not in st.session_state:
    st.session_state.tab1_weight = st.session_state.input_weight
if "tab1_activity" not in st.session_state:
    st.session_state.tab1_activity = st.session_state.input_activity
if "tab1_sugar" not in st.session_state:
    st.session_state.tab1_sugar = st.session_state.input_sugar
if "tab2_age" not in st.session_state:
    st.session_state.tab2_age = st.session_state.input_age
if "tab2_weight" not in st.session_state:
    st.session_state.tab2_weight = st.session_state.input_weight
if "tab2_activity" not in st.session_state:
    st.session_state.tab2_activity = st.session_state.input_activity
if "tab2_sugar" not in st.session_state:
    st.session_state.tab2_sugar = st.session_state.input_sugar


def _sync_inputs(source_tab: str) -> None:
    if source_tab == "tab1":
        st.session_state.input_age = st.session_state.tab1_age
        st.session_state.input_weight = st.session_state.tab1_weight
        st.session_state.input_activity = st.session_state.tab1_activity
        st.session_state.input_sugar = st.session_state.tab1_sugar
    else:
        st.session_state.input_age = st.session_state.tab2_age
        st.session_state.input_weight = st.session_state.tab2_weight
        st.session_state.input_activity = st.session_state.tab2_activity
        st.session_state.input_sugar = st.session_state.tab2_sugar

    st.session_state.tab1_age = st.session_state.input_age
    st.session_state.tab1_weight = st.session_state.input_weight
    st.session_state.tab1_activity = st.session_state.input_activity
    st.session_state.tab1_sugar = st.session_state.input_sugar
    st.session_state.tab2_age = st.session_state.input_age
    st.session_state.tab2_weight = st.session_state.input_weight
    st.session_state.tab2_activity = st.session_state.input_activity
    st.session_state.tab2_sugar = st.session_state.input_sugar


st.sidebar.subheader("Explainable Reasoning Engine")
if "show_reasoning_menu" not in st.session_state:
    st.session_state.show_reasoning_menu = True
if "active_page" not in st.session_state:
    st.session_state.active_page = "What-If Calculator"


def _toggle_reasoning_menu() -> None:
    st.session_state.show_reasoning_menu = not st.session_state.show_reasoning_menu


def _set_active_page(page_name: str) -> None:
    st.session_state.active_page = page_name


parent_label = "▼ Explainable Reasoning Engine" if st.session_state.show_reasoning_menu else "▶ Explainable Reasoning Engine"
st.sidebar.button(parent_label, use_container_width=True, on_click=_toggle_reasoning_menu)

if st.session_state.show_reasoning_menu:
    _, nested_col = st.sidebar.columns([0.12, 0.88])
    nested_col.button(
        "🧪 What-If Calculator",
        key="page_what_if",
        use_container_width=True,
        type="primary" if st.session_state.active_page == "What-If Calculator" else "secondary",
        on_click=_set_active_page,
        args=("What-If Calculator",),
    )
    _, nested_col = st.sidebar.columns([0.12, 0.88])
    nested_col.button(
        "📊 Probabilistic Reasoning",
        key="page_probabilistic",
        use_container_width=True,
        type="primary" if st.session_state.active_page == "Probabilistic Reasoning" else "secondary",
        on_click=_set_active_page,
        args=("Probabilistic Reasoning",),
    )

selected_reasoning_tool = st.session_state.active_page

if selected_reasoning_tool == "What-If Calculator":
    st.sidebar.slider("Age", min_value=10, max_value=80, key="tab1_age", on_change=_sync_inputs, args=("tab1",))
    st.sidebar.slider("Weight (kg)", min_value=30, max_value=150, key="tab1_weight", on_change=_sync_inputs, args=("tab1",))
    st.sidebar.selectbox(
        "Activity Level",
        ["low", "medium", "high"],
        key="tab1_activity",
        on_change=_sync_inputs,
        args=("tab1",),
    )
    st.sidebar.selectbox("Sugar Preference", ["low", "high"], key="tab1_sugar", on_change=_sync_inputs, args=("tab1",))

    if st.sidebar.button("Run What-If Simulation", type="primary", use_container_width=True):
        original_profile = baseline_profile
        original_prediction = st.session_state.last_prediction or generate_recommendation(original_profile, mode="ml-based")

        scenario_profile = UserProfile(
            age=st.session_state.input_age,
            weight=float(st.session_state.input_weight),
            activity_level=st.session_state.input_activity,
            dietary_preference="balanced",
            sugar_preference=st.session_state.input_sugar,
        )
        scenario_prediction = generate_recommendation(scenario_profile, mode="ml-based")
        scenario_explanation = explanation_engine.generate_explanation(scenario_profile, scenario_prediction)
        counterfactual_fn = getattr(explanation_engine, "generate_counterfactual_explanation", None)
        if callable(counterfactual_fn):
            counterfactual = counterfactual_fn(original_profile, scenario_profile)
        else:
            counterfactual = {
                "key_differences": ["Counterfactual comparison unavailable in this runtime."],
                "expected_effect": ["Upgrade src.explainability.explanation_engine to include generate_counterfactual_explanation()."],
                "human_readable": "Counterfactual explanation helper is unavailable; showing recommendation comparison only.",
            }

        st.session_state.reasoning_result = {
            "mode": "what_if",
            "profile": scenario_profile,
            "prediction": scenario_prediction,
            "explanation": scenario_explanation,
            "comparison": {
                "original_profile": original_profile,
                "original_prediction": original_prediction,
                "counterfactual_explanation": counterfactual,
            },
        }

else:
    st.sidebar.slider("Age", min_value=10, max_value=80, key="tab2_age", on_change=_sync_inputs, args=("tab2",))
    st.sidebar.slider("Weight (kg)", min_value=30, max_value=150, key="tab2_weight", on_change=_sync_inputs, args=("tab2",))
    st.sidebar.selectbox(
        "Activity Level",
        ["low", "medium", "high"],
        key="tab2_activity",
        on_change=_sync_inputs,
        args=("tab2",),
    )
    st.sidebar.selectbox("Sugar Preference", ["low", "high"], key="tab2_sugar", on_change=_sync_inputs, args=("tab2",))

    if st.sidebar.button("Compute Probabilities", use_container_width=True):
        scenario_profile = UserProfile(
            age=st.session_state.input_age,
            weight=float(st.session_state.input_weight),
            activity_level=st.session_state.input_activity,
            dietary_preference="balanced",
            sugar_preference=st.session_state.input_sugar,
        )
        scenario_prediction = generate_recommendation(scenario_profile, mode="ml-based")
        scenario_explanation = explanation_engine.generate_explanation(scenario_profile, scenario_prediction)
        st.session_state.reasoning_result = {
            "mode": "bayesian",
            "profile": scenario_profile,
            "prediction": scenario_prediction,
            "explanation": scenario_explanation,
            "comparison": st.session_state.get("reasoning_result", {}).get("comparison"),
        }

st.title("Reasoning Results")
st.caption(f"Selected tool: **{selected_reasoning_tool}**")
current = st.session_state.get("reasoning_result")
if not current:
    if selected_reasoning_tool == "What-If Calculator":
        st.info("Use the sidebar to run a What-If simulation.")
    else:
        st.info("Use the sidebar to compute Bayesian probabilities.")
else:
    if selected_reasoning_tool == "What-If Calculator":
        st.subheader("What-If Calculator")
        if current.get("mode") == "what_if":
            st.success(
                f"Updated ML prediction: **{current['prediction'].get('diet_label', 'n/a').replace('_', ' ').title()}**"
            )
            st.caption("SHAP + Bayesian summary")
            st.write(current["explanation"].get("natural_language", "No explanation available."))
        else:
            st.info("Run the What-If simulation from the sidebar to view results on this page.")

    if selected_reasoning_tool == "Probabilistic Reasoning":
        st.subheader("Probabilistic Reasoning")
        if current.get("mode") == "bayesian":
            bayesian_probs = current.get("explanation", {}).get("trace", {}).get("bayesian_probabilities", {})
            if bayesian_probs:
                ordered = ["balanced", "high_protein", "low_carb", "low_calorie"]
                bayesian_df = pd.DataFrame(
                    [{"Diet Class": diet, "Probability": bayesian_probs.get(diet, 0.0)} for diet in ordered]
                )
                top_diet = max(bayesian_probs.items(), key=lambda item: item[1])[0]
                bayesian_df["Diet Class"] = bayesian_df["Diet Class"].map(
                    lambda diet: f"⭐ {diet}" if diet == top_diet else diet
                )
                st.dataframe(
                    bayesian_df.assign(Probability=bayesian_df["Probability"].map(lambda prob: f"{prob:.1%}")),
                    use_container_width=True,
                    hide_index=True,
                )
                st.success(f"Highest probability class: **{top_diet.replace('_', ' ').title()}**")
                st.write(current.get("explanation", {}).get("trace", {}).get("bayesian_explanation", ""))
        else:
            st.info("Run Probabilistic Reasoning from the sidebar to view Bayesian results on this page.")

    if current.get("mode") == "what_if" and selected_reasoning_tool == "What-If Calculator":
        st.subheader("Final Recommendation")
        st.success(current["prediction"].get("diet_label", "n/a").replace("_", " ").title())

        st.subheader("Explanation Summary")
        st.write(current["explanation"].get("natural_language", "No explanation available."))

        comparison = current.get("comparison")
        if comparison:
            st.subheader("Comparison")
            st.write(
                f"**Before:** {comparison['original_prediction'].get('diet_label', 'n/a').replace('_', ' ').title()} | "
                f"**After:** {current['prediction'].get('diet_label', 'n/a').replace('_', ' ').title()}"
            )
            st.write(comparison["counterfactual_explanation"].get("human_readable", ""))

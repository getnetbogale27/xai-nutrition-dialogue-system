"""Single-file Streamlit app shell with custom sidebar navigation."""

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ui_components import (  # noqa: E402
    initialize_state,
    inject_professional_theme,
    profile_to_table,
    render_author_details,
    render_chat_panel,
    render_evaluation_strip,
    render_explanation_panel,
    render_hero,
    render_kpi_card,
    render_profile_controls,
    render_recommendation_summary,
    render_wur_job_fit_portfolio,
    run_recommendation_pipeline,
)
from src.explainability import explanation_engine  # noqa: E402
from src.recommender.rules import UserProfile, generate_recommendation  # noqa: E402

st.set_page_config(page_title="XAI Nutrition Dialogue System", page_icon="🧠", layout="wide")
inject_professional_theme()
initialize_state()

PAGES = [
    "Recommendation Studio",
    "Explainability Console",
    "Explainable Reasoning Engine",
    "Interactive Dialogue",
    "User Guide",
    "Author Details",
]
PAGE_ICONS = {
    "Recommendation Studio": "📋",
    "Explainability Console": "🔍",
    "Explainable Reasoning Engine": "🧠",
    "Interactive Dialogue": "💬",
    "User Guide": "📘",
    "Author Details": "👤",
}

if "page" not in st.session_state or st.session_state.page not in PAGES:
    st.session_state.page = "Recommendation Studio"


def _set_page(page_name: str) -> None:
    st.session_state.page = page_name


def _init_reasoning_state() -> None:
    baseline_profile = st.session_state.profile or UserProfile(
        age=30,
        weight=70.0,
        activity_level="medium",
        dietary_preference="balanced",
        sugar_preference="low",
    )
    defaults = {
        "reasoning_age": int(baseline_profile.age),
        "reasoning_weight": int(round(baseline_profile.weight)),
        "reasoning_activity": baseline_profile.activity_level,
        "reasoning_sugar": baseline_profile.sugar_preference,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def show_recommendation() -> None:
    """Recommendation page content."""
    render_hero(
        "Recommendation Studio",
        "Design evidence-based dietary pathways with explainable ML and clinical rule engines in one unified command center.",
    )

    rec = st.session_state.last_prediction
    confidence = max(rec.get("probabilities", {}).values(), default=0.0) if rec else 0.0
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_kpi_card("Pipeline Status", "Ready" if rec else "Waiting")
    with k2:
        render_kpi_card("Active Engine", rec.get("mode", "not selected").replace("-", " ").title() if rec else "—")
    with k3:
        render_kpi_card("Top Confidence", f"{confidence:.1%}" if rec else "—")
    with k4:
        render_kpi_card("Evaluated Profiles", "1" if st.session_state.profile else "0")

    st.markdown('<h2 class="section-header">Live Snapshot</h2>', unsafe_allow_html=True)
    overview_left, overview_right = st.columns([1.1, 1])
    with overview_left:
        st.markdown('<div class="panel-card"><div class="panel-title">Current Profile Snapshot</div>', unsafe_allow_html=True)
        profile_df = profile_to_table()
        if profile_df.empty:
            st.info("No active patient profile yet. Configure the patient profile to activate real-time tracking.")
        else:
            st.dataframe(profile_df, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with overview_right:
        st.markdown(
            '<div class="panel-card"><div class="panel-title">Current Recommendation Snapshot</div>',
            unsafe_allow_html=True,
        )
        if rec:
            st.markdown(
                f"""
                <span class="stat-chip">Diet: {rec.get('diet_label', 'n/a').replace('_', ' ').title()}</span>
                <span class="stat-chip">Engine: {rec.get('mode', 'unknown').replace('-', ' ').title()}</span>
                <span class="stat-chip">Confidence: {confidence:.1%}</span>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("#### Clinical Guidance")
            st.success(rec.get("recommendation_text", ""))
        else:
            st.info("No recommendation generated yet. Complete the profile and run the recommendation engine.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.divider()
    st.markdown('<h2 class="section-header">Recommendation Workspace</h2>', unsafe_allow_html=True)
    left, right = st.columns([1.05, 1.15])
    with left:
        profile, mode, generate_clicked = render_profile_controls("recommendation_page")
        if generate_clicked:
            run_recommendation_pipeline(profile, mode)
            st.success("Recommendation pipeline completed successfully. Snapshot and evaluation panels updated.")

    with right:
        render_recommendation_summary()

    st.divider()
    st.markdown('<h2 class="section-header">Evaluation Overview</h2>', unsafe_allow_html=True)
    render_evaluation_strip()


def show_explanations() -> None:
    """Explanations page content."""
    render_hero(
        "Explainability Console",
        "Inspect natural-language rationale, reasoning trace, and feature-level influence.",
    )
    render_explanation_panel()


def show_what_if() -> None:
    """Placeholder implementation for the What-If calculator UI."""
    with st.container(border=True):
        st.markdown("#### What-If Calculator")
        c1, c2 = st.columns(2)
        with c1:
            st.number_input("Age", min_value=10, max_value=80, key="reasoning_age", step=1)
        with c2:
            st.number_input("Weight (kg)", min_value=30, max_value=150, key="reasoning_weight", step=1)
        c3, c4 = st.columns(2)
        with c3:
            st.selectbox("Activity Level", ["low", "medium", "high"], key="reasoning_activity")
        with c4:
            st.selectbox("Sugar Preference", ["low", "high"], key="reasoning_sugar")

        if st.button("Run What-If Simulation", type="primary", use_container_width=True):
            original_profile = st.session_state.profile or UserProfile(
                age=30,
                weight=70.0,
                activity_level="medium",
                dietary_preference="balanced",
                sugar_preference="low",
            )
            original_prediction = st.session_state.last_prediction or generate_recommendation(original_profile, mode="ml-based")

            scenario_profile = UserProfile(
                age=st.session_state.reasoning_age,
                weight=float(st.session_state.reasoning_weight),
                activity_level=st.session_state.reasoning_activity,
                dietary_preference="balanced",
                sugar_preference=st.session_state.reasoning_sugar,
            )
            scenario_prediction = generate_recommendation(scenario_profile, mode="ml-based")
            scenario_explanation = explanation_engine.generate_explanation(scenario_profile, scenario_prediction)
            counterfactual_fn = getattr(explanation_engine, "generate_counterfactual_explanation", None)
            counterfactual = (
                counterfactual_fn(original_profile, scenario_profile)
                if callable(counterfactual_fn)
                else {
                    "human_readable": "Counterfactual helper unavailable; showing recommendation comparison only.",
                }
            )
            st.session_state.reasoning_result = {
                "mode": "what_if",
                "profile": scenario_profile,
                "prediction": scenario_prediction,
                "explanation": scenario_explanation,
                "comparison": {
                    "original_prediction": original_prediction,
                    "counterfactual_explanation": counterfactual,
                },
            }

    current = st.session_state.get("reasoning_result")
    if current and current.get("mode") == "what_if":
        st.success(f"Updated ML prediction: **{current['prediction'].get('diet_label', 'n/a').replace('_', ' ').title()}**")
        st.write(current.get("explanation", {}).get("natural_language", "No explanation available."))


def show_probabilistic() -> None:
    """Placeholder implementation for probabilistic reasoning UI."""
    with st.container(border=True):
        st.markdown("#### Probabilistic Reasoning")
        c1, c2 = st.columns(2)
        with c1:
            st.number_input("Age", min_value=10, max_value=80, key="reasoning_age_prob", value=st.session_state.reasoning_age, step=1)
        with c2:
            st.number_input(
                "Weight (kg)",
                min_value=30,
                max_value=150,
                key="reasoning_weight_prob",
                value=st.session_state.reasoning_weight,
                step=1,
            )
        c3, c4 = st.columns(2)
        with c3:
            st.selectbox(
                "Activity Level",
                ["low", "medium", "high"],
                key="reasoning_activity_prob",
                index=["low", "medium", "high"].index(st.session_state.reasoning_activity),
            )
        with c4:
            st.selectbox(
                "Sugar Preference",
                ["low", "high"],
                key="reasoning_sugar_prob",
                index=["low", "high"].index(st.session_state.reasoning_sugar),
            )

        if st.button("Compute Probabilities", type="primary", use_container_width=True):
            scenario_profile = UserProfile(
                age=st.session_state.reasoning_age_prob,
                weight=float(st.session_state.reasoning_weight_prob),
                activity_level=st.session_state.reasoning_activity_prob,
                dietary_preference="balanced",
                sugar_preference=st.session_state.reasoning_sugar_prob,
            )
            scenario_prediction = generate_recommendation(scenario_profile, mode="ml-based")
            scenario_explanation = explanation_engine.generate_explanation(scenario_profile, scenario_prediction)
            st.session_state.reasoning_result = {
                "mode": "bayesian",
                "profile": scenario_profile,
                "prediction": scenario_prediction,
                "explanation": scenario_explanation,
            }

    current = st.session_state.get("reasoning_result")
    if current and current.get("mode") == "bayesian":
        bayesian_probs = current.get("explanation", {}).get("trace", {}).get("bayesian_probabilities", {})
        if bayesian_probs:
            bayesian_df = pd.DataFrame(
                [{"Diet Class": diet.replace("_", " ").title(), "Probability": f"{prob:.1%}"} for diet, prob in bayesian_probs.items()]
            )
            st.dataframe(bayesian_df, use_container_width=True, hide_index=True)


def show_reasoning_engine() -> None:
    """Dedicated page for explainable reasoning with top-level tabs."""
    _init_reasoning_state()
    st.title("Explainable Reasoning Engine")
    st.caption("Evaluate scenario changes and Bayesian probabilities without sidebar tool controls.")

    what_if_tab, probabilistic_tab = st.tabs(["What-If Calculator", "Probabilistic Reasoning"])
    with what_if_tab:
        show_what_if()
    with probabilistic_tab:
        show_probabilistic()


def show_dialogue() -> None:
    render_hero(
        "Interactive Dialogue",
        "Conduct context-aware follow-up discussion to improve trust and understanding.",
    )
    render_chat_panel()


def show_user_guide() -> None:
    render_hero(
        "WUR DECIDE Portfolio Guide",
        "How this app demonstrates fit for the PhD position: xAI to support better dietary decisions.",
    )
    render_wur_job_fit_portfolio()

    st.markdown("### What this system provides")
    st.markdown(
        """
        - **Personalized nutrition recommendation** using ML-based and rule-based logic.
        - **Transparent explanation layer** with rationale, feature contributions, and trace.
        - **Interactive dialogue support** for follow-up, trust calibration, and what-if analysis.
        - **Evaluation views** connected to consistency, stability, confidence, and user-feedback simulation.
        """
    )


def show_author_details() -> None:
    render_hero(
        "Author Details",
        "Professional profile, research focus, and application-ready positioning.",
    )
    render_author_details()


for page in PAGES:
    st.sidebar.button(
        f"{PAGE_ICONS.get(page, '•')}  {page}",
        key=f"nav_{page}",
        use_container_width=True,
        type="primary" if st.session_state.page == page else "secondary",
        on_click=_set_page,
        args=(page,),
    )

if st.session_state.page == "Recommendation Studio":
    show_recommendation()
elif st.session_state.page == "Explainability Console":
    show_explanations()
elif st.session_state.page == "Explainable Reasoning Engine":
    show_reasoning_engine()
elif st.session_state.page == "Interactive Dialogue":
    show_dialogue()
elif st.session_state.page == "User Guide":
    show_user_guide()
elif st.session_state.page == "Author Details":
    show_author_details()

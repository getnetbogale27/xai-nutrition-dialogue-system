"""Reusable UI helpers for the Streamlit nutrition app."""

from __future__ import annotations

from dataclasses import asdict

import pandas as pd
import streamlit as st

from src.dialogue.chatbot import NutritionChatbot
from src.evaluation.metrics import (
    explanation_consistency_score,
    feature_importance_stability,
    model_confidence_distribution,
    rule_vs_ml_comparison,
)
from src.evaluation.user_study_simulation import simulate_user_feedback
from src.explainability.explanation_engine import generate_explanation
from src.recommender.ml_model import train_model
from src.recommender.rules import UserProfile, generate_recommendation


def inject_professional_theme() -> None:
    """Apply a polished visual style for all pages."""
    st.markdown(
        """
        <style>
            .block-container {padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1200px;}
            .hero-card {
                background: linear-gradient(120deg, #0f172a 0%, #1e293b 60%, #334155 100%);
                color: #f8fafc;
                border-radius: 16px;
                padding: 1rem 1.2rem;
                margin-bottom: 1rem;
                border: 1px solid rgba(148, 163, 184, 0.35);
            }
            .metric-card {
                background: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 14px;
                padding: 0.8rem 1rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def initialize_state() -> None:
    """Initialize shared session values used across pages."""
    defaults = {
        "profile": None,
        "last_prediction": {},
        "last_explanation": {},
        "last_shap_output": [],
        "evaluation_feedback": {},
        "chat_history": [],
        "chatbot": NutritionChatbot(),
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_hero(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="hero-card">
            <h2 style="margin-bottom:0.35rem;">{title}</h2>
            <p style="margin:0; opacity:0.9;">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_author_details() -> None:
    """Render author details in the main content area."""
    author_query = "Getnet B. Begashaw"
    with st.container(border=True):
        st.markdown("### 👤 Getnet B. Begashaw")
        st.markdown("**PhD in Statistics**")
        st.markdown("📧 getnetbogale145@gmail.com")

        st.markdown("#### Professional Links")
        c1, c2 = st.columns(2)
        with c1:
            st.link_button(
                "LinkedIn",
                f"https://www.linkedin.com/search/results/all/?keywords={author_query.replace(' ', '%20')}",
                use_container_width=True,
            )
            st.link_button(
                "GitHub",
                f"https://github.com/search?q={author_query.replace(' ', '+')}",
                use_container_width=True,
            )
        with c2:
            st.link_button(
                "Google Scholar",
                f"https://scholar.google.com/scholar?q={author_query.replace(' ', '+')}",
                use_container_width=True,
            )
            st.link_button(
                "Causal AI Work",
                f"https://www.google.com/search?q={author_query.replace(' ', '+')}+causal+AI",
                use_container_width=True,
            )


def render_profile_controls(key_prefix: str = "main") -> tuple[UserProfile, str, bool]:
    """Render profile controls and return profile, mode and generate click state."""
    with st.container(border=True):
        st.subheader("Patient Profile")
        c1, c2 = st.columns(2)
        age = c1.number_input("Age", min_value=1, max_value=100, value=30, key=f"{key_prefix}_age")
        weight = c2.number_input(
            "Weight (kg)", min_value=20.0, max_value=250.0, value=70.0, key=f"{key_prefix}_weight"
        )

        c3, c4 = st.columns(2)
        activity = c3.selectbox(
            "Activity Level", ["low", "medium", "high"], key=f"{key_prefix}_activity"
        )
        sugar = c4.selectbox("Sugar Preference", ["low", "high"], key=f"{key_prefix}_sugar")

        model_choice = st.radio(
            "Recommendation Engine",
            ["ML (Logistic Regression)", "Rule-based"],
            horizontal=True,
            key=f"{key_prefix}_model",
        )
        mode = "ml-based" if model_choice.startswith("ML") else "rule-based"

        c5, c6 = st.columns([1, 2])
        if c5.button("Train ML Model", key=f"{key_prefix}_train"):
            stats = train_model()
            st.success(f"Model trained (accuracy={stats['accuracy']:.2f}, rows={stats['n_rows']}).")

        generate_clicked = c6.button(
            "Generate Personalized Recommendation",
            type="primary",
            use_container_width=True,
            key=f"{key_prefix}_generate",
        )

    profile = UserProfile(
        age=int(age),
        weight=float(weight),
        activity_level=activity,
        dietary_preference="balanced",
        sugar_preference=sugar,
    )
    return profile, mode, generate_clicked


def run_recommendation_pipeline(profile: UserProfile, mode: str) -> None:
    recommendation = generate_recommendation(profile, mode=mode)
    explanation = generate_explanation(profile, recommendation)

    st.session_state.profile = profile
    st.session_state.last_prediction = recommendation
    st.session_state.last_explanation = explanation
    st.session_state.last_shap_output = explanation.get("trace", {}).get("contributions", [])
    st.session_state.evaluation_feedback = simulate_user_feedback(explanation, recommendation)


def render_recommendation_summary() -> None:
    rec = st.session_state.last_prediction
    if not rec:
        st.info("No recommendation yet. Complete the profile form and generate one.")
        return

    confidence = max(rec.get("probabilities", {}).values(), default=0.0)
    a, b, c = st.columns(3)
    a.metric("Diet Label", rec.get("diet_label", "n/a").replace("_", " ").title())
    b.metric("Engine", rec.get("mode", "unknown").replace("-", " ").title())
    c.metric("Confidence", f"{confidence:.1%}" if rec.get("mode") == "ml-based" else "Rule-based")

    st.markdown("### Recommendation")
    st.success(rec.get("recommendation_text", "No recommendation text available."))


def render_explanation_panel() -> None:
    exp = st.session_state.last_explanation
    if not exp:
        st.info("Generate a recommendation first to inspect explanation details.")
        return

    st.markdown("### Natural-language rationale")
    st.write(exp.get("natural_language", "No explanation available."))

    trace = exp.get("trace", {})
    st.markdown("### Reasoning trace")
    if trace.get("human_readable"):
        for idx, step in enumerate(trace["human_readable"], start=1):
            st.write(f"{idx}. {step}")
    else:
        for idx, item in enumerate(trace.get("contributions", []), start=1):
            detail = item.get("reason") or f"{item.get('feature')} contribution={item.get('contribution', 0):+.2f}"
            st.write(f"{idx}. {detail}")

    shap_rows = st.session_state.last_shap_output
    if shap_rows:
        st.markdown("### Feature contributions")
        shap_df = pd.DataFrame(shap_rows)
        if "contribution" in shap_df.columns:
            st.bar_chart(shap_df.set_index("feature")["contribution"])
        st.dataframe(shap_df, use_container_width=True)


def render_evaluation_strip() -> None:
    rec = st.session_state.last_prediction
    exp = st.session_state.last_explanation
    if not rec or not exp:
        st.caption("Evaluation metrics will appear after the first recommendation.")
        return

    feedback = st.session_state.evaluation_feedback
    consistency = explanation_consistency_score(exp)
    stability = feature_importance_stability(st.session_state.last_shap_output)
    confidence = model_confidence_distribution(rec.get("probabilities", {}))

    ml_feedback = simulate_user_feedback(exp, {**rec, "mode": "ml-based"})
    rule_feedback = simulate_user_feedback(exp, {**rec, "mode": "rule-based"})
    comparison = rule_vs_ml_comparison(ml_feedback, rule_feedback)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Trust", f"{feedback.get('trust_score', 0.0):.2f}")
    c2.metric("Explanation Quality", f"{feedback.get('explanation_quality', 0.0):.2f}")
    c3.metric("Stability", f"{stability:.2f}")
    c4.metric("Max Confidence", f"{confidence['max_confidence']:.2f}")

    st.caption(
        f"Consistency={consistency:.2f} | ML average={comparison['ml_average']:.2f} | "
        f"Rule average={comparison['rule_average']:.2f}"
    )


def render_chat_panel() -> None:
    st.markdown("### Clinical Q&A Dialogue")
    query = st.text_input(
        "Ask follow-up questions",
        placeholder="Why was this recommended? What changes if sugar is reduced? Explain in simple terms.",
    )

    if st.button("Ask Assistant", use_container_width=True):
        if st.session_state.profile and st.session_state.last_prediction:
            response = st.session_state.chatbot.respond(
                question=query,
                profile=st.session_state.profile,
                recommendation=st.session_state.last_prediction,
                explanation=st.session_state.last_explanation,
                shap_summary=st.session_state.last_shap_output,
            )
            st.session_state.chat_history.append((query, response))
            st.session_state.chat_history = st.session_state.chat_history[-3:]
        else:
            st.warning("Generate a recommendation first.")

    for question, answer in st.session_state.chat_history:
        st.markdown(f"**You:** {question}")
        st.markdown(f"**Assistant:** {answer}")


def profile_to_table() -> pd.DataFrame:
    profile = st.session_state.profile
    if not profile:
        return pd.DataFrame()
    return pd.DataFrame([asdict(profile)])

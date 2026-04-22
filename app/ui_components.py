"""Reusable UI helpers for the Streamlit nutrition app."""

from __future__ import annotations

import inspect
from dataclasses import asdict
from html import escape
import re

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


def show_ai_disclaimer() -> None:
    """Display a concise AI safety disclaimer for user-facing outputs."""
    st.info(
        "⚠️ I'm an AI system and may occasionally make mistakes. "
        "Please verify important or health-related information with a qualified professional."
    )


def inject_professional_theme() -> None:
    """Apply a polished visual style for all pages."""
    st.markdown(
        """
        <style>
            .block-container {padding-top: 1.2rem; padding-bottom: 2.4rem; max-width: 1200px;}
            .st-emotion-cache-1lads1q {justify-content: unset !important;}
            .hero-card {
                background: linear-gradient(120deg, #0f172a 0%, #1e293b 60%, #334155 100%);
                color: #f8fafc;
                border-radius: 20px;
                padding: 1.35rem 1.5rem;
                margin-bottom: 1.1rem;
                border: 1px solid rgba(148, 163, 184, 0.35);
            }
            .metric-card {
                background: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 14px;
                padding: 0.8rem 1rem;
            }
            .section-header {
                margin: 0.2rem 0 0.85rem;
                font-size: 2rem;
                font-weight: 700;
                color: #111827;
                letter-spacing: -0.02em;
            }
            .panel-card {
                border: 1px solid #e2e8f0;
                border-radius: 18px;
                padding: 1.05rem 1.15rem;
                background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
                box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
                min-height: 0;
            }
            .explain-layout {
                display: grid;
                grid-template-columns: 2fr 1fr;
                gap: 1rem;
                margin-bottom: 0.8rem;
            }
            .explain-title {
                margin: 0 0 0.35rem;
                font-size: 1.05rem;
                font-weight: 650;
                color: #0f172a;
            }
            .explain-subtitle {
                margin: 0 0 0.75rem;
                color: #475569;
                font-size: 0.92rem;
            }
            .rationale-copy {
                margin: 0;
                color: #1f2937;
                line-height: 1.55;
                font-size: 1rem;
            }
            .impact-list {
                margin: 0.2rem 0 0;
                padding: 0;
                list-style: none;
            }
            .impact-list li {
                display: flex;
                align-items: center;
                justify-content: space-between;
                margin-bottom: 0.45rem;
                padding: 0.45rem 0.55rem;
                border: 1px solid #e2e8f0;
                border-radius: 10px;
                background: #ffffff;
                font-size: 0.9rem;
                color: #1e293b;
            }
            .impact-pill {
                border-radius: 999px;
                padding: 0.12rem 0.5rem;
                font-size: 0.75rem;
                font-weight: 600;
            }
            .impact-positive {
                background: #dcfce7;
                color: #166534;
                border: 1px solid #bbf7d0;
            }
            .impact-negative {
                background: #fee2e2;
                color: #991b1b;
                border: 1px solid #fecaca;
            }
            .explain-kpi-grid {
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 0.7rem;
                margin: 0.25rem 0 1rem;
            }
            .trace-card {
                margin-top: 0.25rem;
                border: 1px solid #e2e8f0;
                border-radius: 14px;
                padding: 0.85rem 0.9rem;
                background: #ffffff;
            }
            .trace-step {
                display: grid;
                grid-template-columns: 30px 1fr;
                gap: 0.6rem;
                align-items: start;
                padding: 0.4rem 0;
                border-bottom: 1px dashed #e2e8f0;
            }
            .trace-step:last-child {
                border-bottom: none;
                padding-bottom: 0;
            }
            .trace-index {
                width: 28px;
                height: 28px;
                border-radius: 50%;
                background: #e0e7ff;
                color: #3730a3;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 0.8rem;
                font-weight: 700;
            }
            .trace-text {
                margin: 0.15rem 0 0;
                color: #1f2937;
            }
            .panel-title {
                font-size: 1.02rem;
                color: #1e293b;
                font-weight: 650;
                margin-bottom: 0.6rem;
            }
            .stat-chip {
                background: #eef2ff;
                border: 1px solid #dbeafe;
                color: #1e3a8a;
                border-radius: 999px;
                display: inline-block;
                padding: 0.3rem 0.65rem;
                font-size: 0.85rem;
                margin-right: 0.35rem;
                margin-bottom: 0.35rem;
            }
            .kpi-card {
                background: #ffffff;
                border: 1px solid #e5e7eb;
                border-radius: 14px;
                padding: 0.8rem 0.9rem;
                height: 100%;
            }
            .kpi-label {
                color: #6b7280;
                font-size: 0.84rem;
                margin: 0;
            }
            .kpi-value {
                color: #0f172a;
                font-size: 1.2rem;
                font-weight: 700;
                margin: 0.2rem 0 0;
            }
            .chat-shell {
                border: 1px solid #dbe4f0;
                border-radius: 20px;
                background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
                padding: 1rem;
                box-shadow: 0 14px 32px rgba(15, 23, 42, 0.08);
            }
            .chat-toolbar {
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 0.6rem;
                margin-bottom: 0.7rem;
                padding-bottom: 0.7rem;
                border-bottom: 1px solid #e5edf6;
            }
            .chat-title {
                margin: 0;
                color: #0f172a;
                font-weight: 700;
                font-size: 1.05rem;
            }
            .chat-subtitle {
                margin: 0.15rem 0 0;
                color: #64748b;
                font-size: 0.85rem;
            }
            .prompt-chip {
                display: inline-block;
                background: #eef4ff;
                border: 1px solid #dbe7ff;
                color: #1e3a8a;
                border-radius: 999px;
                padding: 0.3rem 0.65rem;
                font-size: 0.78rem;
                margin: 0.15rem 0.25rem 0.25rem 0;
                font-weight: 600;
            }
            .chat-history-card {
                border: 1px solid #e2e8f0;
                background: #ffffff;
                border-radius: 16px;
                padding: 0.75rem;
                max-height: 420px;
                overflow-y: auto;
                margin-top: 0.65rem;
            }
            .chat-bubble {
                border-radius: 14px;
                padding: 0.65rem 0.75rem;
                margin-bottom: 0.55rem;
            }
            .chat-bubble.user {
                background: #eff6ff;
                border: 1px solid #dbeafe;
            }
            .chat-bubble.assistant {
                background: #f8fafc;
                border: 1px solid #e2e8f0;
            }
            .chat-role {
                margin: 0 0 0.2rem;
                font-size: 0.78rem;
                font-weight: 700;
                letter-spacing: 0.02em;
                text-transform: uppercase;
                color: #334155;
            }
            .chat-copy {
                margin: 0;
                color: #0f172a;
                line-height: 1.5;
            }
            div[data-testid="stMetric"] {
                background: #ffffff;
                border: 1px solid #e5e7eb;
                border-radius: 14px;
                padding: 0.6rem 0.75rem;
            }
            div[data-testid="stMetricLabel"] p {
                font-size: 1rem;
            }
            div[data-testid="stMetricValue"] {
                font-size: 2.15rem !important;
                line-height: 1.2;
            }
            div[data-testid="stMetricValue"] > div {
                font-size: inherit !important;
            }
            div[data-testid="stDataFrame"] {
                border: 1px solid #e2e8f0;
                border-radius: 12px;
                overflow: hidden;
            }
            section[data-testid="stSidebar"] .stButton {
                text-align: left;
            }
            section[data-testid="stSidebar"] .stButton > button {
                width: 100%;
                text-align: left;
                justify-content: flex-start !important;
                align-items: center;
            }
            section[data-testid="stSidebar"] .stButton > button div[data-testid="stMarkdownContainer"] {
                width: 100%;
            }
            section[data-testid="stSidebar"] .stButton > button div[data-testid="stMarkdownContainer"] p {
                margin: 0;
                text-align: left;
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


def render_kpi_card(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="kpi-card">
            <p class="kpi-label">{label}</p>
            <p class="kpi-value">{value}</p>
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


def render_research_job_fit_portfolio() -> None:
    """Render a portfolio section tailored to transparent AI research applications."""
    st.markdown("### 🎯 Portfolio Fit: Transparent AI for Nutrition")
    st.caption(
        "This project is structured to demonstrate readiness for PhD/PostDoc roles in causal AI, "
        "explainable AI, and human-centered decision support systems."
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_kpi_card("Target Role", "PhD/PostDoc")
    with c2:
        render_kpi_card("Core Domain", "xAI + Nutrition")
    with c3:
        render_kpi_card("AI Methods", "ML + Rule-based")
    with c4:
        render_kpi_card("Interaction", "Dialogue-first")

    st.markdown("#### Requirement-to-Implementation Mapping")
    with st.container(border=True):
        st.markdown(
            """
            - **Transparent recommendation system for nutrition intake** → Hybrid pipeline combining logistic-regression
              predictions with deterministic rule-based reasoning.
            - **Explainability strategies** → Natural-language rationale, contribution ranking, and trace-based explanation
              views for each generated recommendation.
            - **User-facing explanation interfaces** → Dedicated explanation console and conversational follow-up module.
            - **Trust calibration and behaviour outcomes** → Simulated user-study metrics (trust, explanation quality,
              consistency, stability) visible in the evaluation strip.
            - **Technical + human-centred evaluation** → Confidence, consistency, and comparative feedback between
              ML and rule-based modes.
            """
        )

    a, b = st.columns(2)
    with a:
        st.markdown("#### Research Contribution Highlights")
        st.markdown(
            """
            1. **Hybrid xAI architecture** connecting statistical prediction with symbolic logic.
            2. **Dialogue-based explanation workflow** for contestability and user understanding.
            3. **Human-centred evaluation hooks** to support reproducible behavioural experiments.
            4. **Portfolio-ready modularity** through separated `recommender`, `explainability`, `dialogue`,
               and `evaluation` components.
            """
        )

    with b:
        st.markdown("#### Research-Oriented Next Steps")
        st.markdown(
            """
            1. Add **counterfactual recourse prompts** ("what minimal change improves recommendation quality?").
            2. Integrate a **nutrition knowledge graph** for domain-grounded explanations.
            3. Extend fairness checks for **robustness and subgroup reliability**.
            4. Run controlled studies with **trust calibration** and decision-quality outcomes.
            """
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
        health_goal = st.selectbox(
            "Primary Health Goal",
            ["general_wellness", "weight_loss", "muscle_gain", "blood_sugar_control"],
            format_func=lambda value: value.replace("_", " ").title(),
            key=f"{key_prefix}_health_goal",
        )
        conditions = st.multiselect(
            "Medical Conditions (for personalization safety)",
            ["diabetes", "hypertension", "pcos", "high_cholesterol"],
            key=f"{key_prefix}_conditions",
        )
        allergies_text = st.text_input(
            "Allergies (comma separated, e.g., peanut, shellfish)",
            value="",
            key=f"{key_prefix}_allergies",
        )
        excluded_foods_text = st.text_input(
            "Foods to Avoid (comma separated, e.g., beef, dairy)",
            value="",
            key=f"{key_prefix}_excluded_foods",
        )

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

    profile_kwargs = {
        "age": int(age),
        "weight": float(weight),
        "activity_level": activity,
        "dietary_preference": "balanced",
        "sugar_preference": sugar,
        "health_goal": health_goal,
        "allergies": tuple(part.strip().lower() for part in allergies_text.split(",") if part.strip()),
        "excluded_foods": tuple(
            part.strip().lower() for part in excluded_foods_text.split(",") if part.strip()
        ),
        "medical_conditions": tuple(condition.replace("_", " ") for condition in conditions),
    }
    accepted_fields = set(inspect.signature(UserProfile).parameters)
    profile = UserProfile(**{key: value for key, value in profile_kwargs.items() if key in accepted_fields})
    st.session_state.current_draft_profile = profile
    st.session_state.current_draft_mode = mode
    return profile, mode, generate_clicked


def build_personalization_message(profile: UserProfile | None) -> str:
    """Create a professional personalization note, even for legacy predictions."""
    if profile is None:
        return "Personalization active: profile context is applied to improve recommendation relevance."

    goal = str(getattr(profile, "health_goal", "general wellness") or "general wellness").replace("_", " ")
    medical_conditions = tuple(getattr(profile, "medical_conditions", ()) or ())
    allergies = tuple(getattr(profile, "allergies", ()) or ())

    conditions = ", ".join(str(item) for item in medical_conditions) if medical_conditions else "none reported"
    restrictions = ", ".join(str(item) for item in allergies) if allergies else "none reported"

    return (
        f"Personalization active: aligned to goal '{goal}', conditions ({conditions}), "
        f"and allergies/restrictions ({restrictions})."
    )


def run_recommendation_pipeline(profile: UserProfile, mode: str) -> None:
    recommendation = generate_recommendation(profile, mode=mode)
    recommendation.setdefault("personalization_message", build_personalization_message(profile))
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
    st.info(rec.get("personalization_message") or build_personalization_message(st.session_state.profile))
    draft_profile = st.session_state.get("current_draft_profile")
    generated_profile = st.session_state.get("profile")
    if draft_profile is not None and generated_profile is not None and draft_profile != generated_profile:
        st.warning(
            "Form values changed since the last generation. The recommendation summary reflects the previous run. "
            "Click **Generate Personalized Recommendation** to refresh."
        )

    meal_ideas = rec.get("meal_ideas", [])
    if meal_ideas:
        st.markdown("### Personalized Meal Ideas")
        for idx, meal in enumerate(meal_ideas, start=1):
            st.markdown(f"{idx}. {meal}")

    image_links = rec.get("food_image_links", [])
    if image_links:
        st.markdown("### Public Food Image References")
        image_cols = st.columns(min(len(image_links), 2))
        for idx, image_info in enumerate(image_links):
            with image_cols[idx % len(image_cols)]:
                st.image(
                    image_info.get("url", ""),
                    caption=f"{image_info.get('title', 'Food reference')} ({image_info.get('source', 'Public source')})",
                    use_container_width=True,
                )
                st.markdown(f"[Open image link]({image_info.get('url', '')})")


def render_explanation_panel() -> None:
    exp = st.session_state.last_explanation
    if not exp:
        st.info("Generate a recommendation first to inspect explanation details.")
        return

    trace = exp.get("trace", {})
    shap_rows = st.session_state.last_shap_output
    top_feature = "n/a"
    top_contribution = 0.0
    negative_count = 0
    contribution_count = 0

    if shap_rows:
        shap_df = pd.DataFrame(shap_rows)
        if {"feature", "contribution"}.issubset(shap_df.columns):
            contribution_count = len(shap_df)
            ranked = shap_df.reindex(shap_df["contribution"].abs().sort_values(ascending=False).index)
            if not ranked.empty:
                top_feature = str(ranked.iloc[0]["feature"])
                top_contribution = float(ranked.iloc[0]["contribution"])
            negative_count = int((shap_df["contribution"] < 0).sum())
        else:
            shap_df = pd.DataFrame()
    else:
        shap_df = pd.DataFrame()

    st.markdown("### Explainability Studio")
    st.markdown(
        """
        <div class="explain-kpi-grid">
            <div class="kpi-card">
                <p class="kpi-label">Top Driver</p>
                <p class="kpi-value">{top_feature}</p>
            </div>
            <div class="kpi-card">
                <p class="kpi-label">Strongest Influence</p>
                <p class="kpi-value">{top_contribution:+.3f}</p>
            </div>
            <div class="kpi-card">
                <p class="kpi-label">Features Reviewed</p>
                <p class="kpi-value">{contribution_count}</p>
            </div>
        </div>
        """.format(
            top_feature=top_feature.replace("_", " ").title(),
            top_contribution=top_contribution,
            contribution_count=contribution_count,
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="explain-layout">
            <div class="panel-card">
                <h4 class="explain-title">Natural-language rationale</h4>
                <p class="explain-subtitle">Human-friendly summary of why this recommendation was produced.</p>
                <p class="rationale-copy">{exp.get("natural_language", "No explanation available.")}</p>
            </div>
            <div class="panel-card">
                <h4 class="explain-title">Signal balance</h4>
                <p class="explain-subtitle">Direction of factors that increased or reduced confidence.</p>
                <ul class="impact-list">
                    <li><span>Positive influences</span><span class="impact-pill impact-positive">{max(contribution_count - negative_count, 0)}</span></li>
                    <li><span>Negative influences</span><span class="impact-pill impact-negative">{negative_count}</span></li>
                </ul>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Reasoning trace")
    if trace.get("human_readable"):
        steps = trace["human_readable"]
    else:
        steps = []
        for item in trace.get("contributions", []):
            steps.append(item.get("reason") or f"{item.get('feature')} contribution={item.get('contribution', 0):+.2f}")

    if steps:
        def _clean_trace_step(step: object) -> str:
            text = str(step or "").strip()
            if not text:
                return ""
            # If malformed HTML leaks into trace content, keep only readable text.
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
            return text

        clean_steps = [_clean_trace_step(step) for step in steps]
        clean_steps = [step for step in clean_steps if step]
        if clean_steps:
            trace_markup = ['<div class="trace-card">']
            for idx, clean_step in enumerate(clean_steps, start=1):
                trace_markup.append(
                    f'<div class="trace-step"><span class="trace-index">{idx}</span>'
                    f'<p class="trace-text">{escape(clean_step)}</p></div>'
                )
            trace_markup.append("</div>")
            st.markdown("".join(trace_markup), unsafe_allow_html=True)
        else:
            st.caption("No step-by-step trace is available for this recommendation.")
    else:
        st.caption("No step-by-step trace is available for this recommendation.")

    bayesian_probs = trace.get("bayesian_probabilities", {})
    if bayesian_probs:
        st.markdown("## Probabilistic Reasoning (Bayesian View)")
        bayes_df = (
            pd.DataFrame(
                [
                    {"diet": diet, "probability": probability}
                    for diet, probability in bayesian_probs.items()
                ]
            )
            .sort_values("probability", ascending=False)
            .reset_index(drop=True)
        )
        bayes_df["diet"] = bayes_df["diet"].str.replace("_", " ").str.title()
        bayes_df["probability"] = bayes_df["probability"].map(lambda value: f"{value:.1%}")

        top_bayes = max(bayesian_probs.items(), key=lambda item: item[1])[0]
        st.dataframe(bayes_df, use_container_width=True, hide_index=True)
        st.success(f"Most likely diet (Bayesian): **{top_bayes.replace('_', ' ').title()}**")
        st.caption(trace.get("bayesian_explanation", "No Bayesian explanation available."))

    if not shap_df.empty:
        st.markdown("### Feature contributions")
        chart_cols = st.columns([2, 1])
        with chart_cols[0]:
            st.bar_chart(shap_df.set_index("feature")["contribution"])
        with chart_cols[1]:
            st.markdown("#### Ranking by absolute impact")
            ranked_df = shap_df.assign(abs_contribution=shap_df["contribution"].abs()).sort_values(
                "abs_contribution",
                ascending=False,
            )
            st.dataframe(
                ranked_df[["feature", "contribution"]].reset_index(drop=True),
                use_container_width=True,
                hide_index=True,
            )

        with st.expander("View raw contribution table"):
            st.dataframe(shap_df, use_container_width=True, hide_index=True)


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
    show_ai_disclaimer()
    st.markdown("### Clinical Q&A Dialogue")
    st.markdown(
        """
        <div class="chat-shell">
            <div class="chat-toolbar">
                <div>
                    <p class="chat-title">Follow-up Assistant</p>
                    <p class="chat-subtitle">Get patient-friendly reasoning and actionable next steps.</p>
                </div>
            </div>
            <span class="prompt-chip">Explain this recommendation in plain language</span>
            <span class="prompt-chip">How would lower sugar preference change the result?</span>
            <span class="prompt-chip">What 2 improvements should I prioritize first?</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    query = st.text_input(
        "Ask follow-up questions",
        placeholder="Example: Why this plan? What changes if I increase activity level to high?",
    )

    c1, c2 = st.columns([4, 1])
    with c1:
        ask_clicked = st.button("Ask Assistant", type="primary", use_container_width=True)
    with c2:
        clear_clicked = st.button("Clear", use_container_width=True)

    if clear_clicked:
        st.session_state.chat_history = []

    if ask_clicked:
        cleaned_query = query.strip()
        if not cleaned_query:
            st.info("Please enter a question so I can provide a focused answer.")
        elif st.session_state.profile and st.session_state.last_prediction:
            response = st.session_state.chatbot.respond(
                question=cleaned_query,
                profile=st.session_state.profile,
                recommendation=st.session_state.last_prediction,
                explanation=st.session_state.last_explanation,
                shap_summary=st.session_state.last_shap_output,
            )
            st.session_state.chat_history.append((cleaned_query, response))
            st.session_state.chat_history = st.session_state.chat_history[-5:]
        else:
            st.warning("Generate a recommendation first.")

    if st.session_state.chat_history:
        st.markdown('<div class="chat-history-card">', unsafe_allow_html=True)
        for question, answer in reversed(st.session_state.chat_history):
            safe_question = escape(question)
            safe_answer = escape(answer)
            st.markdown(
                f"""
                <div class="chat-bubble user">
                    <p class="chat-role">You</p>
                    <p class="chat-copy">{safe_question}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
                <div class="chat-bubble assistant">
                    <p class="chat-role">Assistant</p>
                    <p class="chat-copy">{safe_answer}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)
        show_ai_disclaimer()
    else:
        st.info("No conversation yet. Ask a follow-up to begin.")


def profile_to_table() -> pd.DataFrame:
    profile = st.session_state.profile
    if not profile:
        return pd.DataFrame()
    return pd.DataFrame([asdict(profile)])

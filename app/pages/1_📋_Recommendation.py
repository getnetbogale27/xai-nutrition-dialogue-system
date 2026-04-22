"""Recommendation workspace page."""

from pathlib import Path
import sys

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ui_components import (
    initialize_state,
    build_personalization_message,
    inject_professional_theme,
    profile_to_table,
    render_evaluation_strip,
    render_hero,
    render_kpi_card,
    render_profile_controls,
    render_recommendation_summary,
    run_recommendation_pipeline,
)

st.set_page_config(page_title="Recommendation", page_icon="📋", layout="wide")
inject_professional_theme()

st.markdown(
    """
    <style>
        .studio-shell {
            border: 1px solid #dbe4f0;
            border-radius: 22px;
            background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
            padding: 1rem;
            box-shadow: 0 18px 36px rgba(15, 23, 42, 0.08);
            margin-bottom: 0.8rem;
        }
        .studio-title {
            margin: 0;
            color: #0f172a;
            font-size: 1rem;
            font-weight: 700;
        }
        .studio-subtitle {
            margin: 0.2rem 0 0.8rem;
            color: #64748b;
            font-size: 0.86rem;
        }
        .workspace-status {
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            padding: 0.85rem;
            background: #ffffff;
            margin-bottom: 0.75rem;
        }
        .workspace-status h4 {
            margin: 0 0 0.45rem;
            font-size: 0.95rem;
            color: #0f172a;
        }
        .workspace-status ul {
            margin: 0;
            padding-left: 1rem;
            color: #475569;
            font-size: 0.88rem;
        }
        .workspace-status li { margin-bottom: 0.28rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

initialize_state()
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

st.divider()
st.markdown('<h2 class="section-header">Recommendation Workspace</h2>', unsafe_allow_html=True)
workspace_left, workspace_right = st.columns([1.2, 1.0], vertical_alignment="top")
with workspace_left:
    st.markdown(
        """
        <div class="studio-shell">
            <p class="studio-title">Workspace Controls</p>
            <p class="studio-subtitle">Capture profile context, choose the engine, then run a personalized recommendation.</p>
        """,
        unsafe_allow_html=True,
    )
    profile, mode, generate_clicked = render_profile_controls("recommendation_page")
    if generate_clicked:
        run_recommendation_pipeline(profile, mode)
        st.success("Recommendation pipeline completed successfully. Snapshot and evaluation panels updated.")
    st.markdown("</div>", unsafe_allow_html=True)

with workspace_right:
    st.markdown(
        """
        <div class="studio-shell">
            <p class="studio-title">Decision Output</p>
            <p class="studio-subtitle">Review generated diet pathway and confidence signals in real time.</p>
        """,
        unsafe_allow_html=True,
    )
    render_recommendation_summary()
    st.markdown(
        """
        <div class="workspace-status">
            <h4>Recommended workflow</h4>
            <ul>
                <li>Complete profile fields with realistic patient context.</li>
                <li>Select ML for probabilistic guidance or Rule-based for deterministic logic.</li>
                <li>Generate recommendation and validate outcomes in Live Snapshot.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

st.divider()
st.markdown('<h2 class="section-header">Live Snapshot</h2>', unsafe_allow_html=True)
overview_left, overview_right = st.columns([1.1, 1], vertical_alignment="top")
with overview_left:
    st.markdown('<div class="panel-card"><div class="panel-title">Current Profile Snapshot</div>', unsafe_allow_html=True)
    profile_df = profile_to_table()
    if profile_df.empty:
        st.info("No active patient profile yet. Configure the patient profile to activate real-time tracking.")
    else:
        st.dataframe(profile_df, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

with overview_right:
    st.markdown('<div class="panel-card"><div class="panel-title">Current Recommendation Snapshot</div>', unsafe_allow_html=True)
    if rec:
        st.markdown(
            f"""
            <span class="stat-chip">Diet: {rec.get("diet_label", "n/a").replace("_", " ").title()}</span>
            <span class="stat-chip">Engine: {rec.get("mode", "unknown").replace("-", " ").title()}</span>
            <span class="stat-chip">Confidence: {confidence:.1%}</span>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("#### Clinical Guidance")
        st.success(rec.get("recommendation_text", ""))
        st.caption(
            rec.get("personalization_message")
            or build_personalization_message(st.session_state.profile)
        )
    else:
        st.info("No recommendation generated yet. Complete the profile and run the recommendation engine.")
    st.markdown("</div>", unsafe_allow_html=True)

st.divider()
st.markdown('<h2 class="section-header">Evaluation Overview</h2>', unsafe_allow_html=True)
render_evaluation_strip()

"""Recommendation workspace page."""

from pathlib import Path
import sys

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ui_components import (
    initialize_state,
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

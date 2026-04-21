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
    render_profile_controls,
    render_recommendation_summary,
    run_recommendation_pipeline,
)

st.set_page_config(page_title="Recommendation", page_icon="📋", layout="wide")
inject_professional_theme()
initialize_state()
render_hero(
    "Recommendation Studio",
    "Generate transparent dietary recommendations with ML or rule-based logic while tracking live profile and outcome snapshots.",
)

overview_left, overview_right = st.columns([1, 1])
with overview_left:
    st.subheader("Current Profile Snapshot")
    profile_df = profile_to_table()
    if profile_df.empty:
        st.info("No active patient profile yet.")
    else:
        st.dataframe(profile_df, use_container_width=True)

with overview_right:
    st.subheader("Current Recommendation Snapshot")
    rec = st.session_state.last_prediction
    if rec:
        st.metric("Diet Label", rec.get("diet_label", "n/a").replace("_", " ").title())
        st.write(rec.get("recommendation_text", ""))
        st.caption(f"Engine: {rec.get('mode', 'unknown')}")
    else:
        st.info("No recommendation generated yet.")

st.divider()
st.subheader("Recommendation Workspace")
left, right = st.columns([1, 1.1])
with left:
    profile, mode, generate_clicked = render_profile_controls("recommendation_page")
    if generate_clicked:
        run_recommendation_pipeline(profile, mode)
        st.success("Recommendation pipeline completed.")

with right:
    render_recommendation_summary()

st.divider()
st.subheader("Evaluation Overview")
render_evaluation_strip()

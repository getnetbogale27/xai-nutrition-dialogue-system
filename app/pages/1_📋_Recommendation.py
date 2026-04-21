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
    "Generate transparent dietary recommendations with ML or rule-based logic.",
)

left, right = st.columns([1, 1.1])
with left:
    profile, mode, generate_clicked = render_profile_controls("recommendation_page")
    if generate_clicked:
        run_recommendation_pipeline(profile, mode)
        st.success("Recommendation pipeline completed.")

with right:
    render_recommendation_summary()

st.divider()
render_evaluation_strip()

"""Landing page for the XAI Nutrition Dialogue application."""

from pathlib import Path
import sys

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ui_components import (
    initialize_state,
    inject_professional_theme,
    profile_to_table,
    render_evaluation_strip,
    render_hero,
)

st.set_page_config(page_title="Main", page_icon="🏠", layout="wide")
inject_professional_theme()
initialize_state()

render_hero(
    "XAI Nutrition Dialogue System",
    "Portfolio project for PostDoc/PhD opportunities in Causal AI and explainable decision support.",
)

c1, c2 = st.columns([1, 1])
with c1:
    st.subheader("Current Profile Snapshot")
    profile_df = profile_to_table()
    if profile_df.empty:
        st.info("No active patient profile yet.")
    else:
        st.dataframe(profile_df, use_container_width=True)

with c2:
    st.subheader("Current Recommendation Snapshot")
    rec = st.session_state.last_prediction
    if rec:
        st.metric("Diet Label", rec.get("diet_label", "n/a").replace("_", " ").title())
        st.write(rec.get("recommendation_text", ""))
        st.caption(f"Engine: {rec.get('mode', 'unknown')}")
    else:
        st.info("No recommendation generated yet.")

st.divider()
st.subheader("Evaluation Overview")
render_evaluation_strip()

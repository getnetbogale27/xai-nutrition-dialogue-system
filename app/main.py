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
    render_hero,
)

st.set_page_config(page_title="Main", page_icon="🏠", layout="wide")
inject_professional_theme()
initialize_state()

render_hero(
    "XAI Nutrition Dialogue System",
    "Portfolio project for PostDoc/PhD opportunities in Causal AI and explainable decision support.",
)

st.info(
    "Snapshot and evaluation panels now live in the **Recommendation** page. "
    "Use the sidebar to open **📋 Recommendation**."
)

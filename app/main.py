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
    render_wur_job_fit_portfolio,
)

st.set_page_config(page_title="Main", page_icon="🏠", layout="wide")
inject_professional_theme()
initialize_state()

render_hero(
    "XAI Nutrition Dialogue System",
    "Portfolio project tailored for the Wageningen DECIDE PhD on transparent, explainable AI for dietary behaviour change.",
)

render_wur_job_fit_portfolio()

st.info("Use the sidebar flow: **📋 Recommendation → 🔍 Explanations → 💬 Dialogue → 📘 User Guide**.")

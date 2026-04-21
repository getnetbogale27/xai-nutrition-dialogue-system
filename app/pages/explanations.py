"""Explanation and transparency page."""

from pathlib import Path
import sys

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ui_components import (
    initialize_state,
    inject_professional_theme,
    render_explanation_panel,
    render_hero,
)

st.set_page_config(page_title="Explanations", page_icon="🔍", layout="wide")
inject_professional_theme()
initialize_state()

render_hero(
    "Explainability Console",
    "Inspect natural-language rationale, reasoning trace, and feature-level influence.",
)

render_explanation_panel()

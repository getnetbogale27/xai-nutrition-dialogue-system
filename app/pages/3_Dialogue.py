"""Dialogue page for follow-up recommendation questions."""

from pathlib import Path
import sys

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ui_components import (
    initialize_state,
    inject_professional_theme,
    render_chat_panel,
    render_hero,
    render_workflow_sidebar,
)

st.set_page_config(page_title="Dialogue", page_icon="💬", layout="wide")
inject_professional_theme()
initialize_state()
render_workflow_sidebar(current_step=3)

render_hero(
    "Interactive Dialogue",
    "Conduct context-aware follow-up discussion to improve trust and understanding.",
)

render_chat_panel()

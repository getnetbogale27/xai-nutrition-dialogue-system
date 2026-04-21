"""Author details page."""

from pathlib import Path
import sys

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ui_components import (
    initialize_state,
    inject_professional_theme,
    render_author_details,
    render_hero,
)

st.set_page_config(page_title="Author Details", page_icon="👤", layout="wide")
inject_professional_theme()
initialize_state()

render_hero(
    "Author Details",
    "Professional profile, research focus, and application-ready positioning.",
)

render_author_details()

with st.container(border=True):
    st.markdown("### 🧭 Positioning for the WUR DECIDE PhD")
    st.markdown(
        """
        - This portfolio demonstrates **hybrid AI system design** (ML + symbolic/rule-based reasoning).
        - The app includes **transparent explanation interfaces** and **dialogue-based explanation support**.
        - The evaluation layer is structured for **trust calibration** and **human-centred assessment**.
        - The architecture is modular and extensible for **knowledge graph integration** and **real-world studies**.
        """
    )

"""Professional user guide page for onboarding and best practices."""

from pathlib import Path
import sys

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ui_components import (
    initialize_state,
    inject_professional_theme,
    render_hero,
)

st.set_page_config(page_title="User Guide", page_icon="📘", layout="wide")
inject_professional_theme()
initialize_state()

render_hero(
    "User Guide",
    "A professional walkthrough for using the XAI Nutrition Dialogue System effectively.",
)

st.markdown("### What this system provides")
st.markdown(
    """
    - **Personalized nutrition recommendation** using either ML-based or rule-based logic.
    - **Transparent explanation layer** including rationale and feature-level contributions.
    - **Interactive dialogue support** for follow-up and what-if analysis.
    """
)

st.markdown("### Quick workflow")
with st.container(border=True):
    st.markdown(
        """
        1. **Recommendation**: Enter profile details and generate a plan.
        2. **Explanations**: Review why the recommendation was produced.
        3. **Dialogue**: Ask clinical or plain-language follow-up questions.
        """
    )

c1, c2 = st.columns(2)
with c1:
    st.markdown("### Input guidance")
    st.markdown(
        """
        - Enter **realistic profile values** (age, weight, activity, sugar preference).
        - Choose **ML engine** for probability-based output and confidence display.
        - Choose **Rule-based engine** for deterministic logic and interpretability.
        """
    )

with c2:
    st.markdown("### Interpretation guidance")
    st.markdown(
        """
        - Use **Diet Label** as the primary recommendation category.
        - Check **Confidence** for ML outputs before making decisions.
        - Use the **Reasoning Trace** and **Feature Contributions** to validate fairness and trust.
        """
    )

st.markdown("### Best practices")
st.info(
    """
    - Regenerate recommendations after any profile change.
    - Compare ML and rule-based outputs when decisions are sensitive.
    - Use Dialogue to test scenarios (e.g., reduced sugar, higher activity).
    - Treat this tool as decision support; final nutrition plans should be reviewed by professionals.
    """
)

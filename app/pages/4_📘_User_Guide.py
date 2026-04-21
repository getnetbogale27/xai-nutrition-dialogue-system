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
    render_wur_job_fit_portfolio,
)

st.set_page_config(page_title="User Guide", page_icon="📘", layout="wide")
inject_professional_theme()
initialize_state()

render_hero(
    "WUR DECIDE Portfolio Guide",
    "How this app demonstrates fit for the PhD position: xAI to support better dietary decisions.",
)

render_wur_job_fit_portfolio()

st.markdown("### What this system provides")
st.markdown(
    """
    - **Personalized nutrition recommendation** using ML-based and rule-based logic.
    - **Transparent explanation layer** with rationale, feature contributions, and trace.
    - **Interactive dialogue support** for follow-up, trust calibration, and what-if analysis.
    - **Evaluation views** connected to consistency, stability, confidence, and user-feedback simulation.
    """
)

st.markdown("### Quick workflow")
with st.container(border=True):
    st.markdown(
        """
        1. **Recommendation**: Configure profile and generate output with ML or rule-based reasoning.
        2. **Explanations**: Inspect transparent rationale and ranked feature influences.
        3. **Dialogue**: Ask plain-language follow-up questions to evaluate understanding.
        4. **Evaluation**: Review trust-related and reliability-oriented metrics.
        """
    )

c1, c2 = st.columns(2)
with c1:
    st.markdown("### Input guidance")
    st.markdown(
        """
        - Enter **realistic profile values** (age, weight, activity, sugar preference).
        - Choose **ML engine** for probabilistic output and confidence-based assessment.
        - Choose **Rule-based engine** for deterministic and interpretable logic.
        - Compare both engines to communicate transparency and contestability.
        """
    )

with c2:
    st.markdown("### Interpretation guidance")
    st.markdown(
        """
        - Use **Diet Label** as the primary recommendation category.
        - Check **Confidence** for ML outputs before considering intervention priority.
        - Use the **Reasoning Trace** and **Feature Contributions** to validate trust and clarity.
        - Use dialogue responses to assess whether explanations are citizen-understandable.
        """
    )

st.markdown("### Best practices")
st.info(
    """
    - Regenerate recommendations after any profile change.
    - Compare ML and rule-based outputs when decisions are sensitive.
    - Use dialogue to test scenario changes (e.g., reduced sugar, higher activity).
    - Treat this tool as decision support; final nutrition plans should be reviewed by professionals.
    - For the WUR application, frame this prototype as a baseline for controlled human-AI studies.
    """
)

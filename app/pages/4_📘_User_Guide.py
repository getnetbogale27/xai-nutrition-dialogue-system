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
    render_research_job_fit_portfolio,
)

st.set_page_config(page_title="User Guide", page_icon="📘", layout="wide")
inject_professional_theme()
initialize_state()

render_hero(
    "Research Portfolio Guide",
    "How this app demonstrates fit for PhD/PostDoc applications in causal and explainable AI for nutrition.",
)

render_research_job_fit_portfolio()

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
    - For applications, frame this prototype as a baseline for controlled human-AI studies.
    """
)

st.markdown("### Healthy Living Insights & Recommendations")
with st.container(border=True):
    st.markdown(
        """
        1. **Maintain a Healthy Weight**  
        Aim for steady, realistic habits instead of quick fixes. Small changes in eating, movement, and sleep can improve energy and support long-term health.

        2. **Be Physically Active Most Days**  
        Include regular movement such as walking, cycling, dancing, or strength exercises. Even short activity sessions during the day can make a meaningful difference.

        3. **Avoid Smoking and Tobacco**  
        Choosing not to smoke protects your heart, lungs, and overall health. If you currently smoke, seeking support to quit is a positive and powerful step.

        4. **Choose a Balanced, Whole-Food Diet**  
        Base your meals on vegetables, fruits, whole grains, legumes, lean proteins, and healthy fats. Limit heavily processed foods and sugary snacks when possible.

        5. **Use Alcohol in Moderation (or Avoid It)**  
        If you drink alcohol, keep amounts low and include alcohol-free days each week. Drinking mindfully supports liver health, sleep quality, and weight goals.

        6. **Prioritize Sleep Quality**  
        Keep a regular sleep schedule and aim for enough rest each night. Good sleep helps with appetite control, mood, concentration, and recovery.

        7. **Stay Hydrated**  
        Drink water consistently throughout the day, especially during physical activity or warm weather. Simple habits like carrying a water bottle can help.

        8. **Manage Stress in Healthy Ways**  
        Practice stress-reducing habits such as breathing exercises, stretching, short walks, or quiet time. Managing stress supports healthier food choices and better overall well-being.

        9. **Keep Up with Preventive Check-Ups**  
        Regular health visits and screenings can identify issues early and support better outcomes. Talk with your healthcare provider about the right check-up schedule for you.

        10. **Support Social Connection and Mental Well-Being**  
        Spend time with people who support and encourage you. Strong social connections can improve resilience, reduce stress, and promote a healthier lifestyle.
        """
    )

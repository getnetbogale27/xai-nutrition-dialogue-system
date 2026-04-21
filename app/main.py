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
    render_workflow_sidebar,
)

st.set_page_config(page_title="XAI Nutrition Dialogue", page_icon="🥗", layout="wide")
inject_professional_theme()
initialize_state()
render_workflow_sidebar(current_step=0)

render_hero(
    "XAI Nutrition Dialogue System",
    "Portfolio project for PostDoc/PhD opportunities in Causal AI and explainable decision support.",
)

st.markdown("### Author")
st.markdown("**Getnet B. Begashaw (PhD in Statistics)**  \n📧 getnetbogale145@gmail.com")

st.markdown("### Professional Links")
author_query = "Getnet B. Begashaw"
link_cols = st.columns(4)
link_cols[0].link_button(
    "LinkedIn",
    f"https://www.linkedin.com/search/results/all/?keywords={author_query.replace(' ', '%20')}",
    use_container_width=True,
)
link_cols[1].link_button(
    "Google Scholar",
    f"https://scholar.google.com/scholar?q={author_query.replace(' ', '+')}",
    use_container_width=True,
)
link_cols[2].link_button(
    "GitHub",
    f"https://github.com/search?q={author_query.replace(' ', '+')}",
    use_container_width=True,
)
link_cols[3].link_button(
    "Causal AI Work",
    f"https://www.google.com/search?q={author_query.replace(' ', '+')}+causal+AI",
    use_container_width=True,
)

st.markdown(
    """
    Use the workflow sidebar top-to-bottom:
    1. **Recommendation** → create or update a personalized plan.
    2. **Explanations** → inspect model reasoning and feature impact.
    3. **Dialogue** → ask follow-up what-if and trust-building questions.
    """
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

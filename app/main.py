"""Streamlit app for nutrition recommendation + explainability + chat."""

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

# Ensure project root is on PYTHONPATH so `src` imports work when running
# `streamlit run app/main.py`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dialogue.chatbot import NutritionChatbot
from src.explainability.explanation_engine import generate_explanation
from src.recommender.ml_model import train_model
from src.recommender.rules import UserProfile, generate_recommendation


st.set_page_config(page_title="XAI Nutrition Dialogue", layout="wide")
st.title("XAI Nutrition Dialogue System")
st.caption("Human-centered nutrition recommendation with transparent explainability")

# Session state initialization for continuous interactions.
if "profile" not in st.session_state:
    st.session_state.profile = None
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = {}
if "last_explanation" not in st.session_state:
    st.session_state.last_explanation = {}
if "last_shap_output" not in st.session_state:
    st.session_state.last_shap_output = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "chatbot" not in st.session_state:
    st.session_state.chatbot = NutritionChatbot()

left, center, right = st.columns([1, 1, 1])

with left:
    st.subheader("Panel 1: User Input")
    age = st.number_input("Age", min_value=1, max_value=100, value=30)
    weight = st.number_input("Weight (kg)", min_value=20.0, max_value=250.0, value=70.0)
    activity_level = st.selectbox("Activity Level", ["low", "medium", "high"])
    sugar_preference = st.selectbox("Sugar Preference", ["low", "high"])
    model_selector = st.radio(
        "Model Selector",
        ["ML (Logistic Regression)", "Rule-based"],
        horizontal=False,
    )
    mode = "ml-based" if model_selector.startswith("ML") else "rule-based"

    if st.button("Train / Refresh ML Model"):
        stats = train_model()
        st.success(f"Model trained. Accuracy={stats['accuracy']:.2f}, rows={stats['n_rows']}")

    generate_clicked = st.button("Generate Recommendation", type="primary")

if generate_clicked:
    profile = UserProfile(
        age=int(age),
        weight=float(weight),
        activity_level=activity_level,
        dietary_preference="balanced",
        sugar_preference=sugar_preference,
    )

    recommendation = generate_recommendation(profile, mode=mode)
    explanation = generate_explanation(profile, recommendation)

    st.session_state.profile = profile
    st.session_state.last_prediction = recommendation
    st.session_state.last_explanation = explanation
    st.session_state.last_shap_output = explanation.get("trace", {}).get("contributions", [])
    st.session_state.chat_history = []

with center:
    st.subheader("Panel 2: Recommendation Output")
    rec = st.session_state.last_prediction

    if rec:
        st.markdown(f"**Predicted Recommendation:** `{rec.get('diet_label', 'n/a')}`")
        st.write(rec.get("recommendation_text", ""))
        st.markdown(f"**Model Used:** `{rec.get('mode', 'unknown')}`")

        if rec.get("mode") == "ml-based":
            confidence = max(rec.get("probabilities", {}).values(), default=0.0)
            st.markdown(f"**Confidence:** `{confidence:.1%}`")
    else:
        st.info("Enter your inputs and click **Generate Recommendation**.")

with right:
    st.subheader("Panel 3: Explainability Dashboard")
    exp = st.session_state.last_explanation

    if exp:
        st.markdown("#### A) Explanation Text")
        st.write(exp.get("natural_language", "No explanation available."))

        st.markdown("#### B) Reasoning Trace")
        trace = exp.get("trace", {})

        if trace.get("human_readable"):
            for i, step in enumerate(trace["human_readable"], start=1):
                st.write(f"{i}. {step}")
        elif trace.get("contributions"):
            for i, item in enumerate(trace["contributions"], start=1):
                reason = item.get("reason") or f"{item.get('feature')} influenced the recommendation"
                st.write(f"{i}. {reason}")

        if st.session_state.last_prediction.get("mode") == "ml-based":
            st.markdown("#### C) SHAP Feature Contributions")
            shap_rows = st.session_state.last_shap_output
            if shap_rows:
                shap_df = pd.DataFrame(shap_rows)
                if "contribution" in shap_df.columns:
                    st.bar_chart(shap_df.set_index("feature")["contribution"])
                st.dataframe(shap_df, use_container_width=True)
            else:
                st.caption("No SHAP output available for the current prediction.")
    else:
        st.info("Explainability details will appear after generating a recommendation.")

st.divider()
st.subheader("Follow-up Dialogue")
question = st.text_input(
    "Ask follow-up questions",
    placeholder="Why was this recommended? / What changed if I reduce sugar? / Explain in simple terms",
)

if st.button("Send Question"):
    if st.session_state.profile and st.session_state.last_prediction:
        bot = st.session_state.chatbot
        response = bot.respond(
            question=question,
            profile=st.session_state.profile,
            recommendation=st.session_state.last_prediction,
            explanation=st.session_state.last_explanation,
            shap_summary=st.session_state.last_shap_output,
        )
        st.session_state.chat_history.append((question, response))
        st.session_state.chat_history = st.session_state.chat_history[-3:]
    else:
        st.warning("Please generate a recommendation first.")

if st.session_state.chat_history:
    for q, a in st.session_state.chat_history:
        st.markdown(f"**You:** {q}")
        st.markdown(f"**Assistant:** {a}")

    memory_preview = st.session_state.chatbot.get_memory()
    if memory_preview:
        st.caption("Chatbot memory stores the last 3 interactions for context-aware responses.")

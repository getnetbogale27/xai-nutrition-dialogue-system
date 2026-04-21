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


st.set_page_config(page_title="XAI Nutrition Dialogue", layout="centered")
st.title("XAI Nutrition Dialogue System")

st.subheader("1) Enter Your Details")
age = st.number_input("Age", min_value=1, max_value=100, value=30)
weight = st.number_input("Weight (kg)", min_value=20.0, max_value=250.0, value=70.0)
activity_level = st.selectbox("Activity Level", ["low", "medium", "high"])
dietary_preference = st.selectbox("Dietary Preference", ["balanced", "vegetarian", "vegan", "low carb"])
sugar_preference = st.selectbox("Sugar Preference", ["low", "high"])
mode = st.radio("Recommender Mode", ["ml-based", "rule-based"], index=0, horizontal=True)

if st.button("Train / Refresh ML Model"):
    stats = train_model()
    st.success(f"Model trained. Accuracy={stats['accuracy']:.2f}, rows={stats['n_rows']}")

if "profile" not in st.session_state:
    st.session_state.profile = None
if "recommendation" not in st.session_state:
    st.session_state.recommendation = {}
if "explanation" not in st.session_state:
    st.session_state.explanation = {}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if st.button("Generate Recommendation"):
    profile = UserProfile(
        age=int(age),
        weight=float(weight),
        activity_level=activity_level,
        dietary_preference=dietary_preference,
        sugar_preference=sugar_preference,
    )
    recommendation = generate_recommendation(profile, mode=mode)
    explanation = generate_explanation(profile, recommendation)

    st.session_state.profile = profile
    st.session_state.recommendation = recommendation
    st.session_state.explanation = explanation
    st.session_state.chat_history = []

if st.session_state.recommendation:
    rec = st.session_state.recommendation
    exp = st.session_state.explanation

    st.subheader("2) Recommendation")
    st.write(f"**Diet label:** {rec['diet_label']}")
    st.write(rec["recommendation_text"])

    if rec.get("probabilities"):
        st.caption("Class probabilities (ML model)")
        st.json(rec["probabilities"])

    st.subheader("3) Why This Recommendation")
    st.write(exp["natural_language"])

    trace = exp.get("trace", {})
    if trace.get("contributions"):
        st.markdown("**Feature contributions**")
        contrib_df = pd.DataFrame(trace["contributions"])
        if "contribution" in contrib_df.columns:
            st.bar_chart(contrib_df.set_index("feature")["contribution"])
            st.dataframe(contrib_df, use_container_width=True)
        if trace.get("human_readable"):
            for line in trace["human_readable"]:
                st.write(f"- {line}")

st.subheader("4) Ask a Follow-up Question")
question = st.text_input("Ask something like: Why this recommendation? or Can I eat less sugar?")

if st.button("Send Question"):
    if st.session_state.profile and st.session_state.recommendation:
        bot = NutritionChatbot()
        response = bot.respond(question, st.session_state.profile, st.session_state.recommendation)
        st.session_state.chat_history.append((question, response))
    else:
        st.warning("Please generate a recommendation first.")

for q, a in st.session_state.chat_history:
    st.markdown(f"**You:** {q}")
    st.markdown(f"**Assistant:** {a}")

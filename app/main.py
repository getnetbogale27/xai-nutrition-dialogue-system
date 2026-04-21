"""Minimal Streamlit app for nutrition recommendation + explanations + chat."""

from pathlib import Path
import sys

import streamlit as st

# Ensure project root is on PYTHONPATH so `src` imports work when running
# `streamlit run app/main.py`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dialogue.chatbot import NutritionChatbot
from src.explainability.explanation_engine import generate_explanation
from src.recommender.rules import UserProfile, generate_recommendation


st.set_page_config(page_title="XAI Nutrition Dialogue MVP", layout="centered")
st.title("XAI Nutrition Dialogue MVP")

st.subheader("1) Enter Your Details")
age = st.number_input("Age", min_value=1, max_value=100, value=30)
weight = st.number_input("Weight (kg)", min_value=20.0, max_value=250.0, value=70.0)
activity_level = st.selectbox("Activity Level", ["low", "moderate", "high"])
dietary_preference = st.selectbox("Dietary Preference", ["balanced", "vegetarian", "vegan", "low carb"])

if "profile" not in st.session_state:
    st.session_state.profile = None
if "recommendation" not in st.session_state:
    st.session_state.recommendation = ""
if "explanation" not in st.session_state:
    st.session_state.explanation = ""
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if st.button("Generate Recommendation"):
    profile = UserProfile(
        age=int(age),
        weight=float(weight),
        activity_level=activity_level,
        dietary_preference=dietary_preference,
    )
    recommendation = generate_recommendation(profile)
    explanation = generate_explanation(profile, recommendation)

    st.session_state.profile = profile
    st.session_state.recommendation = recommendation
    st.session_state.explanation = explanation
    st.session_state.chat_history = []

if st.session_state.recommendation:
    st.subheader("2) Recommendation")
    st.write(st.session_state.recommendation)

    st.subheader("3) Why This Recommendation")
    st.write(st.session_state.explanation)

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

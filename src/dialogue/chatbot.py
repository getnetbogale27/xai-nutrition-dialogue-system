"""Context-aware chatbot for recommendation follow-up questions."""

from __future__ import annotations

from dataclasses import asdict

from src.explainability.explanation_engine import generate_explanation
from src.recommender.rules import UserProfile, generate_recommendation


class NutritionChatbot:
    """Simple multi-turn assistant with short memory and recommendation context."""

    def __init__(self, memory_size: int = 3) -> None:
        self.memory_size = memory_size
        self.memory: list[dict] = []

    def _remember(self, question: str, response: str, recommendation: dict) -> None:
        self.memory.append(
            {
                "question": question,
                "response": response,
                "diet_label": recommendation.get("diet_label", "unknown"),
                "mode": recommendation.get("mode", "unknown"),
            }
        )
        self.memory = self.memory[-self.memory_size :]

    def get_memory(self) -> list[dict]:
        return self.memory

    def _simple_summary(self, recommendation: dict, shap_summary: list[dict]) -> str:
        label = recommendation.get("diet_label", "balanced")
        if not shap_summary:
            return (
                f"In simple terms: your plan is '{label}' because your profile best matches that nutrition style."
            )

        top = shap_summary[:2]
        top_text = ", ".join(item.get("feature", "feature") for item in top)
        return (
            f"In simple terms: the model selected '{label}' mainly because of {top_text}. "
            "These factors had the strongest influence on this recommendation."
        )

    def _what_if_lower_sugar(self, profile: UserProfile, recommendation: dict) -> str:
        mode = recommendation.get("mode", "ml-based")
        adjusted_profile = UserProfile(**{**asdict(profile), "sugar_preference": "low"})

        adjusted_rec = generate_recommendation(adjusted_profile, mode=mode)
        adjusted_exp = generate_explanation(adjusted_profile, adjusted_rec)

        current_label = recommendation.get("diet_label", "unknown")
        new_label = adjusted_rec.get("diet_label", "unknown")

        if current_label == new_label:
            return (
                f"If you reduce sugar, the recommendation stays '{new_label}' in {mode} mode, "
                "but confidence and feature influences can shift slightly. "
                f"Updated explanation: {adjusted_exp['natural_language']}"
            )

        return (
            f"If you reduce sugar, the recommendation changes from '{current_label}' to '{new_label}' in {mode} mode. "
            f"Updated explanation: {adjusted_exp['natural_language']}"
        )

    def respond(
        self,
        question: str,
        profile: UserProfile,
        recommendation: dict,
        explanation: dict | None = None,
        shap_summary: list[dict] | None = None,
    ) -> str:
        """Return context-aware response using recommendation + explanation context."""
        q = question.strip().lower()
        shap_summary = shap_summary or []

        if not q:
            response = "Please type a question about your recommendation."
            self._remember(question, response, recommendation)
            return response

        if "what changed" in q or "reduce sugar" in q or "less sugar" in q:
            response = self._what_if_lower_sugar(profile, recommendation)
            self._remember(question, response, recommendation)
            return response

        if "simple" in q or "simple terms" in q:
            response = self._simple_summary(recommendation, shap_summary)
            self._remember(question, response, recommendation)
            return response

        if "why" in q or "recommended" in q or "recommendation" in q:
            payload = explanation or generate_explanation(profile, recommendation)
            response = payload.get("natural_language", "I could not generate a detailed explanation.")
            self._remember(question, response, recommendation)
            return response

        if "last" in q and self.memory:
            previous = self.memory[-1]
            response = (
                f"Your last recommendation was '{previous['diet_label']}' using {previous['mode']} mode. "
                "Ask 'why was this recommended?' for a detailed explanation."
            )
            self._remember(question, response, recommendation)
            return response

        response = (
            "I can explain your recommendation, compare what happens if sugar is reduced, "
            "provide a simple-language summary, and reference your personalized restrictions."
        )
        self._remember(question, response, recommendation)
        return response

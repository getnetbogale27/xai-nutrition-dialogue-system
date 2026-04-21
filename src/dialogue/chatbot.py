"""Simple rule-based chatbot for recommendation follow-up questions."""

from src.explainability.explanation_engine import generate_explanation
from src.recommender.rules import UserProfile


class NutritionChatbot:
    def respond(self, question: str, profile: UserProfile, recommendation: str) -> str:
        """Return a short response based on user question intent."""
        q = question.strip().lower()

        if not q:
            return "Please type a question about your recommendation."

        if "why" in q or "recommendation" in q:
            return generate_explanation(profile, recommendation)

        if "sugar" in q:
            return (
                "Yes. You can reduce sugar by limiting sweet drinks, desserts, and processed snacks, "
                "while choosing fruit, yogurt, and whole foods."
            )

        if "protein" in q:
            return "Good protein options include eggs, fish, chicken, tofu, beans, and Greek yogurt."

        return "I can explain your plan or answer food questions like sugar and protein choices."

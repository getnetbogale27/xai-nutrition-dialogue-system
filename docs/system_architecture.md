# System Architecture Summary

## Pipeline Overview
The system follows the pipeline below:

**Input → Recommender → Explanation Engine → SHAP → Dialogue → UI**

## Component Flow (Text Diagram)
1. **Input**
   - User enters age, weight, activity level, and sugar preference in the Streamlit interface.
2. **Recommender**
   - The user-selected strategy produces a recommendation:
     - Logistic Regression model (ML-based), or
     - Rule-based logic (fallback / comparator).
3. **Explanation Engine**
   - Builds natural-language rationale and structured trace for the predicted diet label.
4. **SHAP Module**
   - For ML predictions, computes feature contributions and ranking of influential factors.
5. **Dialogue Module**
   - Handles follow-up questions and contextual clarification based on recent interaction memory.
6. **UI Layer (Streamlit)**
   - Presents recommendation output, explainability dashboard, and evaluation panel in one interface.

## Design Intent
The architecture intentionally combines transparent symbolic reasoning and data-driven modeling to provide robust recommendations while preserving user trust and interpretability.

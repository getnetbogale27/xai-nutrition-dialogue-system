# Methodology: Transparent AI for Nutrition Recommendation

## 1. Problem Statement
Personalized nutrition systems often optimize predictive accuracy but provide limited transparency for end users. In practical health settings, users require understandable and trustworthy rationales before adopting dietary recommendations. This project addresses that gap by combining predictive modeling with interpretable reasoning and conversational explanations.

## 2. System Design
The prototype follows a hybrid AI design to balance adaptability and reliability:

- **ML Recommender (Logistic Regression):** Generates data-driven diet labels with class probabilities.
- **Rule-based Fallback:** Ensures deterministic recommendations for edge cases or baseline comparison.
- **SHAP Explainability:** Produces feature-level attributions for model predictions.
- **Explanation Engine:** Converts traces and contributions into natural-language rationales.
- **Dialogue Layer:** Supports follow-up clarification and contextual user interaction.
- **Streamlit Interface:** Integrates recommendations, explainability dashboards, and interaction flow.

This combination supports both operational robustness and human-centered interpretability.

## 3. Explainability Approach
Explainability is implemented at two levels:

1. **Model-level transparency** through SHAP contribution scores for each input feature.
2. **User-level transparency** through natural-language summaries and structured reasoning traces.

For rule-based recommendations, the system reports explicit condition-based reasoning (e.g., activity level, weight, and sugar preference), preserving traceability even without ML inference.

## 4. Evaluation Methodology (Simulation-Based User Study)
To support research-oriented analysis without external human-subject collection, the system includes a simulation module that estimates perceived user outcomes from generated explanations.

### Simulated Constructs
- **Trust score** (0–1)
- **Explanation quality** (0–1)
- **Satisfaction** (0–1)

### Simulation Logic
- ML-based explanations are assigned higher baseline trust/quality due to contribution-level transparency.
- Rule-based explanations receive medium baseline scores due to deterministic but less nuanced explanation depth.
- Missing or unclear explanation text/trace decreases all outcomes.

### System Metrics
In addition to simulated user outcomes, the evaluation layer computes:
- Explanation consistency score
- Feature-importance stability proxy for SHAP outputs
- Model confidence distribution descriptors
- Rule vs ML comparison summary

These metrics provide lightweight but reproducible indicators for comparative system analysis.

## 5. Expected Contributions and DECIDE Alignment
This prototype contributes a reproducible framework for explainable nutrition recommendation that aligns with DECIDE-oriented goals in transparent and responsible AI decision support:

- A practical hybrid architecture combining symbolic and statistical reasoning.
- Integrated explainability that is both machine-auditable and user-facing.
- A simulation-based evaluation protocol for early-stage research iterations.
- An end-to-end demonstrator suitable for publication-oriented prototype reporting.

## 6. Reproducibility Notes
The full pipeline runs locally without external APIs. Data processing, model inference, explainability generation, and evaluation are deterministic under fixed seeds and static datasets, enabling replicable experimentation.

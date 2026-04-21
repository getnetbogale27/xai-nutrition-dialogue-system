"""System-level evaluation metrics for explainability and model behavior."""

from __future__ import annotations

from typing import Any


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def explanation_consistency_score(explanation: dict[str, Any]) -> float:
    """Score consistency between natural language text and trace structure."""
    natural = str(explanation.get("natural_language", "")).strip()
    trace = explanation.get("trace", {}) if isinstance(explanation, dict) else {}
    has_text = len(natural.split()) >= 8
    has_trace = bool(trace.get("human_readable") or trace.get("contributions"))

    if has_text and has_trace:
        return 0.9
    if has_text or has_trace:
        return 0.6
    return 0.2


def feature_importance_stability(shap_contributions: list[dict[str, Any]]) -> float:
    """Estimate stability from SHAP contribution concentration.

    Heuristic: if top absolute contribution dominates heavily, stability is lower;
    more balanced top-3 contributions indicate better stability.
    """
    if not shap_contributions:
        return 0.0

    abs_values = [abs(float(row.get("contribution", 0.0))) for row in shap_contributions]
    total = sum(abs_values)
    if total == 0:
        return 0.5

    top = sorted(abs_values, reverse=True)[:3]
    concentration = top[0] / max(sum(top), 1e-8)
    return _clamp(1.0 - concentration)


def model_confidence_distribution(probabilities: dict[str, float]) -> dict[str, float]:
    """Return basic confidence descriptors from class probabilities."""
    if not probabilities:
        return {"max_confidence": 0.0, "mean_confidence": 0.0, "confidence_gap": 0.0}

    vals = sorted([float(v) for v in probabilities.values()], reverse=True)
    max_conf = vals[0]
    mean_conf = sum(vals) / len(vals)
    gap = vals[0] - vals[1] if len(vals) > 1 else vals[0]

    return {
        "max_confidence": _clamp(max_conf),
        "mean_confidence": _clamp(mean_conf),
        "confidence_gap": _clamp(gap),
    }


def rule_vs_ml_comparison(ml_feedback: dict[str, float], rule_feedback: dict[str, float]) -> dict[str, float]:
    """Compare simulated user-study outcomes for ML vs rule-based strategy."""
    ml_avg = sum(ml_feedback.values()) / len(ml_feedback) if ml_feedback else 0.0
    rule_avg = sum(rule_feedback.values()) / len(rule_feedback) if rule_feedback else 0.0

    return {
        "ml_average": _clamp(ml_avg),
        "rule_average": _clamp(rule_avg),
        "ml_advantage": _clamp((ml_avg - rule_avg + 1.0) / 2.0),
    }

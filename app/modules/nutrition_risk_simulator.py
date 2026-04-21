"""UI module for Nutrition Risk Simulator (XAI)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.risk_simulator import NutritionRiskSimulator, RiskProfile


SIMULATOR = NutritionRiskSimulator()


def _default_profile() -> RiskProfile:
    return RiskProfile(
        sugar_intake_g=40.0,
        fiber_intake_g=18.0,
        fruit_veg_servings=3.0,
        processed_food_servings=3.0,
        activity_level="medium",
        sleep_quality="average",
        smoking_status="non_smoker",
        alcohol_use="moderate",
        stress_level=5,
    )


def _build_profile_from_state() -> RiskProfile:
    return RiskProfile(
        sugar_intake_g=float(st.session_state.risk_sugar_intake_g),
        fiber_intake_g=float(st.session_state.risk_fiber_intake_g),
        fruit_veg_servings=float(st.session_state.risk_fruit_veg_servings),
        processed_food_servings=float(st.session_state.risk_processed_food_servings),
        activity_level=st.session_state.risk_activity_level,
        sleep_quality=st.session_state.risk_sleep_quality,
        smoking_status=st.session_state.risk_smoking_status,
        alcohol_use=st.session_state.risk_alcohol_use,
        stress_level=int(st.session_state.risk_stress_level),
    )


def _ensure_defaults() -> None:
    defaults = _default_profile()
    for key, value in vars(defaults).items():
        session_key = f"risk_{key}"
        if session_key not in st.session_state:
            st.session_state[session_key] = value


def _result_to_frame(result) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Factor": row.factor.replace("_", " ").title(),
                "Contribution": round(row.contribution, 2),
                "Direction": row.direction.title(),
                "Rationale": row.rationale,
            }
            for row in result.contributions
        ]
    )


def render_nutrition_risk_simulator() -> None:
    _ensure_defaults()
    st.title("Nutrition Risk Simulator (XAI)")
    st.caption(
        "Modular, explainable risk scoring with baseline weighted model and one-variable scenario simulation."
    )

    input_tab, results_tab, explain_tab, simulation_tab = st.tabs(
        ["Input", "Results", "Explanation", "Simulation"]
    )

    with input_tab:
        st.markdown("#### Diet Indicators")
        d1, d2 = st.columns(2)
        with d1:
            st.slider("Sugar Intake (g/day)", 0.0, 120.0, key="risk_sugar_intake_g")
            st.slider("Fiber Intake (g/day)", 5.0, 50.0, key="risk_fiber_intake_g")
        with d2:
            st.slider("Fruit & Veg Servings/day", 0.0, 10.0, key="risk_fruit_veg_servings")
            st.slider("Processed Food Servings/day", 0.0, 8.0, key="risk_processed_food_servings")

        st.markdown("#### Activity & Lifestyle Indicators")
        l1, l2 = st.columns(2)
        with l1:
            st.selectbox("Activity Level", ["low", "medium", "high"], key="risk_activity_level")
            st.selectbox("Sleep Quality", ["poor", "average", "good"], key="risk_sleep_quality")
        with l2:
            st.selectbox("Smoking Status", ["non_smoker", "occasional", "regular"], key="risk_smoking_status")
            st.selectbox("Alcohol Use", ["none", "moderate", "high"], key="risk_alcohol_use")

        st.slider("Stress Level (1=low, 10=high)", 1, 10, key="risk_stress_level")

        if st.button("Compute Nutrition Health Score", type="primary", use_container_width=True):
            st.session_state.risk_profile = _build_profile_from_state()
            st.session_state.risk_baseline = SIMULATOR.evaluate(st.session_state.risk_profile)

    baseline = st.session_state.get("risk_baseline")
    if baseline is None:
        st.session_state.risk_profile = _build_profile_from_state()
        st.session_state.risk_baseline = SIMULATOR.evaluate(st.session_state.risk_profile)
        baseline = st.session_state.risk_baseline

    with results_tab:
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Nutrition Health Score", f"{baseline.score:.1f}/100")
        with c2:
            st.metric("Risk Band", baseline.risk_band)
        st.markdown("#### Ranked Factor Contributions")
        st.dataframe(_result_to_frame(baseline), use_container_width=True, hide_index=True)

    with explain_tab:
        st.markdown("#### Explainability Text")
        st.info(baseline.explanation_text)
        top3 = baseline.contributions[:3]
        st.markdown("#### Top Influencers")
        for row in top3:
            st.markdown(f"- **{row.factor.replace('_', ' ').title()}** ({row.contribution:+.2f}): {row.rationale}")

    with simulation_tab:
        st.markdown("#### Scenario Simulation (Adjust One Variable)")
        variables = {
            "sugar_intake_g": "Sugar Intake (g/day)",
            "fiber_intake_g": "Fiber Intake (g/day)",
            "fruit_veg_servings": "Fruit & Veg Servings/day",
            "processed_food_servings": "Processed Food Servings/day",
            "activity_level": "Activity Level",
            "sleep_quality": "Sleep Quality",
            "smoking_status": "Smoking Status",
            "alcohol_use": "Alcohol Use",
            "stress_level": "Stress Level",
        }
        variable = st.selectbox(
            "Variable to modify",
            list(variables.keys()),
            format_func=lambda key: variables[key],
            key="risk_scenario_variable",
        )

        original_profile = st.session_state.get("risk_profile", _build_profile_from_state())

        if variable in {"sugar_intake_g", "fiber_intake_g", "fruit_veg_servings", "processed_food_servings"}:
            ranges = {
                "sugar_intake_g": (0.0, 120.0),
                "fiber_intake_g": (5.0, 50.0),
                "fruit_veg_servings": (0.0, 10.0),
                "processed_food_servings": (0.0, 8.0),
            }
            lo, hi = ranges[variable]
            new_value = st.slider(
                f"New {variables[variable]}",
                lo,
                hi,
                value=float(getattr(original_profile, variable)),
                key="risk_scenario_numeric_value",
            )
        elif variable == "stress_level":
            new_value = st.slider(
                "New Stress Level",
                1,
                10,
                value=int(getattr(original_profile, variable)),
                key="risk_scenario_stress_value",
            )
        else:
            options = {
                "activity_level": ["low", "medium", "high"],
                "sleep_quality": ["poor", "average", "good"],
                "smoking_status": ["non_smoker", "occasional", "regular"],
                "alcohol_use": ["none", "moderate", "high"],
            }
            new_value = st.selectbox(
                f"New {variables[variable]}",
                options[variable],
                index=options[variable].index(getattr(original_profile, variable)),
                key="risk_scenario_category_value",
            )

        if st.button("Run Scenario Simulation", use_container_width=True):
            st.session_state.risk_scenario = SIMULATOR.simulate_one_variable(original_profile, variable, new_value)

        scenario = st.session_state.get("risk_scenario")
        if scenario:
            b, s = scenario["baseline"], scenario["scenario"]
            s1, s2, s3 = st.columns(3)
            s1.metric("Baseline Score", f"{b.score:.1f}")
            s2.metric("Scenario Score", f"{s.score:.1f}", delta=f"{s.score - b.score:+.1f}")
            s3.metric("Scenario Risk Band", s.risk_band)
            st.success(scenario["comparison_text"])

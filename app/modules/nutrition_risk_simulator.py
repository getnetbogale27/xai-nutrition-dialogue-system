"""UI module for Nutrition Risk Simulator (XAI)."""

from __future__ import annotations

from datetime import date
from inspect import signature

import pandas as pd
import streamlit as st

from src.risk_simulator import NutritionRiskSimulator, RiskProfile


SIMULATOR = NutritionRiskSimulator()

ACTIVITY_OPTIONS = ["low", "medium", "high"]
ACTIVITY_LABELS = {
    "low": "Low (<150 min/week moderate activity)",
    "medium": "Medium (150-300 min/week moderate activity)",
    "high": "High (>300 min/week moderate activity)",
}

SLEEP_OPTIONS = ["poor", "average", "good"]
SLEEP_LABELS = {
    "poor": "Poor (<6 hours/night or frequent interruption)",
    "average": "Average (6-7 hours/night, occasional interruption)",
    "good": "Good (7-9 hours/night, restful sleep)",
}

SMOKING_OPTIONS = ["non_smoker", "occasional", "regular"]
SMOKING_LABELS = {
    "non_smoker": "Non-smoker (0 cigarettes/day)",
    "occasional": "Occasional (1-5 cigarettes/day or social smoking)",
    "regular": "Regular (>5 cigarettes/day)",
}

ALCOHOL_OPTIONS = ["none", "moderate", "high"]
ALCOHOL_LABELS = {
    "none": "None (0 drinks/week)",
    "moderate": "Moderate (1-7 drinks/week)",
    "high": "High (8+ drinks/week)",
}

GENDER_OPTIONS = ["female", "male", "non_binary", "prefer_not_to_say"]
GENDER_LABELS = {
    "female": "Female",
    "male": "Male",
    "non_binary": "Non-binary",
    "prefer_not_to_say": "Prefer not to say",
}

COUNTRY_OPTIONS = [
    "",
    "United States",
    "Canada",
    "United Kingdom",
    "Australia",
    "India",
    "Germany",
    "France",
    "Brazil",
    "Japan",
    "Other",
]


def _make_risk_profile(**kwargs) -> RiskProfile:
    """Build a profile while tolerating older RiskProfile signatures.

    Some environments may import a legacy `RiskProfile` class that does not yet
    define demographic fields such as `gender`, `date_of_birth`, `bmi_value`,
    or `country`. Filter kwargs to accepted parameters so UI stays compatible.
    """

    accepted = set(signature(RiskProfile).parameters)
    filtered = {key: value for key, value in kwargs.items() if key in accepted}
    return RiskProfile(**filtered)


def _default_profile() -> RiskProfile:
    return _make_risk_profile(
        sugar_intake_g=40.0,
        fiber_intake_g=18.0,
        fruit_veg_servings=3.0,
        processed_food_servings=3.0,
        activity_level="medium",
        sleep_quality="average",
        smoking_status="non_smoker",
        alcohol_use="moderate",
        stress_level=5,
        gender="prefer_not_to_say",
        date_of_birth=None,
        bmi_value=None,
        country="",
    )


def _build_profile_from_state() -> RiskProfile:
    return _make_risk_profile(
        sugar_intake_g=float(st.session_state.risk_sugar_intake_g),
        fiber_intake_g=float(st.session_state.risk_fiber_intake_g),
        fruit_veg_servings=float(st.session_state.risk_fruit_veg_servings),
        processed_food_servings=float(st.session_state.risk_processed_food_servings),
        activity_level=st.session_state.risk_activity_level,
        sleep_quality=st.session_state.risk_sleep_quality,
        smoking_status=st.session_state.risk_smoking_status,
        alcohol_use=st.session_state.risk_alcohol_use,
        stress_level=int(st.session_state.risk_stress_level),
        gender=st.session_state.risk_gender,
        date_of_birth=st.session_state.risk_date_of_birth,
        bmi_value=st.session_state.risk_bmi_value,
        country=st.session_state.risk_country,
    )


def _compute_bmi(weight_kg: float, height_cm: float) -> float | None:
    if height_cm <= 0:
        return None
    height_m = height_cm / 100.0
    return weight_kg / (height_m * height_m)


def _bmi_category(bmi: float | None) -> str:
    if bmi is None:
        return "Not set"
    if bmi < 18.5:
        return "Underweight"
    if bmi < 25:
        return "Normal"
    if bmi < 30:
        return "Overweight"
    return "Obese"


def _ensure_defaults() -> None:
    defaults = _default_profile()
    for key, value in vars(defaults).items():
        session_key = f"risk_{key}"
        if session_key not in st.session_state:
            st.session_state[session_key] = value
    if "risk_date_of_birth" not in st.session_state:
        st.session_state.risk_date_of_birth = date(1990, 1, 1)
    if "risk_bmi_metric_weight_kg" not in st.session_state:
        st.session_state.risk_bmi_metric_weight_kg = 70.0
    if "risk_bmi_metric_height_cm" not in st.session_state:
        st.session_state.risk_bmi_metric_height_cm = 170.0
    if "risk_bmi_imperial_weight_lb" not in st.session_state:
        st.session_state.risk_bmi_imperial_weight_lb = 154.0
    if "risk_bmi_imperial_height_in" not in st.session_state:
        st.session_state.risk_bmi_imperial_height_in = 67.0
    if "risk_bmi_value" not in st.session_state:
        st.session_state.risk_bmi_value = None


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
        st.markdown("#### Personal & Demographic Inputs")
        p1, p2 = st.columns(2)
        with p1:
            st.selectbox(
                "Gender",
                GENDER_OPTIONS,
                format_func=lambda option: GENDER_LABELS[option],
                key="risk_gender",
            )
            st.date_input(
                "Date of Birth",
                min_value=date(1920, 1, 1),
                max_value=date.today(),
                key="risk_date_of_birth",
            )
        with p2:
            st.selectbox("Country", COUNTRY_OPTIONS, key="risk_country")

            bmi_unit = st.radio("BMI Calculator Units", ["Metric (kg/cm)", "Imperial (lb/in)"], horizontal=True)
            if bmi_unit == "Metric (kg/cm)":
                weight_kg = st.number_input(
                    "Weight (kg)", min_value=10.0, max_value=350.0, step=0.5, key="risk_bmi_metric_weight_kg"
                )
                height_cm = st.number_input(
                    "Height (cm)", min_value=50.0, max_value=250.0, step=0.5, key="risk_bmi_metric_height_cm"
                )
                bmi_value = _compute_bmi(weight_kg, height_cm)
            else:
                weight_lb = st.number_input(
                    "Weight (lb)", min_value=22.0, max_value=770.0, step=1.0, key="risk_bmi_imperial_weight_lb"
                )
                height_in = st.number_input(
                    "Height (in)", min_value=20.0, max_value=100.0, step=0.5, key="risk_bmi_imperial_height_in"
                )
                bmi_value = _compute_bmi(weight_lb * 0.45359237, height_in * 2.54)

            if st.button("Calculate BMI", use_container_width=True):
                st.session_state.risk_bmi_value = bmi_value

            current_bmi = st.session_state.risk_bmi_value
            if current_bmi is not None:
                st.info(f"Current BMI: {current_bmi:.1f} ({_bmi_category(current_bmi)})")
            else:
                st.caption("Click **Calculate BMI** to store BMI in the profile.")

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
            st.selectbox(
                "Activity Level",
                ACTIVITY_OPTIONS,
                format_func=lambda option: ACTIVITY_LABELS[option],
                key="risk_activity_level",
            )
            st.selectbox(
                "Sleep Quality",
                SLEEP_OPTIONS,
                format_func=lambda option: SLEEP_LABELS[option],
                key="risk_sleep_quality",
            )
        with l2:
            st.selectbox(
                "Smoking Status",
                SMOKING_OPTIONS,
                format_func=lambda option: SMOKING_LABELS[option],
                key="risk_smoking_status",
            )
            st.selectbox(
                "Alcohol Use",
                ALCOHOL_OPTIONS,
                format_func=lambda option: ALCOHOL_LABELS[option],
                key="risk_alcohol_use",
            )

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
                "activity_level": ACTIVITY_OPTIONS,
                "sleep_quality": SLEEP_OPTIONS,
                "smoking_status": SMOKING_OPTIONS,
                "alcohol_use": ALCOHOL_OPTIONS,
            }
            option_labels = {
                "activity_level": ACTIVITY_LABELS,
                "sleep_quality": SLEEP_LABELS,
                "smoking_status": SMOKING_LABELS,
                "alcohol_use": ALCOHOL_LABELS,
            }
            new_value = st.selectbox(
                f"New {variables[variable]}",
                options[variable],
                index=options[variable].index(getattr(original_profile, variable)),
                format_func=lambda option: option_labels[variable][option],
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

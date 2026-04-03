# risk_score.py
from __future__ import annotations
from typing import Mapping

def clamp(x: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, x))

def parabolic_index(value: float, optimal: float, tolerance: float) -> float:
    """
    Creates a parabolic curve peaking at 1.0 when value == optimal.
    Tolerance dictates how wide the 'bell' is before hitting 0.
    """
    distance = abs(value - optimal)
    score = 1.0 - ((distance / tolerance) ** 2)
    return clamp(score)

# ----------------------------
# Component indices
# ----------------------------

def season_gate(T30_c: float) -> float:
    """
    Biological feasibility gate. Stable flies lack diapause and freeze tolerance.
    Average temps below 5°C force the entire system risk to 0.
    """
    return clamp((T30_c - 5.0) / 10.0)

def thermal_index_dd15(dd15_sum_30: float) -> float:
    """
    30-day accumulated DD15.
    Optimal: 6.6 DD15/day * 30 days = 198.
    Tolerance: 150 (allows risk to taper off during extreme heat/cold).
    """
    return parabolic_index(dd15_sum_30, optimal=198.0, tolerance=150.0)

def moisture_index(p_mm_2_7w_total: float) -> float:
    """
    2-7 week precip lag scaling (35 days).
    Optimal: 7.4 mm/day * 35 days = 259 mm.
    Tolerance: 200 (penalizes extreme drought or flooding).
    """
    return parabolic_index(p_mm_2_7w_total, optimal=259.0, tolerance=200.0)

def management_index(answers_dict: Mapping[str, float]) -> float:
    """
    Computes weighted average based on the 9 specific management questions.
    Expects answers as normalized floats [0.0 to 1.0], where 1.0 is HIGH RISK (poor management).
    """
    weights = {
        "q1_sanitation": 0.20,
        "q2_manure_tasks": 0.10,
        "q3_calves": 0.15,
        "q4_manure_store": 0.15,
        "q5_fogging": 0.10,
        "q6_spraying": 0.10,
        "q7_feedthrough": 0.05,
        "q8_wasps": 0.10,
        "q9_bait": 0.05
    }
    
    score = 0.0
    for key, weight in weights.items():
        val = clamp(float(answers_dict.get(key, 0.5)))
        score += (val * weight)
        
    return clamp(score)

# ----------------------------
# Final scoring
# ----------------------------

def _band_from_score(score: int) -> str:
    if score <= 10:
        return "Minimal"
    if score <= 40:
        return "Building"
    if score <= 79:
        return "Elevated"
    return "Peak"

def compute_risk_score(
    T30_c: float,
    dd15_sum_30: float,
    p_mm_2_7w_total: float,
    answers_dict: Mapping[str, float],
) -> dict:
    """
    60% Weather, 40% Management.
    All components gated by season feasibility G.
    """
    # 1. Calculate base indices
    G = season_gate(T30_c)
    HT = thermal_index_dd15(dd15_sum_30)
    HP = moisture_index(p_mm_2_7w_total)
    M = management_index(answers_dict)

    # 2. Weather composite (50/50 blend of Temp and Moisture)
    H = 0.5 * HT + 0.5 * HP

    # 3. Apply 60/40 Split and Season Gate
    weather_component = 0.60 * G * H
    management_component = 0.40 * G * M

    # 4. Final Aggregation
    overall = weather_component + management_component

    risk_score = int(round(clamp(overall) * 100.0))
    weather_score = int(round(clamp(weather_component / 0.60) * 100.0)) # Scaled to 0-100 for display
    management_score = int(round(clamp(management_component / 0.40) * 100.0)) # Scaled to 0-100 for display

    return {
        "risk_score": risk_score,
        "band": _band_from_score(risk_score),
        "components": {
            "G": round(G, 2),
            "HT": round(HT, 2),
            "HP": round(HP, 2),
            "M": round(M, 2),
        },
        "weather_score": weather_score,
        "management_score": management_score,
    }
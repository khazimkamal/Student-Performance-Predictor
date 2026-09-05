"""Loads the trained models and turns raw form input into a prediction."""

import math
import os

import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")

# (min, max, label, step) for every input the form accepts. The ranges match the
# real UCI Student Performance columns the model was trained on.
INPUT_SPEC = {
    "study_time":   (1, 4, "Weekly study time", 1),
    "absences":     (0, 93, "Classes missed", 1),
    "failures":     (0, 4, "Past class failures", 1),
    "previous_g1":  (0, 100, "First-period grade", 1),
    "previous_g2":  (0, 100, "Second-period grade", 1),
}

# Counts and ordinal levels in the source data - fractions are not meaningful.
INTEGER_FIELDS = {"study_time", "absences", "failures"}

_bundle = None


class ModelNotTrained(Exception):
    """Raised when model.pkl is missing - run train_model.py first."""


def load_bundle():
    """Load model.pkl once and cache it."""
    global _bundle
    if _bundle is None:
        if not os.path.exists(MODEL_PATH):
            raise ModelNotTrained(
                "model/model.pkl not found. Run `python train_model.py` first."
            )
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


def get_metrics():
    return load_bundle()["metrics"]


# --------------------------------------------------------------------------- #
# Input handling
# --------------------------------------------------------------------------- #
def validate(form):
    """Return (values, errors). Accepts a Flask form or a parsed JSON dict, so
    values may arrive as strings or as numbers."""
    values, errors = {}, {}
    for field, (lo, hi, label, _step) in INPUT_SPEC.items():
        raw = form.get(field)
        raw = "" if raw is None else str(raw).strip()
        if raw == "":
            errors[field] = f"{label} is required."
            continue
        try:
            val = float(raw)
        except ValueError:
            errors[field] = f"{label} must be a number."
            continue
        if not math.isfinite(val):  # "nan" and "inf" survive float()
            errors[field] = f"{label} must be a number."
            continue
        if val < lo or val > hi:
            errors[field] = f"{label} must be between {lo} and {hi}."
            continue
        if field in INTEGER_FIELDS and val != int(val):
            errors[field] = f"{label} must be a whole number."
            continue
        values[field] = val
    return values, errors


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #
def grade_for(score):
    for cutoff, letter in ((90, "A+"), (80, "A"), (70, "B"), (60, "C"), (50, "D")):
        if score >= cutoff:
            return letter
    return "F"


def risk_for(score, fail_prob, pass_mark):
    if score < pass_mark or fail_prob >= 0.50:
        return "High"
    if score < pass_mark + 15 or fail_prob >= 0.20:
        return "Medium"
    return "Low"


def _frame(values, features):
    return pd.DataFrame([[values[f] for f in features]], columns=features)


def predict(values):
    """values: dict of the five features. Returns the full result dict."""
    bundle = load_bundle()
    features = bundle["features"]
    pass_mark = bundle["pass_mark"]

    X = _frame(values, features)
    score = float(np.clip(bundle["regressor"].predict(X)[0], 0, 100))
    fail_prob = float(bundle["classifier"].predict_proba(X)[0][1])

    return {
        **values,
        "predicted_score": round(score, 1),
        "grade": grade_for(score),
        "fail_probability": round(fail_prob * 100, 1),
        "risk_level": risk_for(score, fail_prob, pass_mark),
        "pass_mark": pass_mark,
        "model_used": bundle["best_regressor"],
        "tips": build_tips(values, score, pass_mark),
        "what_if": what_if(values, score),
    }


def _score_only(values):
    bundle = load_bundle()
    X = _frame(values, bundle["features"])
    return float(np.clip(bundle["regressor"].predict(X)[0], 0, 100))


def what_if(values, base_score):
    """Re-run the model with one habit improved at a time, so the student can
    see which change buys the most marks. Only the two habits a student can
    still act on plus the next period's grade - past failures cannot change."""
    scenarios = [
        ("Study 5-10 hours a week (level 3)", "study_time",
         max(values["study_time"], 3)),
        ("Miss no more than 2 classes", "absences", min(values["absences"], 2)),
        ("Lift the second-period grade by 10", "previous_g2",
         min(values["previous_g2"] + 10, 100)),
    ]

    results = []
    for label, field, new_value in scenarios:
        if abs(new_value - values[field]) < 1e-9:
            continue  # already there
        trial = {**values, field: new_value}
        gain = _score_only(trial) - base_score
        if gain > 0.3:
            results.append({"label": label, "gain": round(gain, 1)})

    results.sort(key=lambda r: r["gain"], reverse=True)
    return results[:3]


def build_tips(values, score, pass_mark):
    """Plain rule-based advice to go with the number."""
    tips = []
    if values["study_time"] <= 1:
        tips.append(
            "Under 2 hours of study a week. Moving up one level (2-5 hours) is the "
            "cheapest change on this form."
        )
    if values["absences"] >= 10:
        tips.append(
            f"{values['absences']:.0f} classes missed. Absences are the strongest "
            "non-grade signal in this model - stop the count rising first."
        )
    if values["failures"] >= 1:
        tips.append(
            f"{values['failures']:.0f} past class failure(s). Students who have "
            "failed before score markedly lower here, so ask for support early."
        )
    if values["previous_g2"] < values["previous_g1"] - 5:
        tips.append(
            "The second-period grade dropped below the first. The trend matters more "
            "than either grade on its own - find out what changed."
        )
    if values["previous_g2"] < pass_mark:
        tips.append(
            f"The second-period grade is already below the {pass_mark:.0f} pass mark. "
            "The final grade rarely recovers on its own."
        )
    if values["previous_g1"] >= 70 and values["previous_g2"] >= 70:
        tips.append(
            "Both period grades are strong. Keep the same routine through the final "
            "assessment."
        )

    if not tips:
        tips.append("Nothing on this form stands out as a warning sign. Keep steady.")
    if score < pass_mark:
        tips.insert(0, "Predicted below the pass mark - talk to a tutor this week.")
    return tips

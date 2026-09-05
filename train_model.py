"""
Trains the student performance models on real data.

Source: UCI Student Performance data set (Cortez & Silva, 2008) - 395 real
secondary-school students from two Portuguese schools, Math course.
The raw file ships with the project as data/student-mat.csv.

Two models are produced from the same feature set:
  1. A regressor that predicts the expected final score (0-100).
  2. A classifier that predicts the probability of failing (score < 50).

For the regressor, Linear Regression and Random Forest are both trained and the
one with the better test R2 is kept. Metrics for both are saved so the web app
can show how the comparison went.

Run:  python train_model.py
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "student_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "model", "metrics.json")

# "mat" (Math, 395 students) or "por" (Portuguese, 649 students).
COURSE = "mat"
RAW_PATH = os.path.join(BASE_DIR, "data", f"student-{COURSE}.csv")

DATASET = {
    "name": "UCI Student Performance",
    "course": "Mathematics" if COURSE == "mat" else "Portuguese language",
    "citation": (
        "Cortez, P. and Silva, A. (2008). Using Data Mining to Predict Secondary "
        "School Student Performance. Proceedings of 5th FUture BUsiness "
        "TEChnology Conference, pp. 5-12."
    ),
    "url": "https://archive.ics.uci.edu/dataset/320/student+performance",
    "synthetic": False,
}

FEATURES = [
    "study_time",
    "absences",
    "failures",
    "previous_g1",
    "previous_g2",
]
FEATURE_LABELS = {
    "study_time": "Weekly study time",
    "absences": "Classes missed",
    "failures": "Past class failures",
    "previous_g1": "First-period grade",
    "previous_g2": "Second-period grade",
}
# Grades in the raw data run 0-20; the app works in percent, so a 10/20 pass
# becomes 50/100.
GRADE_SCALE = 5.0
PASS_MARK = 50.0
RANDOM_STATE = 42


# --------------------------------------------------------------------------- #
# 1. Dataset
# --------------------------------------------------------------------------- #
def build_dataset():
    """Read the raw UCI file and keep the five predictors the app asks for.

    The raw file is semicolon-separated with quoted text columns. Only the
    columns below are used; everything else (demographics, family background)
    is left out to keep the form to five inputs.
    """
    try:
        raw = pd.read_csv(RAW_PATH, sep=";")
        if "studytime" not in raw.columns:
            raw = pd.read_csv(RAW_PATH, sep=",")
    except Exception:
        raw = pd.read_csv(RAW_PATH, sep=",")

    df = pd.DataFrame(
        {
            "study_time": raw["studytime"].astype(float),
            "absences": raw["absences"].astype(float),
            "failures": raw["failures"].astype(float),
            "previous_g1": raw["G1"].astype(float) * GRADE_SCALE,
            "previous_g2": raw["G2"].astype(float) * GRADE_SCALE,
            "final_score": raw["G3"].astype(float) * GRADE_SCALE,
        }
    )
    return df.round(1)


def load_dataset():
    """Use data/student_data.csv if it is there, otherwise build it from the
    raw UCI file. Drop in your own CSV with the same six columns to retrain on
    different data - nothing else needs to change."""
    if os.path.exists(DATA_PATH):
        print(f"Loading dataset from {DATA_PATH}")
        return pd.read_csv(DATA_PATH)

    if not os.path.exists(RAW_PATH):
        raise SystemExit(
            f"Neither {DATA_PATH} nor {RAW_PATH} exists. Download the UCI "
            "Student Performance data set and put student-mat.csv in data/."
        )

    print(f"Building dataset from {RAW_PATH}")
    df = build_dataset()
    df.to_csv(DATA_PATH, index=False)
    print(f"Saved {len(df)} real student records to {DATA_PATH}")
    return df


# --------------------------------------------------------------------------- #
# 2. Training
# --------------------------------------------------------------------------- #
def main():
    df = load_dataset()
    X = df[FEATURES]
    y_score = df["final_score"]
    y_fail = (df["final_score"] < PASS_MARK).astype(int)

    X_train, X_test, y_train, y_test, f_train, f_test = train_test_split(
        X, y_score, y_fail, test_size=0.2, random_state=RANDOM_STATE
    )

    # --- regressors -------------------------------------------------------- #
    candidates = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    scores = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        scores[name] = {
            "r2": round(float(r2_score(y_test, pred)), 4),
            "rmse": round(float(np.sqrt(mean_squared_error(y_test, pred))), 3),
            "mae": round(float(mean_absolute_error(y_test, pred)), 3),
        }
        print(f"{name:18s} R2={scores[name]['r2']:.4f}  "
              f"RMSE={scores[name]['rmse']:.3f}  MAE={scores[name]['mae']:.3f}")

    best_name = max(scores, key=lambda n: scores[n]["r2"])
    regressor = candidates[best_name]
    print(f"\nBest regressor: {best_name}")

    # --- classifier (risk of failing) -------------------------------------- #
    classifier = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    classifier.fit(X_train, f_train)
    f_pred = classifier.predict(X_test)
    f_prob = classifier.predict_proba(X_test)[:, 1]
    clf_metrics = {
        "accuracy": round(float(accuracy_score(f_test, f_pred)), 4),
        "roc_auc": round(float(roc_auc_score(f_test, f_prob)), 4),
        "fail_rate_in_data": round(float(y_fail.mean()), 4),
    }
    print(f"Risk classifier    accuracy={clf_metrics['accuracy']:.4f}  "
          f"AUC={clf_metrics['roc_auc']:.4f}")

    # --- what drives the prediction ---------------------------------------- #
    importances = dict(
        zip(FEATURES, [round(float(v), 4) for v in classifier.feature_importances_])
    )

    metrics = {
        "regressors": scores,
        "best_regressor": best_name,
        "classifier": clf_metrics,
        "feature_importance": importances,
        "n_samples": int(len(df)),
        "pass_mark": PASS_MARK,
        "dataset": DATASET,
    }

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(
        {
            "regressor": regressor,
            "classifier": classifier,
            "features": FEATURES,
            "feature_labels": FEATURE_LABELS,
            "best_regressor": best_name,
            "pass_mark": PASS_MARK,
            "metrics": metrics,
        },
        MODEL_PATH,
    )
    with open(METRICS_PATH, "w") as fh:
        json.dump(metrics, fh, indent=2)

    print(f"\nSaved model  -> {MODEL_PATH}")
    print(f"Saved metrics-> {METRICS_PATH}")


if __name__ == "__main__":
    main()

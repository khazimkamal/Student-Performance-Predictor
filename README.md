# EduPredict — Student Performance Prediction

A web app where five details from a student's school record are entered and a
machine learning model predicts their expected final score, grade, and risk of
failing. The model is trained on **real student data**, not a simulation.

**Stack:** HTML + CSS + JavaScript · Python + Flask · scikit-learn · SQLite

---

## Features

- **Prediction form** — weekly study time, classes missed, past class failures,
  first-period grade, second-period grade. Sliders and number boxes stay in sync.
- **Live estimate** — the gauge on the right updates as you drag a slider, before
  you submit (calls the JSON API in the background).
- **Result page** — predicted score out of 100, letter grade, risk level
  (Low / Medium / High), fail probability, and rule-based recommendations.
- **"Biggest wins available"** — re-runs the model with one input improved at a
  time, so the student sees which change buys the most marks.
- **History** — every prediction is saved to SQLite and listed with summary stats;
  rows can be deleted individually or cleared.
- **Model page** — Linear Regression vs Random Forest scores side by side, plus
  feature importances and the data source.
- **JSON API** — `POST /api/predict` and `GET /api/history`.

---

## Setup

```bash
# 1. create and activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# 2. install dependencies
pip install -r requirements.txt

# 3. train the model (reads data/student-mat.csv, writes model/model.pkl)
python train_model.py

# 4. run the app
python app.py
```

Then open **http://127.0.0.1:5000**.

On Windows you can skip steps 2-4 with `run.bat`, which sets everything up and
starts the server.

---

## Tests

```bash
python test_app.py        # 39 tests
python test_app.py -v     # one line per test
```

No test dependencies - it uses the standard library's `unittest` and Flask's test
client, and writes to a temporary database rather than `data/students.db`. The
suite covers every route, input validation and its rejection messages, malformed
API bodies, the risk tiers and grade bands, HTML escaping, the history round trip,
and the shape of the dataset built from the raw UCI file. It needs
`model/model.pkl`, so run `python train_model.py` first.

---

## Project structure

```
PROJECT__00/
├── app.py                  Flask routes
├── predictor.py            input validation, prediction, tips, what-if scenarios
├── database.py             SQLite schema and queries
├── train_model.py          dataset preparation + model training
├── test_app.py             test suite (standard library unittest)
├── requirements.txt
├── run.bat                 one-click setup + start (Windows)
├── data/
│   ├── student-mat.csv     raw UCI data, Math course (395 students)
│   ├── student-por.csv     raw UCI data, Portuguese course (649 students)
│   ├── student.txt         UCI column descriptions
│   ├── student_data.csv    the five features + target (generated)
│   └── students.db         prediction history (created on first run)
├── model/
│   ├── model.pkl           trained models + metadata
│   └── metrics.json        evaluation scores
├── static/
│   ├── css/style.css
│   └── js/main.js
└── templates/
    ├── base.html  index.html  result.html
    ├── history.html  about.html  error.html
```

---

## The data

**Source:** [UCI Student Performance data set](https://archive.ics.uci.edu/dataset/320/student+performance)
— 395 real secondary-school students from two Portuguese schools (Gabriel Pereira
and Mousinho da Silveira), collected in the 2005–06 school year from school
reports and questionnaires.

> Cortez, P. and Silva, A. (2008). *Using Data Mining to Predict Secondary School
> Student Performance.* Proceedings of 5th FUture BUsiness TEChnology Conference,
> pp. 5–12. (CC BY 4.0)

The raw file has 33 columns. `train_model.py` keeps the five that carry the
predictive signal and drops the demographic and family-background columns to keep
the form short:

| Form input | Raw column | Range |
|---|---|---|
| Weekly study time | `studytime` | 1 = <2h, 2 = 2–5h, 3 = 5–10h, 4 = >10h |
| Classes missed | `absences` | 0–93 |
| Past class failures | `failures` | 0–4 (4 means 4 or more) |
| First-period grade | `G1` | 0–20 in the raw data → rescaled to 0–100 |
| Second-period grade | `G2` | 0–20 in the raw data → rescaled to 0–100 |
| *(target)* Final score | `G3` | 0–20 in the raw data → rescaled to 0–100 |

Grades are multiplied by 5 so the whole app can work in percent. The pass mark
follows the original: 10/20 becomes **50/100**. 32.9% of the students in the data
fail by that definition, so the risk classifier sees a reasonably balanced problem.

The Portuguese-course file (`student-por.csv`, 649 students) ships too — set
`COURSE = "por"` at the top of `train_model.py`, delete `data/student_data.csv` and
retrain to use it. The two files are *not* stacked, because 382 students appear in
both and would be counted twice.

**To use your own data:** replace `data/student_data.csv` with a CSV having the
same six columns (`study_time, absences, failures, previous_g1, previous_g2,
final_score`) and re-run `python train_model.py` — nothing else needs to change.

---

## The models

Two models are trained from the same five features on the same 80/20 split
(`random_state=42`, 316 train / 79 test):

| Purpose | Model | Test score |
|---|---|---|
| Predict final score (0–100) | **Random Forest Regressor** | **R² 0.851**, RMSE 8.74, MAE 5.49 |
| — compared against | Linear Regression | R² 0.782, RMSE 10.57, MAE 6.70 |
| Predict risk of failing | Random Forest Classifier | accuracy 0.924, ROC AUC 0.975 |

`train_model.py` keeps whichever regressor has the higher R². On this data Random
Forest wins by about 7 R² points, and the reason is the bottom of the scale: 38 of
the 395 students (9.6%) have a final grade of 0. On those rows Linear Regression
predicts 27 on average — an error of 31.5 points, against 5.0 points on every other
row. A straight line cannot bend down to zero; the forest can. The exact numbers
are written to `model/metrics.json` and shown on the app's Model page.

One caveat about those 38 rows: every one of them also has `absences = 0`, which
looks less like perfect attendance than like a record that stopped being kept.
They are left in because they are part of the published data set, but they are why
both models look sharper than the underlying problem really is — and why "cut your
absences" can occasionally come out as a *negative* what-if gain. The what-if list
only shows changes the model scores above +0.3, so those cases are simply hidden
rather than dressed up.

Feature importance from the risk classifier:

| Feature | Importance |
|---|---|
| Second-period grade | 56.5% |
| First-period grade | 31.2% |
| Classes missed | 6.3% |
| Past class failures | 3.5% |
| Weekly study time | 2.4% |

The two prior grades carry almost 88% of the signal. This is the honest result on
real data, and it is worth reading carefully: the model is mostly saying *"students
who were doing badly keep doing badly"*. Study time barely registers. Treat the
what-if numbers for study time and absences as small nudges, not promises.

Risk levels:

- **High** — predicted score below 50, or fail probability ≥ 50%
- **Medium** — predicted score below 65, or fail probability ≥ 20%
- **Low** — everything else

---

## API

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"study_time":2,"absences":6,"failures":0,"previous_g1":25,"previous_g2":30}'
```

```json
{
  "ok": true,
  "result": {
    "predicted_score": 31.7,
    "grade": "F",
    "risk_level": "High",
    "fail_probability": 97.7,
    "model_used": "Random Forest",
    "tips": ["Predicted below the pass mark - talk to a tutor this week.", "..."],
    "what_if": [{"label": "Lift the second-period grade by 10", "gain": 9.9}]
  }
}
```

Invalid or out-of-range input returns `400` with a per-field `errors` object.
`study_time`, `absences` and `failures` are counts, so they must be whole numbers;
the two grades may be fractional.

`GET /api/history` returns the saved predictions and summary statistics.

---

## Notes

- History rows saved before the switch to real data used different columns. On
  first run they are moved to a `predictions_synthetic` table rather than deleted,
  and the history page starts empty.
- The app runs Flask's development server. For deployment, use a WSGI server
  (`waitress-serve --port=5000 app:app`) and set a real `SECRET_KEY` environment
  variable.
- This is a study aid, not a verdict on any student. The data is real but narrow —
  two schools, one country, one school year — and a prediction from five inputs
  should never be the only basis for a decision about someone's education.

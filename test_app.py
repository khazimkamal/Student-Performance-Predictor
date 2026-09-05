"""
Test suite for EduPredict.

Runs against a throwaway SQLite file, so it never touches data/students.db.
Needs model/model.pkl - run `python train_model.py` first.

    python test_app.py           # all tests
    python test_app.py -v        # one line per test
    python -m unittest test_app.TestValidation   # one group
"""

import os
import shutil
import tempfile
import unittest

import database

# Point the app at a temporary database before it is imported: app.py calls
# database.init_db() at import time.
_TMP_DIR = tempfile.mkdtemp(prefix="edupredict-tests-")
database.DB_PATH = os.path.join(_TMP_DIR, "test.db")

import app as application  # noqa: E402  (must come after the DB_PATH swap)
import predictor  # noqa: E402
import train_model  # noqa: E402

VALID = {
    "study_time": "2",
    "absences": "6",
    "failures": "0",
    "previous_g1": "25",
    "previous_g2": "30",
}


def tearDownModule():
    shutil.rmtree(_TMP_DIR, ignore_errors=True)


class Base(unittest.TestCase):
    """Shared client. Skips everything with a clear message if untrained."""

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(predictor.MODEL_PATH):
            raise unittest.SkipTest("model/model.pkl missing - run python train_model.py")
        application.app.config["TESTING"] = True
        cls.c = application.app.test_client()

    def api(self, **overrides):
        """POST /api/predict with VALID plus any overrides."""
        return self.c.post("/api/predict", json=dict(VALID, **overrides))

    def form(self, **overrides):
        """POST /predict the way the browser form does."""
        return self.c.post("/predict", data=dict(VALID, **overrides))


# --------------------------------------------------------------------------- #
class TestPages(Base):
    def test_form_page(self):
        r = self.c.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn(b"Weekly study time", r.data)
        self.assertIn(b"Second-period grade", r.data)

    def test_model_page_shows_the_comparison(self):
        r = self.c.get("/about")
        self.assertEqual(r.status_code, 200)
        for needle in (b"Linear Regression", b"Random Forest",
                       b"UCI Student Performance", b"Cortez"):
            self.assertIn(needle, r.data)

    def test_model_page_metrics_match_the_saved_ones(self):
        m = predictor.get_metrics()
        body = self.c.get("/about").data.decode()
        self.assertIn(str(m["regressors"]["Random Forest"]["r2"]), body)
        self.assertIn(str(m["regressors"]["Linear Regression"]["r2"]), body)

    def test_history_page(self):
        self.assertEqual(self.c.get("/history").status_code, 200)

    def test_unknown_url_is_404(self):
        r = self.c.get("/no-such-page")
        self.assertEqual(r.status_code, 404)
        self.assertIn(b"Page not found", r.data)


# --------------------------------------------------------------------------- #
class TestValidation(Base):
    def test_accepts_a_typical_student(self):
        d = self.api().get_json()
        self.assertTrue(d["ok"])
        self.assertGreaterEqual(d["result"]["predicted_score"], 0)
        self.assertLessEqual(d["result"]["predicted_score"], 100)

    def test_accepts_the_range_boundaries(self):
        cases = {
            "minimums": {"study_time": "1", "absences": "0", "failures": "0",
                         "previous_g1": "0", "previous_g2": "0"},
            "maximums": {"study_time": "4", "absences": "93", "failures": "4",
                         "previous_g1": "100", "previous_g2": "100"},
        }
        for label, data in cases.items():
            with self.subTest(label):
                self.assertTrue(self.c.post("/api/predict", json=data).get_json()["ok"])

    def test_accepts_numbers_as_well_as_strings(self):
        payload = {k: float(v) for k, v in VALID.items()}
        self.assertTrue(self.c.post("/api/predict", json=payload).get_json()["ok"])

    def test_accepts_a_fractional_grade(self):
        d = self.api(previous_g1="62.5").get_json()
        self.assertTrue(d["ok"])
        self.assertEqual(d["result"]["previous_g1"], 62.5)

    def test_rejects_bad_input(self):
        cases = {
            "above max": ({"study_time": "5"}, "study_time"),
            "below min": ({"absences": "-1"}, "absences"),
            "fractional count": ({"failures": "1.5"}, "failures"),
            "fractional level": ({"study_time": "2.5"}, "study_time"),
            "not a number": ({"previous_g1": "abc"}, "previous_g1"),
            "empty": ({"previous_g2": ""}, "previous_g2"),
            "NaN": ({"study_time": "nan"}, "study_time"),
            "infinity": ({"previous_g1": "inf"}, "previous_g1"),
            "negative infinity": ({"previous_g2": "-inf"}, "previous_g2"),
        }
        for label, (override, field) in cases.items():
            with self.subTest(label):
                r = self.api(**override)
                self.assertEqual(r.status_code, 400)
                self.assertIn(field, r.get_json()["errors"])

                r = self.form(**override)                     # the form path must agree
                self.assertEqual(r.status_code, 400)
                self.assertIn(b"Predict performance", r.data)  # form is re-rendered

    def test_missing_fields_are_all_reported(self):
        r = self.c.post("/api/predict", json={"study_time": 2})
        self.assertEqual(r.status_code, 400)
        self.assertEqual(set(r.get_json()["errors"]), set(VALID) - {"study_time"})

    def test_validate_returns_floats(self):
        values, errors = predictor.validate(VALID)
        self.assertEqual(errors, {})
        self.assertTrue(all(isinstance(v, float) for v in values.values()))


# --------------------------------------------------------------------------- #
class TestApiRobustness(Base):
    def test_malformed_bodies_are_rejected_not_crashed(self):
        for label, body in [("list", [1, 2, 3]), ("string", "hello"),
                            ("number", 5), ("bool", True), ("null", None)]:
            with self.subTest(label):
                self.assertEqual(self.c.post("/api/predict", json=body).status_code, 400)

    def test_broken_json_is_rejected(self):
        r = self.c.post("/api/predict", data="{oops", content_type="application/json")
        self.assertEqual(r.status_code, 400)

    def test_no_body_is_rejected(self):
        self.assertEqual(self.c.post("/api/predict").status_code, 400)


# --------------------------------------------------------------------------- #
class TestPredictions(Base):
    def test_all_three_risk_tiers_are_reachable(self):
        tiers = {
            "High": {"study_time": "1", "absences": "20", "failures": "2",
                     "previous_g1": "20", "previous_g2": "15"},
            "Medium": {"study_time": "2", "absences": "4", "failures": "0",
                       "previous_g1": "50", "previous_g2": "50"},
            "Low": {"study_time": "4", "absences": "1", "failures": "0",
                    "previous_g1": "90", "previous_g2": "95"},
        }
        for want, data in tiers.items():
            with self.subTest(want):
                got = self.c.post("/api/predict", json=data).get_json()["result"]
                self.assertEqual(got["risk_level"], want, got["predicted_score"])

    def test_grade_never_contradicts_the_pass_mark(self):
        for g in range(0, 101, 5):
            d = self.api(previous_g1=g, previous_g2=g).get_json()["result"]
            with self.subTest(grade=g):
                if d["predicted_score"] >= d["pass_mark"]:
                    self.assertNotEqual(d["grade"], "F")
                else:
                    self.assertEqual(d["grade"], "F")

    def test_grade_bands(self):
        for score, want in ((95, "A+"), (85, "A"), (75, "B"), (65, "C"),
                            (55, "D"), (49.9, "F"), (0, "F")):
            with self.subTest(score=score):
                self.assertEqual(predictor.grade_for(score), want)

    def test_risk_rules(self):
        self.assertEqual(predictor.risk_for(30, 0.90, 50), "High")    # below pass mark
        self.assertEqual(predictor.risk_for(80, 0.60, 50), "High")    # probability alone
        self.assertEqual(predictor.risk_for(55, 0.10, 50), "Medium")  # inside the margin
        self.assertEqual(predictor.risk_for(80, 0.30, 50), "Medium")  # probability alone
        self.assertEqual(predictor.risk_for(80, 0.05, 50), "Low")

    def test_result_shape(self):
        r = self.api().get_json()["result"]
        for key in ("predicted_score", "grade", "risk_level", "fail_probability",
                    "pass_mark", "model_used", "tips", "what_if"):
            self.assertIn(key, r)
        self.assertEqual(r["pass_mark"], 50.0)
        self.assertIn(r["model_used"], ("Random Forest", "Linear Regression"))
        self.assertGreaterEqual(len(r["tips"]), 1)

    def test_what_if_only_offers_real_gains(self):
        r = self.api().get_json()["result"]
        gains = [w["gain"] for w in r["what_if"]]
        self.assertLessEqual(len(gains), 3)
        self.assertTrue(all(g > 0 for g in gains), r["what_if"])
        self.assertEqual(gains, sorted(gains, reverse=True))

    def test_a_stronger_student_scores_higher(self):
        weak = self.api(previous_g1="30", previous_g2="30").get_json()["result"]
        strong = self.api(previous_g1="85", previous_g2="90").get_json()["result"]
        self.assertGreater(strong["predicted_score"], weak["predicted_score"])


# --------------------------------------------------------------------------- #
class TestResultPage(Base):
    def test_shows_the_prediction_and_the_inputs(self):
        body = self.form(student_name="Ayesha Khan").data.decode()
        self.assertIn("Ayesha Khan", body)
        self.assertIn("Random Forest", body)
        self.assertIn("First-period grade", body)

    def test_whole_grades_print_without_a_decimal(self):
        body = self.form(previous_g1="45").data.decode()
        self.assertIn(">45<", body)
        self.assertNotIn(">45.0<", body)

    def test_student_name_is_escaped(self):
        body = self.form(student_name="<script>alert(1)</script>").data
        self.assertNotIn(b"<script>alert(1)</script>", body)
        self.assertIn(b"&lt;script&gt;", body)

    def test_long_name_is_capped(self):
        body = self.form(student_name="Q" * 200).data
        self.assertIn(b"Q" * 60, body)
        self.assertNotIn(b"Q" * 61, body)

    def test_blank_name_falls_back(self):
        self.assertIn(b"Anonymous student", self.form(student_name="   ").data)


# --------------------------------------------------------------------------- #
class TestHistory(Base):
    def setUp(self):
        database.clear_predictions()

    def test_empty_state(self):
        self.assertIn(b"No predictions yet", self.c.get("/history").data)
        self.assertEqual(self.c.get("/api/history").get_json()["stats"]["total"], 0)

    def test_a_prediction_is_saved_and_listed(self):
        self.form(student_name="Saved student")
        rows = self.c.get("/api/history").get_json()["rows"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["student_name"], "Saved student")
        self.assertGreaterEqual(set(rows[0]),
                                {"study_time", "absences", "failures",
                                 "previous_g1", "previous_g2", "predicted_score"})
        self.assertIn(b"Saved student", self.c.get("/history").data)

    def test_stats_summarise_the_rows(self):
        self.form(previous_g1="90", previous_g2="95", absences="0")
        self.form(previous_g1="20", previous_g2="15", absences="10")
        stats = self.c.get("/api/history").get_json()["stats"]
        self.assertEqual(stats["total"], 2)
        self.assertEqual(stats["avg_absences"], 5.0)
        self.assertEqual(stats["high_risk"] + stats["medium_risk"] + stats["low_risk"], 2)

    def test_delete_one_row(self):
        self.form()
        self.form()
        rows = self.c.get("/api/history").get_json()["rows"]
        self.c.post(f"/history/{rows[0]['id']}/delete", follow_redirects=True)
        self.assertEqual(self.c.get("/api/history").get_json()["stats"]["total"], 1)

    def test_deleting_a_missing_row_is_harmless(self):
        r = self.c.post("/history/99999/delete", follow_redirects=True)
        self.assertEqual(r.status_code, 200)

    def test_clear_all(self):
        self.form()
        self.c.post("/history/clear", follow_redirects=True)
        self.assertEqual(self.c.get("/api/history").get_json()["stats"]["total"], 0)


# --------------------------------------------------------------------------- #
class TestDataset(unittest.TestCase):
    """The real UCI file is read and reshaped the way the README describes."""

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(train_model.RAW_PATH):
            raise unittest.SkipTest(f"{train_model.RAW_PATH} missing")
        cls.df = train_model.build_dataset()

    def test_row_count(self):
        self.assertEqual(len(self.df), 395)

    def test_columns(self):
        self.assertEqual(list(self.df.columns), train_model.FEATURES + ["final_score"])

    def test_grades_are_rescaled_to_percent(self):
        for col in ("previous_g1", "previous_g2", "final_score"):
            with self.subTest(col):
                self.assertLessEqual(self.df[col].max(), 100)
                self.assertGreaterEqual(self.df[col].min(), 0)
        # 0-20 grades would never exceed 20 once loaded
        self.assertGreater(self.df["final_score"].max(), 20)

    def test_values_stay_inside_the_form_ranges(self):
        for field, (lo, hi, _label, _step) in predictor.INPUT_SPEC.items():
            with self.subTest(field):
                self.assertGreaterEqual(self.df[field].min(), lo)
                self.assertLessEqual(self.df[field].max(), hi)

    def test_no_missing_values(self):
        self.assertEqual(int(self.df.isna().sum().sum()), 0)

    def test_fail_rate_is_balanced_enough_to_classify(self):
        rate = (self.df["final_score"] < train_model.PASS_MARK).mean()
        self.assertGreater(rate, 0.20)
        self.assertLess(rate, 0.50)


if __name__ == "__main__":
    unittest.main(verbosity=2)

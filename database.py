"""SQLite storage for prediction history."""

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "data", "students.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    student_name    TEXT    NOT NULL,
    study_time      REAL    NOT NULL,
    absences        REAL    NOT NULL,
    failures        REAL    NOT NULL,
    previous_g1     REAL    NOT NULL,
    previous_g2     REAL    NOT NULL,
    predicted_score REAL    NOT NULL,
    grade           TEXT    NOT NULL,
    risk_level      TEXT    NOT NULL,
    fail_probability REAL   NOT NULL,
    created_at      TEXT    NOT NULL
);
"""


@contextmanager
def get_connection():
    """Open a connection, commit on success, and always close it again.
    sqlite3's own `with conn` only ends the transaction - it leaves the
    connection open."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def init_db():
    with get_connection() as conn:
        _retire_old_table(conn)
        conn.executescript(SCHEMA)


def _retire_old_table(conn):
    """Rows saved before the switch to the real dataset used different columns
    (study_hours, attendance, ...). Keep them, but out of the way."""
    cols = {r[1] for r in conn.execute("PRAGMA table_info(predictions)")}
    if cols and "study_time" not in cols:
        conn.execute("ALTER TABLE predictions RENAME TO predictions_synthetic")
        print("Archived pre-real-dataset history as table 'predictions_synthetic'.")


def save_prediction(record):
    """Insert one prediction and return its row id."""
    with get_connection() as conn:
        cur = conn.execute(
            """
            INSERT INTO predictions (
                student_name, study_time, absences, failures,
                previous_g1, previous_g2, predicted_score, grade,
                risk_level, fail_probability, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record["student_name"],
                record["study_time"],
                record["absences"],
                record["failures"],
                record["previous_g1"],
                record["previous_g2"],
                record["predicted_score"],
                record["grade"],
                record["risk_level"],
                record["fail_probability"],
                datetime.now().strftime("%Y-%m-%d %H:%M"),
            ),
        )
        return cur.lastrowid


def get_predictions(limit=100):
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_prediction(pred_id):
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM predictions WHERE id = ?", (pred_id,)
        ).fetchone()
    return dict(row) if row else None


def delete_prediction(pred_id):
    with get_connection() as conn:
        conn.execute("DELETE FROM predictions WHERE id = ?", (pred_id,))


def clear_predictions():
    with get_connection() as conn:
        conn.execute("DELETE FROM predictions")


def get_stats():
    """Summary numbers for the dashboard."""
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT COUNT(*)                                   AS total,
                   COALESCE(AVG(predicted_score), 0)          AS avg_score,
                   COALESCE(SUM(risk_level = 'High'), 0)      AS high_risk,
                   COALESCE(SUM(risk_level = 'Medium'), 0)    AS medium_risk,
                   COALESCE(SUM(risk_level = 'Low'), 0)       AS low_risk,
                   COALESCE(AVG(study_time), 0)               AS avg_study,
                   COALESCE(AVG(absences), 0)                 AS avg_absences
            FROM predictions
            """
        ).fetchone()
    stats = dict(row)
    stats["avg_score"] = round(stats["avg_score"], 1)
    stats["avg_study"] = round(stats["avg_study"], 1)
    stats["avg_absences"] = round(stats["avg_absences"], 1)
    return stats

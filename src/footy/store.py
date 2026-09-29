"""Predictions store.

Predictions are committed before kickoff and never revised: the table has
a unique key on (season, home, away) and save_predictions raises rather
than overwrite. Scoring only fills the actual-result columns.
"""

import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from footy.config import DATA_DIR
from footy.model.scoring import actual_outcome

DB_PATH = DATA_DIR / "predictions.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    match_date TEXT NOT NULL,
    season TEXT NOT NULL,
    home TEXT NOT NULL,
    away TEXT NOT NULL,
    p_home REAL NOT NULL,
    p_draw REAL NOT NULL,
    p_away REAL NOT NULL,
    most_likely_home INTEGER NOT NULL,
    most_likely_away INTEGER NOT NULL,
    model_version TEXT NOT NULL,
    actual_home_goals INTEGER,
    actual_away_goals INTEGER,
    outcome TEXT,
    scored_at TEXT,
    UNIQUE (season, home, away)
)
"""

PREDICTION_FIELDS = [
    "match_date", "season", "home", "away", "p_home", "p_draw", "p_away",
    "most_likely_home", "most_likely_away", "model_version",
]


class PredictionExistsError(Exception):
    pass


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def connect(db_path: Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute(SCHEMA)
    return conn


def predicted_keys(db_path: Path = DB_PATH) -> set[tuple[str, str, str]]:
    """(season, home, away) of every stored prediction."""
    with closing(connect(db_path)) as conn:
        rows = conn.execute("SELECT season, home, away FROM predictions")
        return {tuple(row) for row in rows}


def save_predictions(rows: list[dict], db_path: Path = DB_PATH) -> None:
    """Insert predictions atomically. Raises PredictionExistsError, and
    saves none of the batch, if any (season, home, away) already exists."""
    created_at = utc_now()
    values = [
        (created_at, *(row[field] for field in PREDICTION_FIELDS))
        for row in rows
    ]
    columns = ", ".join(["created_at", *PREDICTION_FIELDS])
    placeholders = ", ".join("?" * (len(PREDICTION_FIELDS) + 1))
    conn = connect(db_path)
    try:
        with conn:
            conn.executemany(
                f"INSERT INTO predictions ({columns}) VALUES ({placeholders})",
                values,
            )
    except sqlite3.IntegrityError as error:
        raise PredictionExistsError(
            f"Refusing to overwrite an existing prediction: {error}"
        ) from None
    finally:
        conn.close()


def load_predictions(db_path: Path = DB_PATH) -> pd.DataFrame:
    with closing(connect(db_path)) as conn:
        return pd.read_sql_query(
            "SELECT * FROM predictions ORDER BY match_date, home", conn
        )


def score_pending(
    finished: pd.DataFrame | None = None, db_path: Path = DB_PATH
) -> int:
    """Fill actual results into unscored predictions. Returns rows scored.

    finished defaults to FINISHED matches from the API, in the shared
    schema (season, home, away, home_goals, away_goals).
    """
    if finished is None:
        from footy.data.fetch import finished_matches_frame

        finished = finished_matches_frame()
    if finished.empty:
        return 0

    scored_at = utc_now()
    updates = [
        (
            int(m.home_goals),
            int(m.away_goals),
            actual_outcome(m.home_goals, m.away_goals),
            scored_at,
            m.season,
            m.home,
            m.away,
        )
        for m in finished.itertuples()
    ]
    conn = connect(db_path)
    try:
        with conn:
            before = conn.total_changes
            conn.executemany(
                """
                UPDATE predictions
                SET actual_home_goals = ?, actual_away_goals = ?,
                    outcome = ?, scored_at = ?
                WHERE season = ? AND home = ? AND away = ?
                  AND scored_at IS NULL
                """,
                updates,
            )
            return conn.total_changes - before
    finally:
        conn.close()

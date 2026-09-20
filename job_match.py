"""Matching an analysed resume against specific job descriptions.

Each match is stored so the "job-specific resumes" list survives the session,
alongside any tailored resume generated for that posting.
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

from fastapi import HTTPException, status
from pydantic import BaseModel, Field

from auth import DATABASE_PATH


@contextmanager
def _db():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        with conn:
            yield conn
    finally:
        conn.close()


def init_db():
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS job_matches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                company TEXT NOT NULL DEFAULT '',
                title TEXT NOT NULL DEFAULT '',
                jd_text TEXT NOT NULL,
                match_score INTEGER NOT NULL DEFAULT 0,
                matching_skills TEXT NOT NULL DEFAULT '[]',
                missing_skills TEXT NOT NULL DEFAULT '[]',
                role_summary TEXT NOT NULL DEFAULT '{}',
                optimized_resume TEXT,
                optimized_score INTEGER,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_job_matches_user "
            "ON job_matches (user_id, updated_at DESC)"
        )


# --- Models -----------------------------------------------------------------


class RoleSummary(BaseModel):
    company: str = ""
    title: str = ""
    experience: str = ""
    employment_type: str = ""
    key_skills: list[str] = []
    nice_to_have: list[str] = []


class JobMatch(BaseModel):
    id: int
    company: str
    title: str
    match_score: int
    matching_skills: list[str]
    missing_skills: list[str]
    role_summary: RoleSummary
    has_optimized_resume: bool
    optimized_score: int | None
    created_at: str
    updated_at: str


class JobMatchDetail(JobMatch):
    jd_text: str
    optimized_resume: str | None


class JobMatchRequest(BaseModel):
    job_description: str
    # Typed by the user. Takes precedence over whatever the model reads out of
    # the posting, which often cannot find a company name at all.
    company: str = ""
    title: str = ""


class JobMatchPatch(BaseModel):
    company: str | None = Field(default=None, max_length=120)
    title: str | None = Field(default=None, max_length=160)


# --- Helpers ----------------------------------------------------------------


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _loads(raw: str, fallback):
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return fallback


def _to_match(row: sqlite3.Row) -> JobMatch:
    return JobMatch(
        id=row["id"],
        company=row["company"],
        title=row["title"],
        match_score=row["match_score"],
        matching_skills=_loads(row["matching_skills"], []),
        missing_skills=_loads(row["missing_skills"], []),
        role_summary=RoleSummary(**_loads(row["role_summary"], {})),
        has_optimized_resume=bool(row["optimized_resume"]),
        optimized_score=row["optimized_score"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _fetch(conn: sqlite3.Connection, user_id: int, match_id: int) -> sqlite3.Row:
    row = conn.execute(
        "SELECT * FROM job_matches WHERE id = ? AND user_id = ?", (match_id, user_id)
    ).fetchone()
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Job match not found.")
    return row


def create(
    user_id: int,
    jd_text: str,
    match_score: int,
    matching_skills: list[str],
    missing_skills: list[str],
    role_summary: dict,
) -> JobMatch:
    now = _now()
    with _db() as conn:
        cursor = conn.execute(
            """
            INSERT INTO job_matches
                (user_id, company, title, jd_text, match_score, matching_skills,
                 missing_skills, role_summary, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                role_summary.get("company", ""),
                role_summary.get("title", ""),
                jd_text,
                match_score,
                json.dumps(matching_skills),
                json.dumps(missing_skills),
                json.dumps(role_summary),
                now,
                now,
            ),
        )
        row = _fetch(conn, user_id, cursor.lastrowid)
    return _to_match(row)


def list_matches(user_id: int) -> list[JobMatch]:
    with _db() as conn:
        rows = conn.execute(
            "SELECT * FROM job_matches WHERE user_id = ? ORDER BY updated_at DESC",
            (user_id,),
        ).fetchall()
    return [_to_match(row) for row in rows]


def get_detail(user_id: int, match_id: int) -> JobMatchDetail:
    with _db() as conn:
        row = _fetch(conn, user_id, match_id)
    return JobMatchDetail(
        **_to_match(row).model_dump(),
        jd_text=row["jd_text"],
        optimized_resume=row["optimized_resume"],
    )


def store_optimized(user_id: int, match_id: int, resume_text: str, score: int | None) -> JobMatch:
    with _db() as conn:
        _fetch(conn, user_id, match_id)
        conn.execute(
            "UPDATE job_matches SET optimized_resume = ?, optimized_score = ?, "
            "updated_at = ? WHERE id = ? AND user_id = ?",
            (resume_text, score, _now(), match_id, user_id),
        )
        row = _fetch(conn, user_id, match_id)
    return _to_match(row)


def rename(user_id: int, match_id: int, company: str | None, title: str | None) -> JobMatch:
    fields: dict = {}
    if company is not None:
        fields["company"] = company.strip()
    if title is not None:
        fields["title"] = title.strip()
    if not fields:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Nothing to update.")

    fields["updated_at"] = _now()
    assignments = ", ".join(f"{name} = ?" for name in fields)

    with _db() as conn:
        row = _fetch(conn, user_id, match_id)
        # Keep the stored summary in step with the edited values.
        summary = _loads(row["role_summary"], {})
        if company is not None:
            summary["company"] = fields["company"]
        if title is not None:
            summary["title"] = fields["title"]

        conn.execute(
            f"UPDATE job_matches SET {assignments}, role_summary = ? WHERE id = ? AND user_id = ?",
            (*fields.values(), json.dumps(summary), match_id, user_id),
        )
        row = _fetch(conn, user_id, match_id)
    return _to_match(row)


def delete(user_id: int, match_id: int) -> None:
    with _db() as conn:
        _fetch(conn, user_id, match_id)
        conn.execute(
            "DELETE FROM job_matches WHERE id = ? AND user_id = ?", (match_id, user_id)
        )

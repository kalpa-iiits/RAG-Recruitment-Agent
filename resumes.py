"""Persistent storage for analysed resumes.

The session keeps one analysis in memory and forgets it on restart, so the
Saved Resumes page needs its own table. Mirrors auth.py's SQLite layout and
lives in the same database file.
"""

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from auth import DATABASE_PATH, User, get_current_user

logger = logging.getLogger(__name__)

# How many skills are shown as tags on a saved-resume card.
TAG_LIMIT = 8


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
            CREATE TABLE IF NOT EXISTS resumes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                filename TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT '',
                resume_text TEXT NOT NULL,
                analysis_json TEXT,
                overall_score INTEGER,
                favourite INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_resumes_user ON resumes (user_id, updated_at DESC)"
        )


# --- Models -----------------------------------------------------------------


class SavedResume(BaseModel):
    id: int
    filename: str
    role: str
    overall_score: int | None
    selected: bool
    favourite: bool
    tags: list[str]
    skill_count: int
    created_at: str
    updated_at: str


class SavedResumeDetail(SavedResume):
    resume_text: str
    analysis_result: dict | None


class ResumePatch(BaseModel):
    filename: str | None = Field(default=None, min_length=1, max_length=200)
    role: str | None = Field(default=None, max_length=120)
    favourite: bool | None = None


# --- Helpers ----------------------------------------------------------------


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tags(analysis: dict | None) -> list[str]:
    """Strongest skills first — what the card shows as chips."""
    if not analysis:
        return []
    scores: dict = analysis.get("skill_scores") or {}
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [name for name, _ in ordered[:TAG_LIMIT]]


def _to_summary(row: sqlite3.Row) -> SavedResume:
    analysis = json.loads(row["analysis_json"]) if row["analysis_json"] else None
    return SavedResume(
        id=row["id"],
        filename=row["filename"],
        role=row["role"],
        overall_score=row["overall_score"],
        selected=bool((analysis or {}).get("selected")),
        favourite=bool(row["favourite"]),
        tags=_tags(analysis),
        skill_count=len((analysis or {}).get("skill_scores") or {}),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def save_analysis(user_id: int, filename: str, role: str, resume_text: str, analysis: dict) -> int:
    """Store a freshly analysed resume. Called after every successful analyze."""
    now = _now()
    with _db() as conn:
        cursor = conn.execute(
            """
            INSERT INTO resumes
                (user_id, filename, role, resume_text, analysis_json,
                 overall_score, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                filename or "resume.pdf",
                role or "",
                resume_text or "",
                json.dumps(analysis),
                analysis.get("overall_score"),
                now,
                now,
            ),
        )
    return cursor.lastrowid


def _fetch(conn: sqlite3.Connection, user_id: int, resume_id: int) -> sqlite3.Row:
    row = conn.execute(
        "SELECT * FROM resumes WHERE id = ? AND user_id = ?", (resume_id, user_id)
    ).fetchone()
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Saved resume not found.")
    return row


def get_detail(user_id: int, resume_id: int) -> SavedResumeDetail:
    with _db() as conn:
        row = _fetch(conn, user_id, resume_id)
    analysis = json.loads(row["analysis_json"]) if row["analysis_json"] else None
    return SavedResumeDetail(
        **_to_summary(row).model_dump(),
        resume_text=row["resume_text"],
        analysis_result=analysis,
    )


# --- Routes -----------------------------------------------------------------

router = APIRouter(prefix="/api/resumes", tags=["resumes"])


@router.get("", response_model=list[SavedResume])
def list_resumes(user: User = Depends(get_current_user)):
    """Favourites first, then most recently updated."""
    with _db() as conn:
        rows = conn.execute(
            "SELECT * FROM resumes WHERE user_id = ? ORDER BY favourite DESC, updated_at DESC",
            (user.id,),
        ).fetchall()
    return [_to_summary(row) for row in rows]


@router.get("/{resume_id}", response_model=SavedResumeDetail)
def read_resume(resume_id: int, user: User = Depends(get_current_user)):
    return get_detail(user.id, resume_id)


@router.patch("/{resume_id}", response_model=SavedResume)
def update_resume(
    resume_id: int, body: ResumePatch, user: User = Depends(get_current_user)
):
    """Rename, retag the role, or toggle the favourite star."""
    fields: dict = {}
    if body.filename is not None:
        fields["filename"] = body.filename.strip()
    if body.role is not None:
        fields["role"] = body.role.strip()
    if body.favourite is not None:
        fields["favourite"] = int(body.favourite)

    if not fields:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Nothing to update.")

    fields["updated_at"] = _now()
    assignments = ", ".join(f"{name} = ?" for name in fields)

    with _db() as conn:
        _fetch(conn, user.id, resume_id)
        conn.execute(
            f"UPDATE resumes SET {assignments} WHERE id = ? AND user_id = ?",
            (*fields.values(), resume_id, user.id),
        )
        row = _fetch(conn, user.id, resume_id)
    return _to_summary(row)


@router.delete("/{resume_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_resume(resume_id: int, user: User = Depends(get_current_user)):
    with _db() as conn:
        _fetch(conn, user.id, resume_id)
        conn.execute(
            "DELETE FROM resumes WHERE id = ? AND user_id = ?", (resume_id, user.id)
        )

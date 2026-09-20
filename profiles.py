"""Account profile, preferences and account management.

Kept in its own table rather than added to `users` so existing databases
don't need a migration — auth.py creates `users` with CREATE TABLE IF NOT
EXISTS, which would silently skip new columns.
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

import resumes as resume_store
from auth import DATABASE_PATH, User, get_current_user, password_hash

TEXT_FIELDS = ("full_name", "email", "headline", "location", "linkedin", "website", "about")

DEFAULT_PREFERENCES: dict = {
    "email_analysis_complete": True,
    "email_weekly_tips": False,
    "email_product_updates": False,
    "default_role": "",
    "default_difficulty": "Medium",
    "default_question_count": 5,
}


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
            CREATE TABLE IF NOT EXISTS profiles (
                user_id INTEGER PRIMARY KEY,
                full_name TEXT NOT NULL DEFAULT '',
                email TEXT NOT NULL DEFAULT '',
                headline TEXT NOT NULL DEFAULT '',
                location TEXT NOT NULL DEFAULT '',
                linkedin TEXT NOT NULL DEFAULT '',
                website TEXT NOT NULL DEFAULT '',
                about TEXT NOT NULL DEFAULT '',
                preferences TEXT NOT NULL DEFAULT '{}',
                updated_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            )
            """
        )


# --- Models -----------------------------------------------------------------


class Profile(BaseModel):
    username: str
    member_since: str
    full_name: str = ""
    email: str = ""
    headline: str = ""
    location: str = ""
    linkedin: str = ""
    website: str = ""
    about: str = ""
    preferences: dict = Field(default_factory=dict)
    saved_resume_count: int = 0


class ProfilePatch(BaseModel):
    full_name: str | None = Field(default=None, max_length=120)
    email: str | None = Field(default=None, max_length=200)
    headline: str | None = Field(default=None, max_length=200)
    location: str | None = Field(default=None, max_length=120)
    linkedin: str | None = Field(default=None, max_length=300)
    website: str | None = Field(default=None, max_length=300)
    about: str | None = Field(default=None, max_length=2000)
    preferences: dict | None = None


class PasswordChange(BaseModel):
    current_password: str = Field(min_length=1)
    new_password: str = Field(min_length=8, max_length=128)


class AccountDelete(BaseModel):
    """Deleting an account is irreversible, so the password is required."""

    password: str = Field(min_length=1)


# --- Helpers ----------------------------------------------------------------


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row(conn: sqlite3.Connection, user_id: int) -> sqlite3.Row:
    row = conn.execute("SELECT * FROM profiles WHERE user_id = ?", (user_id,)).fetchone()
    if row is None:
        conn.execute(
            "INSERT INTO profiles (user_id, preferences, updated_at) VALUES (?, ?, ?)",
            (user_id, json.dumps(DEFAULT_PREFERENCES), _now()),
        )
        row = conn.execute(
            "SELECT * FROM profiles WHERE user_id = ?", (user_id,)
        ).fetchone()
    return row


def _to_profile(user: User, row: sqlite3.Row, resume_count: int) -> Profile:
    try:
        stored = json.loads(row["preferences"])
    except (json.JSONDecodeError, TypeError):
        stored = {}
    return Profile(
        username=user.username,
        member_since=user.created_at,
        **{field: row[field] for field in TEXT_FIELDS},
        preferences={**DEFAULT_PREFERENCES, **stored},
        saved_resume_count=resume_count,
    )


def _resume_count(conn: sqlite3.Connection, user_id: int) -> int:
    return conn.execute(
        "SELECT COUNT(*) FROM resumes WHERE user_id = ?", (user_id,)
    ).fetchone()[0]


# --- Routes -----------------------------------------------------------------

router = APIRouter(prefix="/api/profile", tags=["profile"])


@router.get("", response_model=Profile)
def read_profile(user: User = Depends(get_current_user)):
    with _db() as conn:
        row = _row(conn, user.id)
        return _to_profile(user, row, _resume_count(conn, user.id))


@router.patch("", response_model=Profile)
def update_profile(body: ProfilePatch, user: User = Depends(get_current_user)):
    with _db() as conn:
        row = _row(conn, user.id)

        fields: dict = {
            name: (getattr(body, name) or "").strip()
            for name in TEXT_FIELDS
            if getattr(body, name) is not None
        }

        if body.preferences is not None:
            try:
                current = json.loads(row["preferences"])
            except (json.JSONDecodeError, TypeError):
                current = {}
            # Merge, and only keep keys the app knows about.
            merged = {**DEFAULT_PREFERENCES, **current, **body.preferences}
            fields["preferences"] = json.dumps(
                {key: merged[key] for key in DEFAULT_PREFERENCES}
            )

        if not fields:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Nothing to update.")

        fields["updated_at"] = _now()
        assignments = ", ".join(f"{name} = ?" for name in fields)
        conn.execute(
            f"UPDATE profiles SET {assignments} WHERE user_id = ?",
            (*fields.values(), user.id),
        )
        return _to_profile(user, _row(conn, user.id), _resume_count(conn, user.id))


@router.post("/password", status_code=status.HTTP_204_NO_CONTENT)
def change_password(body: PasswordChange, user: User = Depends(get_current_user)):
    with _db() as conn:
        row = conn.execute(
            "SELECT password_hash FROM users WHERE id = ?", (user.id,)
        ).fetchone()
        if row is None or not password_hash.verify(
            body.current_password, row["password_hash"]
        ):
            raise HTTPException(
                status.HTTP_401_UNAUTHORIZED, "Current password is incorrect."
            )
        conn.execute(
            "UPDATE users SET password_hash = ? WHERE id = ?",
            (password_hash.hash(body.new_password), user.id),
        )


@router.get("/export")
def export_account(user: User = Depends(get_current_user)):
    """Everything stored about this account, as JSON."""
    with _db() as conn:
        profile = _to_profile(user, _row(conn, user.id), _resume_count(conn, user.id))

    saved = [
        resume_store.get_detail(user.id, row.id).model_dump()
        for row in resume_store.list_resumes(user)
    ]
    return {
        "exported_at": _now(),
        "account": {
            "username": user.username,
            "created_at": user.created_at,
        },
        "profile": profile.model_dump(),
        "saved_resumes": saved,
    }


@router.post("/delete", status_code=status.HTTP_204_NO_CONTENT)
def delete_account(body: AccountDelete, user: User = Depends(get_current_user)):
    """Irreversible. Cascades to profile and saved resumes."""
    with _db() as conn:
        row = conn.execute(
            "SELECT password_hash FROM users WHERE id = ?", (user.id,)
        ).fetchone()
        if row is None or not password_hash.verify(body.password, row["password_hash"]):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Password is incorrect.")
        conn.execute("DELETE FROM users WHERE id = ?", (user.id,))

"""Account profile, preferences and account management.

Kept in its own table rather than added to `users` so the two concerns stay
separable: auth owns credentials, this owns everything the user can edit
about themselves. Preferences are a JSON column, merged against
DEFAULT_PREFERENCES on the way in and out so an older row never misses a key.
"""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

import db
import resumes as resume_store
import storage
from auth import User, get_current_user, password_hash
from db import session_scope

# The account's email is its login identity and lives on the user row, so
# it is not among the free-text fields the profile form can edit.
TEXT_FIELDS = ("full_name", "headline", "location", "linkedin", "website", "about")

DEFAULT_PREFERENCES: dict = {
    "email_analysis_complete": True,
    "email_weekly_tips": False,
    "email_product_updates": False,
    "default_role": "",
    "default_difficulty": "Medium",
    "default_question_count": 5,
}


# --- Models -----------------------------------------------------------------


class Profile(BaseModel):
    # The address this account signs in with; read-only here.
    email: str
    member_since: str
    full_name: str = ""
    headline: str = ""
    location: str = ""
    linkedin: str = ""
    website: str = ""
    about: str = ""
    preferences: dict = Field(default_factory=dict)
    saved_resume_count: int = 0


class ProfilePatch(BaseModel):
    full_name: str | None = Field(default=None, max_length=120)
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


def _row(session, user_id: int) -> db.Profile:
    """The user's profile row, created on first read."""
    row = session.get(db.Profile, user_id)
    if row is None:
        row = db.Profile(
            user_id=user_id,
            preferences=dict(DEFAULT_PREFERENCES),
            updated_at=db.utcnow(),
        )
        session.add(row)
        session.flush()
    return row


def _to_profile(user: User, row: db.Profile, resume_count: int) -> Profile:
    stored = row.preferences if isinstance(row.preferences, dict) else {}
    return Profile(
        email=user.email,
        member_since=user.created_at,
        **{field: getattr(row, field) for field in TEXT_FIELDS},
        preferences={**DEFAULT_PREFERENCES, **stored},
        saved_resume_count=resume_count,
    )


def _require_account(session, user_id: int) -> db.User:
    row = session.get(db.User, user_id)
    if row is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Account no longer exists.")
    return row


# --- Routes -----------------------------------------------------------------

router = APIRouter(prefix="/api/profile", tags=["profile"])


@router.get("", response_model=Profile)
def read_profile(user: User = Depends(get_current_user)):
    with session_scope() as session:
        row = _row(session, user.id)
        return _to_profile(user, row, resume_store.count_for_user(session, user.id))


@router.patch("", response_model=Profile)
def update_profile(body: ProfilePatch, user: User = Depends(get_current_user)):
    with session_scope() as session:
        row = _row(session, user.id)

        changed = False
        for name in TEXT_FIELDS:
            value = getattr(body, name)
            if value is not None:
                setattr(row, name, value.strip())
                changed = True

        if body.preferences is not None:
            current = row.preferences if isinstance(row.preferences, dict) else {}
            # Merge, and only keep keys the app knows about.
            merged = {**DEFAULT_PREFERENCES, **current, **body.preferences}
            row.preferences = {key: merged[key] for key in DEFAULT_PREFERENCES}
            changed = True

        if not changed:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Nothing to update.")

        row.updated_at = db.utcnow()
        session.flush()
        return _to_profile(user, row, resume_store.count_for_user(session, user.id))


@router.post("/password", status_code=status.HTTP_204_NO_CONTENT)
def change_password(body: PasswordChange, user: User = Depends(get_current_user)):
    with session_scope() as session:
        account = _require_account(session, user.id)
        if not password_hash.verify(body.current_password, account.password_hash):
            raise HTTPException(
                status.HTTP_401_UNAUTHORIZED, "Current password is incorrect."
            )
        account.password_hash = password_hash.hash(body.new_password)


@router.get("/export")
def export_account(user: User = Depends(get_current_user)):
    """Everything stored about this account, as JSON."""
    with session_scope() as session:
        profile = _to_profile(
            user, _row(session, user.id), resume_store.count_for_user(session, user.id)
        )

    saved = [
        resume_store.get_detail(user.id, resume_id).model_dump()
        for resume_id in resume_store.ids_for_user(user.id)
    ]
    return {
        "exported_at": _now(),
        "account": {
            "email": user.email,
            "created_at": user.created_at,
        },
        "profile": profile.model_dump(),
        "saved_resumes": saved,
    }


@router.post("/delete", status_code=status.HTTP_204_NO_CONTENT)
def delete_account(body: AccountDelete, user: User = Depends(get_current_user)):
    """Irreversible. Cascades to profile, saved resumes and job matches."""
    with session_scope() as session:
        account = _require_account(session, user.id)
        if not password_hash.verify(body.password, account.password_hash):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Password is incorrect.")
        session.delete(account)

    # The database cascade cannot reach S3, so the stored PDFs would otherwise
    # outlive the account that owns them.
    storage.delete_user_files(user.id)

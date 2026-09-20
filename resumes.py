"""Persistent storage for analysed resumes.

The session keeps one analysis in memory and forgets it on restart, so the
Saved Resumes page needs its own table. The analysis itself lands in a JSON
column (JSONB on Postgres) because its shape follows the prompts in
agents.py; only the fields the list view sorts and filters on are columns.

The original PDF goes to S3 (see storage.py) and the row keeps its key. The
`resume_url` the API returns is presigned per request, so the bucket stays
private and a link copied out of a response cannot be shared indefinitely.
"""

import logging
from collections.abc import Sequence

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
from sqlalchemy import func, select

import db
import pagination
import storage
from auth import User, get_current_user
from db import session_scope

logger = logging.getLogger(__name__)

# How many skills are shown as tags on a saved-resume card.
TAG_LIMIT = 8


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
    # Time-limited download links, or None when nothing was stored.
    resume_url: str | None
    jd_url: str | None
    improved_url: str | None
    created_at: str
    updated_at: str


class SavedResumeDetail(SavedResume):
    resume_text: str
    analysis_result: dict | None
    improved_text: str | None


class ResumePatch(BaseModel):
    filename: str | None = Field(default=None, min_length=1, max_length=200)
    role: str | None = Field(default=None, max_length=120)
    favourite: bool | None = None


# --- Helpers ----------------------------------------------------------------


def tags_for(analysis: dict | None) -> list[str]:
    """Strongest skills first — what the card shows as chips."""
    if not analysis:
        return []
    scores: dict = analysis.get("skill_scores") or {}
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [name for name, _ in ordered[:TAG_LIMIT]]


def _link(key: str | None) -> str | None:
    return storage.presigned_url(key) if key else None


def _to_summary(row: db.Resume) -> SavedResume:
    analysis = row.analysis or {}
    return SavedResume(
        id=row.id,
        filename=row.filename,
        role=row.role,
        overall_score=row.overall_score,
        selected=bool(analysis.get("selected")),
        favourite=bool(row.favourite),
        tags=tags_for(row.analysis),
        skill_count=len(analysis.get("skill_scores") or {}),
        resume_url=_link(row.resume_key),
        jd_url=_link(row.jd_key),
        improved_url=_link(row.improved_key),
        created_at=db.iso(row.created_at),
        updated_at=db.iso(row.updated_at),
    )


def _fetch(session, user_id: int, resume_id: int) -> db.Resume:
    row = session.scalar(
        select(db.Resume).where(db.Resume.id == resume_id, db.Resume.user_id == user_id)
    )
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Saved resume not found.")
    return row


# --- Storage ----------------------------------------------------------------


def save_analysis(
    user_id: int,
    filename: str,
    role: str,
    resume_text: str,
    analysis: dict,
    stored_file: storage.StoredFile | None = None,
    jd_file: storage.StoredFile | None = None,
) -> int:
    """Store a freshly analysed resume. Called after every successful analyze."""
    now = db.utcnow()
    with session_scope() as session:
        row = db.Resume(
            user_id=user_id,
            filename=filename or "resume.pdf",
            role=role or "",
            resume_text=resume_text or "",
            analysis=analysis,
            overall_score=analysis.get("overall_score"),
            resume_key=stored_file.key if stored_file else None,
            resume_url=stored_file.url if stored_file else None,
            jd_key=jd_file.key if jd_file else None,
            jd_url=jd_file.url if jd_file else None,
            created_at=now,
            updated_at=now,
        )
        session.add(row)
        session.flush()
        return row.id


def attach_improved(
    user_id: int,
    resume_id: int,
    improved_text: str,
    stored_file: storage.StoredFile | None,
) -> None:
    """Record a rewrite against the saved resume it came from.

    Replaces any earlier rewrite, deleting the superseded PDF so regenerating
    does not leave a trail of orphaned objects in the bucket.
    """
    with session_scope() as session:
        row = _fetch(session, user_id, resume_id)
        stale_key = row.improved_key if stored_file else None

        row.improved_text = improved_text
        if stored_file:
            row.improved_key = stored_file.key
            row.improved_url = stored_file.url
        row.updated_at = db.utcnow()

    if stale_key and stored_file and stale_key != stored_file.key:
        storage.delete(stale_key)


def get_detail(user_id: int, resume_id: int) -> SavedResumeDetail:
    with session_scope() as session:
        row = _fetch(session, user_id, resume_id)
        return SavedResumeDetail(
            **_to_summary(row).model_dump(),
            resume_text=row.resume_text,
            analysis_result=row.analysis,
            improved_text=row.improved_text,
        )


def latest_id(user_id: int) -> int | None:
    """The user's most recently analysed resume.

    Used when the in-memory session has forgotten which row it is working on
    — it holds that id only until the server restarts or the session times
    out, and a rewrite generated afterwards would otherwise be stored nowhere.
    """
    with session_scope() as session:
        return session.scalar(
            select(db.Resume.id)
            .where(db.Resume.user_id == user_id)
            .order_by(db.Resume.id.desc())
            .limit(1)
        )


def summaries_for(session, user_id: int, ids: Sequence[int]) -> dict[int, SavedResume]:
    """Full models for a handful of ids, keyed by id.

    The caller owns the ordering; this only hydrates. Scoped to the user
    again so a stray id from elsewhere cannot widen what a caller sees.
    """
    if not ids:
        return {}
    rows = session.scalars(
        select(db.Resume).where(db.Resume.id.in_(ids), db.Resume.user_id == user_id)
    ).all()
    return {row.id: _to_summary(row) for row in rows}


def _hits(row, needle: str) -> bool:
    """Filename, role and skill tags — what the search box has always matched."""
    haystack = [row.filename, row.role, *tags_for(row.analysis)]
    return any(needle in (text or "").lower() for text in haystack)


def list_page(
    user_id: int,
    *,
    limit: int = pagination.DEFAULT_LIMIT,
    offset: int = 0,
    q: str = "",
) -> pagination.Page[SavedResume]:
    """One page of saved resumes: favourites first, then most recently updated.

    Two passes on purpose. The first reads only the columns the filter needs,
    leaving the extracted resume text — the largest column on the table — in
    the database; the analysis JSON joins it only when there is something to
    search its skill tags for. The second hydrates just the rows that made
    the page, which is where the cost is: each row presigns up to three S3
    links, so that now happens `limit` times per request rather than once per
    resume the user owns.
    """
    needle = q.strip().lower()
    with session_scope() as session:
        columns = [db.Resume.id, db.Resume.filename, db.Resume.role]
        if needle:
            columns.append(db.Resume.analysis)

        # id breaks ties so a row cannot drift between pages on equal stamps.
        rows = session.execute(
            select(*columns)
            .where(db.Resume.user_id == user_id)
            .order_by(
                db.Resume.favourite.desc(),
                db.Resume.updated_at.desc(),
                db.Resume.id.desc(),
            )
        ).all()

        if needle:
            rows = [row for row in rows if _hits(row, needle)]

        page_ids = [row.id for row in rows[offset : offset + limit]]
        by_id = summaries_for(session, user_id, page_ids)
        return pagination.Page[SavedResume](
            items=[by_id[rid] for rid in page_ids if rid in by_id],
            total=len(rows),
            limit=limit,
            offset=offset,
        )


def ids_for_user(user_id: int) -> list[int]:
    """Every saved resume id, newest first.

    For the account export, which wants the whole library rather than a page
    of it — so it does not go through the paginated listing.
    """
    with session_scope() as session:
        return list(
            session.scalars(
                select(db.Resume.id)
                .where(db.Resume.user_id == user_id)
                .order_by(db.Resume.updated_at.desc(), db.Resume.id.desc())
            ).all()
        )


def count_for_user(session, user_id: int) -> int:
    return session.scalar(
        select(func.count()).select_from(db.Resume).where(db.Resume.user_id == user_id)
    )


# --- Routes -----------------------------------------------------------------

router = APIRouter(prefix="/api/resumes", tags=["resumes"])


@router.get("", response_model=pagination.Page[SavedResume])
def list_resumes(
    user: User = Depends(get_current_user),
    limit: int = Query(pagination.DEFAULT_LIMIT, ge=1, le=pagination.MAX_LIMIT),
    offset: int = Query(0, ge=0),
    q: str = Query("", max_length=200, description="Filename, role or skill tag."),
):
    """One page of saved resumes: favourites first, then most recently updated."""
    return list_page(user.id, limit=limit, offset=offset, q=q)


@router.get("/{resume_id}", response_model=SavedResumeDetail)
def read_resume(resume_id: int, user: User = Depends(get_current_user)):
    return get_detail(user.id, resume_id)


@router.patch("/{resume_id}", response_model=SavedResume)
def update_resume(
    resume_id: int, body: ResumePatch, user: User = Depends(get_current_user)
):
    """Rename, retag the role, or toggle the favourite star."""
    if body.filename is None and body.role is None and body.favourite is None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Nothing to update.")

    with session_scope() as session:
        row = _fetch(session, user.id, resume_id)
        if body.filename is not None:
            row.filename = body.filename.strip()
        if body.role is not None:
            row.role = body.role.strip()
        if body.favourite is not None:
            row.favourite = body.favourite
        row.updated_at = db.utcnow()
        session.flush()
        return _to_summary(row)


# Which stored file a download refers to.
_DOWNLOAD_KINDS = {
    "original": "resume_key",
    "jd": "jd_key",
    "improved": "improved_key",
}


@router.get("/{resume_id}/download")
def download_resume(
    resume_id: int,
    kind: str = "original",
    user: User = Depends(get_current_user),
):
    """Redirect to a freshly signed link for one of this resume's PDFs."""
    column = _DOWNLOAD_KINDS.get(kind)
    if column is None:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"kind must be one of: {', '.join(_DOWNLOAD_KINDS)}.",
        )

    with session_scope() as session:
        key = getattr(_fetch(session, user.id, resume_id), column)

    link = _link(key)
    if not link:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, f"No stored {kind} file for this resume."
        )
    return RedirectResponse(link, status_code=status.HTTP_307_TEMPORARY_REDIRECT)


@router.delete("/{resume_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_resume(resume_id: int, user: User = Depends(get_current_user)):
    with session_scope() as session:
        row = _fetch(session, user.id, resume_id)
        keys = [row.resume_key, row.jd_key, row.improved_key]
        session.delete(row)
    # After the row is gone: a failed S3 delete leaves an orphan object, which
    # is recoverable, while the reverse would leave a row pointing at nothing.
    for key in filter(None, keys):
        storage.delete(key)

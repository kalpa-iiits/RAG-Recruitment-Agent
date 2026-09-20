"""Matching an analysed resume against specific job descriptions.

Each match is stored so the "job-specific resumes" list survives the session,
alongside any tailored resume generated for that posting. The skill lists and
the parsed role summary are JSON columns rather than encoded strings.
"""

from collections.abc import Sequence

from fastapi import HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select

import db
import pagination
import storage
from db import session_scope


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
    # Time-limited link to the tailored resume PDF.
    optimized_url: str | None
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


def _to_match(row: db.JobMatch) -> JobMatch:
    summary = row.role_summary if isinstance(row.role_summary, dict) else {}
    return JobMatch(
        id=row.id,
        company=row.company,
        title=row.title,
        match_score=row.match_score,
        matching_skills=row.matching_skills or [],
        missing_skills=row.missing_skills or [],
        role_summary=RoleSummary(**summary),
        has_optimized_resume=bool(row.optimized_resume),
        optimized_score=row.optimized_score,
        optimized_url=(
            storage.presigned_url(row.optimized_key) if row.optimized_key else None
        ),
        created_at=db.iso(row.created_at),
        updated_at=db.iso(row.updated_at),
    )


def _fetch(session, user_id: int, match_id: int) -> db.JobMatch:
    row = session.scalar(
        select(db.JobMatch).where(
            db.JobMatch.id == match_id, db.JobMatch.user_id == user_id
        )
    )
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Job match not found.")
    return row


# --- Storage ----------------------------------------------------------------


def create(
    user_id: int,
    jd_text: str,
    match_score: int,
    matching_skills: list[str],
    missing_skills: list[str],
    role_summary: dict,
) -> JobMatch:
    now = db.utcnow()
    with session_scope() as session:
        row = db.JobMatch(
            user_id=user_id,
            company=role_summary.get("company", ""),
            title=role_summary.get("title", ""),
            jd_text=jd_text,
            match_score=match_score,
            matching_skills=list(matching_skills),
            missing_skills=list(missing_skills),
            role_summary=role_summary,
            created_at=now,
            updated_at=now,
        )
        session.add(row)
        session.flush()
        return _to_match(row)


def summaries_for(session, user_id: int, ids: Sequence[int]) -> dict[int, JobMatch]:
    """Full models for a handful of ids, keyed by id.

    The caller owns the ordering; this only hydrates. Scoped to the user
    again so a stray id from elsewhere cannot widen what a caller sees.
    """
    if not ids:
        return {}
    rows = session.scalars(
        select(db.JobMatch).where(
            db.JobMatch.id.in_(ids), db.JobMatch.user_id == user_id
        )
    ).all()
    return {row.id: _to_match(row) for row in rows}


def _hits(row, needle: str) -> bool:
    """Same fields the Job Match search box has always matched on."""
    summary = row.role_summary if isinstance(row.role_summary, dict) else {}
    haystack = [row.company, row.title, *(summary.get("key_skills") or [])]
    return any(needle in (text or "").lower() for text in haystack)


def list_matches(
    user_id: int,
    *,
    limit: int = pagination.DEFAULT_LIMIT,
    offset: int = 0,
    optimized_only: bool = False,
    q: str = "",
) -> pagination.Page[JobMatch]:
    """One page of a user's job matches, most recently updated first.

    Two passes on purpose. The first reads only the columns the filter needs,
    so the job descriptions and tailored resumes — by far the largest columns
    on the table — never leave the database. The second hydrates just the
    rows that made the page, which is where the cost is: every stored PDF on
    a row gets a presigned link, so that now happens `limit` times per
    request instead of once per match the user owns.
    """
    needle = q.strip().lower()
    with session_scope() as session:
        stmt = select(
            db.JobMatch.id,
            db.JobMatch.company,
            db.JobMatch.title,
            db.JobMatch.role_summary,
        ).where(db.JobMatch.user_id == user_id)

        if optimized_only:
            # Mirrors has_optimized_resume, which treats an empty string as
            # "nothing generated yet".
            stmt = stmt.where(
                db.JobMatch.optimized_resume.is_not(None),
                db.JobMatch.optimized_resume != "",
            )

        # id breaks ties so a row cannot drift between pages on equal stamps.
        rows = session.execute(
            stmt.order_by(db.JobMatch.updated_at.desc(), db.JobMatch.id.desc())
        ).all()

        if needle:
            rows = [row for row in rows if _hits(row, needle)]

        page_ids = [row.id for row in rows[offset : offset + limit]]
        by_id = summaries_for(session, user_id, page_ids)
        return pagination.Page[JobMatch](
            items=[by_id[match_id] for match_id in page_ids if match_id in by_id],
            total=len(rows),
            limit=limit,
            offset=offset,
        )


def get_detail(user_id: int, match_id: int) -> JobMatchDetail:
    with session_scope() as session:
        row = _fetch(session, user_id, match_id)
        return JobMatchDetail(
            **_to_match(row).model_dump(),
            jd_text=row.jd_text,
            optimized_resume=row.optimized_resume,
        )


def store_optimized(
    user_id: int,
    match_id: int,
    resume_text: str,
    score: int | None,
    stored_file: storage.StoredFile | None = None,
) -> JobMatch:
    """Save the tailored resume, replacing any earlier one for this posting."""
    with session_scope() as session:
        row = _fetch(session, user_id, match_id)
        stale_key = row.optimized_key if stored_file else None

        row.optimized_resume = resume_text
        row.optimized_score = score
        if stored_file:
            row.optimized_key = stored_file.key
            row.optimized_url = stored_file.url
        row.updated_at = db.utcnow()
        session.flush()
        match = _to_match(row)

    if stale_key and stored_file and stale_key != stored_file.key:
        storage.delete(stale_key)
    return match


def rename(user_id: int, match_id: int, company: str | None, title: str | None) -> JobMatch:
    if company is None and title is None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Nothing to update.")

    with session_scope() as session:
        row = _fetch(session, user_id, match_id)
        # Keep the stored summary in step with the edited values.
        summary = dict(row.role_summary) if isinstance(row.role_summary, dict) else {}

        if company is not None:
            row.company = company.strip()
            summary["company"] = row.company
        if title is not None:
            row.title = title.strip()
            summary["title"] = row.title

        row.role_summary = summary
        row.updated_at = db.utcnow()
        session.flush()
        return _to_match(row)


def delete(user_id: int, match_id: int) -> None:
    with session_scope() as session:
        row = _fetch(session, user_id, match_id)
        key = row.optimized_key
        session.delete(row)
    if key:
        storage.delete(key)

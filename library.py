"""The Saved Resumes feed: analysed resumes and tailored ones in one list.

The page interleaves both kinds under a single sort, so neither table can
paginate on its own — a page of the merged list is not a page of either
source, and two independently paged endpoints cannot be zipped back together
without re-reading both in full. The union is built here instead.

Two passes, as elsewhere. The first reads only the small columns the filters
and the sort need from each table, leaving the resume text, the job
descriptions and the generated resumes in the database; the JSON columns
join it only when there is a search term to match their skill tags against.
The second hydrates just the rows that reached the page, which is where the
cost is: every stored PDF on a row gets a presigned S3 link, so that now
happens `limit` times per request rather than once per row the user owns.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cmp_to_key
from typing import Literal

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy import select

import db
import job_match
import pagination
import resumes
from auth import User, get_current_user
from db import session_scope

Kind = Literal["base", "tailored"]
Source = Literal["all", "base", "tailored"]
Sort = Literal["modified", "score", "name"]

# At or above this, a resume is presented as a good match for its role.
GOOD_SCORE = 75


# --- Models -----------------------------------------------------------------


class LibraryItem(BaseModel):
    """One row of the feed, whichever table it came from.

    The headings are resolved here rather than in the client because they are
    what the name sort and the search run against — deriving them twice would
    let the two drift apart.
    """

    key: str
    kind: Kind
    heading: str
    subheading: str
    company: str
    role: str
    score: int | None
    good: bool
    favourite: bool
    updated_at: str
    # Exactly one of these is set: the payload that row's actions work on.
    base: resumes.SavedResume | None = None
    match: job_match.JobMatch | None = None


class LibraryPage(pagination.Page[LibraryItem]):
    # Every role in the library, not only on this page, for the role filter.
    roles: list[str]
    # Rows before any filter, which tells an empty search apart from an
    # empty library — the two want very different empty states.
    total_all: int


# --- The union --------------------------------------------------------------


@dataclass
class _Ref:
    """A row reduced to what filtering and sorting need, nothing more."""

    kind: Kind
    id: int
    heading: str
    subheading: str
    company: str
    role: str
    score: int | None
    favourite: bool
    updated_at: str
    # Only populated when there is a search term to match them against.
    tags: list[str] = field(default_factory=list)


def _resume_refs(session, user_id: int, with_tags: bool) -> list[_Ref]:
    columns = [
        db.Resume.id,
        db.Resume.filename,
        db.Resume.role,
        db.Resume.overall_score,
        db.Resume.favourite,
        db.Resume.updated_at,
    ]
    if with_tags:
        columns.append(db.Resume.analysis)

    rows = session.execute(
        select(*columns)
        .where(db.Resume.user_id == user_id)
        .order_by(db.Resume.id.desc())
    ).all()

    return [
        _Ref(
            kind="base",
            id=row.id,
            heading=row.filename,
            subheading=row.role or "No role set",
            company="",
            role=row.role,
            score=row.overall_score,
            favourite=bool(row.favourite),
            updated_at=db.iso(row.updated_at),
            tags=resumes.tags_for(row.analysis) if with_tags else [],
        )
        for row in rows
    ]


def _match_refs(session, user_id: int, with_tags: bool) -> list[_Ref]:
    columns = [
        db.JobMatch.id,
        db.JobMatch.company,
        db.JobMatch.title,
        db.JobMatch.match_score,
        db.JobMatch.optimized_score,
        db.JobMatch.updated_at,
    ]
    if with_tags:
        columns.append(db.JobMatch.role_summary)

    rows = session.execute(
        select(*columns)
        .where(
            db.JobMatch.user_id == user_id,
            # A match only becomes a resume once one has been generated for
            # it; until then it lives on the Job Match page alone.
            db.JobMatch.optimized_resume.is_not(None),
            db.JobMatch.optimized_resume != "",
        )
        .order_by(db.JobMatch.id.desc())
    ).all()

    refs = []
    for row in rows:
        summary = row.role_summary if with_tags else None
        refs.append(
            _Ref(
                kind="tailored",
                id=row.id,
                heading=row.company or "Company not stated",
                subheading=row.title or "Role not stated",
                company=row.company,
                role=row.title,
                score=row.optimized_score if row.optimized_score is not None else row.match_score,
                # Tailored resumes cannot be starred; only base ones can.
                favourite=False,
                updated_at=db.iso(row.updated_at),
                tags=list((summary or {}).get("key_skills") or []),
            )
        )
    return refs


def _keep(ref: _Ref, source: str, role: str, needle: str) -> bool:
    if source != "all" and ref.kind != source:
        return False
    if role and ref.role != role:
        return False
    if not needle:
        return True
    haystack = [ref.heading, ref.subheading, ref.company, *ref.tags]
    return any(needle in (text or "").lower() for text in haystack)


def _cmp(left, right) -> int:
    return (left > right) - (left < right)


def _comparator(sort: str) -> Callable[[_Ref, _Ref], int]:
    """Favourites stay pinned whichever sort is active."""

    def compare(a: _Ref, b: _Ref) -> int:
        if a.favourite != b.favourite:
            return -1 if a.favourite else 1
        if sort == "score":
            # Unscored rows sort last rather than as a zero.
            return _cmp(b.score if b.score is not None else -1,
                        a.score if a.score is not None else -1)
        if sort == "name":
            return _cmp(a.heading.casefold(), b.heading.casefold())
        return _cmp(b.updated_at, a.updated_at)

    return compare


def feed(
    user_id: int,
    *,
    source: str = "all",
    role: str = "",
    q: str = "",
    sort: str = "modified",
    limit: int = pagination.DEFAULT_LIMIT,
    offset: int = 0,
) -> LibraryPage:
    needle = q.strip().lower()
    with session_scope() as session:
        # Both halves arrive in a fixed order and Python's sort is stable, so
        # rows that tie on the sort key land the same way on every request —
        # without that, paging could show one row twice and skip another.
        refs = _resume_refs(session, user_id, bool(needle))
        refs += _match_refs(session, user_id, bool(needle))

        roles = sorted({ref.role for ref in refs if ref.role})
        total_all = len(refs)

        kept = [ref for ref in refs if _keep(ref, source, role, needle)]
        kept.sort(key=cmp_to_key(_comparator(sort)))
        page = kept[offset : offset + limit]

        saved = resumes.summaries_for(
            session, user_id, [ref.id for ref in page if ref.kind == "base"]
        )
        matches = job_match.summaries_for(
            session, user_id, [ref.id for ref in page if ref.kind == "tailored"]
        )
        items = [_item(ref, saved, matches) for ref in page]
        items = [item for item in items if item is not None]

    return LibraryPage(
        items=items,
        total=len(kept),
        limit=limit,
        offset=offset,
        roles=roles,
        total_all=total_all,
    )


def _item(ref: _Ref, saved: dict, matches: dict) -> LibraryItem | None:
    """Pair a ref with its hydrated payload, or drop it if it went away."""
    base = saved.get(ref.id) if ref.kind == "base" else None
    match = matches.get(ref.id) if ref.kind == "tailored" else None
    if base is None and match is None:
        return None

    if base is not None:
        # "Selected" is the analysis's own verdict, which is not a threshold
        # on the score the way a match score is.
        good = base.selected
    else:
        good = ref.score is not None and ref.score >= GOOD_SCORE

    return LibraryItem(
        key=f"r-{ref.id}" if ref.kind == "base" else f"j-{ref.id}",
        kind=ref.kind,
        heading=ref.heading,
        subheading=ref.subheading,
        company=ref.company,
        role=ref.role,
        score=ref.score,
        good=good,
        favourite=ref.favourite,
        updated_at=ref.updated_at,
        base=base,
        match=match,
    )


# --- Routes -----------------------------------------------------------------

router = APIRouter(prefix="/api/library", tags=["library"])


@router.get("", response_model=LibraryPage)
def read_library(
    user: User = Depends(get_current_user),
    source: Source = "all",
    role: str = Query("", max_length=160),
    q: str = Query("", max_length=200, description="Heading, company or skill tag."),
    sort: Sort = "modified",
    limit: int = Query(pagination.DEFAULT_LIMIT, ge=1, le=pagination.MAX_LIMIT),
    offset: int = Query(0, ge=0),
):
    """One page of the merged Saved Resumes feed."""
    return feed(
        user.id, source=source, role=role, q=q, sort=sort, limit=limit, offset=offset
    )

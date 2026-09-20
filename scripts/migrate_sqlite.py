#!/usr/bin/env python
"""Copy the old raw-sqlite3 database into whatever DATABASE_URL points at.

The original schema stored JSON as TEXT and timestamps as ISO strings; this
reads those, converts them, and writes them through the SQLAlchemy models, so
it works equally well into a fresh SQLite file or into Amazon RDS.

    python scripts/migrate_sqlite.py --source data/users.db

Row ids are preserved, because the frontend holds on to them.
"""

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import func, select, text  # noqa: E402

import db  # noqa: E402

TEXT_FIELDS = ("full_name", "email", "headline", "location", "linkedin", "website", "about")


def _dt(value) -> datetime:
    """ISO string from the old schema into an aware UTC datetime."""
    if not value:
        return db.utcnow()
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return db.utcnow()
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _json(value, fallback):
    if value in (None, ""):
        return fallback
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return fallback


def _tables(conn) -> set[str]:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    return {row[0] for row in rows}


def _columns(conn, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def migrate(source: Path, force: bool) -> None:
    if not source.exists():
        sys.exit(f"No such database: {source}")

    old = sqlite3.connect(source)
    old.row_factory = sqlite3.Row
    present = _tables(old)

    db.init_db()
    target = db.engine.url.render_as_string(hide_password=True)
    print(f"source: {source}\ntarget: {target}\n")

    with db.session_scope() as session:
        existing = session.scalar(select(func.count()).select_from(db.User))
        if existing and not force:
            sys.exit(
                f"Target already has {existing} user(s). "
                "Re-run with --force to add to it anyway."
            )

        counts = {}

        if "users" in present:
            rows = old.execute("SELECT * FROM users").fetchall()
            for row in rows:
                session.merge(
                    db.User(
                        id=row["id"],
                        username=row["username"],
                        password_hash=row["password_hash"],
                        created_at=_dt(row["created_at"]),
                    )
                )
            counts["users"] = len(rows)

        if "profiles" in present:
            rows = old.execute("SELECT * FROM profiles").fetchall()
            available = _columns(old, "profiles")
            for row in rows:
                session.merge(
                    db.Profile(
                        user_id=row["user_id"],
                        **{
                            field: (row[field] or "")
                            for field in TEXT_FIELDS
                            if field in available
                        },
                        preferences=_json(row["preferences"], {}),
                        updated_at=_dt(row["updated_at"]),
                    )
                )
            counts["profiles"] = len(rows)

        if "resumes" in present:
            rows = old.execute("SELECT * FROM resumes").fetchall()
            for row in rows:
                session.merge(
                    db.Resume(
                        id=row["id"],
                        user_id=row["user_id"],
                        filename=row["filename"],
                        role=row["role"] or "",
                        resume_text=row["resume_text"] or "",
                        analysis=_json(row["analysis_json"], None),
                        overall_score=row["overall_score"],
                        favourite=bool(row["favourite"]),
                        created_at=_dt(row["created_at"]),
                        updated_at=_dt(row["updated_at"]),
                    )
                )
            counts["resumes"] = len(rows)

        if "job_matches" in present:
            rows = old.execute("SELECT * FROM job_matches").fetchall()
            for row in rows:
                session.merge(
                    db.JobMatch(
                        id=row["id"],
                        user_id=row["user_id"],
                        company=row["company"] or "",
                        title=row["title"] or "",
                        jd_text=row["jd_text"] or "",
                        match_score=row["match_score"] or 0,
                        matching_skills=_json(row["matching_skills"], []),
                        missing_skills=_json(row["missing_skills"], []),
                        role_summary=_json(row["role_summary"], {}),
                        optimized_resume=row["optimized_resume"],
                        optimized_score=row["optimized_score"],
                        created_at=_dt(row["created_at"]),
                        updated_at=_dt(row["updated_at"]),
                    )
                )
            counts["job_matches"] = len(rows)

    old.close()

    if not db.IS_SQLITE:
        _resync_sequences()

    for table, count in counts.items():
        print(f"  {table:<12} {count}")
    print("\nDone.")


def _resync_sequences() -> None:
    """Explicit ids leave Postgres' sequences behind, so the next INSERT collides."""
    with db.engine.begin() as conn:
        for table in ("users", "resumes", "job_matches"):
            conn.execute(
                text(
                    "SELECT setval(pg_get_serial_sequence(:t, 'id'), "
                    "COALESCE((SELECT MAX(id) FROM " + table + "), 1))"
                ),
                {"t": table},
            )
    print("Postgres id sequences resynced.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "users.db",
        help="the old sqlite file (default: data/users.db)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="write even though the target already has users",
    )
    args = parser.parse_args()
    migrate(args.source, args.force)

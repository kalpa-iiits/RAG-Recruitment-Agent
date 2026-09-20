"""Database engine, session factory and schema.

Every table in the app is defined here. `DATABASE_URL` picks the backend:
the default SQLite file keeps local development zero-setup, and a
`postgresql+psycopg://` URL points the identical schema at Amazon RDS.

Anything the app filters or sorts on is a real column; the free-form parts
of an analysis stay in a JSON column, which becomes JSONB on Postgres so it
can be indexed and queried into later.
"""

import logging
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from dotenv import load_dotenv
from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    create_engine,
    desc,
    event,
    func,
    inspect,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
    sessionmaker,
)

load_dotenv()

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"
DEFAULT_SQLITE_URL = f"sqlite:///{DATA_DIR / 'cvexpert.db'}"


def _normalise(url: str) -> str:
    """Accept the URL forms RDS and Heroku-style tooling hand out.

    SQLAlchemy 2 needs an explicit driver, and `psycopg` (v3) is the one we
    depend on, so `postgres://` and bare `postgresql://` are rewritten rather
    than failing at connect time with a driver error.
    """
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://") :]
    if url.startswith("postgresql://"):
        url = "postgresql+psycopg://" + url[len("postgresql://") :]
    return url


DATABASE_URL = _normalise(os.getenv("DATABASE_URL", "").strip() or DEFAULT_SQLITE_URL)
IS_SQLITE = DATABASE_URL.startswith("sqlite")


def _engine_kwargs() -> dict:
    if IS_SQLITE:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        # FastAPI serves requests on a threadpool, so connections cross threads.
        return {"connect_args": {"check_same_thread": False}}
    return {
        # RDS drops idle connections and reboots during maintenance windows;
        # without these a pooled connection comes back dead.
        "pool_pre_ping": True,
        "pool_recycle": 1800,
        "pool_size": 5,
        "max_overflow": 10,
    }


engine = create_engine(DATABASE_URL, echo=False, future=True, **_engine_kwargs())
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, future=True)


if IS_SQLITE:

    @event.listens_for(engine, "connect")
    def _sqlite_pragmas(dbapi_connection, _record):
        """SQLite ignores foreign keys unless asked, making CASCADE a no-op."""
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys = ON")
        cursor.close()


@contextmanager
def session_scope() -> Iterator[Session]:
    """Commit on success, roll back on error, always close."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# --- Column types -----------------------------------------------------------

# JSON everywhere, JSONB on Postgres: same Python dicts, but indexable and
# queryable server-side once this runs on RDS.
JSONColumn = JSON().with_variant(JSONB, "postgresql")

TZDateTime = DateTime(timezone=True)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def iso(value: datetime | None) -> str:
    """Timestamps as the API has always returned them: ISO-8601 in UTC.

    SQLite has no timezone-aware type and hands back naive datetimes, which
    were written as UTC; Postgres hands back timestamptz in the connection's
    timezone. Both are normalised here so the wire format never depends on
    the backend or on the server's locale.
    """
    if value is None:
        return ""
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


# --- Schema -----------------------------------------------------------------


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    # The login identity. Accounts predating email login keep whatever
    # username they registered with; only new sign-ups must be addresses.
    email: Mapped[str] = mapped_column(String(200), nullable=False)
    password_hash: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(TZDateTime, nullable=False, default=utcnow)

    profile: Mapped["Profile"] = relationship(
        back_populates="user", uselist=False, cascade="all, delete-orphan"
    )
    resumes: Mapped[list["Resume"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    job_matches: Mapped[list["JobMatch"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )


# Replaces SQLite's `COLLATE NOCASE`, which Postgres does not have. Every
# lookup by email must go through lower() to use this index.
Index("uq_users_email_lower", func.lower(User.email), unique=True)


class Profile(Base):
    __tablename__ = "profiles"

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    full_name: Mapped[str] = mapped_column(String(120), nullable=False, default="")
    email: Mapped[str] = mapped_column(String(200), nullable=False, default="")
    headline: Mapped[str] = mapped_column(String(200), nullable=False, default="")
    location: Mapped[str] = mapped_column(String(120), nullable=False, default="")
    linkedin: Mapped[str] = mapped_column(String(300), nullable=False, default="")
    website: Mapped[str] = mapped_column(String(300), nullable=False, default="")
    about: Mapped[str] = mapped_column(Text, nullable=False, default="")
    preferences: Mapped[dict] = mapped_column(JSONColumn, nullable=False, default=dict)
    updated_at: Mapped[datetime] = mapped_column(
        TZDateTime, nullable=False, default=utcnow, onupdate=utcnow
    )

    user: Mapped[User] = relationship(back_populates="profile")


class Resume(Base):
    __tablename__ = "resumes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    filename: Mapped[str] = mapped_column(String(200), nullable=False)
    role: Mapped[str] = mapped_column(String(120), nullable=False, default="")
    resume_text: Mapped[str] = mapped_column(Text, nullable=False, default="")
    # The agent's output verbatim. Its shape moves with the prompts, so it is
    # stored whole rather than spread across columns.
    analysis: Mapped[dict | None] = mapped_column(JSONColumn, nullable=True)
    # Lifted out of the analysis so the list view can sort without parsing it.
    overall_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # Where the original PDF lives in S3. Null when S3 is not configured, or
    # for rows analysed before it was. The key is canonical; resume_url is the
    # object's address, which on a private bucket is not itself downloadable.
    resume_key: Mapped[str | None] = mapped_column(String(512), nullable=True)
    resume_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    # The job description PDF, when one was uploaded instead of picking a role.
    jd_key: Mapped[str | None] = mapped_column(String(512), nullable=True)
    jd_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    # The rewrite. Text so it can be re-rendered or diffed, plus the PDF that
    # was generated from it. Previously this lived only in the session and was
    # lost on timeout.
    improved_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    improved_key: Mapped[str | None] = mapped_column(String(512), nullable=True)
    improved_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    favourite: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(TZDateTime, nullable=False, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        TZDateTime, nullable=False, default=utcnow, onupdate=utcnow
    )

    user: Mapped[User] = relationship(back_populates="resumes")

    __table_args__ = (Index("idx_resumes_user", "user_id", desc("updated_at")),)


class JobMatch(Base):
    __tablename__ = "job_matches"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    company: Mapped[str] = mapped_column(String(120), nullable=False, default="")
    title: Mapped[str] = mapped_column(String(160), nullable=False, default="")
    jd_text: Mapped[str] = mapped_column(Text, nullable=False)
    match_score: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    matching_skills: Mapped[list] = mapped_column(JSONColumn, nullable=False, default=list)
    missing_skills: Mapped[list] = mapped_column(JSONColumn, nullable=False, default=list)
    role_summary: Mapped[dict] = mapped_column(JSONColumn, nullable=False, default=dict)
    optimized_resume: Mapped[str | None] = mapped_column(Text, nullable=True)
    optimized_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # The tailored resume rendered to PDF and stored.
    optimized_key: Mapped[str | None] = mapped_column(String(512), nullable=True)
    optimized_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(TZDateTime, nullable=False, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        TZDateTime, nullable=False, default=utcnow, onupdate=utcnow
    )

    user: Mapped[User] = relationship(back_populates="job_matches")

    __table_args__ = (Index("idx_job_matches_user", "user_id", desc("updated_at")),)


# Columns added after the first release. create_all() only creates whole
# tables, so an existing database needs these bolted on. Additive and
# nullable only — anything more involved wants a real migration tool.
_ADDED_COLUMNS: dict[str, dict[str, str]] = {
    "resumes": {
        "resume_key": "VARCHAR(512)",
        "resume_url": "TEXT",
        "jd_key": "VARCHAR(512)",
        "jd_url": "TEXT",
        "improved_text": "TEXT",
        "improved_key": "VARCHAR(512)",
        "improved_url": "TEXT",
    },
    "job_matches": {
        "optimized_key": "VARCHAR(512)",
        "optimized_url": "TEXT",
    },
}


def _add_missing_columns() -> None:
    inspector = inspect(engine)
    for table, columns in _ADDED_COLUMNS.items():
        if not inspector.has_table(table):
            continue
        present = {column["name"] for column in inspector.get_columns(table)}
        for name, sql_type in columns.items():
            if name in present:
                continue
            with engine.begin() as conn:
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {sql_type}"))
            logger.info("Added column %s.%s", table, name)


def _rename_username_to_email() -> None:
    """Accounts were keyed by a username before login moved to email.

    Renames the column in place rather than adding a second one, so there is
    only ever one login identity. Existing values carry over untouched: an
    account registered as "demo" still signs in as "demo" — the address
    format is only required of new registrations.

    Runs before create_all(), which skips tables that already exist and so
    would leave the old column in place under a model that no longer has it.
    """
    inspector = inspect(engine)
    if not inspector.has_table("users"):
        return
    columns = {column["name"] for column in inspector.get_columns("users")}
    if "email" in columns or "username" not in columns:
        return

    with engine.begin() as conn:
        conn.execute(text("DROP INDEX IF EXISTS uq_users_username_lower"))
        conn.execute(text("ALTER TABLE users RENAME COLUMN username TO email"))
        if not IS_SQLITE:
            # SQLite ignores declared lengths; Postgres would hold it at 50.
            conn.execute(
                text("ALTER TABLE users ALTER COLUMN email TYPE VARCHAR(200)")
            )
    logger.info("Renamed users.username to users.email")


def _ensure_email_index() -> None:
    """Add the unique index to a database that was migrated, not created.

    create_all() builds indexes only alongside a table it creates itself, so
    a renamed column arrives without one. IF NOT EXISTS rather than an
    inspector check: SQLite does not report indexes over an expression like
    lower(email), so a fresh database would look like it were missing one.
    """
    if not inspect(engine).has_table("users"):
        return
    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE UNIQUE INDEX IF NOT EXISTS uq_users_email_lower "
                "ON users (lower(email))"
            )
        )


def init_db() -> None:
    """Create anything missing. Safe to call on every boot."""
    _rename_username_to_email()
    Base.metadata.create_all(engine)
    _add_missing_columns()
    _ensure_email_index()
    logger.info("Database ready: %s", engine.url.render_as_string(hide_password=True))

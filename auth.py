import logging
import os
import secrets
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

import jwt
from dotenv import load_dotenv
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pwdlib import PasswordHash
from pydantic import BaseModel, Field

load_dotenv()

logger = logging.getLogger(__name__)

DATABASE_PATH = Path(
    os.getenv("DATABASE_PATH", Path(__file__).parent / "data" / "users.db")
)
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
ALLOW_REGISTRATION = os.getenv("ALLOW_REGISTRATION", "true").lower() == "true"

JWT_SECRET = os.getenv("JWT_SECRET")
if not JWT_SECRET:
    JWT_SECRET = secrets.token_urlsafe(64)
    logger.warning(
        "JWT_SECRET is not set; using a random secret. "
        "All tokens will be invalidated when the server restarts."
    )

password_hash = PasswordHash.recommended()
# Verified against when the username doesn't exist, so a failed login takes the
# same time either way and doesn't reveal which usernames are registered.
_DUMMY_HASH = password_hash.hash(secrets.token_urlsafe(16))

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


# --- Storage ----------------------------------------------------------------


@contextmanager
def _db():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def init_db():
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )


# --- Models -----------------------------------------------------------------


class UserCreate(BaseModel):
    username: str = Field(min_length=3, max_length=50, pattern=r"^[A-Za-z0-9_.-]+$")
    password: str = Field(min_length=8, max_length=128)


class User(BaseModel):
    id: int
    username: str
    created_at: str


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


def _row_to_user(row: sqlite3.Row) -> User:
    return User(id=row["id"], username=row["username"], created_at=row["created_at"])


# --- Dependencies -----------------------------------------------------------


def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    credentials_error = HTTPException(
        status.HTTP_401_UNAUTHORIZED,
        "Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[ALGORITHM])
        user_id = int(payload["sub"])
    except (jwt.InvalidTokenError, KeyError, ValueError):
        raise credentials_error

    with _db() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    if row is None:
        raise credentials_error
    return _row_to_user(row)


# --- Routes -----------------------------------------------------------------

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/register", response_model=User, status_code=status.HTTP_201_CREATED)
def register(body: UserCreate):
    if not ALLOW_REGISTRATION:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Registration is disabled.")
    created_at = datetime.now(timezone.utc).isoformat()
    try:
        with _db() as conn:
            cursor = conn.execute(
                "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
                (body.username, password_hash.hash(body.password), created_at),
            )
    except sqlite3.IntegrityError:
        raise HTTPException(status.HTTP_409_CONFLICT, "Username is already taken.")
    return User(id=cursor.lastrowid, username=body.username, created_at=created_at)


@router.post("/login", response_model=Token)
def login(form: OAuth2PasswordRequestForm = Depends()):
    """OAuth2 password flow: form-encoded `username` and `password`"""
    with _db() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE username = ?", (form.username,)
        ).fetchone()

    valid = password_hash.verify(
        form.password, row["password_hash"] if row else _DUMMY_HASH
    )
    if row is None or not valid:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    token = jwt.encode(
        {"sub": str(row["id"]), "exp": datetime.now(timezone.utc) + expires},
        JWT_SECRET,
        algorithm=ALGORITHM,
    )
    return Token(access_token=token, expires_in=int(expires.total_seconds()))


@router.get("/me", response_model=User)
def me(user: User = Depends(get_current_user)):
    return user

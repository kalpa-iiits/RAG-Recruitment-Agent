import logging
import os
import secrets
from datetime import datetime, timedelta, timezone

import jwt
from dotenv import load_dotenv
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pwdlib import PasswordHash
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

import db
from db import session_scope

load_dotenv()

logger = logging.getLogger(__name__)

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


def _to_user(row: db.User) -> User:
    return User(id=row.id, username=row.username, created_at=db.iso(row.created_at))


# --- Storage ----------------------------------------------------------------


def find_by_username(session, username: str) -> db.User | None:
    """Case-insensitive, matching the unique index on lower(username)."""
    return session.scalar(
        select(db.User).where(func.lower(db.User.username) == username.lower())
    )


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

    with session_scope() as session:
        row = session.get(db.User, user_id)
        if row is None:
            raise credentials_error
        return _to_user(row)


# --- Routes -----------------------------------------------------------------

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/register", response_model=User, status_code=status.HTTP_201_CREATED)
def register(body: UserCreate):
    if not ALLOW_REGISTRATION:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Registration is disabled.")
    try:
        with session_scope() as session:
            row = db.User(
                username=body.username,
                password_hash=password_hash.hash(body.password),
                created_at=db.utcnow(),
            )
            session.add(row)
            session.flush()
            return _to_user(row)
    except IntegrityError:
        raise HTTPException(status.HTTP_409_CONFLICT, "Username is already taken.")


@router.post("/login", response_model=Token)
def login(form: OAuth2PasswordRequestForm = Depends()):
    """OAuth2 password flow: form-encoded `username` and `password`"""
    with session_scope() as session:
        row = find_by_username(session, form.username)
        stored_hash = row.password_hash if row else _DUMMY_HASH
        user_id = row.id if row else None

    valid = password_hash.verify(form.password, stored_hash)
    if user_id is None or not valid:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    token = jwt.encode(
        {"sub": str(user_id), "exp": datetime.now(timezone.utc) + expires},
        JWT_SECRET,
        algorithm=ALGORITHM,
    )
    return Token(access_token=token, expires_in=int(expires.total_seconds()))


@router.get("/me", response_model=User)
def me(user: User = Depends(get_current_user)):
    return user

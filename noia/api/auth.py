"""
noia/api/auth.py
─────────────────
JWT-based authentication and role-based access control (RBAC) for the NOIA API.

Roles:
  - noc_engineer   : Full query/summarise/recommend/generate access
  - team_lead      : All engineer permissions + index stats + audit log
  - read_only      : Query and summarise only; no recommend or generate

In a production deployment, users would be managed in a database with hashed
passwords. For the Build-a-thon demo, a small in-memory user store is used.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext

from config.settings import get_settings

settings = get_settings()

# ── Password hashing ──────────────────────────────────────────────────────────

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(plain: str) -> str:
    return _pwd_context.hash(plain)

def verify_password(plain: str, hashed: str) -> bool:
    return _pwd_context.verify(plain, hashed)


# ── Demo user store ────────────────────────────────────────────────────────────
# In production: replace with a proper database query.

_USERS: dict[str, dict] = {
    "noc_engineer_demo": {
        "username": "noc_engineer_demo",
        "hashed_password": hash_password("noia_demo_2025"),
        "role": "noc_engineer",
        "full_name": "NOC Engineer (Demo)",
    },
    "team_lead_demo": {
        "username": "team_lead_demo",
        "hashed_password": hash_password("lead_demo_2025"),
        "role": "team_lead",
        "full_name": "NOC Team Lead (Demo)",
    },
    "readonly_demo": {
        "username": "readonly_demo",
        "hashed_password": hash_password("readonly_2025"),
        "role": "read_only",
        "full_name": "Read-Only Observer (Demo)",
    },
}

# Role → allowed endpoints (endpoint tag strings)
ROLE_PERMISSIONS: dict[str, set[str]] = {
    "noc_engineer": {"query", "rca", "summarise", "recommend", "generate_article", "health"},
    "team_lead":    {"query", "rca", "summarise", "recommend", "generate_article",
                     "health", "stats", "audit"},
    "read_only":    {"query", "summarise", "health"},
}

# ── OAuth2 scheme ─────────────────────────────────────────────────────────────

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/token")


# ── Token creation ────────────────────────────────────────────────────────────

def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    """Create a signed JWT access token."""
    to_encode = data.copy()
    expire = datetime.now(tz=timezone.utc) + (
        expires_delta or timedelta(minutes=settings.access_token_expire_minutes)
    )
    to_encode["exp"] = expire
    return jwt.encode(to_encode, settings.api_secret_key, algorithm=settings.api_algorithm)


# ── Token verification ────────────────────────────────────────────────────────

def _decode_token(token: str) -> dict:
    """Decode and validate a JWT token. Raises HTTPException on failure."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired authentication token.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(
            token,
            settings.api_secret_key,
            algorithms=[settings.api_algorithm],
        )
        username: str | None = payload.get("sub")
        if not username:
            raise credentials_exception
        return payload
    except JWTError:
        raise credentials_exception


# ── FastAPI dependencies ──────────────────────────────────────────────────────

def get_current_user(token: Annotated[str, Depends(oauth2_scheme)]) -> dict:
    """Dependency: decode token and return the user dict."""
    payload  = _decode_token(token)
    username = payload.get("sub")
    user     = _USERS.get(username)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found.",
        )
    return user


def require_permission(permission: str):
    """
    Dependency factory: return a FastAPI dependency that enforces a specific
    permission string against the current user's role.

    Usage::

        @router.post("/rca")
        def rca_endpoint(
            _: Annotated[dict, Depends(require_permission("rca"))],
            ...
        ):
    """
    def _check(user: Annotated[dict, Depends(get_current_user)]) -> dict:
        role = user.get("role", "read_only")
        allowed = ROLE_PERMISSIONS.get(role, set())
        if permission not in allowed:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{role}' does not have permission for '{permission}'.",
            )
        return user
    return _check


def authenticate_user(username: str, password: str) -> dict | None:
    """Verify credentials and return the user dict, or None on failure."""
    user = _USERS.get(username)
    if not user:
        return None
    if not verify_password(password, user["hashed_password"]):
        return None
    return user

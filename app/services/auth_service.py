"""
Pathos AI — Authentication Service
======================================
Password hashing (bcrypt via passlib) and JWT access/refresh token
issuance & verification. Kept as a standalone module so it has no FastAPI
dependency and can be unit tested in isolation.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone

from jose import JWTError, jwt
from passlib.context import CryptContext

from app.config import settings
from app.schemas import TokenPair, TokenPayload

logger = logging.getLogger("pathos_ai.auth")

_pwd_context = CryptContext(schemes=[settings.password_hash_scheme], deprecated="auto")


class AuthError(Exception):
    """Raised for any authentication failure — routers translate this to HTTP 401."""


def hash_password(plain_password: str) -> str:
    return _pwd_context.hash(plain_password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        return _pwd_context.verify(plain_password, hashed_password)
    except Exception:
        logger.exception("password_verification_error")
        return False


def _create_token(subject: uuid.UUID, token_type: str, expires_delta: timedelta) -> str:
    now = datetime.now(timezone.utc)
    payload = TokenPayload(
        sub=str(subject),
        iat=int(now.timestamp()),
        exp=int((now + expires_delta).timestamp()),
        type=token_type,
        jti=str(uuid.uuid4()),
    )
    return jwt.encode(
        payload.model_dump(),
        settings.jwt_secret_key.get_secret_value(),
        algorithm=settings.jwt_algorithm,
    )


def create_token_pair(user_id: uuid.UUID) -> TokenPair:
    access_token = _create_token(
        user_id, "access", timedelta(minutes=settings.access_token_expire_minutes)
    )
    refresh_token = _create_token(
        user_id, "refresh", timedelta(days=settings.refresh_token_expire_days)
    )
    return TokenPair(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in_seconds=settings.access_token_expire_minutes * 60,
    )


def decode_token(token: str, expected_type: str = "access") -> TokenPayload:
    try:
        raw_payload = jwt.decode(
            token,
            settings.jwt_secret_key.get_secret_value(),
            algorithms=[settings.jwt_algorithm],
        )
        payload = TokenPayload.model_validate(raw_payload)
    except JWTError as exc:
        raise AuthError("Invalid or expired token.") from exc

    if payload.type != expected_type:
        raise AuthError(f"Expected a {expected_type} token, got {payload.type}.")

    return payload

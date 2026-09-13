import uuid
from collections.abc import Awaitable, Callable
from typing import TypeVar

import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ValidationError
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.user import User, UserRole
from app.security import decode_access_token

_ModelT = TypeVar("_ModelT", bound=BaseModel)

bearer_scheme = HTTPBearer()

credentials_exception = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Could not validate credentials",
    headers={"WWW-Authenticate": "Bearer"},
)


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> User:
    try:
        payload = decode_access_token(credentials.credentials)
        user_id = uuid.UUID(payload["sub"])
    except (jwt.PyJWTError, KeyError, ValueError):
        raise credentials_exception

    # Looked up by primary key only for the token's own subject — this is the
    # one place a user may be fetched without an explicit organization_id
    # filter, because the id came from a token WE signed, embedding that
    # user's organization_id. Every other lookup in the app must go through
    # app.repositories.base.TenantScopedRepository.
    user = db.get(User, user_id)
    if user is None or not user.is_active:
        raise credentials_exception
    return user


def require_role(role: UserRole) -> Callable[[User], User]:
    def dependency(current_user: User = Depends(get_current_user)) -> User:
        if current_user.role != role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions",
            )
        return current_user

    return dependency


def validate_body_before_gateway(
    model_cls: type[_ModelT],
) -> Callable[[Request], Awaitable[_ModelT]]:
    """Factory for a Depends() that validates the raw request body against
    `model_cls`, for use — listed FIRST — in any endpoint whose signature
    also declares Depends(get_gateway) (or another dependency that can
    itself raise before FastAPI gets to validate an ordinary Pydantic body
    parameter).

    Why this has to be a dependency at all: FastAPI's solve_dependencies()
    always finishes resolving *every* Depends() in a path operation (in
    signature order) before it validates an automatic Pydantic body
    parameter. That ordering holds no matter where the body parameter
    appears in the function signature, so declaring the body as a plain
    `payload: SomeModel` parameter can never make its validation run before
    Depends(get_gateway) — get_gateway would always be called first
    regardless of parameter order.

    Concretely, that means a malformed request (e.g. empty `question`,
    which SomeModel's own Field constraints should reject with 422) would
    still invoke get_gateway() — which raises 503 when GROQ_API_KEY isn't
    set — before Pydantic ever got a chance to validate the body. Declaring
    the body's validation as its own Depends(), placed first in the
    endpoint's signature, guarantees it resolves before get_current_user,
    get_db, get_gateway, and any other Depends() in that signature — so bad
    input is always rejected with 422 first, gateway included.
    """

    async def _validate(request: Request) -> _ModelT:
        try:
            raw_body = await request.json()
        except Exception as exc:
            raise RequestValidationError(
                [
                    {
                        "type": "json_invalid",
                        "loc": ("body",),
                        "msg": "JSON decode error",
                        "input": {},
                    }
                ]
            ) from exc
        try:
            return model_cls.model_validate(raw_body)
        except ValidationError as exc:
            raise RequestValidationError(exc.errors()) from exc

    return _validate

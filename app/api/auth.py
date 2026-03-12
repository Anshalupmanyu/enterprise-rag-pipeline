"""
Enterprise RAG Pipeline — Authentication API Routes

Endpoints for user registration and JWT token issuance.
"""

from fastapi import APIRouter, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from fastapi import Depends

from app.core.security import (
    UserCreate,
    UserResponse,
    Token,
    hash_password,
    verify_password,
    create_access_token,
    get_users_db,
)

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user",
)
async def register(user: UserCreate):
    """
    Register a new user account.

    - **username**: Unique username (3+ characters)
    - **password**: Strong password (6+ characters)
    """
    users_db = get_users_db()

    if len(user.username) < 3:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Username must be at least 3 characters",
        )
    if len(user.password) < 6:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Password must be at least 6 characters",
        )
    if user.username in users_db:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Username already registered",
        )

    users_db[user.username] = {
        "username": user.username,
        "hashed_password": hash_password(user.password),
        "is_active": True,
    }
    return UserResponse(username=user.username, is_active=True)


@router.post(
    "/token",
    response_model=Token,
    summary="Login and get access token",
)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Authenticate with username/password and receive a JWT access token.

    Use the returned token in the `Authorization: Bearer <token>` header
    for all protected endpoints.
    """
    users_db = get_users_db()
    user = users_db.get(form_data.username)

    if not user or not verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(data={"sub": user["username"]})
    return Token(access_token=access_token)

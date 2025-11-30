from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr
from datetime import timedelta, date
import models
from schemas import Token, UserLogin, AuthUserResponse
from database import get_db
from routers.utils import (
    get_current_user,
    create_access_token,
    ACCESS_TOKEN_EXPIRE_MINUTES,
)
from dotenv import load_dotenv

load_dotenv()
router = APIRouter()

# Signup Request Schema
class UserSignup(BaseModel):
    email: EmailStr
    password: str
    first_name: str
    last_name: str
    dob: date
    gender: str

def authenticate_user(db: Session, email: str, password: str):
    """
    Authenticate a user by email and password.
    Returns user if authentication successful, None otherwise.
    """
    user = (
        db.query(models.User)
        .filter(models.User.email == email, models.User.password == password, models.User.is_deleted == False)
        .first()
    )

    if not user:
        return None
    return user


# Endpoints


@router.post("/login", response_model=Token)
async def login(user_data: UserLogin, db: Session = Depends(get_db)):
    """
    Authenticate user and return JWT token.

    The token should be included in subsequent requests as:
    Authorization: Bearer <token>
    """
    print(user_data)
    # Authenticate user
    user = authenticate_user(db, user_data.email, user_data.password)

    if not user:
        print("HELLO")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"user_id": user.user_id, "email": user.email},
        expires_delta=access_token_expires,
    )

    # Prepare user data for response
    user_response = {
        "user_id": user.user_id,
        "email": user.email,
        "full_name": user.first_name + " " + user.last_name,
        "created_at": user.created.isoformat() if user.created else None,
    }

    return {"access_token": access_token, "token_type": "bearer", "user": user_response}


@router.post("/signup", response_model=Token, status_code=status.HTTP_201_CREATED)
async def signup(user_data: UserSignup, db: Session = Depends(get_db)):
    """
    Register a new user account.

    Returns JWT token for immediate login after signup.
    
    Required fields:
    - email: User's email address
    - password: User's password (plain text, will be stored as-is)
    - first_name: User's first name
    - last_name: User's last name
    - dob: User's date of birth (YYYY-MM-DD)
    - gender: User's gender (Male, Female, Other, Prefer not to say)
    """
    # Check if user already exists
    existing_user = (
        db.query(models.User)
        .filter(models.User.email == user_data.email, models.User.is_deleted == False)
        .first()
    )

    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Email already registered"
        )

    # Validate date of birth (user must be at least 13 years old)
    from datetime import datetime
    today = datetime.today().date()
    age = today.year - user_data.dob.year
    if today.month < user_data.dob.month or (today.month == user_data.dob.month and today.day < user_data.dob.day):
        age -= 1
    
    if age < 13:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You must be at least 13 years old to create an account"
        )

    # Validate gender
    valid_genders = ["Male", "Female", "Other", "Prefer not to say"]
    if user_data.gender not in valid_genders:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Gender must be one of: {', '.join(valid_genders)}"
        )

    # Create new user
    # NOTE: Password is stored as plain text as per requirements
    new_user = models.User(
        email=user_data.email,
        password=user_data.password,  # Plain text password (not hashed)
        first_name=user_data.first_name,
        last_name=user_data.last_name,
        dob=user_data.dob,
        gender=user_data.gender,
        is_deleted=False,
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"user_id": new_user.user_id, "email": new_user.email},
        expires_delta=access_token_expires,
    )

    # Prepare user data for response
    user_response = {
        "user_id": new_user.user_id,
        "email": new_user.email,
        "full_name": f"{new_user.first_name} {new_user.last_name}",
        "created_at": new_user.created.isoformat() if new_user.created else None,
    }

    return {"access_token": access_token, "token_type": "bearer", "user": user_response}


@router.post("/logout")
async def logout():
    """
    Logout endpoint.

    Since we're using JWT tokens, the client just needs to delete the token.
    This endpoint exists for completeness and future enhancements (like token blacklisting).
    """
    return {"message": "Successfully logged out"}


@router.get("/me", response_model=AuthUserResponse)
async def get_current_user_info(current_user: models.User = Depends(get_current_user)):
    """
    Get current authenticated user information.

    Requires valid JWT token in Authorization header.
    """
    return AuthUserResponse(
        user_id=current_user.user_id,
        email=current_user.email,
        full_name=f"{current_user.first_name} {current_user.last_name}",
        created_at=(
            current_user.created.isoformat() if current_user.created else None
        ),
    )
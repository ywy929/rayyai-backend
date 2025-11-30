from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from datetime import datetime, timezone

import models
import schemas
from database import get_db
from routers.utils import get_current_user, create_access_token

router = APIRouter()

@router.get("", response_model=schemas.UserResponse)
async def get_user_profile(
    current_user: models.User = Depends(get_current_user)
):
    """Retrieve the current user's profile details"""
    return current_user

@router.post("", response_model=schemas.UserResponse, status_code=201)
async def create_user(
    user: schemas.UserCreate,
    db: Session = Depends(get_db)
):
    """Create a new user account"""
    # Check if user already exists
    existing_user = db.query(models.User).filter(
        models.User.email == user.email
    ).first()
    if existing_user:
        raise HTTPException(
            status_code=400,
            detail="Email already registered"
        )
    
    # Create new user
    db_user = models.User(**user.model_dump())
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

@router.put("", response_model=schemas.UserResponse)
async def update_user_profile(
    user_update: schemas.UserUpdate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """Update the current user's profile information"""
    for key, value in user_update.model_dump(exclude_unset=True).items():
        setattr(current_user, key, value)
    
    current_user.updated = datetime.now(timezone.utc)
    db.commit()
    db.refresh(current_user)
    return current_user
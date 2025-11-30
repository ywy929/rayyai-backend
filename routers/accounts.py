from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import List
import models
import schemas
from database import get_db
from routers.utils import get_current_user, calculate_account_balance
from datetime import date

router = APIRouter()


@router.post("/", response_model=schemas.AccountResponse, status_code=status.HTTP_201_CREATED)
def create_account(
    account: schemas.AccountCreate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """Create a new account for the current user"""
    import logging
    logger = logging.getLogger(__name__)

    # Create the account
    db_account = models.Account(
        user_id=current_user.user_id,
        **account.model_dump(exclude_unset=True)
    )

    db.add(db_account)
    db.commit()
    db.refresh(db_account)

    logger.info(f"[create_account] Created account: account_id={db_account.account_id}, account_type={db_account.account_type}, account_name={db_account.account_name}")

    # If this is a credit account, try to link it to the most recent unlinked credit card
    if db_account.account_type == 'credit' and not db_account.card_id:
        logger.info(f"[create_account] Looking for unlinked credit card for user {current_user.user_id}")

        unlinked_credit_card = db.query(models.UserCreditCard).filter(
            models.UserCreditCard.user_id == current_user.user_id,
            models.UserCreditCard.is_deleted == False
        ).outerjoin(
            models.Account,
            models.Account.card_id == models.UserCreditCard.card_id
        ).filter(
            models.Account.card_id.is_(None)
        ).order_by(models.UserCreditCard.card_id.desc()).first()

        if unlinked_credit_card:
            logger.info(f"[create_account] Found unlinked credit card: card_id={unlinked_credit_card.card_id}, card_name={unlinked_credit_card.card_name}")
            db_account.card_id = unlinked_credit_card.card_id
            db.commit()
            db.refresh(db_account)
            logger.info(f"[create_account] ✓ Linked account {db_account.account_id} to credit card {unlinked_credit_card.card_id}")
        else:
            logger.info(f"[create_account] No unlinked credit card found for user {current_user.user_id}")

    return db_account


@router.get("/", response_model=List[schemas.AccountResponse])
def get_all_accounts(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """Get all accounts for the current user"""

    accounts = db.query(models.Account).filter(
        models.Account.user_id == current_user.user_id,
        models.Account.is_deleted == False
    ).all()

    return accounts


@router.get("/{account_id}", response_model=schemas.AccountResponse)
def get_account_by_id(
    account_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """Get a specific account by ID"""

    account = db.query(models.Account).filter(
        models.Account.account_id == account_id,
        models.Account.user_id == current_user.user_id,
        models.Account.is_deleted == False
    ).first()

    if not account:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Account not found or doesn't belong to you"
        )

    return account


@router.put("/{account_id}", response_model=schemas.AccountResponse)
def update_account(
    account_id: int,
    account_update: schemas.AccountUpdate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """Update an existing account"""

    # Get the account
    db_account = db.query(models.Account).filter(
        models.Account.account_id == account_id,
        models.Account.user_id == current_user.user_id,
        models.Account.is_deleted == False
    ).first()

    if not db_account:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Account not found or doesn't belong to you"
        )

    # Update fields
    update_data = account_update.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_account, field, value)

    db.commit()
    db.refresh(db_account)

    return db_account


@router.delete("/{account_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_account(
    account_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """Soft delete an account"""

    # Get the account
    db_account = db.query(models.Account).filter(
        models.Account.account_id == account_id,
        models.Account.user_id == current_user.user_id,
        models.Account.is_deleted == False
    ).first()

    if not db_account:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Account not found or doesn't belong to you"
        )

    # Soft delete
    db_account.is_deleted = True
    db.commit()

    return None


@router.get("/{account_id}/balance")
def get_account_balance(
    account_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """Get the current balance for an account"""

    # Verify account belongs to user
    account = db.query(models.Account).filter(
        models.Account.account_id == account_id,
        models.Account.user_id == current_user.user_id,
        models.Account.is_deleted == False
    ).first()

    if not account:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Account not found or doesn't belong to you"
        )

    # Calculate balance
    balance = calculate_account_balance(db, account_id)

    return {
        "account_id": account_id,
        "account_name": account.account_name,
        "balance": balance
    }

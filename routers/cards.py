import json
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional

from database import get_db
import models, schemas
from routers.utils import get_current_user
from services.rag_service import RAGService

# Match the tag name used in main.py to avoid duplicates
router = APIRouter(
    tags=["Credit Cards"]
)


def _parse_numeric_value(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return None
        sign = 1
        upper = cleaned.upper()
        if upper.endswith("CR"):
            cleaned = cleaned[:-2].strip()
        elif upper.endswith("DR"):
            cleaned = cleaned[:-2].strip()
            sign = -1
        cleaned = (
            cleaned.replace("RM", "")
            .replace("MYR", "")
            .replace(",", "")
            .replace(" ", "")
        )
        if cleaned.startswith("(") and cleaned.endswith(")"):
            cleaned = cleaned[1:-1]
            sign *= -1
        if cleaned in ("", "-", "--"):
            return None
        try:
            return float(cleaned) * sign
        except ValueError:
            return None
    return None


def _load_extracted_data(data):
    if data is None:
        return None
    if isinstance(data, str):
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            return None
    return data


def _build_statement_metrics(extracted_data: dict) -> Optional[dict]:
    if not extracted_data:
        return None
    summary = extracted_data.get("credit_card_summary") or {}
    terms = extracted_data.get("credit_card_terms") or {}

    credit_limit = (
        _parse_numeric_value(summary.get("credit_limit"))
        or _parse_numeric_value(terms.get("credit_limit"))
    )
    available_credit = (
        _parse_numeric_value(summary.get("available_credit"))
        or _parse_numeric_value(terms.get("available_credit"))
    )
    outstanding_balance = (
        _parse_numeric_value(summary.get("outstanding_balance"))
        or _parse_numeric_value(summary.get("total_amount_due"))
        or _parse_numeric_value(terms.get("total_amount_due"))
        or _parse_numeric_value(terms.get("current_balance"))
        or _parse_numeric_value(summary.get("current_balance"))
    )
    current_balance = (
        _parse_numeric_value(summary.get("current_balance"))
        or _parse_numeric_value(terms.get("current_balance"))
        or _parse_numeric_value(extracted_data.get("closing_balance"))
    )

    if outstanding_balance is None:
        outstanding_balance = current_balance

    if available_credit is None and credit_limit is not None and outstanding_balance is not None:
        available_credit = max(credit_limit - outstanding_balance, 0)

    if (
        credit_limit is None
        and available_credit is None
        and outstanding_balance is None
        and current_balance is None
    ):
        return None

    return {
        "credit_limit": credit_limit,
        "available_credit": available_credit,
        "outstanding_balance": outstanding_balance,
        "current_balance": current_balance,
    }


# ========== GET ALL USER CARDS ==========
@router.get("/", response_model=List[schemas.UserCreditCardResponse])
def get_all_cards(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get all credit cards for the authenticated user."""
    cards = db.query(models.UserCreditCard).filter(
        models.UserCreditCard.user_id == current_user.user_id,
        models.UserCreditCard.is_deleted == False
    ).all()
    return cards


# ========== CREATE NEW USER CARD ==========
@router.post("/", response_model=schemas.UserCreditCardResponse)
def create_card(
    card: schemas.UserCreditCardCreate,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new credit card for the authenticated user."""
    import logging
    logger = logging.getLogger(__name__)

    # Create card with user_id from authenticated user
    card_data = card.model_dump()
    card_data['user_id'] = current_user.user_id

    logger.info(f"[create_card] Creating credit card for user {current_user.user_id}")
    new_card = models.UserCreditCard(**card_data)
    db.add(new_card)
    db.commit()
    db.refresh(new_card)
    logger.info(f"[create_card] Created credit card with card_id={new_card.card_id}")

    # Find the most recently created credit account without a card_id and link it to this credit card
    logger.info(f"[create_card] Looking for unlinked credit account for user {current_user.user_id}")
    unlinked_credit_account = db.query(models.Account).filter(
        models.Account.user_id == current_user.user_id,
        models.Account.account_type == 'credit',
        models.Account.card_id.is_(None),
        models.Account.is_deleted == False
    ).order_by(models.Account.account_id.desc()).first()

    if unlinked_credit_account:
        logger.info(f"[create_card] Found unlinked credit account: account_id={unlinked_credit_account.account_id}, account_name={unlinked_credit_account.account_name}")
        # Link the account to the newly created credit card
        unlinked_credit_account.card_id = new_card.card_id
        db.commit()
        db.refresh(unlinked_credit_account)
        logger.info(f"[create_card] ✓ Linked account {unlinked_credit_account.account_id} to credit card {new_card.card_id}")
    else:
        logger.warning(f"[create_card] No unlinked credit account found for user {current_user.user_id}. Credit card created without linked account.")

    return new_card


# ========== GET MARKET CARDS ==========
@router.get("/market", response_model=List[schemas.MarketCreditCardResponse])
def get_market_cards(db: Session = Depends(get_db)):
    """
    Retrieve a list of all available credit cards in the market for comparison.
    This does not include AI-based analysis or user-specific data.
    """
    market_cards = db.query(models.MarketCreditCard).filter(models.MarketCreditCard.is_deleted == False).all()
    return market_cards


# ========== GET CARDS OVERVIEW/SUMMARY ==========
@router.get("/overview", response_model=schemas.CardsOverviewResponse)
def get_cards_overview(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get a summary overview of all user's credit cards.

    Returns:
        Dictionary containing:
        - cards: List of all user's credit cards
        - summary: Aggregate statistics (total limit, total balance, utilization, etc.)
        - upcoming_payments: Upcoming payments in the next 30 days
    """
    cards = db.query(models.UserCreditCard).filter(
        models.UserCreditCard.user_id == current_user.user_id,
        models.UserCreditCard.is_deleted == False
    ).all()

    # Calculate summary statistics
    total_limit = sum(card.credit_limit or 0 for card in cards)
    total_balance = sum(card.current_balance or 0 for card in cards)

    # Calculate available credit for each card and total
    card_items = []
    for card in cards:
        limit = float(card.credit_limit or 0)
        balance = float(card.current_balance or 0)
        available = max(0.0, limit - balance)  # Available credit = limit - balance
        utilization = round((balance / limit * 100) if limit > 0 else 0, 2)

        card_items.append({
            "card_id": int(card.card_id),
            "card_name": str(card.card_name),
            "bank_name": str(card.bank_name) if card.bank_name else None,
            "credit_limit": round(limit, 2),
            "current_balance": round(balance, 2),
            "available_credit": round(available, 2),
            "utilization_pct": utilization,
            "next_payment_date": card.next_payment_date.isoformat() if card.next_payment_date else None,
            "next_payment_amount": round(float(card.next_payment_amount), 2) if card.next_payment_amount is not None else None,
        })

    total_available = sum(item["available_credit"] for item in card_items)

    # Calculate utilization percentage
    utilization_pct = (total_balance / total_limit * 100) if total_limit > 0 else 0

    # Get upcoming payments (next 30 days)
    from datetime import datetime, timedelta
    today = datetime.today().date()
    thirty_days = today + timedelta(days=30)

    upcoming_payments = []
    for card in cards:
        if card.next_payment_date and card.next_payment_amount:
            if today <= card.next_payment_date <= thirty_days:
                upcoming_payments.append({
                    "card_name": str(card.card_name),
                    "bank_name": str(card.bank_name) if card.bank_name else None,
                    "amount": round(float(card.next_payment_amount), 2),
                    "due_date": card.next_payment_date.isoformat(),
                    "days_until_due": int((card.next_payment_date - today).days)
                })

    # Sort by due date
    upcoming_payments.sort(key=lambda x: x["due_date"])

    # Calculate monthly spending on credit cards (current month only)
    today = datetime.today().date()
    first_day_of_month = today.replace(day=1)

    # Query expenses paid with credit cards in current month
    monthly_credit_spending = db.query(func.sum(models.Expense.amount)).filter(
        models.Expense.user_id == current_user.user_id,
        models.Expense.card_id.isnot(None),  # Only expenses paid with credit cards
        models.Expense.date_spent >= first_day_of_month,
        models.Expense.date_spent <= today,
        models.Expense.is_deleted == False
    ).scalar() or 0.0

    return {
        "cards": card_items,
        "summary": {
            "total_cards": len(cards),
            "total_limit": round(total_limit, 2),
            "total_balance": round(total_balance, 2),
            "total_available": round(total_available, 2),
            "utilization_pct": round(utilization_pct, 2),
            "upcoming_payments_count": len(upcoming_payments),
            "upcoming_payments_total": round(sum(p["amount"] for p in upcoming_payments), 2),
            "monthly_spending": round(float(monthly_credit_spending), 2),
        },
        "upcoming_payments": upcoming_payments[:5],  # Return up to 5 upcoming payments
    }


# ========== GET A SPECIFIC USER CARD ==========
@router.get("/{card_id}", response_model=schemas.UserCreditCardResponse)
def get_card(
    card_id: int,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get a specific credit card for the authenticated user."""
    card = db.query(models.UserCreditCard).filter(
        models.UserCreditCard.card_id == card_id,
        models.UserCreditCard.user_id == current_user.user_id,
        models.UserCreditCard.is_deleted == False
    ).first()
    if not card:
        raise HTTPException(status_code=404, detail="Card not found or access denied")
    return card


# ========== UPDATE CARD ==========
@router.put("/{card_id}", response_model=schemas.UserCreditCardResponse)
def update_card(
    card_id: int,
    updated_card: schemas.UserCreditCardUpdate,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update a credit card for the authenticated user."""
    card = db.query(models.UserCreditCard).filter(
        models.UserCreditCard.card_id == card_id,
        models.UserCreditCard.user_id == current_user.user_id,
        models.UserCreditCard.is_deleted == False
    ).first()
    if not card:
        raise HTTPException(status_code=404, detail="Card not found or access denied")

    # Update only the fields provided (exclude user_id to prevent tampering)
    for key, value in updated_card.dict(exclude_unset=True).items():
        if key != 'user_id':  # Prevent user_id modification
            setattr(card, key, value)

    db.commit()
    db.refresh(card)
    return card


# ========== DELETE CARD ==========
@router.delete("/{card_id}")
def delete_card(
    card_id: int,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete (soft delete) a credit card for the authenticated user."""
    card = db.query(models.UserCreditCard).filter(
        models.UserCreditCard.card_id == card_id,
        models.UserCreditCard.user_id == current_user.user_id,
        models.UserCreditCard.is_deleted == False
    ).first()
    if not card:
        raise HTTPException(status_code=404, detail="Card not found or access denied")

    card.is_deleted = True
    db.commit()
    return {"message": "Card deleted successfully"}


# ========== GET CARD HISTORY ==========
@router.get("/{card_id}/history")
def get_card_history(
    card_id: int,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get terms history for a specific credit card."""
    # First verify the card belongs to the user
    card = db.query(models.UserCreditCard).filter(
        models.UserCreditCard.card_id == card_id,
        models.UserCreditCard.user_id == current_user.user_id,
        models.UserCreditCard.is_deleted == False
    ).first()
    if not card:
        raise HTTPException(status_code=404, detail="Card not found or access denied")

    # Get history records for this card
    history = db.query(models.UserCreditCardTermsHistory).filter(
        models.UserCreditCardTermsHistory.card_id == card_id,
        models.UserCreditCardTermsHistory.is_deleted == False
    ).order_by(models.UserCreditCardTermsHistory.effective_date.desc()).all()

    return history


# ========== CREATE CARD TERMS HISTORY ==========
@router.post("/{card_id}/history", response_model=schemas.UserCreditCardTermsHistoryResponse)
def create_card_terms_history(
    card_id: int,
    terms: schemas.UserCreditCardTermsHistoryCreate,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new terms history record for a credit card."""
    # First verify the card belongs to the user
    card = db.query(models.UserCreditCard).filter(
        models.UserCreditCard.card_id == card_id,
        models.UserCreditCard.user_id == current_user.user_id,
        models.UserCreditCard.is_deleted == False
    ).first()
    if not card:
        raise HTTPException(status_code=404, detail="Card not found or access denied")

    # Create terms history record
    terms_data = terms.model_dump()
    terms_data['card_id'] = card_id  # Override to ensure it matches the URL parameter

    new_terms = models.UserCreditCardTermsHistory(**terms_data)
    db.add(new_terms)
    db.commit()
    db.refresh(new_terms)
    return new_terms


# ========== AI-POWERED CREDIT CARD RECOMMENDATIONS ==========
@router.get("/recommendations/ai", response_model=Dict[str, Any])
def get_ai_card_recommendations(
    max_results: int = 5,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get AI-powered credit card recommendations based on user's financial profile.

    This endpoint:
    - Analyzes user's income, spending patterns, and credit utilization
    - Queries market credit cards from MongoDB
    - Calculates match scores for each card
    - Returns top N recommendations with reasoning and highlighted benefits

    Args:
        max_results: Maximum number of recommendations to return (default: 5, max: 10)

    Returns:
        Dictionary containing:
        - recommendations: List of matched cards with scores and reasoning
        - user_profile_summary: User's financial profile used for matching
        - total_cards_analyzed: Number of cards considered
        - message: Summary message
    """
    # Validate max_results
    if max_results < 1 or max_results > 10:
        raise HTTPException(
            status_code=400,
            detail="max_results must be between 1 and 10"
        )

    # Initialize RAG service
    rag_service = RAGService(db)

    # Get recommendations
    recommendations = rag_service.recommend_credit_cards(
        user_id=current_user.user_id,
        max_results=max_results
    )

    # Check for errors
    if 'error' in recommendations:
        raise HTTPException(
            status_code=500,
            detail=recommendations.get('message', 'Failed to generate recommendations')
        )

    return recommendations

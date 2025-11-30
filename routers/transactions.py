from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from sqlalchemy import func, or_
from typing import List, Optional
from datetime import date
import logging

import models
import schemas
from database import get_db
from routers.utils import get_current_user

router = APIRouter()
logger = logging.getLogger(__name__)

NEEDS_CATEGORY_KEYWORDS = {
    "rent",
    "mortgage",
    "housing",
    "utilities",
    "electricity",
    "water",
    "internet",
    "groceries",
    "grocery",
    "supermarket",
    "transportation",
    "fuel",
    "gas",
    "public transport",
    "healthcare",
    "medical",
    "insurance",
    "education",
    "tuition",
    "loan payment",
    "debt",
    "childcare",
    "necessities",
    "bill",
    "bills",
    "phone",
    "mobile",
    "salary advance"
}

WANTS_CATEGORY_KEYWORDS = {
    "entertainment",
    "shopping",
    "shop",
    "mall",
    "travel",
    "vacation",
    "holiday",
    "luxury",
    "gifts",
    "hobbies",
    "dining out",
    "dining",
    "fancy dining",
    "fine dining",
    "coffee",
    "cafe",
    "bistro",
    "restaurant",
    "takeout",
    "subscription",
    "gym",
    "fitness",
    "electronics",
    "clothing",
    "apparel",
    "beauty",
    "spa",
    "makeup",
    "gaming",
    "dine",
    "food court",
    "foodcourt"
}

TRANSFER_KEYWORDS = {
    "internal transfer",
    "self transfer",
    "fund transfer",
    "instant transfer",
    "duitnow transfer",
    "duit now transfer",
    "ibg transfer",
    "own account",
    "own acct",
    "tabung haji",
    "asb",
    "sspni",
    "ssp1m",
    "saving",
    "savings",
    "auto-save",
    "autosave",
    "auto save",
    "standing instruction",
    "top up",
    "top-up",
    "topup",
    "cash deposit",
    "deposit to",
    "stash",
    "goal transfer",
    "rainy day",
}

TRANSFER_PAIR_HINTS = {
    "transfer": {"saving", "savings", "self", "own", "internal", "tabung", "asb", "stash", "goal", "duitnow", "duit now", "investment", "fund"},
    "deposit": {"saving", "savings", "own", "stash", "goal", "tabung", "asb"},
}


def _looks_like_transfer(text: str) -> bool:
    lowered = text.lower()
    if not lowered:
        return False

    if any(keyword in lowered for keyword in TRANSFER_KEYWORDS):
        return True

    for anchor, companions in TRANSFER_PAIR_HINTS.items():
        if anchor in lowered and any(companion in lowered for companion in companions):
            return True

    return False


def infer_expense_type(category: Optional[str], description: Optional[str], amount: Optional[float] = None) -> Optional[str]:
    """
    Basic heuristic to classify an expense into needs or wants based on category/description keywords.
    Returns None for transfer/savings shuffles so they can be excluded from spend analytics.
    Defaults to 'needs' if no clear signal found to avoid understating essentials.
    
    Large dining transactions (above RM 50) are considered wants as they are typically discretionary.
    Shopping and entertainment are always considered wants.
    """
    text = " ".join(
        segment for segment in [category or "", description or ""]
        if segment
    ).strip().lower()

    if not text:
        return "needs"

    if _looks_like_transfer(text):
        return None

    # Check for shopping first - always wants
    # Check both category and description for shopping keywords
    shopping_keywords = ["shopping", "shop", "mall", "purchase", "buy", "retail", "store", "fashion", "clothing", "apparel", "online shopping", "e-commerce", "marketplace"]
    category_lower = (category or "").lower()
    # If category is "Shopping", it's always wants
    if category_lower == "shopping":
        return "wants"
    # Also check description for shopping keywords
    if any(keyword in text for keyword in shopping_keywords):
        return "wants"

    # Check for dining - always wants if it's clearly dining out
    # Common dining keywords
    dining_keywords = ["restaurant", "cafe", "bistro", "dining", "dine", "food court", "foodcourt", "takeout", "dining out"]
    is_dining = any(keyword in text for keyword in dining_keywords)
    
    # If it's clearly a dining transaction, classify as wants (regardless of amount)
    # OR if it's a dining transaction and amount is large (above RM 50), classify as wants
    if is_dining:
        if amount is None or amount > 50:
            return "wants"
        # For smaller dining amounts, still consider wants if it's clearly dining out
        if any(keyword in text for keyword in ["restaurant", "cafe", "bistro", "dining out", "takeout"]):
            return "wants"

    # Check WANTS keywords (entertainment, travel, etc.)
    for keyword in WANTS_CATEGORY_KEYWORDS:
        if keyword in text:
            return "wants"

    # Check NEEDS keywords (but exclude if it's clearly dining)
    # Only check needs if it's not a dining-related transaction
    if not is_dining:
        for keyword in NEEDS_CATEGORY_KEYWORDS:
            if keyword in text:
                return "needs"

    # Default to needs for safety (to avoid understating essentials)
    return "needs"

# ============ INCOME ENDPOINTS ============

@router.post("/income", response_model=schemas.IncomeResponse, status_code=status.HTTP_201_CREATED)
def create_income(
    income: schemas.IncomeCreate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """Create a new income record"""
    logger.info(f"Creating income transaction - Amount: RM{income.amount:.2f}, Date: {income.date_received}, Payer: {income.payer}, Account: {income.account_id}")

    # Verify the account belongs to the user
    account = db.query(models.Account).filter(
        models.Account.account_id == income.account_id,
        models.Account.user_id == current_user.user_id
    ).first()

    if not account:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Account not found or doesn't belong to you"
        )

    # Check for duplicate income transactions (same user, account, amount, date, payer)
    existing_income = db.query(models.Income).filter(
        models.Income.user_id == current_user.user_id,
        models.Income.account_id == income.account_id,
        models.Income.amount == income.amount,
        models.Income.date_received == income.date_received,
        models.Income.payer == income.payer,
        models.Income.is_deleted == False
    ).first()

    if existing_income:
        logger.warning(f"Duplicate income detected - Existing ID: {existing_income.income_id}, Amount: RM{income.amount:.2f}, Date: {income.date_received}, Payer: {income.payer}")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Duplicate income transaction detected. A similar income (ID: {existing_income.income_id}) with the same amount (RM{income.amount:.2f}), date ({income.date_received}), and payer ({income.payer}) already exists."
        )

    # Create the income record
    db_income = models.Income(
        user_id=current_user.user_id,
        **income.model_dump(exclude_unset=True)
    )

    db.add(db_income)
    db.commit()
    db.refresh(db_income)

    logger.info(f"Successfully created income transaction - ID: {db_income.income_id}, Amount: RM{income.amount:.2f}")

    return db_income


@router.get("/income", response_model=List[schemas.IncomeResponse])
def get_all_incomes(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records to return"),
    category: Optional[str] = Query(None, description="Filter by category"),
    start_date: Optional[date] = Query(None, description="Filter by start date"),
    end_date: Optional[date] = Query(None, description="Filter by end date")
):
    """Get all income records for the current user with optional filters"""
    # Validate date range
    if start_date and end_date and start_date > end_date:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid date range: start_date ({start_date}) must be before or equal to end_date ({end_date})"
        )
    
    query = db.query(models.Income).filter(
        models.Income.user_id == current_user.user_id,
        models.Income.is_deleted == False
    )

    # Apply filters
    if category:
        query = query.filter(models.Income.category == category)
    if start_date:
        query = query.filter(models.Income.date_received >= start_date)
    if end_date:
        query = query.filter(models.Income.date_received <= end_date)

    # Order by date (most recent first) and apply pagination
    incomes = query.order_by(models.Income.date_received.desc()).offset(skip).limit(limit).all()

    return incomes


@router.get("/income/{income_id}", response_model=schemas.IncomeResponse)
def get_income_by_id(
    income_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """Get a specific income record by ID"""
    income = db.query(models.Income).filter(
        models.Income.income_id == income_id,
        models.Income.user_id == current_user.user_id,
        models.Income.is_deleted == False
    ).first()

    if not income:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Income record not found"
        )

    return income


@router.put("/income/{income_id}", response_model=schemas.IncomeResponse)
def update_income(
    income_id: int,
    income_update: schemas.IncomeUpdate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """Update an existing income record"""
    # Find the income record
    db_income = db.query(models.Income).filter(
        models.Income.income_id == income_id,
        models.Income.user_id == current_user.user_id
    ).first()

    if not db_income:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Income record not found"
        )

    # If updating account_id, verify the new account belongs to the user
    if income_update.account_id is not None:
        account = db.query(models.Account).filter(
            models.Account.account_id == income_update.account_id,
            models.Account.user_id == current_user.user_id
        ).first()

        if not account:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Account not found or doesn't belong to you"
            )

    # Update only the fields that are provided
    update_data = income_update.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_income, field, value)

    db.commit()
    db.refresh(db_income)

    return db_income


@router.delete("/income/{income_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_income(
    income_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """Delete an income record"""
    db_income = db.query(models.Income).filter(
        models.Income.income_id == income_id,
        models.Income.user_id == current_user.user_id
    ).first()

    if not db_income:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Income record not found"
        )

    db.delete(db_income)
    db.commit()

    return None


# ============ EXPENSE ENDPOINTS ============

@router.post("/expense", response_model=schemas.ExpenseResponse, status_code=status.HTTP_201_CREATED)
def create_expense(
    expense: schemas.ExpenseCreate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """Create a new expense record"""
    logger.info(f"Creating expense transaction - Amount: RM{expense.amount:.2f}, Date: {expense.date_spent}, Seller: {expense.seller}, Account: {expense.account_id}")

    # Verify the account belongs to the user
    account = db.query(models.Account).filter(
        models.Account.account_id == expense.account_id,
        models.Account.user_id == current_user.user_id
    ).first()

    if not account:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Account not found or doesn't belong to you"
        )

    # If card_id is provided, verify it belongs to the user
    if expense.card_id:
        card = db.query(models.UserCreditCard).filter(
            models.UserCreditCard.card_id == expense.card_id,
            models.UserCreditCard.user_id == current_user.user_id
        ).first()

        if not card:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Credit card not found or doesn't belong to you"
            )

    # Check for duplicate expense transactions (same user, account, amount, date, seller)
    existing_expense = db.query(models.Expense).filter(
        models.Expense.user_id == current_user.user_id,
        models.Expense.account_id == expense.account_id,
        models.Expense.amount == expense.amount,
        models.Expense.date_spent == expense.date_spent,
        models.Expense.seller == expense.seller,
        models.Expense.is_deleted == False
    ).first()

    if existing_expense:
        logger.warning(f"Duplicate expense detected - Existing ID: {existing_expense.expense_id}, Amount: RM{expense.amount:.2f}, Date: {expense.date_spent}, Seller: {expense.seller}")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Duplicate expense transaction detected. A similar expense (ID: {existing_expense.expense_id}) with the same amount (RM{expense.amount:.2f}), date ({expense.date_spent}), and seller ({expense.seller}) already exists."
        )

    # Create the expense record
    # Use absolute value for categorization (expenses are stored as negative)
    inferred_type = expense.expense_type or infer_expense_type(expense.category, expense.description, abs(expense.amount))

    db_expense = models.Expense(
        user_id=current_user.user_id,
        **expense.model_dump(exclude={"expense_type"}),
        expense_type=inferred_type
    )

    db.add(db_expense)
    db.commit()
    db.refresh(db_expense)

    logger.info(f"Successfully created expense transaction - ID: {db_expense.expense_id}, Amount: RM{expense.amount:.2f}")

    return db_expense


@router.get("/expense", response_model=List[schemas.ExpenseResponse])
def get_all_expenses(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records to return"),
    category: Optional[str] = Query(None, description="Filter by category"),
    expense_type: Optional[str] = Query(None, description="Filter by type (needs or wants)"),
    start_date: Optional[date] = Query(None, description="Filter by start date"),
    end_date: Optional[date] = Query(None, description="Filter by end date"),
    min_amount: Optional[float] = Query(None, ge=0, description="Minimum amount filter"),
    max_amount: Optional[float] = Query(None, ge=0, description="Maximum amount filter")
):
    """Get all expense records for the current user with optional filters"""
    # Validate date range
    if start_date and end_date and start_date > end_date:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid date range: start_date ({start_date}) must be before or equal to end_date ({end_date})"
        )
    
    query = db.query(models.Expense).filter(
        models.Expense.user_id == current_user.user_id,
        models.Expense.is_deleted == False
    )

    # Apply filters
    if category:
        query = query.filter(models.Expense.category == category)
    if expense_type:
        query = query.filter(models.Expense.expense_type == expense_type)
    if start_date:
        query = query.filter(models.Expense.date_spent >= start_date)
    if end_date:
        query = query.filter(models.Expense.date_spent <= end_date)
    if min_amount is not None:
        query = query.filter(models.Expense.amount >= min_amount)
    if max_amount is not None:
        query = query.filter(models.Expense.amount <= max_amount)

    # Order by date (most recent first) and apply pagination
    expenses = query.order_by(models.Expense.date_spent.desc()).offset(skip).limit(limit).all()

    return expenses


@router.get("/expense/{expense_id}", response_model=schemas.ExpenseResponse)
def get_expense_by_id(
    expense_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """Get a specific expense record by ID"""
    expense = db.query(models.Expense).filter(
        models.Expense.expense_id == expense_id,
        models.Expense.user_id == current_user.user_id,
        models.Expense.is_deleted == False
    ).first()

    if not expense:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Expense record not found"
        )

    return expense


@router.put("/expense/{expense_id}", response_model=schemas.ExpenseResponse)
def update_expense(
    expense_id: int,
    expense_update: schemas.ExpenseUpdate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """Update an existing expense record"""
    # Find the expense record
    db_expense = db.query(models.Expense).filter(
        models.Expense.expense_id == expense_id,
        models.Expense.user_id == current_user.user_id
    ).first()

    if not db_expense:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Expense record not found"
        )

    # If updating account_id, verify the new account belongs to the user
    if expense_update.account_id is not None:
        account = db.query(models.Account).filter(
            models.Account.account_id == expense_update.account_id,
            models.Account.user_id == current_user.user_id
        ).first()

        if not account:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Account not found or doesn't belong to you"
            )

    # If updating card_id, verify the new card belongs to the user
    if expense_update.card_id is not None:
        card = db.query(models.UserCreditCard).filter(
            models.UserCreditCard.card_id == expense_update.card_id,
            models.UserCreditCard.user_id == current_user.user_id
        ).first()

        if not card:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Credit card not found or doesn't belong to you"
            )

    # Update only the fields that are provided
    update_data = expense_update.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_expense, field, value)

    if "expense_type" not in update_data:
        needs_reclassify = any(field in update_data for field in ("category", "description", "amount"))
        if needs_reclassify and (db_expense.category or db_expense.description):
            # Use absolute value for categorization (expenses are stored as negative)
            db_expense.expense_type = infer_expense_type(db_expense.category, db_expense.description, abs(db_expense.amount))

    db.commit()
    db.refresh(db_expense)

    return db_expense


@router.delete("/expense/{expense_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_expense(
    expense_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """Delete an expense record"""
    db_expense = db.query(models.Expense).filter(
        models.Expense.expense_id == expense_id,
        models.Expense.user_id == current_user.user_id
    ).first()

    if not db_expense:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Expense record not found"
        )

    db.delete(db_expense)
    db.commit()

    return None


# ============ BULK DELETE ENDPOINTS ============

@router.post("/bulk-delete", status_code=status.HTTP_200_OK)
def bulk_delete_transactions(
    request: dict,  # Request body with transaction_ids list
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """
    Delete multiple transactions in bulk.
    Accepts transaction IDs in format "income-{id}", "expense-{id}", or "transfer-{id}"
    Request body: {"transaction_ids": ["income-123", "expense-456", "transfer-789", ...]}
    """
    transaction_ids = request.get("transaction_ids", [])
    if not transaction_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="transaction_ids list is required"
        )
    
    deleted_incomes = 0
    deleted_expenses = 0
    deleted_transfers = 0
    not_found = []

    for transaction_id in transaction_ids:
        try:
            if transaction_id.startswith("income-"):
                income_id = int(transaction_id.replace("income-", ""))
                db_income = db.query(models.Income).filter(
                    models.Income.income_id == income_id,
                    models.Income.user_id == current_user.user_id,
                    models.Income.is_deleted == False
                ).first()
                if db_income:
                    db_income.is_deleted = True
                    deleted_incomes += 1
                else:
                    not_found.append(transaction_id)
            elif transaction_id.startswith("expense-"):
                expense_id = int(transaction_id.replace("expense-", ""))
                db_expense = db.query(models.Expense).filter(
                    models.Expense.expense_id == expense_id,
                    models.Expense.user_id == current_user.user_id,
                    models.Expense.is_deleted == False
                ).first()
                if db_expense:
                    db_expense.is_deleted = True
                    deleted_expenses += 1
                else:
                    not_found.append(transaction_id)
            elif transaction_id.startswith("transfer-"):
                transfer_id = int(transaction_id.replace("transfer-", ""))
                db_transfer = db.query(models.Transfer).filter(
                    models.Transfer.transfer_id == transfer_id,
                    models.Transfer.user_id == current_user.user_id,
                    models.Transfer.is_deleted == False
                ).first()
                if db_transfer:
                    db_transfer.is_deleted = True
                    deleted_transfers += 1
                else:
                    not_found.append(transaction_id)
            else:
                not_found.append(transaction_id)
        except (ValueError, AttributeError):
            not_found.append(transaction_id)

    db.commit()

    return {
        "success": True,
        "deleted_incomes": deleted_incomes,
        "deleted_expenses": deleted_expenses,
        "deleted_transfers": deleted_transfers,
        "total_deleted": deleted_incomes + deleted_expenses + deleted_transfers,
        "not_found": not_found
    }


@router.delete("/delete-all", status_code=status.HTTP_200_OK)
def delete_all_transactions(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """
    Delete all transactions for the current user.
    This is a destructive operation - use with caution!
    """
    # Soft delete all incomes
    deleted_incomes = db.query(models.Income).filter(
        models.Income.user_id == current_user.user_id,
        models.Income.is_deleted == False
    ).update({"is_deleted": True})

    # Soft delete all expenses
    deleted_expenses = db.query(models.Expense).filter(
        models.Expense.user_id == current_user.user_id,
        models.Expense.is_deleted == False
    ).update({"is_deleted": True})

    # Soft delete all transfers
    deleted_transfers = db.query(models.Transfer).filter(
        models.Transfer.user_id == current_user.user_id,
        models.Transfer.is_deleted == False
    ).update({"is_deleted": True})

    db.commit()

    return {
        "success": True,
        "deleted_incomes": deleted_incomes,
        "deleted_expenses": deleted_expenses,
        "deleted_transfers": deleted_transfers,
        "total_deleted": deleted_incomes + deleted_expenses + deleted_transfers,
        "message": f"Successfully deleted {deleted_incomes + deleted_expenses + deleted_transfers} transactions"
    }


@router.post("/expense/reclassify")
def reclassify_expenses(
    force: bool = Query(False, description="Reclassify even if an expense already has a type"),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    """
    Re-run needs vs wants inference across existing expenses for the user.
    By default only fills in missing types; set force=true to overwrite existing labels.
    """
    expenses = db.query(models.Expense).filter(
        models.Expense.user_id == current_user.user_id,
        models.Expense.is_deleted.is_(False),
    ).all()

    total = len(expenses)
    updated = 0

    for expense in expenses:
        # Use absolute value for categorization (expenses are stored as negative)
        inferred = infer_expense_type(expense.category, expense.description, abs(expense.amount))
        if force or expense.expense_type is None:
            if expense.expense_type != inferred:
                expense.expense_type = inferred
                updated += 1

    if updated > 0:
        db.commit()

    return {
        "total_expenses": total,
        "updated_expenses": updated,
        "force_mode": force,
    }

# ============ TRANSFER ENDPOINTS ============

@router.post("/transfer", response_model=schemas.TransferResponse, status_code=status.HTTP_201_CREATED)
def create_transfer(
    transfer: schemas.TransferCreate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """Create a new transfer record"""
    # Verify the account belongs to the user
    account = db.query(models.Account).filter(
        models.Account.account_id == transfer.account_id,
        models.Account.user_id == current_user.user_id
    ).first()

    if not account:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Account not found or doesn't belong to you"
        )

    # Create the transfer record
    db_transfer = models.Transfer(
        user_id=current_user.user_id,
        **transfer.model_dump(exclude_unset=True)
    )

    db.add(db_transfer)
    db.commit()
    db.refresh(db_transfer)

    return db_transfer


@router.get("/transfer", response_model=List[schemas.TransferResponse])
def get_all_transfers(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records to return"),
    transfer_type: Optional[str] = Query(None, description="Filter by transfer type (intra_person or inter_person)"),
    start_date: Optional[date] = Query(None, description="Filter by start date"),
    end_date: Optional[date] = Query(None, description="Filter by end date")
):
    """Get all transfer records for the current user with optional filters"""
    # Validate date range
    if start_date and end_date and start_date > end_date:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid date range: start_date ({start_date}) must be before or equal to end_date ({end_date})"
        )
    
    query = db.query(models.Transfer).filter(
        models.Transfer.user_id == current_user.user_id,
        models.Transfer.is_deleted == False
    )

    # Apply filters
    if transfer_type:
        if transfer_type not in ['intra_person', 'inter_person']:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="transfer_type must be 'intra_person' or 'inter_person'"
            )
        query = query.filter(models.Transfer.transfer_type == transfer_type)

    if start_date:
        query = query.filter(models.Transfer.date_transferred >= start_date)
    
    if end_date:
        query = query.filter(models.Transfer.date_transferred <= end_date)

    transfers = query.order_by(models.Transfer.date_transferred.desc()).offset(skip).limit(limit).all()
    return transfers


@router.get("/transfer/{transfer_id}", response_model=schemas.TransferResponse)
def get_transfer_by_id(
    transfer_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """Get a specific transfer by ID"""
    transfer = db.query(models.Transfer).filter(
        models.Transfer.transfer_id == transfer_id,
        models.Transfer.user_id == current_user.user_id,
        models.Transfer.is_deleted == False
    ).first()

    if not transfer:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Transfer not found"
        )

    return transfer


@router.put("/transfer/{transfer_id}", response_model=schemas.TransferResponse)
def update_transfer(
    transfer_id: int,
    transfer_update: schemas.TransferUpdate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """Update a transfer record"""
    transfer = db.query(models.Transfer).filter(
        models.Transfer.transfer_id == transfer_id,
        models.Transfer.user_id == current_user.user_id,
        models.Transfer.is_deleted == False
    ).first()

    if not transfer:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Transfer not found"
        )

    # Verify account if being updated
    if transfer_update.account_id and transfer_update.account_id != transfer.account_id:
        account = db.query(models.Account).filter(
            models.Account.account_id == transfer_update.account_id,
            models.Account.user_id == current_user.user_id
        ).first()

        if not account:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Account not found or doesn't belong to you"
            )

    # Update fields
    update_data = transfer_update.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(transfer, field, value)

    db.commit()
    db.refresh(transfer)

    return transfer


@router.delete("/transfer/{transfer_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_transfer(
    transfer_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """Soft delete a transfer record"""
    transfer = db.query(models.Transfer).filter(
        models.Transfer.transfer_id == transfer_id,
        models.Transfer.user_id == current_user.user_id,
        models.Transfer.is_deleted == False
    ).first()

    if not transfer:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Transfer not found"
        )

    transfer.is_deleted = True
    db.commit()

    return None

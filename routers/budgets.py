# ============ IMPORTS ============
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from sqlalchemy import and_, desc, func
from typing import List, Optional
from datetime import date
import models
import schemas
from database import get_db
from routers.utils import get_current_user

# ============ ROUTER SETUP ============
# Create router instance for budget-related endpoints
# Note: No prefix here since it's added in main.py
router = APIRouter()

# ============ HELPER FUNCTIONS ============

def calculate_expenses_by_category(
    db: Session,
    user_id: int,
    category: str,
    start_date: date,
    end_date: date
) -> float:
    """
    Calculate total expenses for a user in a specific category within a date range.
    Only includes non-deleted expenses from actual transactions.

    Handles backward compatibility by checking both new and old category names:
    - "Food & Dining" will match both "Food & Dining" and "Food"
    - "Others" will match both "Others" and "Other"

    Args:
        db: Database session
        user_id: User ID to filter expenses
        category: Expense category to filter by
        start_date: Start date for the period
        end_date: End date for the period

    Returns:
        float: Total expense amount (only from active transactions)
    """
    # Reverse mapping: new category -> old category
    category_reverse_map = {
        "Food & Dining": "Food",
        "Others": "Other",
    }

    # Build list of categories to check (both new and old if applicable)
    categories_to_check = [category]
    if category in category_reverse_map:
        categories_to_check.append(category_reverse_map[category])

    # Sum expense amounts (expenses are stored as positive values in DB)
    # Check both new and old category names for backward compatibility
    total = db.query(func.sum(models.Expense.amount)).filter(
        and_(
            models.Expense.user_id == user_id,
            models.Expense.category.in_(categories_to_check),  # Match either new or old category
            models.Expense.date_spent >= start_date,
            models.Expense.date_spent <= end_date,
            models.Expense.is_deleted == False  # Exclude soft-deleted transactions
        )
    ).scalar()

    return float(total) if total else 0.0

def calculate_budget_utilization(
    db: Session,
    user_id: int,
    budget: models.Budget
) -> dict:
    """
    Calculate budget utilization metrics including spending and alerts.
    
    Args:
        db: Database session
        user_id: User ID
        budget: Budget model instance
        
    Returns:
        dict: Budget utilization metrics
    """
    spent_amount = calculate_expenses_by_category(
        db, user_id, budget.category, budget.period_start, budget.period_end
    )
    
    remaining_amount = budget.limit_amount - spent_amount
    percentage_used = (spent_amount / budget.limit_amount) * 100 if budget.limit_amount > 0 else 0
    
    # Determine status and alert type
    if spent_amount >= budget.limit_amount:
        status = "over_budget"
        alert_type = "danger"
    elif percentage_used >= budget.alert_threshold:
        status = "at_risk"
        alert_type = "warning"
    else:
        status = "on_track"
        alert_type = "info"
    
    # Calculate days remaining
    today = date.today()
    if budget.period_end >= today:
        days_remaining = (budget.period_end - today).days
    else:
        days_remaining = 0
    
    # Calculate daily allowance
    if days_remaining > 0:
        daily_allowance = remaining_amount / days_remaining
    else:
        daily_allowance = 0.0
    
    return {
        "spent_amount": spent_amount,
        "remaining_amount": remaining_amount,
        "percentage_used": percentage_used,
        "status": status,
        "days_remaining": days_remaining,
        "daily_allowance": daily_allowance,
        "alert_type": alert_type
    }


# ============ CRUD ENDPOINTS ============

@router.get("/", response_model=schemas.BudgetList)
async def get_budgets(
    skip: int = Query(0, ge=0, description="Number of budgets to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Number of budgets to return"),
    category: Optional[str] = Query(None, description="Filter by category"),
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get all budgets for the authenticated user with optional filtering and pagination.
    
    Args:
        skip: Number of budgets to skip for pagination
        limit: Maximum number of budgets to return
        category: Optional filter by budget category
        current_user: Authenticated user (from dependency)
        db: Database session (from dependency)
        
    Returns:
        schemas.BudgetList: Paginated list of budgets with metadata
    """
    # Start with base query for current user's budgets
    query = db.query(models.Budget).filter(models.Budget.user_id == current_user.user_id)
    
    # Apply optional category filter
    if category:
        query = query.filter(models.Budget.category == category)
    
    # Get total count for pagination metadata
    total = query.count()
    
    # Apply pagination and sorting (newest first)
    budgets = query.order_by(desc(models.Budget.created)).offset(skip).limit(limit).all()
    
    # Calculate spending for each budget and convert to response format
    budget_responses = []
    for budget in budgets:
        utilization = calculate_budget_utilization(db, current_user.user_id, budget)
        budget_dict = {
            "budget_id": budget.budget_id,
            "user_id": budget.user_id,
            "name": budget.name,
            "category": budget.category,
            "limit_amount": budget.limit_amount,
            "period_start": budget.period_start,
            "period_end": budget.period_end,
            "alert_threshold": budget.alert_threshold,
            "created": budget.created,
            "spent_amount": utilization["spent_amount"],
            "remaining_amount": utilization["remaining_amount"],
            "percentage_used": utilization["percentage_used"],
            "status": utilization["status"],
            "days_remaining": utilization["days_remaining"],
            "daily_allowance": utilization["daily_allowance"],
            "alert_type": utilization["alert_type"],
        }
        budget_responses.append(schemas.BudgetResponse.model_validate(budget_dict))
    
    return schemas.BudgetList(
        budgets=budget_responses,
        total=total,
        skip=skip,
        limit=limit,
        has_more=skip + limit < total
    )


@router.get("", response_model=schemas.BudgetList)
async def get_budgets_without_trailing_slash(
    skip: int = Query(0, ge=0, description="Number of budgets to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Number of budgets to return"),
    category: Optional[str] = Query(None, description="Filter by category"),
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Convenience alias for clients hitting `/budgets` without a trailing slash."""
    return await get_budgets(skip=skip, limit=limit, category=category, current_user=current_user, db=db)


# ============ UTILITY ENDPOINTS ============

@router.get("/categories", response_model=List[str])
async def get_budget_categories():
    """
    Get list of available budget categories.
    These categories are predefined and validated in the schemas.
    
    Returns:
        List[str]: Array of valid budget category names
    """
    return schemas.BUDGET_CATEGORIES

@router.get("/periods", response_model=List[str])
async def get_budget_periods():
    """
    Get list of available budget periods.
    These periods are predefined and validated in the schemas.
    
    Returns:
        List[str]: Array of valid period options
    """
    return schemas.BUDGET_PERIODS


@router.get("/{budget_id}", response_model=schemas.BudgetResponse)
async def get_budget(
    budget_id: int,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get a specific budget by ID for the authenticated user with calculated spending.
    
    Args:
        budget_id: ID of the budget to retrieve
        current_user: Authenticated user (from dependency)
        db: Database session (from dependency)
        
    Returns:
        schemas.BudgetResponse: The requested budget data with calculated spending metrics
        
    Raises:
        HTTPException: 404 if budget not found or doesn't belong to user
    """
    # Query budget with user ownership check
    budget = db.query(models.Budget).filter(
        and_(
            models.Budget.budget_id == budget_id,
            models.Budget.user_id == current_user.user_id
        )
    ).first()
    
    if not budget:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Budget not found"
        )
    
    # Calculate spending for the budget
    utilization = calculate_budget_utilization(db, current_user.user_id, budget)
    budget_dict = {
        "budget_id": budget.budget_id,
        "user_id": budget.user_id,
        "name": budget.name,
        "category": budget.category,
        "limit_amount": budget.limit_amount,
        "period_start": budget.period_start,
        "period_end": budget.period_end,
        "alert_threshold": budget.alert_threshold,
        "created": budget.created,
        "spent_amount": utilization["spent_amount"],
        "remaining_amount": utilization["remaining_amount"],
        "percentage_used": utilization["percentage_used"],
        "status": utilization["status"],
        "days_remaining": utilization["days_remaining"],
        "daily_allowance": utilization["daily_allowance"],
        "alert_type": utilization["alert_type"],
    }
    return schemas.BudgetResponse.model_validate(budget_dict)


@router.post("/", response_model=schemas.BudgetResponse, status_code=status.HTTP_201_CREATED)
async def create_budget(
    budget_data: schemas.BudgetCreate,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Create a new budget for the authenticated user.
    
    Args:
        budget_data: Budget creation data (validated by Pydantic)
        current_user: Authenticated user (from dependency)
        db: Database session (from dependency)
        
    Returns:
        schemas.BudgetResponse: The created budget data
        
    Note:
        All budget data is validated by the BudgetCreate schema including
        category validation against predefined list and period validation.
    """
    # Create new budget instance
    budget = models.Budget(
        user_id=current_user.user_id,
        name=budget_data.name,
        limit_amount=budget_data.limit_amount,
        category=budget_data.category,
        period_start=budget_data.period_start,
        period_end=budget_data.period_end,
        alert_threshold=budget_data.alert_threshold
    )
    
    # Save to database
    db.add(budget)
    db.commit()
    db.refresh(budget)
    
    # Calculate spending for the newly created budget
    utilization = calculate_budget_utilization(db, current_user.user_id, budget)
    budget_dict = {
        "budget_id": budget.budget_id,
        "user_id": budget.user_id,
        "name": budget.name,
        "category": budget.category,
        "limit_amount": budget.limit_amount,
        "period_start": budget.period_start,
        "period_end": budget.period_end,
        "alert_threshold": budget.alert_threshold,
        "created": budget.created,
        "spent_amount": utilization["spent_amount"],
        "remaining_amount": utilization["remaining_amount"],
        "percentage_used": utilization["percentage_used"],
        "status": utilization["status"],
        "days_remaining": utilization["days_remaining"],
        "daily_allowance": utilization["daily_allowance"],
        "alert_type": utilization["alert_type"],
    }
    return schemas.BudgetResponse.model_validate(budget_dict)


@router.put("/{budget_id}", response_model=schemas.BudgetResponse)
async def update_budget(
    budget_id: int,
    budget_data: schemas.BudgetUpdate,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Update an existing budget for the authenticated user.
    
    Args:
        budget_id: ID of the budget to update
        budget_data: Budget update data (validated by Pydantic)
        current_user: Authenticated user (from dependency)
        db: Database session (from dependency)
        
    Returns:
        schemas.BudgetResponse: The updated budget data
        
    Raises:
        HTTPException: 404 if budget not found or doesn't belong to user
        
    Note:
        Only provided fields will be updated (partial updates supported).
        All data is validated by the BudgetUpdate schema.
    """
    # Find budget with ownership check
    budget = db.query(models.Budget).filter(
        and_(
            models.Budget.budget_id == budget_id,
            models.Budget.user_id == current_user.user_id
        )
    ).first()
    
    if not budget:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Budget not found"
        )
    
    # Update only provided fields
    update_data = budget_data.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(budget, field, value)
    
    # Save changes
    db.commit()
    db.refresh(budget)
    
    # Calculate spending for the updated budget
    utilization = calculate_budget_utilization(db, current_user.user_id, budget)
    budget_dict = {
        "budget_id": budget.budget_id,
        "user_id": budget.user_id,
        "name": budget.name,
        "category": budget.category,
        "limit_amount": budget.limit_amount,
        "period_start": budget.period_start,
        "period_end": budget.period_end,
        "alert_threshold": budget.alert_threshold,
        "created": budget.created,
        "spent_amount": utilization["spent_amount"],
        "remaining_amount": utilization["remaining_amount"],
        "percentage_used": utilization["percentage_used"],
        "status": utilization["status"],
        "days_remaining": utilization["days_remaining"],
        "daily_allowance": utilization["daily_allowance"],
        "alert_type": utilization["alert_type"],
    }
    return schemas.BudgetResponse.model_validate(budget_dict)


@router.delete("/{budget_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_budget(
    budget_id: int,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Delete a budget for the authenticated user.
    
    Args:
        budget_id: ID of the budget to delete
        current_user: Authenticated user (from dependency)
        db: Database session (from dependency)
        
    Raises:
        HTTPException: 404 if budget not found or doesn't belong to user
        
    Note:
        Returns 204 No Content on successful deletion.
    """
    # Find budget with ownership check
    budget = db.query(models.Budget).filter(
        and_(
            models.Budget.budget_id == budget_id,
            models.Budget.user_id == current_user.user_id
        )
    ).first()
    
    if not budget:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Budget not found"
        )
    
    # Delete budget
    db.delete(budget)
    db.commit()


# ============ ANALYTICS ENDPOINTS ============

@router.get("/summary/overview", response_model=schemas.BudgetSummary)
async def get_budget_summary(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get comprehensive budget summary with real spending calculations.
    
    Args:
        current_user: Authenticated user (from dependency)
        db: Database session (from dependency)
        
    Returns:
        schemas.BudgetSummary: Complete budget overview with spending metrics
    """
    # Get all budgets for the user
    budgets = db.query(models.Budget).filter(
        models.Budget.user_id == current_user.user_id
    ).all()
    
    if not budgets:
        return schemas.BudgetSummary(
            total_budgets=0,
            active_budgets=0,
            total_budget_amount=0.0,
            total_spent_amount=0.0,
            total_remaining_amount=0.0,
            budgets_over_budget=0,
            budgets_at_risk=0,
            average_utilization=0.0
        )
    
    # Calculate summary metrics
    total_budgets = len(budgets)
    total_budget_amount = sum(b.limit_amount for b in budgets)
    total_spent_amount = 0.0
    budgets_over_budget = 0
    budgets_at_risk = 0
    total_utilization = 0.0
    
    for budget in budgets:
        utilization_data = calculate_budget_utilization(db, current_user.user_id, budget)
        spent = utilization_data["spent_amount"]
        percentage_used = utilization_data["percentage_used"]
        
        total_spent_amount += spent
        
        # Count budget statuses
        if spent >= budget.limit_amount:
            budgets_over_budget += 1
        elif percentage_used >= budget.alert_threshold:
            budgets_at_risk += 1
        
        total_utilization += percentage_used
    
    # Calculate final metrics
    total_remaining_amount = total_budget_amount - total_spent_amount
    active_budgets = total_budgets  # All budgets are considered active
    average_utilization = total_utilization / total_budgets if total_budgets > 0 else 0.0
    
    return schemas.BudgetSummary(
        total_budgets=total_budgets,
        active_budgets=active_budgets,
        total_budget_amount=total_budget_amount,
        total_spent_amount=total_spent_amount,
        total_remaining_amount=total_remaining_amount,
        budgets_over_budget=budgets_over_budget,
        budgets_at_risk=budgets_at_risk,
        average_utilization=average_utilization
    )

@router.get("/{budget_id}/details", response_model=schemas.BudgetAlert)
async def get_budget_details(
    budget_id: int,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get detailed budget performance with spending analysis and alerts.
    
    Args:
        budget_id: ID of the budget to get details for
        current_user: Authenticated user (from dependency)
        db: Database session (from dependency)
        
    Returns:
        schemas.BudgetAlert: Detailed budget performance with spending metrics
        
    Raises:
        HTTPException: 404 if budget not found or doesn't belong to user
    """
    # Find budget with ownership check
    budget = db.query(models.Budget).filter(
        and_(
            models.Budget.budget_id == budget_id,
            models.Budget.user_id == current_user.user_id
        )
    ).first()
    
    if not budget:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Budget not found"
        )
    
    # Calculate budget utilization
    utilization_data = calculate_budget_utilization(db, current_user.user_id, budget)
    
    return schemas.BudgetAlert(
        budget_id=budget.budget_id,
        name=budget.name,
        category=budget.category,
        limit_amount=budget.limit_amount,
        spent_amount=utilization_data["spent_amount"],
        remaining_amount=utilization_data["remaining_amount"],
        percentage_used=utilization_data["percentage_used"],
        status=utilization_data["status"],
        days_remaining=utilization_data["days_remaining"],
        daily_allowance=utilization_data["daily_allowance"],
        alert_type=utilization_data["alert_type"]
    )

@router.get("/alerts/active", response_model=List[schemas.BudgetAlert])
async def get_active_budget_alerts(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get active budget alerts for budgets that are over budget or at risk.
    
    Args:
        current_user: Authenticated user (from dependency)
        db: Database session (from dependency)
        
    Returns:
        List[schemas.BudgetAlert]: List of budgets that need attention
    """
    # Get all budgets for the user
    budgets = db.query(models.Budget).filter(
        models.Budget.user_id == current_user.user_id
    ).all()
    
    alerts = []
    for budget in budgets:
        utilization_data = calculate_budget_utilization(db, current_user.user_id, budget)
        
        # Only include budgets that need attention (over budget or at risk)
        if utilization_data["alert_type"] in ["danger", "warning"]:
            alert = schemas.BudgetAlert(
                budget_id=budget.budget_id,
                name=budget.name,
                category=budget.category,
                limit_amount=budget.limit_amount,
                spent_amount=utilization_data["spent_amount"],
                remaining_amount=utilization_data["remaining_amount"],
                percentage_used=utilization_data["percentage_used"],
                status=utilization_data["status"],
                days_remaining=utilization_data["days_remaining"],
                daily_allowance=utilization_data["daily_allowance"],
                alert_type=utilization_data["alert_type"]
            )
            alerts.append(alert)
    
    # Sort alerts by severity (danger first, then warning)
    alerts.sort(key=lambda x: (x.alert_type == "warning", x.percentage_used), reverse=True)
    
    return alerts
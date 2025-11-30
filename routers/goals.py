# ============ IMPORTS ============
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_
from typing import List, Optional
from datetime import date, datetime, timedelta
import math
import models
import schemas
from database import get_db
from routers.utils import get_current_user

# ============ ROUTER SETUP ============
# Create router instance for goal-related endpoints
# Note: No prefix here since it's added in main.py
router = APIRouter(tags=["Goals"])

# ============ HELPER FUNCTIONS ============

def calculate_goal_metrics(goal: models.Goal) -> dict:
    """
    Calculate various metrics for a goal including progress, time remaining, and completion status.
    
    Args:
        goal: The Goal model instance to calculate metrics for
        
    Returns:
        dict: Dictionary containing calculated metrics
            - progress_percentage: Percentage of target amount achieved
            - days_remaining: Days left until target date (if set)
            - monthly_required: Monthly contribution needed to reach target on time
            - is_completed: Boolean indicating if goal is fully funded
    """
    # Calculate progress percentage (0-100)
    progress_percentage = (goal.current_amount / goal.target_amount) * 100 if goal.target_amount > 0 else 0
    
    # Initialize optional time-based calculations
    days_remaining = None
    monthly_required = None
    
    # Calculate time-based metrics if target date is set
    if goal.target_date:
        today = date.today()
        days_remaining = (goal.target_date - today).days
        
        # Calculate monthly contribution needed if goal is not yet complete
        if days_remaining > 0 and goal.current_amount < goal.target_amount:
            remaining_amount = goal.target_amount - goal.current_amount
            months_remaining = max(1, math.ceil(days_remaining / 30))  # At least 1 month
            monthly_required = remaining_amount / months_remaining
    
    # Determine if goal is completed
    is_completed = goal.current_amount >= goal.target_amount
    
    return {
        "progress_percentage": round(progress_percentage, 2),
        "days_remaining": days_remaining,
        "monthly_required": round(monthly_required, 2) if monthly_required else None,
        "is_completed": is_completed
    }

def create_goal_response(goal: models.Goal) -> schemas.GoalResponse:
    """
    Create a GoalResponse schema from a Goal model instance.
    Includes calculated metrics for progress tracking.
    
    Args:
        goal: The Goal model instance to convert
        
    Returns:
        schemas.GoalResponse: Formatted response with all goal data and calculated metrics
    """
    # Calculate dynamic metrics for the goal
    metrics = calculate_goal_metrics(goal)
    
    # Create response object with all goal data and calculated metrics
    return schemas.GoalResponse(
        # Basic goal information from model
        goal_id=goal.goal_id,
        user_id=goal.user_id,
        goal_name=goal.goal_name,
        description=goal.description,
        category=goal.category,
        priority=goal.priority,
        target_amount=goal.target_amount,
        current_amount=goal.current_amount,
        target_date=goal.target_date,
        created_at=goal.created_at,
        
        # Calculated metrics from helper function
        is_completed=metrics["is_completed"],
        progress_percentage=metrics["progress_percentage"],
        days_remaining=metrics["days_remaining"],
        monthly_required=metrics["monthly_required"]
    )

# ============ GOAL ENDPOINTS ============

@router.post("/", response_model=schemas.GoalResponse, status_code=status.HTTP_201_CREATED)
def create_goal(
    goal: schemas.GoalCreate,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Create a new financial goal for the authenticated user.
    
    Args:
        goal: Goal creation data from request body
        current_user: Authenticated user (from dependency)
        db: Database session (from dependency)
        
    Returns:
        schemas.GoalResponse: Created goal with calculated metrics
        
    Raises:
        HTTPException: If validation fails or database error occurs
    """
    # Create new Goal model instance with user data
    db_goal = models.Goal(
        user_id=current_user.user_id,  # Associate with current user
        goal_name=goal.goal_name,
        description=goal.description,
        category=goal.category,
        priority=goal.priority,
        target_amount=goal.target_amount,
        current_amount=goal.current_amount,
        target_date=goal.target_date
    )
    
    # Save to database
    db.add(db_goal)
    db.commit()
    db.refresh(db_goal)  # Refresh to get auto-generated fields (like ID)
    
    # Return formatted response with calculated metrics
    return create_goal_response(db_goal)

# ============ UTILITY ENDPOINTS ============

@router.get("/categories", response_model=List[str])
def get_goal_categories():
    """
    Get list of available goal categories.
    These categories are predefined and validated in the schemas.
    
    Returns:
        List[str]: Array of valid goal category names
    """
    return schemas.GOAL_CATEGORIES

@router.get("/priorities", response_model=List[str])
def get_goal_priorities():
    """
    Get list of available goal priority levels.
    These priorities are predefined and validated in the schemas.
    
    Returns:
        List[str]: Array of valid priority levels (low, medium, high)
    """
    return schemas.GOAL_PRIORITIES

# ============ CRUD ENDPOINTS ============

@router.get("/", response_model=schemas.GoalStats)
def get_goals(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
    completed: Optional[bool] = Query(None, description="Filter by completion status"),
    category: Optional[str] = Query(None, description="Filter by category"),
    priority: Optional[str] = Query(None, description="Filter by priority")
):
    """
    Get all goals for the authenticated user with optional filtering and summary statistics.
    
    Args:
        current_user: Authenticated user (from dependency)
        db: Database session (from dependency)
        completed: Optional filter for completed/incomplete goals
        category: Optional filter by goal category
        priority: Optional filter by goal priority
        
    Returns:
        schemas.GoalStats: Summary statistics and filtered list of goals with calculated metrics
    """
    # Start with base query for current user's goals
    query = db.query(models.Goal).filter(models.Goal.user_id == current_user.user_id)
    
    # Apply optional filters
    if completed is not None:
        if completed:
            # Filter for completed goals (current_amount >= target_amount)
            query = query.filter(models.Goal.current_amount >= models.Goal.target_amount)
        else:
            # Filter for incomplete goals (current_amount < target_amount)
            query = query.filter(models.Goal.current_amount < models.Goal.target_amount)
    
    if category:
        query = query.filter(models.Goal.category == category)
    
    if priority:
        query = query.filter(models.Goal.priority == priority)
    
    # Execute query to get filtered goals
    goals = query.all()
    
    # Calculate summary statistics for all goals (not just filtered ones)
    total_goals = len(goals)
    active_goals = len([g for g in goals if g.current_amount < g.target_amount])
    completed_goals = total_goals - active_goals
    
    # Calculate financial totals
    total_target_amount = sum(g.target_amount for g in goals)
    total_current_amount = sum(g.current_amount for g in goals)
    overall_progress = (total_current_amount / total_target_amount * 100) if total_target_amount > 0 else 0
    
    # Create summary statistics object
    summary = schemas.GoalSummary(
        total_goals=total_goals,
        active_goals=active_goals,
        completed_goals=completed_goals,
        total_target_amount=round(total_target_amount, 2),
        total_current_amount=round(total_current_amount, 2),
        overall_progress_percentage=round(overall_progress, 2)
    )
    
    # Convert goals to response format with calculated metrics
    goal_responses = [create_goal_response(goal) for goal in goals]
    
    # Return summary and goal list
    return schemas.GoalStats(summary=summary, goals=goal_responses)

@router.get("/{goal_id}", response_model=schemas.GoalResponse)
def get_goal(
    goal_id: int,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get a specific goal by ID for the authenticated user.
    
    Args:
        goal_id: The ID of the goal to retrieve
        current_user: Authenticated user (from dependency)
        db: Database session (from dependency)
        
    Returns:
        schemas.GoalResponse: Goal data with calculated metrics
        
    Raises:
        HTTPException: 404 if goal not found or doesn't belong to user
    """
    # Query for goal with both ID and user ownership verification
    goal = db.query(models.Goal).filter(
        and_(
            models.Goal.goal_id == goal_id,
            models.Goal.user_id == current_user.user_id  # Ensure user owns this goal
        )
    ).first()
    
    # Return 404 if goal doesn't exist or doesn't belong to user
    if not goal:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Goal not found"
        )
    
    # Return goal with calculated metrics
    return create_goal_response(goal)

@router.put("/{goal_id}", response_model=schemas.GoalResponse)
def update_goal(
    goal_id: int,
    goal_update: schemas.GoalUpdate,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Update a specific goal by ID for the authenticated user.
    Only updates fields that are provided in the request body (partial update).
    
    Args:
        goal_id: The ID of the goal to update
        goal_update: Update data from request body (optional fields)
        current_user: Authenticated user (from dependency)
        db: Database session (from dependency)
        
    Returns:
        schemas.GoalResponse: Updated goal with recalculated metrics
        
    Raises:
        HTTPException: 404 if goal not found or doesn't belong to user
    """
    # Query for goal with both ID and user ownership verification
    goal = db.query(models.Goal).filter(
        and_(
            models.Goal.goal_id == goal_id,
            models.Goal.user_id == current_user.user_id  # Ensure user owns this goal
        )
    ).first()
    
    # Return 404 if goal doesn't exist or doesn't belong to user
    if not goal:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Goal not found"
        )
    
    # Update only provided fields (partial update using Pydantic v2 method)
    update_data = goal_update.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(goal, field, value)
    
    # Save changes to database
    db.commit()
    db.refresh(goal)  # Refresh to get any auto-updated fields
    
    # Return updated goal with recalculated metrics
    return create_goal_response(goal)

@router.delete("/{goal_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_goal(
    goal_id: int,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Delete a specific goal by ID for the authenticated user.
    
    Args:
        goal_id: The ID of the goal to delete
        current_user: Authenticated user (from dependency)
        db: Database session (from dependency)
        
    Returns:
        None: 204 No Content on successful deletion
        
    Raises:
        HTTPException: 404 if goal not found or doesn't belong to user
    """
    # Query for goal with both ID and user ownership verification
    goal = db.query(models.Goal).filter(
        and_(
            models.Goal.goal_id == goal_id,
            models.Goal.user_id == current_user.user_id  # Ensure user owns this goal
        )
    ).first()
    
    # Return 404 if goal doesn't exist or doesn't belong to user
    if not goal:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Goal not found"
        )
    
    # Delete the goal from database
    db.delete(goal)
    db.commit()
    
    # Return 204 No Content (no response body)
    return None

# ============ ACTION ENDPOINTS ============

@router.post("/{goal_id}/contribute", response_model=schemas.GoalResponse)
def contribute_to_goal(
    goal_id: int,
    contribution: schemas.GoalContribute,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Add money to a specific goal (contribution functionality).
    This increases the current_amount of the goal and recalculates metrics.
    
    Args:
        goal_id: The ID of the goal to contribute to
        contribution: Contribution data containing the amount to add
        current_user: Authenticated user (from dependency)
        db: Database session (from dependency)
        
    Returns:
        schemas.GoalResponse: Updated goal with new current_amount and recalculated metrics
        
    Raises:
        HTTPException: 404 if goal not found or doesn't belong to user
    """
    # Query for goal with both ID and user ownership verification
    goal = db.query(models.Goal).filter(
        and_(
            models.Goal.goal_id == goal_id,
            models.Goal.user_id == current_user.user_id  # Ensure user owns this goal
        )
    ).first()
    
    # Return 404 if goal doesn't exist or doesn't belong to user
    if not goal:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Goal not found"
        )
    
    # Add contribution amount to current goal amount
    goal.current_amount += contribution.amount
    
    # Save changes to database
    db.commit()
    db.refresh(goal)  # Refresh to get any auto-updated fields
    
    # Return updated goal with recalculated metrics (including completion status)
    return create_goal_response(goal)

# ============ STATISTICS ENDPOINTS ============

@router.get("/stats/summary", response_model=schemas.GoalSummary)
def get_goals_summary(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get summary statistics for all goals belonging to the authenticated user.
    This endpoint provides aggregated data without returning individual goal details.
    
    Args:
        current_user: Authenticated user (from dependency)
        db: Database session (from dependency)
        
    Returns:
        schemas.GoalSummary: Aggregated statistics for all user goals
    """
    # Get all goals for the current user
    goals = db.query(models.Goal).filter(models.Goal.user_id == current_user.user_id).all()
    
    # Calculate basic counts
    total_goals = len(goals)
    active_goals = len([g for g in goals if g.current_amount < g.target_amount])
    completed_goals = total_goals - active_goals
    
    # Calculate financial totals
    total_target_amount = sum(g.target_amount for g in goals)
    total_current_amount = sum(g.current_amount for g in goals)
    overall_progress = (total_current_amount / total_target_amount * 100) if total_target_amount > 0 else 0
    
    # Return summary statistics
    return schemas.GoalSummary(
        total_goals=total_goals,
        active_goals=active_goals,
        completed_goals=completed_goals,
        total_target_amount=round(total_target_amount, 2),
        total_current_amount=round(total_current_amount, 2),
        overall_progress_percentage=round(overall_progress, 2)
    )
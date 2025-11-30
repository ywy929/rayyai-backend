"""
RAG (Retrieval-Augmented Generation) Service
Retrieves and formats user financial data for LLM context
"""
from typing import Dict, Any, List, Optional
from datetime import date, datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_
import models
from routers.utils import calculate_account_balance
from database import get_mongo_db
import logging
import json
import re

FINANCIAL_CONTEXT = {
    "budgeting_rules": {
        "50_30_20_rule": {
            "description": "50% needs, 30% wants, 20% savings",
            "needs_percentage": 0.50,
            "wants_percentage": 0.30,
            "savings_percentage": 0.20,
        },
        "emergency_fund": {
            "description": "Maintain 3-6 months of expenses as an emergency fund",
            "min_months": 3,
            "max_months": 6,
        },
    },
    "financial_health_indicators": {
        "savings_rate": {
            "minimum": 0.10,
            "good": 0.20,
            "excellent": 0.30,
        },
        "credit_utilization": {
            "excellent": 0.10,
            "good": 0.30,
            "warning": 0.50,
            "poor": 0.70,
        },
        "debt_to_income_ratio": {
            "healthy": 0.36,
            "warning": 0.43,
            "critical": 0.50,
        },
    },
    "credit_score_ranges": {
        "excellent": (800, 850),
        "very_good": (740, 799),
        "good": (670, 739),
        "fair": (580, 669),
        "poor": (300, 579),
    },
}

RECOMMENDATION_LIBRARY = {
    "high_debt": [
        "Prioritize payments on high-interest balances first.",
        "Consider a debt snowball payoff plan for quick wins.",
        "Negotiate with creditors for lower interest rates when possible.",
        "Evaluate consolidation options if they reduce overall interest.",
    ],
    "low_savings": [
        "Automate transfers into savings on payday.",
        "Allocate at least 20% of income toward savings when feasible.",
        "Build an emergency fund covering 3-6 months of expenses.",
        "Review discretionary spending to redirect funds into savings.",
    ],
    "overspending": [
        "Track category-level spending to identify trends.",
        "Set category alerts when you approach budget thresholds.",
        "Apply a 24-hour cooling-off period before non-essential purchases.",
        "Look for lower-cost alternatives for frequent discretionary expenses.",
    ],
    "credit_card": [
        "Aim to keep credit utilization below 30%, ideally under 10%.",
        "Pay the full statement balance monthly to avoid interest charges.",
        "Enable autopay for at least the minimum payment as a safety net.",
        "Request strategic limit increases to improve utilization ratios.",
    ],
    "goal_achievement": [
        "Break large goals into milestones with target dates.",
        "Automate contributions toward high-priority goals.",
        "Review progress monthly and adjust contributions if needed.",
        "Celebrate milestones to maintain motivation.",
    ],
}

logger = logging.getLogger(__name__)

class RAGService:
    """Service for retrieving and formatting user financial context"""
    
    def __init__(self, db: Session):
        """
        Initialize RAG service.
        
        Args:
            db: Database session
        """
        self.db = db
    
    @staticmethod
    def _safe_percentage(numerator: Optional[float], denominator: Optional[float]) -> float:
        if not denominator or denominator == 0:
            return 0.0
        try:
            return float(numerator) / float(denominator)
        except (TypeError, ZeroDivisionError):
            return 0.0
    
    def get_time_context(self) -> Dict[str, Any]:
        """
        Provide current time context information.
        
        Returns:
            Dictionary containing key date references
        """
        now = datetime.now()
        today = now.date()
        first_day_of_month = today.replace(day=1)
        first_day_of_year = today.replace(month=1, day=1)
        thirty_days_ago = today - timedelta(days=30)
        ninety_days_ago = today - timedelta(days=90)
        quarter = (now.month - 1) // 3 + 1
        
        return {
            "today": today.isoformat(),
            "current_year": now.year,
            "current_month": now.strftime("%B %Y"),
            "current_quarter": f"Q{quarter} {now.year}",
            "first_day_of_month": first_day_of_month.isoformat(),
            "first_day_of_year": first_day_of_year.isoformat(),
            "thirty_days_ago": thirty_days_ago.isoformat(),
            "ninety_days_ago": ninety_days_ago.isoformat(),
            "day_of_week": now.strftime("%A"),
        }
    
    def _compute_needs_wants_percentages(self, needs_vs_wants: Dict[str, float]) -> Dict[str, float]:
        total = sum(needs_vs_wants.values())
        if total == 0:
            return {"needs": 0.0, "wants": 0.0}
        
        needs_amount = needs_vs_wants.get("Needs", 0.0)
        wants_amount = needs_vs_wants.get("Wants", 0.0)
        return {
            "needs": round(self._safe_percentage(needs_amount, total) * 100, 2),
            "wants": round(self._safe_percentage(wants_amount, total) * 100, 2),
        }
    
    def _compute_savings_rate(self, total_income: float, total_expenses: float) -> float:
        savings = max(total_income - total_expenses, 0.0)
        return round(self._safe_percentage(savings, total_income) * 100, 2)
    
    def _aggregate_credit_utilization(self, credit_cards: List[Dict[str, Any]]) -> Dict[str, float]:
        if not credit_cards:
            return {"average": 0.0, "max": 0.0}
        
        utilizations = [card.get("utilization_percentage", 0.0) for card in credit_cards]
        return {
            "average": round(sum(utilizations) / len(utilizations), 2),
            "max": round(max(utilizations), 2),
        }
    
    def generate_best_practice_analysis(
        self,
        total_income: float,
        total_expenses: float,
        spending_summary: Dict[str, Any],
        credit_cards: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compare user data against financial best practices.
        """
        needs_vs_wants = spending_summary.get("needs_vs_wants", {})
        needs_wants_percentages = self._compute_needs_wants_percentages(needs_vs_wants)
        savings_rate = self._compute_savings_rate(total_income, total_expenses)
        
        cards = credit_cards.get("cards", [])
        utilization_stats = self._aggregate_credit_utilization(cards)
        total_utilization = credit_cards.get("total_utilization", 0.0)
        
        indicators = FINANCIAL_CONTEXT["financial_health_indicators"]
        budgeting_rule = FINANCIAL_CONTEXT["budgeting_rules"]["50_30_20_rule"]
        
        needs_status = "on_track"
        if needs_wants_percentages["needs"] > budgeting_rule["needs_percentage"] * 100:
            needs_status = "high"
        
        wants_status = "on_track"
        if needs_wants_percentages["wants"] > budgeting_rule["wants_percentage"] * 100:
            wants_status = "high"
        
        savings_status = "low"
        if savings_rate >= indicators["savings_rate"]["excellent"] * 100:
            savings_status = "excellent"
        elif savings_rate >= indicators["savings_rate"]["good"] * 100:
            savings_status = "good"
        elif savings_rate >= indicators["savings_rate"]["minimum"] * 100:
            savings_status = "adequate"
        
        utilization_status = "healthy"
        if total_utilization >= indicators["credit_utilization"]["poor"] * 100:
            utilization_status = "critical"
        elif total_utilization >= indicators["credit_utilization"]["warning"] * 100:
            utilization_status = "warning"
        elif total_utilization >= indicators["credit_utilization"]["good"] * 100:
            utilization_status = "monitor"
        
        return {
            "needs_vs_wants": {
                "percentages": needs_wants_percentages,
                "status": {"needs": needs_status, "wants": wants_status},
                "guideline": {
                    "needs": budgeting_rule["needs_percentage"] * 100,
                    "wants": budgeting_rule["wants_percentage"] * 100,
                    "savings": budgeting_rule["savings_percentage"] * 100,
                },
            },
            "savings_rate": {
                "value": savings_rate,
                "status": savings_status,
                "benchmarks": {
                    "minimum": indicators["savings_rate"]["minimum"] * 100,
                    "good": indicators["savings_rate"]["good"] * 100,
                    "excellent": indicators["savings_rate"]["excellent"] * 100,
                },
            },
            "credit_utilization": {
                "overall": total_utilization,
                "average": utilization_stats["average"],
                "max": utilization_stats["max"],
                "status": utilization_status,
                "benchmarks": {
                    "excellent": indicators["credit_utilization"]["excellent"] * 100,
                    "good": indicators["credit_utilization"]["good"] * 100,
                    "warning": indicators["credit_utilization"]["warning"] * 100,
                },
            },
        }
    
    def generate_recommendations(self, analysis: Dict[str, Any], budgets: Dict[str, Any]) -> List[str]:
        """
        Provide actionable recommendations based on analysis results.
        """
        recommendations: List[str] = []
        
        needs_status = analysis["needs_vs_wants"]["status"]["needs"]
        wants_status = analysis["needs_vs_wants"]["status"]["wants"]
        savings_status = analysis["savings_rate"]["status"]
        utilization_status = analysis["credit_utilization"]["status"]
        
        if savings_status in {"low"}:
            recommendations.extend(RECOMMENDATION_LIBRARY["low_savings"])
        if wants_status == "high":
            recommendations.extend(RECOMMENDATION_LIBRARY["overspending"])
        if needs_status == "high":
            recommendations.append(
                "Review essential spending categories to identify opportunities for renegotiation or optimization."
            )
        if utilization_status in {"warning", "critical"}:
            recommendations.extend(RECOMMENDATION_LIBRARY["credit_card"])
        
        over_budget = budgets.get("over_budget_count", 0)
        near_limit = budgets.get("near_limit_count", 0)
        if over_budget > 0 or near_limit > 0:
            recommendations.extend([
                "Review budgets that are near or over their limits and adjust allocations or spending accordingly.",
                "Consider staggered budget reviews throughout the month to prevent surprises near period end.",
            ])
        
        # De-duplicate while preserving order
        seen = set()
        unique_recommendations = []
        for rec in recommendations:
            if rec not in seen:
                unique_recommendations.append(rec)
                seen.add(rec)
        
        if not unique_recommendations:
            unique_recommendations = [
                "Continue monitoring spending and savings trends; current metrics align with key benchmarks."
            ]
        
        return unique_recommendations
    
    def get_financial_context_reference(self) -> str:
        """
        Return a compact textual summary of financial best practices.
        """
        parts = ["=== FINANCIAL BEST PRACTICES REFERENCE ==="]
        rule = FINANCIAL_CONTEXT["budgeting_rules"]["50_30_20_rule"]
        parts.append(
            f"50/30/20 Rule → Needs: {rule['needs_percentage']*100:.0f}%, "
            f"Wants: {rule['wants_percentage']*100:.0f}%, Savings: {rule['savings_percentage']*100:.0f}%"
        )
        savings_rate = FINANCIAL_CONTEXT["financial_health_indicators"]["savings_rate"]
        parts.append(
            "Savings Rate Benchmarks → Minimum 10%, Good 20%, Excellent 30%+"
        )
        utilization = FINANCIAL_CONTEXT["financial_health_indicators"]["credit_utilization"]
        parts.append(
            "Credit Utilization → Excellent <10%, Good <30%, Warning >50%"
        )
        debt_ratio = FINANCIAL_CONTEXT["financial_health_indicators"]["debt_to_income_ratio"]
        parts.append(
            f"Debt-to-Income Ratio → Healthy <{debt_ratio['healthy']*100:.0f}%, "
            f"Warning >{debt_ratio['warning']*100:.0f}%, Critical >{debt_ratio['critical']*100:.0f}%"
        )
        return "\n".join(parts)
    
    def get_user_accounts(self, user_id: int) -> List[Dict[str, Any]]:
        """
        Get all user accounts with balances.
        
        Args:
            user_id: User ID
            
        Returns:
            List of account dictionaries
        """
        accounts = self.db.query(models.Account).filter(
            models.Account.user_id == user_id,
            models.Account.is_deleted == False
        ).all()
        
        account_data = []
        for account in accounts:
            try:
                balance = calculate_account_balance(self.db, account.account_id)
            except:
                balance = 0.0
            
            account_data.append({
                "account_id": account.account_id,
                "account_name": account.account_name,
                "account_type": account.account_type,
                "account_no": account.account_no,
                "balance": balance,
                "card_id": account.card_id
            })
        
        return account_data
    
    def get_recent_transactions(
        self,
        user_id: int,
        days: int = 90,
        limit: int = 10000
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get recent transactions (income and expenses).
        
        Args:
            user_id: User ID
            days: Number of days to look back
            limit: Maximum number of transactions per type
            
        Returns:
            Dictionary with 'income' and 'expense' lists
        """
        cutoff_date = date.today() - timedelta(days=days)
        
        # Get recent income
        incomes = self.db.query(models.Income).filter(
            models.Income.user_id == user_id,
            models.Income.is_deleted == False,
            models.Income.date_received >= cutoff_date
        ).order_by(models.Income.date_received.desc()).limit(limit).all()
        
        # Get recent expenses
        expenses = self.db.query(models.Expense).filter(
            models.Expense.user_id == user_id,
            models.Expense.is_deleted == False,
            models.Expense.date_spent >= cutoff_date
        ).order_by(models.Expense.date_spent.desc()).limit(limit).all()
        
        income_data = [
            {
                "income_id": inc.income_id,
                "amount": inc.amount,
                "description": inc.description,
                "category": inc.category,
                "date_received": inc.date_received.isoformat(),
                "payer": inc.payer,
                "account_id": inc.account_id
            }
            for inc in incomes
        ]
        
        expense_data = [
            {
                "expense_id": exp.expense_id,
                "amount": exp.amount,
                "description": exp.description,
                "category": exp.category,
                "expense_type": exp.expense_type,
                "date_spent": exp.date_spent.isoformat(),
                "seller": exp.seller,
                "location": exp.location,
                "account_id": exp.account_id,
                "is_reimbursable": exp.is_reimbursable,
                "tax_deductible": exp.tax_deductible
            }
            for exp in expenses
        ]
        
        return {
            "income": income_data,
            "expense": expense_data
        }
    
    def get_spending_summary(self, user_id: int, days: int = 30) -> Dict[str, Any]:
        """
        Get spending summary by category.
        
        Args:
            user_id: User ID
            days: Number of days to analyze
            
        Returns:
            Spending summary dictionary
        """
        cutoff_date = date.today() - timedelta(days=days)
        
        # Get spending by category
        category_spending = self.db.query(
            models.Expense.category,
            func.sum(models.Expense.amount).label('total')
        ).filter(
            models.Expense.user_id == user_id,
            models.Expense.is_deleted == False,
            models.Expense.date_spent >= cutoff_date
        ).group_by(models.Expense.category).all()
        
        # Get needs vs wants breakdown
        needs_wants = self.db.query(
            models.Expense.expense_type,
            func.sum(models.Expense.amount).label('total')
        ).filter(
            models.Expense.user_id == user_id,
            models.Expense.is_deleted == False,
            models.Expense.date_spent >= cutoff_date,
            models.Expense.expense_type.isnot(None)
        ).group_by(models.Expense.expense_type).all()
        
        total_spending = sum(item.total for item in category_spending)
        needs_vs_wants_dict = {
            typ: float(total)
            for typ, total in needs_wants
        }
        
        return {
            "period_days": days,
            "total_spending": float(total_spending),
            "by_category": {
                cat: float(total)
                for cat, total in category_spending
            },
            "needs_vs_wants": needs_vs_wants_dict,
            "needs_vs_wants_percentages": self._compute_needs_wants_percentages(
                needs_vs_wants_dict
            )
        }
    
    def get_budgets_status(self, user_id: int) -> List[Dict[str, Any]]:
        """
        Get all active budgets with current status.
        
        Args:
            user_id: User ID
            
        Returns:
            List of budget dictionaries with status
        """
        today = date.today()
        
        budgets = self.db.query(models.Budget).filter(
            models.Budget.user_id == user_id,
            models.Budget.is_deleted == False,
            models.Budget.period_start <= today,
            models.Budget.period_end >= today
        ).all()
        
        budget_data = []
        for budget in budgets:
            # Calculate spent amount for this budget
            spent = self.db.query(func.sum(models.Expense.amount)).filter(
                models.Expense.user_id == user_id,
                models.Expense.is_deleted == False,
                models.Expense.category == budget.category,
                models.Expense.date_spent >= budget.period_start,
                models.Expense.date_spent <= budget.period_end
            ).scalar() or 0.0
            
            remaining = budget.limit_amount - spent
            percentage_used = (spent / budget.limit_amount * 100) if budget.limit_amount > 0 else 0
            
            budget_data.append({
                "budget_id": budget.budget_id,
                "name": budget.name,
                "category": budget.category,
                "limit_amount": budget.limit_amount,
                "spent_amount": float(spent),
                "remaining_amount": float(remaining),
                "percentage_used": round(percentage_used, 2),
                "period_start": budget.period_start.isoformat(),
                "period_end": budget.period_end.isoformat(),
                "alert_threshold": budget.alert_threshold,
                "is_over_budget": spent > budget.limit_amount,
                "is_near_limit": percentage_used >= (budget.alert_threshold * 100)
            })
        
        return budget_data
    
    def get_goals_status(self, user_id: int) -> List[Dict[str, Any]]:
        """
        Get all active goals with progress.
        
        Args:
            user_id: User ID
            
        Returns:
            List of goal dictionaries with status
        """
        goals = self.db.query(models.Goal).filter(
            models.Goal.user_id == user_id,
            models.Goal.is_deleted == False
        ).all()
        
        goal_data = []
        for goal in goals:
            progress_percentage = (goal.current_amount / goal.target_amount * 100) if goal.target_amount > 0 else 0
            is_completed = goal.current_amount >= goal.target_amount
            
            days_remaining = None
            if goal.target_date:
                days_remaining = (goal.target_date - date.today()).days
            
            goal_data.append({
                "goal_id": goal.goal_id,
                "goal_name": goal.goal_name,
                "description": goal.description,
                "category": goal.category,
                "priority": goal.priority,
                "target_amount": goal.target_amount,
                "current_amount": goal.current_amount,
                "progress_percentage": round(progress_percentage, 2),
                "target_date": goal.target_date.isoformat() if goal.target_date else None,
                "days_remaining": days_remaining,
                "is_completed": is_completed
            })
        
        return goal_data
    
    def get_credit_cards(self, user_id: int) -> List[Dict[str, Any]]:
        """
        Get user credit cards with balances and payment info.
        
        Args:
            user_id: User ID
            
        Returns:
            List of credit card dictionaries
        """
        cards = self.db.query(models.UserCreditCard).filter(
            models.UserCreditCard.user_id == user_id,
            models.UserCreditCard.is_deleted == False
        ).all()
        
        card_data = []
        for card in cards:
            utilization = (card.current_balance / card.credit_limit * 100) if card.credit_limit > 0 else 0
            
            card_data.append({
                "card_id": card.card_id,
                "card_name": card.card_name,
                "bank_name": card.bank_name,
                "card_brand": card.card_brand,
                "credit_limit": card.credit_limit,
                "current_balance": card.current_balance,
                "available_credit": card.credit_limit - card.current_balance,
                "utilization_percentage": round(utilization, 2),
                "annual_fee": card.annual_fee,
                "next_payment_amount": card.next_payment_amount,
                "next_payment_date": card.next_payment_date.isoformat() if card.next_payment_date else None,
                "expiry_month": card.expiry_month,
                "expiry_year": card.expiry_year,
                "benefits": card.benefits
            })
        
        return card_data
    
    def get_financial_summary(self, user_id: int) -> Dict[str, Any]:
        """
        Get comprehensive financial summary.
        
        Args:
            user_id: User ID
            
        Returns:
            Complete financial summary dictionary
        """
        accounts = self.get_user_accounts(user_id)
        transactions = self.get_recent_transactions(user_id, days=90)
        spending_summary = self.get_spending_summary(user_id, days=30)
        budgets = self.get_budgets_status(user_id)
        goals = self.get_goals_status(user_id)
        credit_cards = self.get_credit_cards(user_id)
        
        # Calculate total balance
        total_balance = sum(acc["balance"] for acc in accounts)
        
        # Calculate total income and expenses
        total_income = sum(t["amount"] for t in transactions["income"])
        total_expenses = sum(t["amount"] for t in transactions["expense"])
        
        credit_cards_total_limit = sum(c["credit_limit"] for c in credit_cards)
        credit_cards_total_balance = sum(c["current_balance"] for c in credit_cards)
        total_utilization = round(
            (
                credit_cards_total_balance /
                credit_cards_total_limit * 100
            ) if credit_cards and credit_cards_total_limit > 0 else 0.0,
            2
        )
        
        credit_cards_summary = {
            "total_count": len(credit_cards),
            "total_limit": credit_cards_total_limit,
            "total_balance": credit_cards_total_balance,
            "total_utilization": total_utilization,
            "cards": credit_cards
        }
        
        budgets_summary = {
            "active_count": len(budgets),
            "over_budget_count": sum(1 for b in budgets if b["is_over_budget"]),
            "near_limit_count": sum(1 for b in budgets if b["is_near_limit"]),
            "budgets": budgets
        }
        
        analysis = self.generate_best_practice_analysis(
            total_income=total_income,
            total_expenses=total_expenses,
            spending_summary=spending_summary,
            credit_cards=credit_cards_summary
        )
        
        recommendations = self.generate_recommendations(analysis, budgets_summary)
        time_context = self.get_time_context()
        
        return {
            "snapshot_date": date.today().isoformat(),
            "accounts": {
                "total_count": len(accounts),
                "total_balance": total_balance,
                "accounts": accounts
            },
            "transactions": {
                "recent_income": len(transactions["income"]),
                "recent_expenses": len(transactions["expense"]),
                "total_income_90d": total_income,
                "total_expenses_90d": total_expenses,
                "net_flow_90d": total_income - total_expenses
            },
            "spending_summary": spending_summary,
            "budgets": budgets_summary,
            "goals": {
                "total_count": len(goals),
                "completed_count": sum(1 for g in goals if g["is_completed"]),
                "goals": goals
            },
            "credit_cards": credit_cards_summary,
            "analysis": analysis,
            "recommendations": recommendations,
            "time_context": time_context,
            "financial_context_reference": self.get_financial_context_reference()
        }
    
    def format_context_for_llm(self, financial_data: Dict[str, Any]) -> str:
        """
        Format financial data as text for LLM context.
        
        Args:
            financial_data: Financial summary dictionary
            
        Returns:
            Formatted text context
        """
        context_parts = []
        
        # Accounts section
        accounts = financial_data.get("accounts", {})
        if accounts.get("accounts"):
            context_parts.append("=== ACCOUNTS ===")
            context_parts.append(f"Total Balance: RM{accounts.get('total_balance', 0):,.2f}")
            context_parts.append(f"Number of Accounts: {accounts.get('total_count', 0)}")
            for acc in accounts.get("accounts", [])[:10]:  # Limit to 10 most important
                context_parts.append(
                    f"- {acc['account_name']} ({acc['account_type']}): "
                    f"RM{acc['balance']:,.2f}"
                )
        
        # Recent transactions
        transactions = financial_data.get("transactions", {})
        if transactions.get("recent_income") or transactions.get("recent_expenses"):
            context_parts.append("\n=== RECENT TRANSACTIONS (Last 90 Days) ===")
            context_parts.append(
                f"Income: RM{transactions.get('total_income_90d', 0):,.2f} "
                f"({transactions.get('recent_income', 0)} transactions)"
            )
            context_parts.append(
                f"Expenses: RM{transactions.get('total_expenses_90d', 0):,.2f} "
                f"({transactions.get('recent_expenses', 0)} transactions)"
            )
            context_parts.append(
                f"Net Flow: RM{transactions.get('net_flow_90d', 0):,.2f}"
            )

            # Include detailed transaction list (recent 50 for context)
            income_list = transactions.get("income", [])[:30]
            expense_list = transactions.get("expense", [])[:50]

            if expense_list:
                context_parts.append("\nRecent Expenses (Last 50):")
                for exp in expense_list:
                    merchant = exp.get("seller", "Unknown")
                    amount = exp.get("amount", 0)
                    category = exp.get("category", "Uncategorized")
                    date = exp.get("date_spent", "Unknown")
                    description = exp.get("description", "")
                    context_parts.append(
                        f"  - {date}: {merchant} - RM{amount:,.2f} ({category})"
                        + (f" - {description}" if description and description != merchant else "")
                    )

            if income_list:
                context_parts.append("\nRecent Income (Last 30):")
                for inc in income_list:
                    payer = inc.get("payer", "Unknown")
                    amount = inc.get("amount", 0)
                    category = inc.get("category", "Uncategorized")
                    date = inc.get("date_received", "Unknown")
                    description = inc.get("description", "")
                    context_parts.append(
                        f"  - {date}: {payer} - RM{amount:,.2f} ({category})"
                        + (f" - {description}" if description and description != payer else "")
                    )
        
        # Spending summary
        spending = financial_data.get("spending_summary", {})
        if spending.get("by_category"):
            context_parts.append("\n=== SPENDING BY CATEGORY (Last 30 Days) ===")
            context_parts.append(f"Total Spending: RM{spending.get('total_spending', 0):,.2f}")
            for category, amount in sorted(
                spending.get("by_category", {}).items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]:
                context_parts.append(f"- {category}: RM{amount:,.2f}")
        
        # Budgets
        budgets = financial_data.get("budgets", {}).get("budgets", [])
        if budgets:
            context_parts.append("\n=== ACTIVE BUDGETS ===")
            for budget in budgets[:10]:
                status = "OVER BUDGET" if budget["is_over_budget"] else (
                    "NEAR LIMIT" if budget["is_near_limit"] else "OK"
                )
                context_parts.append(
                    f"- {budget['name']} ({budget['category']}): "
                    f"RM{budget['spent_amount']:,.2f} / RM{budget['limit_amount']:,.2f} "
                    f"({budget['percentage_used']:.1f}%) - {status}"
                )
        
        # Goals
        goals = financial_data.get("goals", {}).get("goals", [])
        if goals:
            context_parts.append("\n=== FINANCIAL GOALS ===")
            for goal in goals[:10]:
                status = "COMPLETED" if goal["is_completed"] else "IN PROGRESS"
                context_parts.append(
                    f"- {goal['goal_name']} ({goal['category']}): "
                    f"RM{goal['current_amount']:,.2f} / RM{goal['target_amount']:,.2f} "
                    f"({goal['progress_percentage']:.1f}%) - {status}"
                )
                if goal.get("target_date"):
                    days = goal.get("days_remaining", 0)
                    context_parts.append(f"  Target Date: {goal['target_date']} ({days} days remaining)")
        
        # Credit Cards
        cards = financial_data.get("credit_cards", {}).get("cards", [])
        if cards:
            context_parts.append("\n=== CREDIT CARDS ===")
            for card in cards[:5]:
                context_parts.append(
                    f"- {card['card_name']} ({card['bank_name']}): "
                    f"RM{card['current_balance']:,.2f} / RM{card['credit_limit']:,.2f} "
                    f"({card['utilization_percentage']:.1f}% utilization)"
                )
                if card.get("annual_fee") is not None:
                    context_parts.append(f"  Annual Fee: RM{card['annual_fee']:,.2f}")
                if card.get("next_payment_date"):
                    context_parts.append(
                        f"  Next Payment: RM{card['next_payment_amount']:,.2f} "
                        f"on {card['next_payment_date']}"
                    )
                # Include card benefits for comparison purposes
                if card.get("benefits"):
                    benefits = card.get("benefits")
                    if isinstance(benefits, dict):
                        context_parts.append(f"  Benefits: {json.dumps(benefits, ensure_ascii=False)}")
                    elif isinstance(benefits, str):
                        context_parts.append(f"  Benefits: {benefits}")
        
        time_context = financial_data.get("time_context")
        if time_context:
            context_parts.append("\n=== TIME CONTEXT ===")
            display_order = [
                ("today", "Today's Date"),
                ("day_of_week", "Day of Week"),
                ("current_month", "Current Month"),
                ("current_quarter", "Current Quarter"),
                ("current_year", "Current Year"),
                ("first_day_of_month", "First Day of Month"),
                ("first_day_of_year", "First Day of Year"),
                ("thirty_days_ago", "30 Days Ago"),
                ("ninety_days_ago", "90 Days Ago"),
            ]
            for key, label in display_order:
                if key in time_context:
                    context_parts.append(f"- {label}: {time_context[key]}")
        
        analysis = financial_data.get("analysis")
        if analysis:
            context_parts.append("\n=== BEST PRACTICE COMPARISON ===")
            needs_vs_wants = analysis.get("needs_vs_wants", {})
            if needs_vs_wants:
                percentages = needs_vs_wants.get("percentages", {})
                status = needs_vs_wants.get("status", {})
                guideline = needs_vs_wants.get("guideline", {})
                context_parts.append(
                    f"Needs: {percentages.get('needs', 0.0):.1f}% "
                    f"(Status: {status.get('needs', 'n/a')}, "
                    f"Guideline ≤ {guideline.get('needs', 0.0):.0f}%)"
                )
                context_parts.append(
                    f"Wants: {percentages.get('wants', 0.0):.1f}% "
                    f"(Status: {status.get('wants', 'n/a')}, "
                    f"Guideline ≤ {guideline.get('wants', 0.0):.0f}%)"
                )
                context_parts.append(
                    f"Savings Target: {guideline.get('savings', 0.0):.0f}%"
                )
            
            savings = analysis.get("savings_rate", {})
            if savings:
                context_parts.append(
                    f"Savings Rate: {savings.get('value', 0.0):.1f}% "
                    f"(Status: {savings.get('status', 'n/a')}, "
                    f"Good ≥ {savings.get('benchmarks', {}).get('good', 0.0):.0f}%)"
                )
            
            utilization = analysis.get("credit_utilization", {})
            if utilization:
                context_parts.append(
                    f"Credit Utilization Overall: {utilization.get('overall', 0.0):.1f}% "
                    f"(Status: {utilization.get('status', 'n/a')})"
                )
                context_parts.append(
                    f"Average Card Utilization: {utilization.get('average', 0.0):.1f}% | "
                    f"Max Card: {utilization.get('max', 0.0):.1f}%"
                )
                context_parts.append(
                    f"Benchmark → Good < {utilization.get('benchmarks', {}).get('good', 0.0):.0f}%"
                )
        
        recommendations = financial_data.get("recommendations")
        if recommendations:
            context_parts.append("\n=== RECOMMENDATIONS ===")
            for rec in recommendations[:8]:
                context_parts.append(f"- {rec}")
        
        reference = financial_data.get("financial_context_reference")
        if reference:
            context_parts.append("\n" + reference)
        
        return "\n".join(context_parts)

    def suggest_budgets_from_spending(self, user_id: int, period_days: int = 90) -> List[Dict[str, Any]]:
        """
        Suggest budget allocations based on historical spending patterns.

        Args:
            user_id: User ID
            period_days: Number of days to analyze for patterns

        Returns:
            List of suggested budgets with justifications
        """
        from datetime import timedelta
        cutoff_date = date.today() - timedelta(days=period_days)

        # Get spending by category over the period
        category_spending = self.db.query(
            models.Expense.category,
            func.sum(models.Expense.amount).label('total'),
            func.count(models.Expense.expense_id).label('transaction_count'),
            func.avg(models.Expense.amount).label('avg_amount')
        ).filter(
            models.Expense.user_id == user_id,
            models.Expense.is_deleted == False,
            models.Expense.date_spent >= cutoff_date
        ).group_by(models.Expense.category).all()

        # Check existing budgets
        existing_budgets = self.db.query(models.Budget).filter(
            models.Budget.user_id == user_id,
            models.Budget.is_deleted == False
        ).all()

        existing_categories = {b.category for b in existing_budgets}

        suggestions = []
        for category, total, txn_count, avg_amount in category_spending:
            # Skip if budget already exists
            if category in existing_categories:
                continue

            # Calculate monthly average (convert from period_days to 30 days)
            monthly_average = (total / period_days) * 30

            # Add 10% buffer for flexibility
            suggested_limit = monthly_average * 1.1

            # Round to nearest 50
            suggested_limit = round(suggested_limit / 50) * 50

            if suggested_limit > 0:
                suggestions.append({
                    "category": category,
                    "suggested_limit": suggested_limit,
                    "monthly_average": round(monthly_average, 2),
                    "historical_total": round(total, 2),
                    "transaction_count": txn_count,
                    "avg_transaction": round(avg_amount, 2),
                    "reasoning": f"Based on {period_days} days of spending data, you spent RM{monthly_average:.2f}/month on {category}. Suggested budget includes 10% buffer."
                })

        # Sort by monthly average (highest first)
        suggestions.sort(key=lambda x: x['monthly_average'], reverse=True)

        return suggestions

    def suggest_goals_from_context(self, user_id: int) -> List[Dict[str, Any]]:
        """
        Suggest financial goals based on user's financial context.

        Args:
            user_id: User ID

        Returns:
            List of suggested goals with justifications
        """
        # Get financial summary
        financial_data = self.get_financial_summary(user_id)

        suggestions = []

        # Get existing goals to avoid duplicates
        existing_goals = self.db.query(models.Goal).filter(
            models.Goal.user_id == user_id,
            models.Goal.is_deleted == False
        ).all()

        existing_goal_categories = {g.category for g in existing_goals}

        # Analyze savings rate and suggest emergency fund
        transactions = financial_data.get("transactions", {})
        total_income = transactions.get("total_income_90d", 0)
        total_expenses = transactions.get("total_expenses_90d", 0)

        # Calculate monthly averages
        monthly_income = (total_income / 90) * 30
        monthly_expenses = (total_expenses / 90) * 30

        # Suggest emergency fund (3-6 months of expenses)
        if "Emergency Fund" not in existing_goal_categories and monthly_expenses > 0:
            emergency_target = monthly_expenses * 6
            suggestions.append({
                "goal_name": "Emergency Fund",
                "category": "Emergency Fund",
                "priority": "high",
                "target_amount": round(emergency_target, 2),
                "current_amount": 0,
                "target_date": None,
                "description": f"Build an emergency fund to cover 6 months of expenses (RM{monthly_expenses:.2f}/month)",
                "reasoning": f"Financial experts recommend having 3-6 months of expenses saved. Based on your average monthly spending of RM{monthly_expenses:.2f}, you should aim for RM{emergency_target:.2f}."
            })

        # Suggest savings goal based on 50/30/20 rule
        if "Savings" not in existing_goal_categories and monthly_income > 0:
            recommended_savings = monthly_income * 0.20  # 20% of income
            yearly_savings = recommended_savings * 12

            suggestions.append({
                "goal_name": "Annual Savings Goal",
                "category": "Savings",
                "priority": "medium",
                "target_amount": round(yearly_savings, 2),
                "current_amount": 0,
                "target_date": date(date.today().year, 12, 31).isoformat(),
                "description": f"Save 20% of your monthly income (RM{recommended_savings:.2f}/month)",
                "reasoning": f"Following the 50/30/20 rule, you should save 20% of your income. Based on your average monthly income of RM{monthly_income:.2f}, this means RM{recommended_savings:.2f}/month."
            })

        # Analyze credit card debt and suggest debt payoff goal
        credit_cards = financial_data.get("credit_cards", {})
        total_cc_balance = credit_cards.get("total_balance", 0)

        if "Debt Payoff" not in existing_goal_categories and total_cc_balance > 0:
            suggestions.append({
                "goal_name": "Credit Card Debt Payoff",
                "category": "Debt Payoff",
                "priority": "high",
                "target_amount": round(total_cc_balance, 2),
                "current_amount": 0,
                "target_date": None,
                "description": f"Pay off all credit card balances (total: RM{total_cc_balance:.2f})",
                "reasoning": f"You have RM{total_cc_balance:.2f} in credit card debt. Paying this off should be a high priority to avoid interest charges."
            })

        # Suggest retirement savings if income is substantial
        if "Retirement" not in existing_goal_categories and monthly_income > 5000:
            retirement_monthly = monthly_income * 0.10  # 10% for retirement
            retirement_yearly = retirement_monthly * 12

            suggestions.append({
                "goal_name": "Retirement Fund",
                "category": "Retirement",
                "priority": "medium",
                "target_amount": round(retirement_yearly, 2),
                "current_amount": 0,
                "target_date": date(date.today().year, 12, 31).isoformat(),
                "description": f"Contribute 10% of income towards retirement (RM{retirement_monthly:.2f}/month)",
                "reasoning": f"With a monthly income of RM{monthly_income:.2f}, setting aside 10% (RM{retirement_monthly:.2f}/month) for retirement is recommended."
            })

        return suggestions

    def get_budget_goal_suggestions(self, user_id: int) -> Dict[str, Any]:
        """
        Get comprehensive budget and goal suggestions based on user's financial data.

        Args:
            user_id: User ID

        Returns:
            Dictionary with budget and goal suggestions
        """
        budget_suggestions = self.suggest_budgets_from_spending(user_id, period_days=90)
        goal_suggestions = self.suggest_goals_from_context(user_id)

        return {
            "budget_suggestions": budget_suggestions,
            "goal_suggestions": goal_suggestions,
            "suggestion_count": {
                "budgets": len(budget_suggestions),
                "goals": len(goal_suggestions)
            },
            "message": f"Found {len(budget_suggestions)} budget suggestions and {len(goal_suggestions)} goal suggestions based on your financial data"
        }

    def format_suggestions_for_llm(self, suggestions: Dict[str, Any]) -> str:
        """
        Format budget and goal suggestions as text for LLM context.

        Args:
            suggestions: Dictionary from get_budget_goal_suggestions()

        Returns:
            Formatted text context
        """
        context_parts = []

        budget_suggestions = suggestions.get("budget_suggestions", [])
        if budget_suggestions:
            context_parts.append("=== SUGGESTED BUDGETS ===")
            context_parts.append("Based on historical spending patterns, here are recommended budgets:")
            for suggestion in budget_suggestions[:5]:  # Top 5
                context_parts.append(
                    f"- {suggestion['category']}: RM{suggestion['suggested_limit']:,.2f}/month "
                    f"(avg: RM{suggestion['monthly_average']:,.2f}, {suggestion['transaction_count']} transactions)"
                )
                context_parts.append(f"  Reasoning: {suggestion['reasoning']}")

        goal_suggestions = suggestions.get("goal_suggestions", [])
        if goal_suggestions:
            context_parts.append("\n=== SUGGESTED GOALS ===")
            context_parts.append("Based on your financial profile, consider these goals:")
            for suggestion in goal_suggestions:
                context_parts.append(
                    f"- {suggestion['goal_name']} ({suggestion['category']}, Priority: {suggestion['priority']}): "
                    f"RM{suggestion['target_amount']:,.2f}"
                )
                context_parts.append(f"  Description: {suggestion['description']}")
                context_parts.append(f"  Reasoning: {suggestion['reasoning']}")

        if not budget_suggestions and not goal_suggestions:
            context_parts.append("No budget or goal suggestions available at this time.")

        return "\n".join(context_parts)

    def get_market_credit_cards_from_mongo(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Query market credit cards from MongoDB.

        Args:
            filters: Optional MongoDB query filters

        Returns:
            List of credit card documents from MongoDB
        """
        try:
            mongo_db = get_mongo_db()
            collection = mongo_db["credit_cards_collection"]

            query = filters or {}
            cards = list(collection.find(query).limit(50))

            # Convert MongoDB _id to string for JSON serialization
            for card in cards:
                if '_id' in card:
                    card['_id'] = str(card['_id'])

            return cards
        except Exception as e:
            logger.error(f"Error querying MongoDB for credit cards: {e}")
            return []

    def _extract_income_from_criteria(self, eligibility_criteria: Dict[str, Any]) -> Optional[float]:
        """
        Extract minimum income requirement from eligibility criteria.

        Args:
            eligibility_criteria: Card eligibility criteria dict

        Returns:
            Minimum annual income as float, or None
        """
        income_str = eligibility_criteria.get('Minimum Annual Income', '')
        if not income_str:
            return None

        # Extract numbers from string like "RM 24,000" or "24000"
        numbers = re.findall(r'[\d,]+', str(income_str))
        if numbers:
            try:
                return float(numbers[0].replace(',', ''))
            except ValueError:
                return None
        return None

    def _calculate_card_match_score(
        self,
        card: Dict[str, Any],
        user_profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate match score between a credit card and user profile.

        Args:
            card: Credit card document from MongoDB
            user_profile: User's financial profile

        Returns:
            Dictionary with match_score (0-100), reasoning, and highlighted benefits
        """
        score = 0
        reasons = []
        highlighted_benefits = []

        monthly_income = user_profile.get('monthly_income', 0)
        annual_income = monthly_income * 12
        spending_categories = user_profile.get('spending_categories', {})
        credit_utilization = user_profile.get('credit_utilization', 0)
        has_debt = user_profile.get('has_debt', False)

        # 1. Income Eligibility Check (30 points)
        min_income = self._extract_income_from_criteria(card.get('eligibility_criteria', {}))
        if min_income:
            if annual_income >= min_income:
                score += 30
                reasons.append(f"You meet the minimum income requirement (RM{min_income:,.0f})")
            elif annual_income >= min_income * 0.8:
                score += 15
                reasons.append(f"Your income is close to the requirement (RM{min_income:,.0f})")
            else:
                score += 0
                reasons.append(f"Income below minimum requirement (RM{min_income:,.0f})")
        else:
            score += 20  # No income requirement specified
            reasons.append("No strict income requirement")

        # 2. Annual Fee Analysis (20 points)
        annual_fee = card.get('annual_fee', 0)
        if annual_fee == 0:
            score += 20
            reasons.append("No annual fee - great for cost savings")
            highlighted_benefits.append("No Annual Fee")
        elif annual_fee <= 200:
            score += 15
            reasons.append(f"Low annual fee (RM{annual_fee})")
        elif annual_fee <= 500:
            score += 10
            reasons.append(f"Moderate annual fee (RM{annual_fee})")
        else:
            score += 5
            reasons.append(f"Premium card with higher fee (RM{annual_fee})")

        # 3. Benefits Alignment with Spending Patterns (30 points)
        benefits = card.get('benefits', {})
        benefits_text = json.dumps(benefits).lower()

        # Check spending categories against card benefits
        category_matches = 0
        for category, amount in sorted(spending_categories.items(), key=lambda x: x[1], reverse=True)[:3]:
            category_lower = category.lower()

            # Map spending categories to benefit keywords
            benefit_keywords = {
                'petrol': ['petrol', 'fuel', 'gas'],
                'fuel': ['petrol', 'fuel', 'gas'],
                'groceries': ['grocery', 'groceries', 'supermarket'],
                'dining': ['dining', 'restaurant', 'food'],
                'food': ['dining', 'restaurant', 'food'],
                'shopping': ['shopping', 'retail', 'online'],
                'travel': ['travel', 'flight', 'hotel', 'miles'],
                'entertainment': ['entertainment', 'movie', 'cinema'],
                'utilities': ['utilities', 'bills'],
                'transport': ['transport', 'grab', 'taxi', 'public transport']
            }

            keywords = benefit_keywords.get(category_lower, [category_lower])
            if any(keyword in benefits_text for keyword in keywords):
                category_matches += 1
                score += 10
                reasons.append(f"Great benefits for {category} (your top spending category)")
                # Extract relevant benefit
                for benefit_key, benefit_value in benefits.items():
                    if any(keyword in str(benefit_value).lower() for keyword in keywords):
                        highlighted_benefits.append(f"{benefit_key}: {benefit_value}")
                        break

        if category_matches == 0:
            # Check for general cashback
            if 'cashback' in benefits_text:
                score += 10
                reasons.append("General cashback benefits available")
                highlighted_benefits.append("General Cashback")

        # 4. Debt/Utilization Considerations (20 points)
        if has_debt or credit_utilization > 50:
            # Prefer low-fee cards for debt management
            if annual_fee <= 200:
                score += 20
                reasons.append("Low fee suitable for debt management")
            else:
                score += 5
        else:
            # Can afford premium cards
            if annual_fee > 200:
                score += 10
                reasons.append("Premium benefits worth the annual fee")
            else:
                score += 15

        # 5. Special promotions or features (bonus points)
        promotions = card.get('promotions', [])
        if promotions:
            score += 5
            reasons.append(f"Active promotions available ({len(promotions)} offers)")

        # Cap score at 100
        score = min(score, 100)

        return {
            'match_score': round(score, 1),
            'reasoning': reasons,
            'highlighted_benefits': highlighted_benefits[:5]  # Top 5 benefits
        }

    def _build_recommendation_prompt(
        self,
        user_profile: Dict[str, Any],
        all_cards: List[Dict[str, Any]],
        max_results: int
    ) -> str:
        """
        Build AI prompt for credit card recommendations.

        Args:
            user_profile: User's financial profile
            all_cards: List of available credit cards
            max_results: Maximum number of recommendations

        Returns:
            Formatted prompt string for Gemini AI
        """
        monthly_income = user_profile.get('monthly_income', 0)
        spending_categories = user_profile.get('spending_categories', {})
        credit_utilization = user_profile.get('credit_utilization', 0)
        has_debt = user_profile.get('has_debt', False)

        # Format spending categories with ALL categories for better value calculation
        spending_text = ""
        all_spending_text = ""
        if spending_categories:
            top_3 = sorted(spending_categories.items(), key=lambda x: x[1], reverse=True)[:3]
            spending_text = ", ".join([f"{cat}: RM{amt:,.2f}" for cat, amt in top_3])

            # Include ALL spending categories for accurate value calculation
            all_categories = sorted(spending_categories.items(), key=lambda x: x[1], reverse=True)
            all_spending_text = "\n".join([f"  - {cat}: RM{amt:,.2f}/month" for cat, amt in all_categories])
        else:
            spending_text = "No spending data available"
            all_spending_text = "  - No spending data available"

        # Format cards as simplified JSON (to reduce token usage)
        cards_simplified = []
        for card in all_cards[:20]:  # Limit to top 20 cards to avoid token limits
            cards_simplified.append({
                'card_id': str(card.get('_id', '')),
                'card_name': card.get('card_name', 'Unknown'),
                'bank_name': card.get('bank_name', 'Unknown'),
                'card_brand': card.get('card_brand', 'Unknown'),
                'annual_fee': card.get('annual_fee', 0),
                'eligibility_criteria': card.get('eligibility_criteria', {}),
                'benefits': card.get('benefits', {}),
                'promotions': card.get('promotions', [])
            })

        cards_json = json.dumps(cards_simplified, indent=2)

        prompt = f"""You are a professional financial advisor speaking directly to a Malaysian credit card user. Analyze their profile and recommend the best cards in a personalized, conversational tone.

USER'S FINANCIAL PROFILE:
- Monthly Income: RM{monthly_income:,.2f}
- Annual Income: RM{monthly_income * 12:,.2f}
- Top Monthly Spending: {spending_text}
- Credit Utilization: {credit_utilization:.1f}%
- Debt Status: {"Has existing credit card debt" if has_debt else "No debt"}

DETAILED MONTHLY SPENDING BY CATEGORY:
{all_spending_text}

AVAILABLE CARDS IN MALAYSIAN MARKET:
{cards_json}

YOUR TASK:
Recommend the top {max_results} credit cards that best match THIS USER's specific financial situation and spending habits. Write as if you're speaking directly to them.

PERSONALIZATION REQUIREMENTS:
1. Always use "you", "your", "you're" when referring to the user
2. Reference their ACTUAL spending amounts and categories specifically
3. Calculate exact savings based on THEIR spending (e.g., "Your RM800 monthly petrol spending × 8% cashback = RM768/year you'll save")
4. Compare cards to THEIR current situation (income level, debt status, spending patterns)
5. Be conversational and relatable - write like a trusted financial advisor talking to a friend
6. If they have high utilization/debt, show empathy and focus on practical solutions
7. If they have specific high-spending categories, emphasize how the card maximizes THOSE rewards

MATCHING CRITERIA:
1. Income Eligibility: Must meet minimum requirements
2. Spending Alignment: Card benefits must match THEIR top spending categories
3. Fee vs Value: Calculate if annual fees are worth it based on THEIR usage
4. Debt-Conscious: Prioritize low/no fees if they have existing debt
5. Malaysian Relevance: Local merchants, banks, and promotions they'll actually use

For each card, provide:
1. Match score (0-100): How well it fits THEIR specific profile
2. Primary reason: One compelling sentence about why THIS card is perfect for THEM
3. Detailed reasoning (4-5 points):
   - Start with how it matches their top spending category
   - Include specific calculations using THEIR actual spending amounts
   - Compare to their current financial situation
   - Mention practical benefits for their lifestyle
   - Address any fees and whether they're justified for THEM
4. Highlighted benefits (2-3): Only the benefits most relevant to THEIR spending

EXAMPLES OF PERSONALIZED REASONING:
❌ BAD (Generic): "This card offers 8% cashback on petrol"
✅ GOOD: "With your RM800/month petrol spending, you'll earn RM768/year with this card's 8% cashback at Shell and Petronas"

❌ BAD: "Low annual fee"
✅ GOOD: "At RM0 annual fee, you're saving RM200/year compared to premium cards while still getting cashback on your top spending categories"

❌ BAD: "Good for people with debt"
✅ GOOD: "Since you currently have credit card debt, this card's RM0 annual fee means more of your payments go toward reducing your balance instead of fees"

OUTPUT FORMAT:
Return ONLY a valid JSON array with no additional text:
[
  {{
    "card_id": "...",
    "card_name": "Card Name",
    "bank_name": "Bank Name",
    "card_brand": "Visa/Mastercard/etc",
    "annual_fee": 0,
    "value": 912,
    "match_score": 92,
    "primary_reason": "Perfect for your RM800/month petrol spending with 8% cashback, saving you RM768 yearly",
    "reasoning": [
      "Your monthly petrol spending of RM800 × 8% cashback = RM768/year in rewards at Shell and Petronas stations",
      "With your RM5,000 monthly income, you easily qualify for this card (minimum RM24,000/year required)",
      "RM0 annual fee means you keep all your rewards, unlike premium cards charging RM200+",
      "Your RM600/month grocery spending gets an additional 2% cashback = RM144/year extra savings",
      "Total estimated value for YOU: RM912/year with zero fees - exceptional return for your spending pattern"
    ],
    "highlighted_benefits": [
      "8% Petrol Cashback (your top spending)",
      "2% Grocery Cashback",
      "No Annual Fee"
    ]
  }}
]

CRITICAL - VALUE CALCULATION:
The "value" field is MANDATORY and must be a number (not a string) representing the estimated total annual rewards/cashback in RM.

HOW TO CALCULATE VALUE:
1. Look at the user's DETAILED MONTHLY SPENDING BY CATEGORY above
2. For each category, check if the card offers cashback/rewards for that category
3. Calculate: (Monthly Spending × Cashback Rate × 12 months)
4. Sum all categories to get total annual value
5. Example:
   - Petrol: RM800/month × 8% cashback × 12 = RM768/year
   - Groceries: RM600/month × 2% cashback × 12 = RM144/year
   - Total value = RM912/year

If user has NO spending data or card has NO benefits matching their spending, set value to 0.
The value should reflect ACTUAL rewards they will receive based on THEIR spending, not generic potential."""

        return prompt

    def _parse_ai_recommendations(self, ai_response: str) -> List[Dict[str, Any]]:
        """
        Parse AI response into structured recommendations.

        Args:
            ai_response: Raw response from Gemini AI

        Returns:
            List of recommendation dictionaries
        """
        try:
            # Try to extract JSON from response
            # Sometimes AI wraps JSON in markdown code blocks
            response_text = ai_response.strip()

            # Remove markdown code blocks if present
            if response_text.startswith('```'):
                # Find the actual JSON content
                lines = response_text.split('\n')
                json_lines = []
                in_json = False
                for line in lines:
                    if line.strip().startswith('```'):
                        in_json = not in_json
                        continue
                    if in_json:
                        json_lines.append(line)
                response_text = '\n'.join(json_lines)

            # Parse JSON
            recommendations = json.loads(response_text)

            if not isinstance(recommendations, list):
                logger.error(f"AI response is not a list: {type(recommendations)}")
                return []

            # Validate and clean each recommendation
            validated_recommendations = []
            for rec in recommendations:
                if not isinstance(rec, dict):
                    continue

                # Ensure required fields exist
                validated_rec = {
                    'card_id': rec.get('card_id', ''),
                    'card_name': rec.get('card_name', 'Unknown'),
                    'bank_name': rec.get('bank_name', 'Unknown'),
                    'card_brand': rec.get('card_brand', 'Unknown'),
                    'annual_fee': rec.get('annual_fee', 0),
                    'value': rec.get('value', 0),  # Estimated annual value
                    'match_score': min(max(rec.get('match_score', 0), 0), 100),  # Clamp 0-100
                    'reasoning': rec.get('reasoning', []),
                    'highlighted_benefits': rec.get('highlighted_benefits', []),
                    'primary_reason': rec.get('primary_reason', '')
                }

                # Ensure reasoning and benefits are lists
                if not isinstance(validated_rec['reasoning'], list):
                    validated_rec['reasoning'] = [str(validated_rec['reasoning'])]
                if not isinstance(validated_rec['highlighted_benefits'], list):
                    validated_rec['highlighted_benefits'] = [str(validated_rec['highlighted_benefits'])]

                validated_recommendations.append(validated_rec)

            return validated_recommendations

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse AI response as JSON: {e}")
            logger.error(f"AI response was: {ai_response[:500]}")  # Log first 500 chars
            return []
        except Exception as e:
            logger.error(f"Error parsing AI recommendations: {e}")
            return []

    def _get_ai_card_recommendations(
        self,
        user_profile: Dict[str, Any],
        all_cards: List[Dict[str, Any]],
        max_results: int
    ) -> List[Dict[str, Any]]:
        """
        Use Gemini AI to analyze and recommend credit cards.

        Args:
            user_profile: User's financial profile
            all_cards: List of available credit cards
            max_results: Maximum number of recommendations

        Returns:
            List of AI-recommended cards with scores and reasoning
        """
        try:
            from services.gemini_service import get_gemini_service

            # Get Gemini service
            gemini = get_gemini_service()

            # Build AI prompt
            prompt = self._build_recommendation_prompt(user_profile, all_cards, max_results)

            logger.info(f"Requesting AI recommendations for user with income RM{user_profile.get('monthly_income', 0):,.2f}")

            # Get AI response (synchronous)
            response = gemini.generate_content_sync(prompt)

            if not response:
                logger.error("Empty response from Gemini AI")
                raise ValueError("Empty AI response")

            # Parse AI recommendations
            recommendations = self._parse_ai_recommendations(response)

            if not recommendations:
                logger.error("Failed to parse AI recommendations")
                raise ValueError("Invalid AI response format")

            logger.info(f"AI successfully recommended {len(recommendations)} cards")

            # Enrich recommendations with full card data from MongoDB
            enriched_recommendations = []
            card_lookup = {str(card.get('_id', '')): card for card in all_cards}

            for rec in recommendations:
                card_id = rec.get('card_id', '')
                full_card = card_lookup.get(card_id, {})

                # Merge AI analysis with full card data
                enriched_rec = {
                    'card_id': card_id,
                    'card_name': rec.get('card_name', full_card.get('card_name', 'Unknown')),
                    'bank_name': rec.get('bank_name', full_card.get('bank_name', 'Unknown')),
                    'card_brand': rec.get('card_brand', full_card.get('card_brand', 'Unknown')),
                    'annual_fee': rec.get('annual_fee', full_card.get('annual_fee', 0)),
                    'match_score': rec.get('match_score', 0),
                    'reasoning': rec.get('reasoning', []),
                    'highlighted_benefits': rec.get('highlighted_benefits', []),
                    'primary_reason': rec.get('primary_reason', ''),
                    'eligibility_criteria': full_card.get('eligibility_criteria', {}),
                    'benefits': full_card.get('benefits', {}),
                    'promotions': full_card.get('promotions', [])
                }

                enriched_recommendations.append(enriched_rec)

            return enriched_recommendations

        except Exception as e:
            logger.error(f"AI recommendation failed: {e}")
            # Re-raise to trigger fallback in recommend_credit_cards
            raise

    def _get_rule_based_recommendations(
        self,
        user_profile: Dict[str, Any],
        all_cards: List[Dict[str, Any]],
        max_results: int
    ) -> List[Dict[str, Any]]:
        """
        Fallback rule-based recommendation system.

        This is the original scoring algorithm, kept as a fallback
        when AI recommendations fail.

        Args:
            user_profile: User's financial profile
            all_cards: List of available credit cards
            max_results: Maximum number of recommendations

        Returns:
            List of recommended cards using rule-based scoring
        """
        # Score each card using rule-based algorithm
        scored_cards = []
        for card in all_cards:
            match_result = self._calculate_card_match_score(card, user_profile)

            scored_cards.append({
                'card_id': str(card.get('_id', '')),
                'card_name': card.get('card_name', 'Unknown'),
                'bank_name': card.get('bank_name', 'Unknown'),
                'card_brand': card.get('card_brand', 'Unknown'),
                'annual_fee': card.get('annual_fee', 0),
                'match_score': match_result['match_score'],
                'reasoning': match_result['reasoning'],
                'highlighted_benefits': match_result['highlighted_benefits'],
                'eligibility_criteria': card.get('eligibility_criteria', {}),
                'benefits': card.get('benefits', {}),
                'promotions': card.get('promotions', [])
            })

        # Sort by match score (descending)
        scored_cards.sort(key=lambda x: x['match_score'], reverse=True)

        # Return top N recommendations
        return scored_cards[:max_results]

    def recommend_credit_cards(self, user_id: int, max_results: int = 5, use_ai: bool = True) -> Dict[str, Any]:
        """
        Recommend credit cards based on user's financial profile using AI or rule-based matching.

        Args:
            user_id: User ID
            max_results: Maximum number of recommendations (default 5)
            use_ai: Use AI-powered recommendations (default True). Falls back to rule-based if AI fails.

        Returns:
            Dictionary with recommended cards, match scores, reasoning, and metadata
        """
        try:
            # Get user's financial profile
            financial_data = self.get_financial_summary(user_id)

            # Calculate monthly income
            transactions = financial_data.get("transactions", {})
            total_income = transactions.get("total_income_90d", 0)
            monthly_income = (total_income / 90) * 30 if total_income > 0 else 0

            # Get spending patterns
            spending = financial_data.get("spending_summary", {})
            spending_categories = spending.get("by_category", {})

            # Get credit card status
            credit_cards = financial_data.get("credit_cards", {})
            total_utilization = credit_cards.get("total_utilization", 0)
            total_balance = credit_cards.get("total_balance", 0)

            # Build user profile for matching
            user_profile = {
                'monthly_income': monthly_income,
                'spending_categories': spending_categories,
                'credit_utilization': total_utilization,
                'has_debt': total_balance > 0
            }

            # Get all market credit cards from MongoDB
            all_cards = self.get_market_credit_cards_from_mongo()

            if not all_cards:
                return {
                    'recommendations': [],
                    'user_profile_summary': user_profile,
                    'ai_powered': False,
                    'message': 'No credit cards available in the database'
                }

            # Try AI-powered recommendations first
            ai_powered = False
            recommendations = []

            if use_ai:
                try:
                    logger.info(f"Attempting AI-powered recommendations for user {user_id}")
                    recommendations = self._get_ai_card_recommendations(
                        user_profile, all_cards, max_results
                    )
                    ai_powered = True
                    logger.info(f"AI recommendations successful: {len(recommendations)} cards")
                except Exception as ai_error:
                    logger.warning(f"AI recommendations failed, falling back to rule-based: {ai_error}")
                    # Fallback to rule-based
                    recommendations = self._get_rule_based_recommendations(
                        user_profile, all_cards, max_results
                    )
                    ai_powered = False
            else:
                # Use rule-based directly if AI is disabled
                logger.info(f"Using rule-based recommendations for user {user_id}")
                recommendations = self._get_rule_based_recommendations(
                    user_profile, all_cards, max_results
                )
                ai_powered = False

            return {
                'recommendations': recommendations,
                'ai_powered': ai_powered,
                'total_cards_analyzed': len(all_cards),
                'user_profile_summary': {
                    'monthly_income': round(monthly_income, 2),
                    'annual_income': round(monthly_income * 12, 2),
                    'top_spending_categories': list(spending_categories.keys())[:3],
                    'credit_utilization': round(total_utilization, 2),
                    'has_existing_debt': total_balance > 0
                },
                'message': f'Found {len(recommendations)} {"AI-recommended" if ai_powered else "recommended"} credit cards based on your financial profile'
            }

        except Exception as e:
            logger.error(f"Error generating credit card recommendations: {e}")
            return {
                'recommendations': [],
                'ai_powered': False,
                'error': str(e),
                'message': 'Failed to generate credit card recommendations'
            }

    def format_card_recommendations_for_llm(self, recommendations_data: Dict[str, Any]) -> str:
        """
        Format credit card recommendations as text for LLM context.

        Args:
            recommendations_data: Dictionary from recommend_credit_cards()

        Returns:
            Formatted text context
        """
        context_parts = []

        recommendations = recommendations_data.get('recommendations', [])
        profile = recommendations_data.get('user_profile_summary', {})

        if not recommendations:
            return "No credit card recommendations available at this time."

        context_parts.append("=== CREDIT CARD RECOMMENDATIONS ===")
        context_parts.append(f"Based on your financial profile:")
        context_parts.append(f"- Monthly Income: RM{profile.get('monthly_income', 0):,.2f}")
        context_parts.append(f"- Annual Income: RM{profile.get('annual_income', 0):,.2f}")

        top_categories = profile.get('top_spending_categories', [])
        if top_categories:
            context_parts.append(f"- Top Spending: {', '.join(top_categories)}")

        context_parts.append(f"- Credit Utilization: {profile.get('credit_utilization', 0):.1f}%")
        context_parts.append("")

        for i, card in enumerate(recommendations, 1):
            context_parts.append(f"{i}. {card['card_name']} ({card['bank_name']}) - {card['card_brand']}")
            context_parts.append(f"   Match Score: {card['match_score']}/100")
            context_parts.append(f"   Annual Fee: RM{card['annual_fee']}")

            # Reasoning
            if card.get('reasoning'):
                context_parts.append("   Why this card:")
                for reason in card['reasoning'][:3]:  # Top 3 reasons
                    context_parts.append(f"   - {reason}")

            # Highlighted benefits
            if card.get('highlighted_benefits'):
                context_parts.append("   Key Benefits:")
                for benefit in card['highlighted_benefits'][:3]:  # Top 3 benefits
                    context_parts.append(f"   - {benefit}")

            context_parts.append("")

        return "\n".join(context_parts)


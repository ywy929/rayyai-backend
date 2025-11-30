"""
Action Execution Engine
Parses and executes financial actions from LLM responses
"""
from typing import Dict, Any, List, Optional
from datetime import date, datetime
from sqlalchemy.orm import Session
import models
import schemas
import logging
import json
import re
from services.processors import detect_card_brand

logger = logging.getLogger(__name__)

class ActionExecutor:
    """Service for executing financial actions requested by AI"""
    
    def __init__(self, db: Session, user_id: int):
        """
        Initialize action executor.
        
        Args:
            db: Database session
            user_id: User ID for permission checks
        """
        self.db = db
        self.user_id = user_id
    
    def parse_action_request(self, llm_response: str) -> List[Dict[str, Any]]:
        """
        Parse action requests from LLM response.
        Looks for structured action blocks in the response.
        
        Args:
            llm_response: LLM response text
            
        Returns:
            List of parsed actions
        """
        actions = []
        
        # Pattern 1: JSON action blocks
        json_pattern = r'<action>(.*?)</action>'
        json_matches = re.findall(json_pattern, llm_response, re.DOTALL)
        
        for match in json_matches:
            try:
                action_data = json.loads(match.strip())
                if isinstance(action_data, dict) and "action" in action_data:
                    actions.append(action_data)
            except json.JSONDecodeError:
                continue
        
        # Pattern 2: Function call format
        function_pattern = r'(create_budget|update_budget|delete_budget|create_goal|update_goal|delete_goal|categorize_transaction|create_credit_card|update_credit_card|delete_credit_card|analyze_credit_utilization|confirm_statement_import)\s*\(([^)]+)\)'
        function_matches = re.findall(function_pattern, llm_response, re.IGNORECASE)
        
        for func_name, params_str in function_matches:
            try:
                # Try to parse parameters
                params = self._parse_function_params(params_str)
                actions.append({
                    "action": func_name,
                    "parameters": params
                })
            except Exception as e:
                logger.warning(f"Failed to parse function call: {e}")
                continue
        
        return actions
    
    def _parse_function_params(self, params_str: str) -> Dict[str, Any]:
        """Parse function parameters from string."""
        params = {}
        # Simple parsing - can be enhanced
        # Format: key="value" or key=value
        param_pattern = r'(\w+)=["\']?([^,"\']+)["\']?'
        matches = re.findall(param_pattern, params_str)
        for key, value in matches:
            # Try to convert to appropriate type
            try:
                if value.replace('.', '').isdigit():
                    params[key] = float(value) if '.' in value else int(value)
                else:
                    params[key] = value
            except:
                params[key] = value
        return params
    
    async def execute_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a single action.
        
        Args:
            action: Action dictionary with 'action' and 'parameters'
            
        Returns:
            Execution result dictionary
        """
        action_type = action.get("action", "").lower()
        params = action.get("parameters", {})
        
        try:
            if action_type == "create_budget":
                return await self._create_budget(params)
            elif action_type == "update_budget":
                return await self._update_budget(params)
            elif action_type == "delete_budget":
                return await self._delete_budget(params)
            elif action_type == "create_goal":
                return await self._create_goal(params)
            elif action_type == "update_goal":
                return await self._update_goal(params)
            elif action_type == "delete_goal":
                return await self._delete_goal(params)
            elif action_type == "categorize_transaction":
                return await self._categorize_transaction(params)
            elif action_type == "create_expense":
                return await self._create_expense(params)
            elif action_type == "create_income":
                return await self._create_income(params)
            elif action_type == "create_credit_card":
                return await self._create_credit_card(params)
            elif action_type == "update_credit_card":
                return await self._update_credit_card(params)
            elif action_type == "delete_credit_card":
                return await self._delete_credit_card(params)
            elif action_type == "analyze_credit_utilization":
                return await self._analyze_credit_utilization(params)
            elif action_type == "confirm_statement_import":
                return await self._confirm_statement_import(params)
            else:
                return {
                    "success": False,
                    "error": f"Unknown action type: {action_type}"
                }
        except Exception as e:
            logger.error(f"Error executing action {action_type}: {e}")
            return {
                "success": False,
                "error": str(e),
                "action": action_type
            }
    
    async def _create_budget(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new budget."""
        # Validate required fields
        required = ["name", "limit_amount", "category", "period_start", "period_end", "alert_threshold"]
        missing = [f for f in required if f not in params]
        if missing:
            return {"success": False, "error": f"Missing required fields: {missing}"}

        # Create budget
        budget = models.Budget(
            user_id=self.user_id,
            name=params["name"],
            limit_amount=float(params["limit_amount"]),
            category=params["category"],
            period_start=datetime.strptime(params["period_start"], "%Y-%m-%d").date(),
            period_end=datetime.strptime(params["period_end"], "%Y-%m-%d").date(),
            alert_threshold=float(params["alert_threshold"])  # Expects 0-100 percentage value
        )
        
        self.db.add(budget)
        self.db.commit()
        self.db.refresh(budget)
        
        return {
            "success": True,
            "action": "create_budget",
            "budget_id": budget.budget_id,
            "message": f"Budget '{budget.name}' created successfully"
        }
    
    async def _update_budget(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Update an existing budget."""
        budget_id = params.get("budget_id")
        if not budget_id:
            return {"success": False, "error": "budget_id is required"}
        
        budget = self.db.query(models.Budget).filter(
            models.Budget.budget_id == budget_id,
            models.Budget.user_id == self.user_id,
            models.Budget.is_deleted == False
        ).first()
        
        if not budget:
            return {"success": False, "error": "Budget not found"}
        
        # Update fields
        if "name" in params:
            budget.name = params["name"]
        if "limit_amount" in params:
            budget.limit_amount = float(params["limit_amount"])
        if "category" in params:
            budget.category = params["category"]
        
        self.db.commit()
        
        return {
            "success": True,
            "action": "update_budget",
            "budget_id": budget.budget_id,
            "message": f"Budget '{budget.name}' updated successfully"
        }
    
    async def _create_goal(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new financial goal."""
        required = ["goal_name", "description", "category", "priority", "target_amount"]
        missing = [f for f in required if f not in params]
        if missing:
            return {"success": False, "error": f"Missing required fields: {missing}"}
        
        goal = models.Goal(
            user_id=self.user_id,
            goal_name=params["goal_name"],
            description=params["description"],
            category=params["category"],
            priority=params["priority"],
            target_amount=float(params["target_amount"]),
            current_amount=float(params.get("current_amount", 0)),
            target_date=datetime.strptime(params["target_date"], "%Y-%m-%d").date() if params.get("target_date") else None
        )
        
        self.db.add(goal)
        self.db.commit()
        self.db.refresh(goal)
        
        return {
            "success": True,
            "action": "create_goal",
            "goal_id": goal.goal_id,
            "message": f"Goal '{goal.goal_name}' created successfully"
        }
    
    async def _update_goal(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Update an existing goal."""
        goal_id = params.get("goal_id")
        if not goal_id:
            return {"success": False, "error": "goal_id is required"}
        
        goal = self.db.query(models.Goal).filter(
            models.Goal.goal_id == goal_id,
            models.Goal.user_id == self.user_id,
            models.Goal.is_deleted == False
        ).first()
        
        if not goal:
            return {"success": False, "error": "Goal not found"}
        
        # Update fields
        if "goal_name" in params:
            goal.goal_name = params["goal_name"]
        if "description" in params:
            goal.description = params["description"]
        if "target_amount" in params:
            goal.target_amount = float(params["target_amount"])
        if "current_amount" in params:
            goal.current_amount = float(params["current_amount"])
        
        self.db.commit()
        
        return {
            "success": True,
            "action": "update_goal",
            "goal_id": goal.goal_id,
            "message": f"Goal '{goal.goal_name}' updated successfully"
        }
    
    async def _categorize_transaction(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Categorize a transaction."""
        transaction_id = params.get("transaction_id")
        category = params.get("category")
        
        if not transaction_id or not category:
            return {"success": False, "error": "transaction_id and category are required"}
        
        # Try to find as expense first
        expense = self.db.query(models.Expense).filter(
            models.Expense.expense_id == transaction_id,
            models.Expense.user_id == self.user_id,
            models.Expense.is_deleted == False
        ).first()
        
        if expense:
            expense.category = category
            if "expense_type" in params:
                expense.expense_type = params["expense_type"]
            self.db.commit()
            return {
                "success": True,
                "action": "categorize_transaction",
                "transaction_id": transaction_id,
                "message": f"Transaction categorized as '{category}'"
            }
        
        # Try as income
        income = self.db.query(models.Income).filter(
            models.Income.income_id == transaction_id,
            models.Income.user_id == self.user_id,
            models.Income.is_deleted == False
        ).first()
        
        if income:
            income.category = category
            self.db.commit()
            return {
                "success": True,
                "action": "categorize_transaction",
                "transaction_id": transaction_id,
                "message": f"Transaction categorized as '{category}'"
            }
        
        return {"success": False, "error": "Transaction not found"}
    
    async def _create_expense(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create an expense record."""
        required = ["account_id", "amount", "description", "category", "date_spent", "seller"]
        missing = [f for f in required if f not in params]
        if missing:
            return {"success": False, "error": f"Missing required fields: {missing}"}
        
        expense = models.Expense(
            user_id=self.user_id,
            account_id=int(params["account_id"]),
            amount=float(params["amount"]),
            description=params["description"],
            category=params["category"],
            date_spent=datetime.strptime(params["date_spent"], "%Y-%m-%d").date(),
            seller=params["seller"],
            location=params.get("location"),
            expense_type=params.get("expense_type")
        )
        
        self.db.add(expense)
        self.db.commit()
        self.db.refresh(expense)
        
        return {
            "success": True,
            "action": "create_expense",
            "expense_id": expense.expense_id,
            "message": f"Expense '{expense.description}' created successfully"
        }
    
    async def _create_income(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create an income record."""
        required = ["account_id", "amount", "category", "date_received", "payer"]
        missing = [f for f in required if f not in params]
        if missing:
            return {"success": False, "error": f"Missing required fields: {missing}"}
        
        income = models.Income(
            user_id=self.user_id,
            account_id=int(params["account_id"]),
            amount=float(params["amount"]),
            category=params["category"],
            date_received=datetime.strptime(params["date_received"], "%Y-%m-%d").date(),
            payer=params["payer"],
            description=params.get("description")
        )
        
        self.db.add(income)
        self.db.commit()
        self.db.refresh(income)
        
        return {
            "success": True,
            "action": "create_income",
            "income_id": income.income_id,
            "message": f"Income from '{income.payer}' created successfully"
        }
    
    async def _delete_budget(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Delete a budget (soft delete)."""
        budget_id = params.get("budget_id")
        if not budget_id:
            return {"success": False, "error": "budget_id is required"}

        budget = self.db.query(models.Budget).filter(
            models.Budget.budget_id == budget_id,
            models.Budget.user_id == self.user_id,
            models.Budget.is_deleted == False
        ).first()

        if not budget:
            return {"success": False, "error": "Budget not found"}

        budget.is_deleted = True
        self.db.commit()

        return {
            "success": True,
            "action": "delete_budget",
            "budget_id": budget.budget_id,
            "message": f"Budget '{budget.name}' deleted successfully"
        }

    async def _delete_goal(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Delete a goal (soft delete)."""
        goal_id = params.get("goal_id")
        if not goal_id:
            return {"success": False, "error": "goal_id is required"}

        goal = self.db.query(models.Goal).filter(
            models.Goal.goal_id == goal_id,
            models.Goal.user_id == self.user_id,
            models.Goal.is_deleted == False
        ).first()

        if not goal:
            return {"success": False, "error": "Goal not found"}

        goal.is_deleted = True
        self.db.commit()

        return {
            "success": True,
            "action": "delete_goal",
            "goal_id": goal.goal_id,
            "message": f"Goal '{goal.goal_name}' deleted successfully"
        }
    
    async def _create_credit_card(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new credit card."""
        required = ["card_name", "bank_name", "card_brand", "credit_limit", "current_balance"]
        missing = [f for f in required if f not in params]
        if missing:
            return {"success": False, "error": f"Missing required fields: {missing}"}

        # Parse next_payment_date if provided
        next_payment_date = None
        if params.get("next_payment_date"):
            try:
                next_payment_date = datetime.strptime(params["next_payment_date"], "%Y-%m-%d").date()
            except:
                pass

        # Apply card_brand fallback if not detected, empty, or "Unknown"
        card_brand = params["card_brand"]
        if not card_brand or card_brand == '' or (isinstance(card_brand, str) and card_brand.lower() == 'unknown'):
            card_brand = detect_card_brand(
                account_name=params.get("card_name"),
                account_type=params.get("account_type"),
                card_brand=None
            )
            logger.info(f"Applied card_brand fallback for credit card creation: {card_brand} (was: {params.get('card_brand')})")

        card = models.UserCreditCard(
            user_id=self.user_id,
            card_number=params.get("card_number", f"****-****-****-{params.get('last_four', '0000')}"),
            card_name=params["card_name"],
            bank_name=params["bank_name"],
            card_brand=card_brand,
            credit_limit=float(params["credit_limit"]),
            current_balance=float(params["current_balance"]),
            annual_fee=float(params.get("annual_fee", 0)),
            expiry_month=int(params["expiry_month"]) if params.get("expiry_month") else None,
            expiry_year=int(params["expiry_year"]) if params.get("expiry_year") else None,
            next_payment_amount=float(params["next_payment_amount"]) if params.get("next_payment_amount") else None,
            next_payment_date=next_payment_date,
            benefits=params.get("benefits", {})
        )
        
        self.db.add(card)
        self.db.commit()
        self.db.refresh(card)
        
        return {
            "success": True,
            "action": "create_credit_card",
            "card_id": card.card_id,
            "message": f"Credit card '{card.card_name}' added successfully"
        }
    
    async def _update_credit_card(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Update an existing credit card."""
        card_id = params.get("card_id")
        if not card_id:
            return {"success": False, "error": "card_id is required"}
        
        card = self.db.query(models.UserCreditCard).filter(
            models.UserCreditCard.card_id == card_id,
            models.UserCreditCard.user_id == self.user_id,
            models.UserCreditCard.is_deleted == False
        ).first()
        
        if not card:
            return {"success": False, "error": "Credit card not found"}
        
        # Update fields
        if "card_name" in params:
            card.card_name = params["card_name"]
        if "bank_name" in params:
            card.bank_name = params["bank_name"]
        if "card_brand" in params:
            card.card_brand = params["card_brand"]
        if "credit_limit" in params:
            card.credit_limit = float(params["credit_limit"])
        if "current_balance" in params:
            card.current_balance = float(params["current_balance"])
        if "annual_fee" in params:
            card.annual_fee = float(params["annual_fee"])
        if "expiry_month" in params:
            card.expiry_month = int(params["expiry_month"])
        if "expiry_year" in params:
            card.expiry_year = int(params["expiry_year"])
        if "next_payment_amount" in params:
            card.next_payment_amount = float(params["next_payment_amount"])
        if "next_payment_date" in params:
            try:
                card.next_payment_date = datetime.strptime(params["next_payment_date"], "%Y-%m-%d").date()
            except:
                pass
        if "benefits" in params:
            card.benefits = params["benefits"]
        
        self.db.commit()
        
        return {
            "success": True,
            "action": "update_credit_card",
            "card_id": card.card_id,
            "message": f"Credit card '{card.card_name}' updated successfully"
        }
    
    async def _delete_credit_card(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Delete a credit card (soft delete)."""
        card_id = params.get("card_id")
        if not card_id:
            return {"success": False, "error": "card_id is required"}
        
        card = self.db.query(models.UserCreditCard).filter(
            models.UserCreditCard.card_id == card_id,
            models.UserCreditCard.user_id == self.user_id,
            models.UserCreditCard.is_deleted == False
        ).first()
        
        if not card:
            return {"success": False, "error": "Credit card not found"}
        
        card.is_deleted = True
        self.db.commit()
        
        return {
            "success": True,
            "action": "delete_credit_card",
            "card_id": card.card_id,
            "message": f"Credit card '{card.card_name}' removed successfully"
        }
    
    async def _analyze_credit_utilization(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze credit card utilization and provide recommendations."""
        card_id = params.get("card_id")
        
        if card_id:
            # Analyze specific card
            card = self.db.query(models.UserCreditCard).filter(
                models.UserCreditCard.card_id == card_id,
                models.UserCreditCard.user_id == self.user_id,
                models.UserCreditCard.is_deleted == False
            ).first()
            
            if not card:
                return {"success": False, "error": "Credit card not found"}
            
            utilization = (card.current_balance / card.credit_limit * 100) if card.credit_limit > 0 else 0
            
            return {
                "success": True,
                "action": "analyze_credit_utilization",
                "card_id": card.card_id,
                "card_name": card.card_name,
                "utilization": round(utilization, 2),
                "current_balance": card.current_balance,
                "credit_limit": card.credit_limit,
                "available_credit": card.credit_limit - card.current_balance,
                "next_payment_date": card.next_payment_date.isoformat() if card.next_payment_date else None,
                "next_payment_amount": card.next_payment_amount,
                "message": f"Credit card '{card.card_name}' analysis completed"
            }
        else:
            # Analyze all cards
            cards = self.db.query(models.UserCreditCard).filter(
                models.UserCreditCard.user_id == self.user_id,
                models.UserCreditCard.is_deleted == False
            ).all()
            
            if not cards:
                return {"success": False, "error": "No credit cards found"}
            
            total_balance = sum(c.current_balance for c in cards)
            total_limit = sum(c.credit_limit for c in cards)
            total_utilization = (total_balance / total_limit * 100) if total_limit > 0 else 0
            
            card_details = []
            for card in cards:
                utilization = (card.current_balance / card.credit_limit * 100) if card.credit_limit > 0 else 0
                card_details.append({
                    "card_id": card.card_id,
                    "card_name": card.card_name,
                    "utilization": round(utilization, 2),
                    "current_balance": card.current_balance,
                    "credit_limit": card.credit_limit,
                    "next_payment_date": card.next_payment_date.isoformat() if card.next_payment_date else None,
                    "next_payment_amount": card.next_payment_amount
                })
            
            return {
                "success": True,
                "action": "analyze_credit_utilization",
                "total_cards": len(cards),
                "total_utilization": round(total_utilization, 2),
                "total_balance": total_balance,
                "total_limit": total_limit,
                "available_credit": total_limit - total_balance,
                "cards": card_details,
                "message": "Credit utilization analysis completed for all cards"
            }

    async def _confirm_statement_import(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Confirm and import transactions from a previously uploaded statement.
        This is called after user reviews the preview and confirms the import.

        Args:
            params: Must contain statement_id

        Returns:
            Result dictionary with import statistics
        """
        # Validate required fields
        statement_id = params.get("statement_id")
        if not statement_id:
            return {"success": False, "error": "Missing required field: statement_id"}

        # Get statement from database
        statement = self.db.query(models.Statement).filter(
            models.Statement.statement_id == statement_id,
            models.Statement.user_id == self.user_id,
            models.Statement.is_deleted == False
        ).first()

        if not statement:
            return {"success": False, "error": "Statement not found"}

        # Check if already imported
        if statement.processing_status == 'imported':
            return {
                "success": False,
                "error": "Statement already imported. Use force_reimport=true to re-import."
            }

        # Check if extraction data exists
        if not statement.extracted_data:
            return {
                "success": False,
                "error": "Statement not yet extracted. Please upload the file first."
            }

        try:
            # Import the transactions using existing logic from statements.py
            from routers.utils import map_account_type

            result = statement.extracted_data
            transactions = result.get('transactions', [])

            if not transactions:
                return {
                    "success": False,
                    "error": "No transactions found in statement"
                }

            # Get or create account
            target_account = None
            if result.get('account_info'):
                account_info = result['account_info']
                if account_info.get('account_number'):
                    existing_account = self.db.query(models.Account).filter(
                        models.Account.user_id == self.user_id,
                        models.Account.account_no == account_info['account_number'],
                        models.Account.is_deleted == False
                    ).first()
                    if existing_account:
                        target_account = existing_account

                if not target_account:
                    extracted_type = account_info.get('account_type', '')
                    standard_type, subtype = map_account_type(extracted_type)
                    account_name = account_info.get('account_name') or f"{extracted_type} Account"
                    new_account = models.Account(
                        user_id=self.user_id,
                        account_no=account_info.get('account_number', ''),
                        account_name=account_name,
                        account_type=standard_type,
                        account_subtype=subtype,
                        is_deleted=False
                    )
                    self.db.add(new_account)
                    self.db.flush()
                    target_account = new_account

                # Update account balance
                if result.get('closing_balance') is not None:
                    target_account.account_balance = result['closing_balance']

            # Create transactions
            created_incomes = 0
            created_expenses = 0
            skipped = 0

            for txn in transactions:
                if not txn.get('date') or not txn.get('amount') or not txn.get('description'):
                    skipped += 1
                    continue

                try:
                    txn_date = datetime.strptime(txn['date'], '%Y-%m-%d').date()
                except:
                    skipped += 1
                    continue

                if txn['type'] == 'credit' and txn['amount'] > 0:
                    income = models.Income(
                        user_id=self.user_id,
                        account_id=target_account.account_id if target_account else None,
                        statement_id=statement.statement_id,
                        amount=abs(txn['amount']),
                        description=txn['description'][:255],
                        category=txn.get('category', 'Other'),
                        date_received=txn_date,
                        payer=txn.get('payer', ''),
                        reference_no=txn.get('reference', ''),
                        is_deleted=False,
                        created=datetime.now()
                    )
                    self.db.add(income)
                    created_incomes += 1
                elif txn['type'] == 'debit' and txn['amount'] < 0:
                    expense = models.Expense(
                        user_id=self.user_id,
                        account_id=target_account.account_id if target_account else None,
                        statement_id=statement.statement_id,
                        amount=abs(txn['amount']),
                        description=txn['description'][:255],
                        category=txn.get('category', 'Other'),
                        expense_type='needs',
                        date_spent=txn_date,
                        seller=txn.get('seller', ''),
                        location=txn.get('location', ''),
                        reference_no=txn.get('reference', ''),
                        tax_amount=0.0,
                        tax_deductible=False,
                        is_reimbursable=False,
                        is_deleted=False,
                        created=datetime.now()
                    )
                    self.db.add(expense)
                    created_expenses += 1
                else:
                    skipped += 1

            # Update statement status
            statement.processing_status = 'imported'
            self.db.commit()

            return {
                "success": True,
                "action": "confirm_statement_import",
                "statement_id": statement.statement_id,
                "incomes_created": created_incomes,
                "expenses_created": created_expenses,
                "total_imported": created_incomes + created_expenses,
                "skipped": skipped,
                "account_name": target_account.account_name if target_account else None,
                "account_balance": target_account.account_balance if target_account else None,
                "message": f"Successfully imported {created_incomes + created_expenses} transactions ({created_incomes} incomes, {created_expenses} expenses)"
            }

        except Exception as e:
            logger.error(f"Error importing statement {statement_id}: {e}", exc_info=True)
            self.db.rollback()
            return {
                "success": False,
                "error": f"Import failed: {str(e)}"
            }


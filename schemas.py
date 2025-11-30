from pydantic import BaseModel, EmailStr, Field, field_validator, field_serializer
from typing import Optional, List, Dict, Any, Literal
from datetime import date, datetime
from enum import Enum

# ============ ENUMS ============
class GenderEnum(str, Enum):
    male = "Male"
    female = "Female"
    other = "Other"
    prefer_not_to_say = "Prefer not to say"


def _normalize_gender(value):
    if value is None or isinstance(value, GenderEnum):
        return value

    if isinstance(value, str):
        normalized = value.strip().lower()
        mapping = {
            "male": GenderEnum.male,
            "female": GenderEnum.female,
            "other": GenderEnum.other,
            "prefer not to say": GenderEnum.prefer_not_to_say,
            "prefer_not_to_say": GenderEnum.prefer_not_to_say,
            "prefer-not-to-say": GenderEnum.prefer_not_to_say,
        }
        if normalized in mapping:
            return mapping[normalized]

    raise ValueError("Gender must be one of: Male, Female, Other, Prefer not to say")

class AccountTypeEnum(str, Enum):
    savings = "savings"
    current = "current"
    credit = "credit"
    investment = "investment"
    ewallet = "ewallet"
    cash = "cash"

class StatementTypeEnum(str, Enum):
    ctos = "CTOS"
    ccris = "CCRIS"
    bank = "bank"
    credit_card = "credit_card"
    ewallet = "ewallet"
    receipt = "receipt"

class TransactionTypeEnum(str, Enum):
    income = "income"
    expense = "expense"

class ExpenseTypeEnum(str, Enum):
    needs = "needs"
    wants = "wants"

# ============ AUTH SCHEMAS ============
class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserSignup(BaseModel):
    email: EmailStr
    password: str
    full_name: str

class Token(BaseModel):
    access_token: str
    token_type: str
    user: dict

class AuthUserResponse(BaseModel):
    """Response schema for auth endpoints (/auth/me) - contains computed full_name"""
    user_id: int
    email: str
    full_name: str
    created_at: str

    class Config:
        from_attributes = True

# ============ USER SCHEMAS ============
class UserBase(BaseModel):
    first_name: str = Field(..., min_length=1, max_length=100, description="User's first name")
    last_name: str = Field(..., min_length=1, max_length=100, description="User's last name")
    email: EmailStr = Field(..., description="User's email address")
    dob: date = Field(..., description="User's date of birth")
    gender: GenderEnum = Field(..., description="User's gender")

    @field_validator("gender", mode="before")
    @classmethod
    def _normalize_gender_field(cls, value):
        return _normalize_gender(value)

class UserCreate(UserBase):
    password: str = Field(..., min_length=8, max_length=72, description="User password (8-72 characters)")

class UserUpdate(BaseModel):
    first_name: Optional[str] = Field(None, min_length=1, max_length=100)
    last_name: Optional[str] = Field(None, min_length=1, max_length=100)
    email: Optional[EmailStr] = None
    dob: Optional[date] = None
    gender: Optional[GenderEnum] = None
    password: Optional[str] = Field(None, min_length=8, max_length=72)

    @field_validator("gender", mode="before")
    @classmethod
    def _normalize_gender_field(cls, value):
        return _normalize_gender(value)

class UserResponse(UserBase):
    user_id: int
    created: datetime
    updated: datetime

    class Config:
        from_attributes = True

class UserProfileResponse(UserResponse):
    total_accounts: int = 0
    total_balance: float = 0.0
    total_credit_cards: int = 0
    active_goals: int = 0
    active_budgets: int = 0

    class Config:
        from_attributes = True

# ============ ACCOUNT SCHEMAS ============
class AccountBase(BaseModel):
    account_name: str = Field(..., description="Name of the account")
    account_type: AccountTypeEnum = Field(..., description="Standard account type (savings, current, credit, ewallet, investment, cash)")
    account_subtype: Optional[str] = Field(None, description="Specific account variant (e.g., 'Islamic Savings', 'Visa Platinum')")
    account_no: Optional[str] = Field(None, description="Account number (if applicable)")

class AccountCreate(AccountBase):
    card_id: Optional[int] = Field(None, description="Associated credit card ID (if applicable)")

class AccountUpdate(BaseModel):
    account_name: Optional[str] = Field(None, description="Name of the account")
    account_type: Optional[AccountTypeEnum] = Field(None, description="Standard account type")
    account_subtype: Optional[str] = Field(None, description="Specific account variant")
    account_no: Optional[str] = Field(None, description="Account number")
    card_id: Optional[int] = Field(None, description="Associated credit card ID")

class AccountResponse(AccountBase):
    account_id: int
    user_id: int
    card_id: Optional[int] = None
    is_deleted: bool = False

    class Config:
        from_attributes = True

class AccountBalanceResponse(BaseModel):
    account_id: int
    account_name: str
    balance: float

class TotalBalanceResponse(BaseModel):
    total_balance: float
    accounts: List[AccountBalanceResponse]

# ============ BALANCE SNAPSHOT SCHEMAS ============
class BalanceSnapshotCreate(BaseModel):
    account_id: int
    snapshot_date: date
    closing_balance: float

class BalanceSnapshotResponse(BaseModel):
    snapshot_id: int
    account_id: int
    snapshot_date: date
    closing_balance: float
    created: datetime

    class Config:
        from_attributes = True
# ============ INCOME SCHEMAS ============
class IncomeBase(BaseModel):
    account_id: int = Field(..., description="ID of the account receiving income")
    amount: float = Field(..., gt=0, description="Income amount (must be positive)")
    description: Optional[str] = Field(None, description="Description of income")
    category: str = Field(..., description="Income category (e.g., Salary, Freelance, Investment)")
    date_received: date = Field(..., description="Date when income was received")
    payer: str = Field(..., description="Name of the payer/source")
    department: Optional[str] = Field(None, description="Department (if applicable)")
    project: Optional[str] = Field(None, description="Project name (if applicable)")
    reference_no: Optional[str] = Field(None, description="Reference or transaction number")

class IncomeCreate(IncomeBase):
    statement_id: Optional[int] = Field(None, description="Associated statement ID (if from statement)")

class IncomeUpdate(BaseModel):
    account_id: Optional[int] = None
    amount: Optional[float] = Field(None, gt=0)
    description: Optional[str] = None
    category: Optional[str] = None
    date_received: Optional[date] = None
    payer: Optional[str] = None
    department: Optional[str] = None
    project: Optional[str] = None
    reference_no: Optional[str] = None
    statement_id: Optional[int] = None

class IncomeResponse(IncomeBase):
    income_id: int
    user_id: int
    statement_id: Optional[int] = None
    is_deleted: bool = False
    created: datetime

    class Config:
        from_attributes = True

# ============ EXPENSE SCHEMAS ============
class ExpenseBase(BaseModel):
    account_id: int = Field(..., description="ID of the account the expense was paid from")
    amount: float = Field(..., gt=0, description="Expense amount (must be positive)")
    description: str = Field(..., description="Description of the expense")
    category: str = Field(..., description="Expense category (e.g., Food & Dining, Transportation)")
    date_spent: date = Field(..., description="Date when expense occurred")
    seller: str = Field(..., description="Seller/merchant name")
    tax_amount: Optional[float] = Field(0.0, ge=0, description="Tax amount")
    tax_deductible: Optional[bool] = Field(False, description="Is this tax deductible?")
    is_reimbursable: Optional[bool] = Field(False, description="Is this reimbursable?")
    expense_type: Optional[ExpenseTypeEnum] = Field(None, description="Type: needs or wants")
    location: Optional[str] = Field(None, description="Location of purchase")
    reference_no: Optional[str] = Field(None, description="Receipt or reference number")
    card_id: Optional[int] = Field(None, description="Credit card used (if paying card debt)")

class ExpenseCreate(ExpenseBase):
    statement_id: Optional[int] = Field(None, description="Associated statement ID (if from statement)")

class ExpenseUpdate(BaseModel):
    account_id: Optional[int] = None
    amount: Optional[float] = Field(None, gt=0)
    description: Optional[str] = None
    category: Optional[str] = None
    date_spent: Optional[date] = None
    seller: Optional[str] = None
    tax_amount: Optional[float] = Field(None, ge=0)
    tax_deductible: Optional[bool] = None
    is_reimbursable: Optional[bool] = None
    expense_type: Optional[ExpenseTypeEnum] = None
    location: Optional[str] = None
    reference_no: Optional[str] = None
    card_id: Optional[int] = None
    statement_id: Optional[int] = None

class ExpenseResponse(ExpenseBase):
    expense_id: int
    user_id: int
    statement_id: Optional[int] = None
    created: datetime

    class Config:
        from_attributes = True

# ============ TRANSFER SCHEMAS ============
class TransferTypeEnum(str, Enum):
    intra_person = "intra_person"  # Transfer to own account/savings
    inter_person = "inter_person"  # Transfer to another person

class TransferBase(BaseModel):
    account_id: int = Field(..., description="ID of the account the transfer was made from")
    amount: float = Field(..., gt=0, description="Transfer amount (must be positive)")
    description: str = Field(..., description="Description of the transfer")
    category: str = Field(default="Transfer", description="Transfer category (usually 'Transfer')")
    transfer_type: TransferTypeEnum = Field(..., description="Type: intra_person (own account) or inter_person (another person)")
    date_transferred: date = Field(..., description="Date when transfer occurred")
    recipient_account_name: Optional[str] = Field(None, description="Name of recipient account if available")
    recipient_account_no: Optional[str] = Field(None, description="Recipient account number if available")
    reference_no: Optional[str] = Field(None, description="Reference number for the transfer")

class TransferCreate(TransferBase):
    statement_id: Optional[int] = Field(None, description="Associated statement ID (if from statement)")

class TransferUpdate(BaseModel):
    account_id: Optional[int] = None
    amount: Optional[float] = Field(None, gt=0)
    description: Optional[str] = None
    category: Optional[str] = None
    transfer_type: Optional[TransferTypeEnum] = None
    date_transferred: Optional[date] = None
    recipient_account_name: Optional[str] = None
    recipient_account_no: Optional[str] = None
    reference_no: Optional[str] = None
    statement_id: Optional[int] = None

class TransferResponse(TransferBase):
    transfer_id: int
    user_id: int
    statement_id: Optional[int] = None
    is_deleted: bool = False
    created: datetime

    class Config:
        from_attributes = True

# ============ STATEMENT SCHEMAS ============
class StatementBase(BaseModel):
    statement_type: Optional[str] = None
    statement_url: str
    display_name: Optional[str] = None
    period_start: Optional[date] = None
    period_end: Optional[date] = None
    credit_score: Optional[int] = None
    score_text: Optional[str] = None

class StatementCreate(StatementBase):
    pass

class StatementResponse(StatementBase):
    statement_id: int
    user_id: int
    date_uploaded: datetime

    class Config:
        from_attributes = True

# ============ CREDIT CARD SCHEMAS ============

class UserCreditCardBase(BaseModel):
    card_number: str
    card_name: str
    bank_name: str
    card_brand: str
    expiry_month: int
    expiry_year: int
    credit_limit: float
    annual_fee: float
    current_balance: float
    next_payment_amount: Optional[float] = None
    next_payment_date: Optional[date] = None
    benefits: Dict[str, Any]

class UserCreditCardCreate(UserCreditCardBase):
    user_id: int

class UserCreditCardUpdate(BaseModel):
    card_name: Optional[str] = None
    bank_name: Optional[str] = None
    card_brand: Optional[str] = None
    expiry_month: Optional[int] = None
    expiry_year: Optional[int] = None
    credit_limit: Optional[float] = None
    annual_fee: Optional[float] = None
    current_balance: Optional[float] = None
    next_payment_amount: Optional[float] = None
    next_payment_date: Optional[date] = None
    benefits: Optional[Dict[str, Any]] = None

class UserCreditCardResponse(UserCreditCardBase):
    card_id: int
    user_id: int
    created: datetime

    class Config:
        from_attributes = True  # replaces orm_mode in Pydantic v2




# ============ CREDIT CARD TERMS HISTORY SCHEMAS ============
class UserCreditCardTermsHistoryBase(BaseModel):
    effective_date: date
    interest_rate: float
    minimum_payment: Optional[float] = None

class UserCreditCardTermsHistoryCreate(UserCreditCardTermsHistoryBase):
    card_id: int

class UserCreditCardTermsHistoryResponse(UserCreditCardTermsHistoryBase):
    term_history_id: int
    card_id: int
    created: datetime

    class Config:
        from_attributes = True


class CreditCardOverviewCard(BaseModel):
    account_name: Optional[str] = None
    statement_date: Optional[date] = None
    credit_limit: Optional[float] = None
    outstanding_balance: Optional[float] = None
    available_credit: Optional[float] = None


class CreditCardOverviewTotals(BaseModel):
    credit_limit: float = 0.0
    outstanding_balance: float = 0.0
    available_credit: float = 0.0
    utilization_pct: float = 0.0
    cards_count: int = 0
    statement_accounts: int = 0


class CreditCardOverviewResponse(BaseModel):
    totals: CreditCardOverviewTotals
    cards: List[CreditCardOverviewCard]
    monthly_spending_30d: float = 0.0
    transaction_window_days: int = 30

# ============ CARDS OVERVIEW SCHEMAS ============
class CardOverviewItem(BaseModel):
    card_id: int
    card_name: str
    bank_name: Optional[str] = None
    credit_limit: Optional[float] = None
    current_balance: Optional[float] = None
    available_credit: Optional[float] = None
    utilization_pct: float
    next_payment_date: Optional[str] = None
    next_payment_amount: Optional[float] = None

class CardOverviewSummary(BaseModel):
    total_cards: int
    total_limit: float
    total_balance: float
    total_available: float
    utilization_pct: float
    upcoming_payments_count: int
    upcoming_payments_total: float
    monthly_spending: float = 0.0  # Credit card spending for current month

class UpcomingPayment(BaseModel):
    card_name: str
    bank_name: Optional[str] = None
    amount: float
    due_date: str
    days_until_due: int

class CardsOverviewResponse(BaseModel):
    cards: List[CardOverviewItem]
    summary: CardOverviewSummary
    upcoming_payments: List[UpcomingPayment]

# ============ GOAL SCHEMAS ============

# Goal categories matching frontend exactly
GOAL_CATEGORIES = [
    "Emergency Fund",
    "Vacation", 
    "Car Purchase",
    "Home Down Payment",
    "Education",
    "Retirement",
    "Investment",
    "Other"
]

# Goal priorities matching frontend exactly
GOAL_PRIORITIES = ["low", "medium", "high"]

class GoalBase(BaseModel):
    goal_name: str = Field(..., min_length=1, max_length=100)
    description: str = Field(..., min_length=1, max_length=500)
    category: str = Field(..., description="Goal category from predefined list")
    priority: str = Field(..., description="Goal priority: low, medium, or high")
    target_amount: float = Field(..., gt=0)
    current_amount: float = Field(default=0.0, ge=0)
    target_date: Optional[date] = None
    
    @field_validator('category')
    @classmethod
    def validate_category(cls, v):
        if v not in GOAL_CATEGORIES:
            raise ValueError(f'Category must be one of: {", ".join(GOAL_CATEGORIES)}')
        return v
    
    @field_validator('priority')
    @classmethod
    def validate_priority(cls, v):
        if v not in GOAL_PRIORITIES:
            raise ValueError(f'Priority must be one of: {", ".join(GOAL_PRIORITIES)}')
        return v

class GoalCreate(GoalBase):
    pass

class GoalUpdate(BaseModel):
    goal_name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = Field(None, min_length=1, max_length=500)
    category: Optional[str] = Field(None, description="Goal category from predefined list")
    priority: Optional[str] = Field(None, description="Goal priority: low, medium, or high")
    target_amount: Optional[float] = Field(None, gt=0)
    current_amount: Optional[float] = Field(None, ge=0)
    target_date: Optional[date] = None
    
    @field_validator('category')
    @classmethod
    def validate_category(cls, v):
        if v is not None and v not in GOAL_CATEGORIES:
            raise ValueError(f'Category must be one of: {", ".join(GOAL_CATEGORIES)}')
        return v
    
    @field_validator('priority')
    @classmethod
    def validate_priority(cls, v):
        if v is not None and v not in GOAL_PRIORITIES:
            raise ValueError(f'Priority must be one of: {", ".join(GOAL_PRIORITIES)}')
        return v

class GoalContribute(BaseModel):
    amount: float = Field(..., gt=0)

class GoalResponse(GoalBase):
    goal_id: int
    user_id: int
    created_at: datetime
    is_completed: bool
    progress_percentage: float
    days_remaining: Optional[int]
    monthly_required: Optional[float]

    class Config:
        from_attributes = True

class GoalSummary(BaseModel):
    total_goals: int
    active_goals: int
    completed_goals: int
    total_target_amount: float
    total_current_amount: float
    overall_progress_percentage: float

class GoalStats(BaseModel):
    summary: GoalSummary
    goals: List[GoalResponse]

# Note: GoalCategory and GoalCategoriesResponse removed as we now use simple List[str] responses

# ============ BUDGET SCHEMAS ============

# Budget/Expense categories matching frontend expense categories
# These categories are used for both budgets and expenses to ensure consistency
BUDGET_CATEGORIES = [
    "Groceries",
    "Transportation",
    "Entertainment",
    "Utilities",
    "Shopping",
    "Food & Dining",
    "Health & Fitness",
    "Travel",
    "Education",
    "Housing",
    "Insurance",
    "Personal Care",
    "Others"
]

# Budget periods
BUDGET_PERIODS = ["weekly", "monthly", "quarterly", "yearly"]

class BudgetBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    limit_amount: float = Field(..., gt=0)
    category: str = Field(..., description="Budget category from predefined list")
    period_start: date
    period_end: date
    alert_threshold: float = Field(..., ge=0, le=100)
    
    @field_validator('category')
    @classmethod
    def validate_category(cls, v):
        if v not in BUDGET_CATEGORIES:
            raise ValueError(f'Category must be one of: {", ".join(BUDGET_CATEGORIES)}')
        return v

class BudgetCreate(BudgetBase):
    pass

class BudgetUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    limit_amount: Optional[float] = Field(None, gt=0)
    category: Optional[str] = Field(None, description="Budget category from predefined list")
    period_start: Optional[date] = None
    period_end: Optional[date] = None
    alert_threshold: Optional[float] = Field(None, ge=0, le=100)
    
    @field_validator('category')
    @classmethod
    def validate_category(cls, v):
        if v is not None and v not in BUDGET_CATEGORIES:
            raise ValueError(f'Category must be one of: {", ".join(BUDGET_CATEGORIES)}')
        return v

class BudgetResponse(BudgetBase):
    budget_id: int
    user_id: int
    created: datetime
    spent_amount: Optional[float] = Field(None, description="Calculated spent amount based on transactions")
    remaining_amount: Optional[float] = Field(None, description="Remaining budget amount")
    percentage_used: Optional[float] = Field(None, description="Percentage of budget used")
    status: Optional[str] = Field(None, description="Budget status: on_track, at_risk, or over_budget")
    days_remaining: Optional[int] = Field(None, description="Days remaining in budget period")
    daily_allowance: Optional[float] = Field(None, description="Daily spending allowance")
    alert_type: Optional[str] = Field(None, description="Alert type: info, warning, or danger")

    @field_validator('category', mode='before')
    @classmethod
    def migrate_old_category(cls, v):
        """Migrate old category names to new ones for backward compatibility"""
        # Mapping of old category names to new ones
        category_migration = {
            "Food": "Food & Dining",
            "Other": "Others",
        }
        return category_migration.get(v, v)

    class Config:
        from_attributes = True

class BudgetList(BaseModel):
    budgets: List[BudgetResponse]
    total: int
    skip: int
    limit: int
    has_more: bool

class BudgetSummary(BaseModel):
    total_budgets: int
    active_budgets: int
    total_budget_amount: float
    total_spent_amount: float
    total_remaining_amount: float
    budgets_over_budget: int
    budgets_at_risk: int
    average_utilization: float

class BudgetAlert(BaseModel):
    budget_id: int
    name: str
    category: str
    limit_amount: float
    spent_amount: float
    remaining_amount: float
    percentage_used: float
    status: str
    days_remaining: int
    daily_allowance: float
    alert_type: str

# ============ SMART ANALYSIS SCHEMAS ============

class SmartAnalysisRequest(BaseModel):
    view_mode: Literal["monthly", "yearly"]
    selected_date: date


class NeedsVsWantsInsightsRequest(BaseModel):
    view_mode: Literal["monthly", "yearly"]
    selected_date: date


class NeedsVsWantsInsightsResponse(BaseModel):
    generated_at: datetime
    period_label: str
    summary: str = Field(..., description="Brief summary of spending patterns for the period")
    localized_guidance: List[str] = Field(default_factory=list, description="Cultural and localized tips")
    spend_optimization: List[str] = Field(default_factory=list, description="Actionable optimization tips")
    model_usage: Optional[Dict[str, Any]] = None


class SmartAnalysisResponse(BaseModel):
    generated_at: datetime
    period_label: str
    summary_title: Optional[str] = None
    analysis_points: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    seasonal_signals: List[str] = Field(default_factory=list)
    savings_opportunities: List[str] = Field(default_factory=list)
    risk_alerts: List[str] = Field(default_factory=list)
    cultural_notes: List[str] = Field(default_factory=list)
    tone: Optional[str] = None
    ai_summary_markdown: Optional[str] = None
    model_usage: Optional[Dict[str, Any]] = None


# ============ CHAT SCHEMAS ============

class ChatMessageBase(BaseModel):
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata for structured data")

class ChatMessageCreate(ChatMessageBase):
    conversation_id: Optional[int] = Field(None, description="Conversation ID (auto-created if not provided)")

class ChatMessageResponse(ChatMessageBase):
    message_id: int
    conversation_id: int
    token_count: Optional[int] = None
    created_at: datetime

    class Config:
        from_attributes = True
        populate_by_name = True

class ChatConversationBase(BaseModel):
    title: Optional[str] = Field(None, description="Conversation title (auto-generated from first message)")

class ChatConversationCreate(ChatConversationBase):
    pass

class ChatConversationResponse(ChatConversationBase):
    conversation_id: int
    user_id: int
    created_at: datetime
    updated_at: datetime
    message_count: int = 0

    class Config:
        from_attributes = True

class ChatConversationListResponse(BaseModel):
    conversations: List[ChatConversationResponse]
    total: int

class ChatSendMessageRequest(BaseModel):
    message: str = Field(..., description="User message content")
    conversation_id: Optional[int] = Field(None, description="Conversation ID (creates new if not provided)")

class ChatSendMessageResponse(BaseModel):
    message: ChatMessageResponse
    assistant_response: ChatMessageResponse
    conversation: ChatConversationResponse
    actions_executed: Optional[List[Dict[str, Any]]] = Field(None, description="List of actions executed by AI")

class ContextSummaryBase(BaseModel):
    summary_type: str = Field(..., description="Type of summary: 'financial_snapshot', 'conversation_summary', etc.")
    data_snapshot_date: date = Field(..., description="Date when data was captured")

class ContextSummaryCreate(ContextSummaryBase):
    summary_content: str = Field(..., description="Summary content (JSON or text)")

class ContextSummaryResponse(ContextSummaryBase):
    summary_id: int
    user_id: int
    summary_content: str
    created_at: datetime
    expires_at: Optional[datetime] = None

    class Config:
        from_attributes = True

class UserEmbeddingCacheBase(BaseModel):
    entity_type: str = Field(..., description="Type of entity: 'transaction', 'goal', 'budget', etc.")
    entity_id: int = Field(..., description="ID of the entity")
    embedding_text: str = Field(..., description="Text that was embedded")

class UserEmbeddingCacheCreate(UserEmbeddingCacheBase):
    embedding_vector: Optional[List[float]] = Field(None, description="Vector embedding")
    metadata_json: Optional[Dict[str, Any]] = None

class UserEmbeddingCacheResponse(UserEmbeddingCacheBase):
    cache_id: int
    user_id: int
    embedding_vector: Optional[List[float]] = None
    metadata_json: Optional[Dict[str, Any]] = None
    created_at: datetime

    class Config:
        from_attributes = True

# ============ SUSPICIOUS TRANSACTIONS SCHEMAS ============
class TransactionForAnalysis(BaseModel):
    id: str = Field(..., description="Transaction ID")
    date: str = Field(..., description="Transaction date")
    amount: float = Field(..., description="Transaction amount")
    description: str = Field(..., description="Transaction description")
    category: str = Field(..., description="Transaction category")
    type: str = Field(..., description="Transaction type (income/expense/transfer)")

class SuspiciousTransactionResult(BaseModel):
    id: str = Field(..., description="Transaction ID")
    reason: str = Field(..., description="Brief reason for flagging")
    severity: Literal["high", "medium"] = Field(..., description="Severity level")
    details: str = Field(..., description="Detailed explanation")

class SuspiciousTransactionsRequest(BaseModel):
    transactions: List[TransactionForAnalysis] = Field(..., description="List of transactions to analyze")

class SuspiciousTransactionsResponse(BaseModel):
    suspicious_transactions: List[SuspiciousTransactionResult] = Field(..., description="List of suspicious transactions")
    analyzed_count: int = Field(..., description="Number of transactions analyzed")
    model_usage: Optional[Dict[str, Any]] = Field(None, description="AI model usage metadata")
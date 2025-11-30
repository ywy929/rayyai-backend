from sqlalchemy import Column, Integer, String, Float, Date, DateTime, Boolean, Text, ForeignKey, JSON, CheckConstraint, ARRAY
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from database import Base

class User(Base):
    __tablename__ = "user"
    
    user_id = Column(Integer, primary_key=True, index=True)
    first_name = Column(String, nullable=False)
    last_name = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False, index=True)
    password = Column(String, nullable=False) 
    dob = Column(Date, nullable=False)
    gender = Column(String, nullable=False)
    address = Column(String, nullable=True)
    is_deleted = Column(Boolean, nullable=False, default=False)  # Soft delete flag
    created = Column(DateTime, nullable=False, server_default=func.now())
    updated = Column(DateTime, nullable=False, server_default=func.now(), onupdate=func.now())
    
    # Relationships
    accounts = relationship("Account", back_populates="user")
    incomes = relationship("Income", back_populates="user")
    expenses = relationship("Expense", back_populates="user")
    transfers = relationship("Transfer", back_populates="user")
    statements = relationship("Statement", back_populates="user")
    ai_analyses = relationship("AIAnalysis", back_populates="user")
    user_credit_cards = relationship("UserCreditCard", back_populates="user")
    goals = relationship("Goal", back_populates="user")
    budgets = relationship("Budget", back_populates="user")
    conversations = relationship("ChatConversation", back_populates="user")
    context_summaries = relationship("ContextSummary", back_populates="user")
    embedding_caches = relationship("UserEmbeddingCache", back_populates="user")


class Account(Base):
    __tablename__ = "account"

    account_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("user.user_id"), nullable=False)
    account_no = Column(String, nullable=True)
    account_name = Column(String, nullable=False)
    account_type = Column(String, nullable=False)  # Standard type: savings, current, credit, ewallet, investment, cash
    account_subtype = Column(String, nullable=True)  # Specific variant: "Islamic Savings", "Visa Platinum", etc.
    account_balance = Column(Float, nullable=True)
    card_id = Column(Integer, ForeignKey("user_credit_card.card_id"), nullable=True)
    is_deleted = Column(Boolean, nullable=False, default=False)  # Soft delete flag

    # Relationships
    user = relationship("User", back_populates="accounts")
    card = relationship("UserCreditCard", foreign_keys=[card_id])
    incomes = relationship("Income", back_populates="account")
    expenses = relationship("Expense", back_populates="account")
    transfers = relationship("Transfer", back_populates="account")
    balance_snapshots = relationship("AccountBalanceSnapshot", back_populates="account")


class AccountBalanceSnapshot(Base):
    __tablename__ = "account_balance_snapshot"
    
    snapshot_id = Column(Integer, primary_key=True, index=True)
    account_id = Column(Integer, ForeignKey("account.account_id"), nullable=False)
    snapshot_date = Column(Date, nullable=False)
    closing_balance = Column(Float, nullable=False)
    is_deleted = Column(Boolean, nullable=False, default=False)  # Soft delete flag
    created = Column(DateTime, nullable=False, server_default=func.now())
    
    # Relationships
    account = relationship("Account", back_populates="balance_snapshots")


class Income(Base):
    __tablename__ = "income"
    
    income_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("user.user_id"), nullable=False)
    account_id = Column(Integer, ForeignKey("account.account_id"), nullable=False)
    statement_id = Column(Integer, ForeignKey("statement.statement_id"), nullable=True)
    amount = Column(Float, nullable=False)
    description = Column(String, nullable=True)
    category = Column(String, nullable=False)
    date_received = Column(Date, nullable=False)
    payer = Column(String, nullable=False)
    department = Column(String, nullable=True)
    project = Column(String, nullable=True)
    reference_no = Column(String, nullable=True)
    is_deleted = Column(Boolean, nullable=False, default=False)  # Soft delete flag
    created = Column(DateTime, nullable=False, server_default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="incomes")
    account = relationship("Account", back_populates="incomes")
    statement = relationship("Statement", back_populates="incomes")


class Expense(Base):
    __tablename__ = "expense"
    
    expense_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("user.user_id"), nullable=False)
    account_id = Column(Integer, ForeignKey("account.account_id"), nullable=False)
    statement_id = Column(Integer, ForeignKey("statement.statement_id"), nullable=True)
    amount = Column(Float, nullable=False)
    tax_amount = Column(Float, nullable=True, default=0.00)
    tax_deductible = Column(Boolean, nullable=True, default=False)
    is_reimbursable = Column(Boolean, nullable=True, default=False)
    description = Column(String, nullable=False)
    category = Column(String, nullable=False)  # "Food & Dining", "Transportation", etc.
    expense_type = Column(String, nullable=True)  # "needs" or "wants"
    date_spent = Column(Date, nullable=False)
    seller = Column(String, nullable=False)
    location = Column(String, nullable=True)
    reference_no = Column(String, nullable=True)
    card_id = Column(Integer, ForeignKey("user_credit_card.card_id"), nullable=True)  # Payment to clear card debt
    is_deleted = Column(Boolean, nullable=False, default=False)  # Soft delete flag
    created = Column(DateTime, nullable=False, server_default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="expenses")
    account = relationship("Account", back_populates="expenses")
    statement = relationship("Statement", back_populates="expenses")
    card = relationship("UserCreditCard", foreign_keys=[card_id])
    
    __table_args__ = (
        CheckConstraint(
            "expense_type IN ('needs', 'wants') OR expense_type IS NULL",
            name="check_expense_type"
        ),
    )


class Transfer(Base):
    __tablename__ = "transfer"
    
    transfer_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("user.user_id"), nullable=False)
    account_id = Column(Integer, ForeignKey("account.account_id"), nullable=False)
    statement_id = Column(Integer, ForeignKey("statement.statement_id"), nullable=True)
    amount = Column(Float, nullable=False)
    description = Column(String, nullable=False)
    category = Column(String, nullable=False, default="Transfer")  # Usually "Transfer" but can be more specific
    transfer_type = Column(String, nullable=False)  # "intra_person" (to own account) or "inter_person" (to another person)
    date_transferred = Column(Date, nullable=False)
    recipient_account_name = Column(String, nullable=True)  # Name of recipient account if available
    recipient_account_no = Column(String, nullable=True)  # Account number if available
    reference_no = Column(String, nullable=True)
    is_deleted = Column(Boolean, nullable=False, default=False)  # Soft delete flag
    created = Column(DateTime, nullable=False, server_default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="transfers")
    account = relationship("Account", back_populates="transfers")
    statement = relationship("Statement", back_populates="transfers")
    
    __table_args__ = (
        CheckConstraint(
            "transfer_type IN ('intra_person', 'inter_person')",
            name="check_transfer_type"
        ),
    )


class Statement(Base):
    __tablename__ = "statement"

    statement_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("user.user_id"), nullable=False)
    statement_type = Column(String, nullable=False)  # CTOS, CCRIS, bank, credit_card, ewallet, receipt
    statement_url = Column(String, nullable=False)
    file_hash = Column(String, nullable=True, index=True)  # SHA-256 hash for duplicate detection
    display_name = Column(String, nullable=True)  # User-friendly display name for statements
    period_start = Column(Date, nullable=True)
    period_end = Column(Date, nullable=True)
    credit_score = Column(Integer, nullable=True)
    score_text = Column(String, nullable=True)
    is_deleted = Column(Boolean, nullable=False, default=False)
    date_uploaded = Column(DateTime, nullable=False, server_default=func.now())

    # AI extraction caching and processing status
    extracted_data = Column(JSON, nullable=True)  # Cache Gemini extraction results to avoid re-processing
    processing_status = Column(String, nullable=False, default='pending')  # pending, extracting, extracted, imported, failed
    processing_error = Column(Text, nullable=True)  # Store error message if processing fails
    last_processed = Column(DateTime, nullable=True)  # Timestamp of last extraction attempt

    # Relationships
    user = relationship("User", back_populates="statements")
    incomes = relationship("Income", back_populates="statement")
    expenses = relationship("Expense", back_populates="statement")
    transfers = relationship("Transfer", back_populates="statement")
    ai_analyses = relationship("AIAnalysis", back_populates="statement")

    __table_args__ = (
        CheckConstraint(
            "processing_status IN ('pending', 'extracting', 'extracted', 'imported', 'failed')",
            name="check_processing_status"
        ),
    )


# ============ CTOS DETAILED DATA MODELS ============

class CTOSPersonalInfo(Base):
    """Personal identification details extracted from CTOS report"""
    __tablename__ = "ctos_personal_info"
    
    personal_info_id = Column(Integer, primary_key=True, index=True)
    statement_id = Column(Integer, ForeignKey("statement.statement_id"), nullable=False, unique=True)
    full_name = Column(String, nullable=True)
    ic_nric = Column(String, nullable=True)  # Malaysian IC/MyKad number
    date_of_birth = Column(Date, nullable=True)
    nationality = Column(String, nullable=True, default="Malaysia")
    address_line1 = Column(String, nullable=True)
    address_line2 = Column(String, nullable=True)
    source = Column(String, nullable=True, default="CCRIS")  # Source of data (CCRIS, etc.)
    is_deleted = Column(Boolean, nullable=False, default=False)
    created = Column(DateTime, nullable=False, server_default=func.now())
    updated = Column(DateTime, nullable=False, server_default=func.now(), onupdate=func.now())
    
    # Relationships
    statement = relationship("Statement", foreign_keys=[statement_id])


class CTOSScore(Base):
    """CTOS Score and risk factors"""
    __tablename__ = "ctos_score"
    
    score_id = Column(Integer, primary_key=True, index=True)
    statement_id = Column(Integer, ForeignKey("statement.statement_id"), nullable=False, unique=True)
    ctos_score = Column(Integer, nullable=True)  # Score between 300-850
    score_text = Column(String, nullable=True)  # "Excellent", "Very Good", "Good", "Fair", "Poor"
    risk_factors = Column(JSON, nullable=True)  # Array of risk factors like ["Too many recent credit applications", "High loan utilisation"]
    is_deleted = Column(Boolean, nullable=False, default=False)
    created = Column(DateTime, nullable=False, server_default=func.now())
    updated = Column(DateTime, nullable=False, server_default=func.now(), onupdate=func.now())
    
    # Relationships
    statement = relationship("Statement", foreign_keys=[statement_id])


class CTOSLegalRecords(Base):
    """Bankruptcy, legal and special attention records"""
    __tablename__ = "ctos_legal_records"
    
    legal_records_id = Column(Integer, primary_key=True, index=True)
    statement_id = Column(Integer, ForeignKey("statement.statement_id"), nullable=False, unique=True)
    is_bankrupt = Column(Boolean, nullable=False, default=False)
    legal_records_personal_24m = Column(Integer, nullable=False, default=0)  # Personal legal records in last 24 months
    legal_records_non_personal_24m = Column(Integer, nullable=False, default=0)  # Non-personal legal records in last 24 months
    has_special_attention_accounts = Column(Boolean, nullable=False, default=False)
    has_trade_referee_listing = Column(Boolean, nullable=False, default=False)
    is_deleted = Column(Boolean, nullable=False, default=False)
    created = Column(DateTime, nullable=False, server_default=func.now())
    updated = Column(DateTime, nullable=False, server_default=func.now(), onupdate=func.now())
    
    # Relationships
    statement = relationship("Statement", foreign_keys=[statement_id])


class CTOSCreditFacilitySummary(Base):
    """Credit facility summary (CCRIS overview)"""
    __tablename__ = "ctos_credit_facility_summary"
    
    summary_id = Column(Integer, primary_key=True, index=True)
    statement_id = Column(Integer, ForeignKey("statement.statement_id"), nullable=False, unique=True)
    total_outstanding_balance = Column(Float, nullable=True)  # Total current liabilities
    total_credit_limit = Column(Float, nullable=True)
    credit_applications_12m_total = Column(Integer, nullable=False, default=0)
    credit_applications_12m_approved = Column(Integer, nullable=False, default=0)
    credit_applications_12m_pending = Column(Integer, nullable=False, default=0)
    is_deleted = Column(Boolean, nullable=False, default=False)
    created = Column(DateTime, nullable=False, server_default=func.now())
    updated = Column(DateTime, nullable=False, server_default=func.now(), onupdate=func.now())
    
    # Relationships
    statement = relationship("Statement", foreign_keys=[statement_id])


class CTOSCreditFacility(Base):
    """Individual credit facility details from CCRIS"""
    __tablename__ = "ctos_credit_facility"
    
    facility_id = Column(Integer, primary_key=True, index=True)
    statement_id = Column(Integer, ForeignKey("statement.statement_id"), nullable=False)
    facility_number = Column(Integer, nullable=True)  # Facility #1, #2, etc.
    facility_type = Column(String, nullable=True)  # CRDTCARD, OTLNFNCE, PCPASCAR, PELNFNCE, etc.
    facility_name = Column(String, nullable=True)  # "Credit Card", "Term Financing", "Car Loan", etc.
    bank_name = Column(String, nullable=True)  # e.g., "Maybank Islamic"
    account_number = Column(String, nullable=True)  # Account number associated with the facility
    account_name = Column(String, nullable=True)  # Account name/holder name for the facility
    credit_limit = Column(Float, nullable=True)
    outstanding_balance = Column(Float, nullable=True)
    collateral_type = Column(String, nullable=True)  # "Clean (00)", "Unit Trust (23)", "Motor Vehicle (JPJ) (30)", etc.
    collateral_code = Column(String, nullable=True)  # "00", "23", "30", etc.
    conduct_12m = Column(JSON, nullable=True)  # Array of 12 months of payment conduct (0 = good, 1+ = missed payments)
    is_deleted = Column(Boolean, nullable=False, default=False)
    created = Column(DateTime, nullable=False, server_default=func.now())
    updated = Column(DateTime, nullable=False, server_default=func.now(), onupdate=func.now())
    
    # Relationships
    statement = relationship("Statement", foreign_keys=[statement_id])


class CTOSCreditUtilisation(Base):
    """Credit utilisation metrics derived from CCRIS data"""
    __tablename__ = "ctos_credit_utilisation"
    
    utilisation_id = Column(Integer, primary_key=True, index=True)
    statement_id = Column(Integer, ForeignKey("statement.statement_id"), nullable=False, unique=True)
    earliest_known_facility_date = Column(Date, nullable=True)
    total_outstanding = Column(Float, nullable=True)
    outstanding_percentage_of_limit = Column(Float, nullable=True)  # e.g., 90.0 for 90%
    number_of_unsecured_facilities = Column(Integer, nullable=False, default=0)
    number_of_secured_facilities = Column(Integer, nullable=False, default=0)
    avg_utilisation_credit_card_6m = Column(Float, nullable=True)  # Average credit card utilisation over last 6 months
    avg_utilisation_revolving_6m = Column(Float, nullable=True)  # Average revolving credit utilisation over last 6 months
    is_deleted = Column(Boolean, nullable=False, default=False)
    created = Column(DateTime, nullable=False, server_default=func.now())
    updated = Column(DateTime, nullable=False, server_default=func.now(), onupdate=func.now())
    
    # Relationships
    statement = relationship("Statement", foreign_keys=[statement_id])


class CTOSLoanApplication(Base):
    """Loan/credit application details"""
    __tablename__ = "ctos_loan_application"
    
    application_id = Column(Integer, primary_key=True, index=True)
    statement_id = Column(Integer, ForeignKey("statement.statement_id"), nullable=False)
    application_date = Column(Date, nullable=True)
    application_type = Column(String, nullable=True)  # "credit_card", "personal_loan", etc.
    amount = Column(Float, nullable=True)  # Application amount
    status = Column(String, nullable=True)  # "Approved", "Pending", "Rejected"
    lender_name = Column(String, nullable=True)
    is_deleted = Column(Boolean, nullable=False, default=False)
    created = Column(DateTime, nullable=False, server_default=func.now())
    updated = Column(DateTime, nullable=False, server_default=func.now(), onupdate=func.now())
    
    # Relationships
    statement = relationship("Statement", foreign_keys=[statement_id])


class CTOSEmploymentInfo(Base):
    """Employment and business information"""
    __tablename__ = "ctos_employment_info"
    
    employment_info_id = Column(Integer, primary_key=True, index=True)
    statement_id = Column(Integer, ForeignKey("statement.statement_id"), nullable=False, unique=True)
    has_directorships = Column(Boolean, nullable=False, default=False)
    directorships_count = Column(Integer, nullable=False, default=0)
    has_business_interests = Column(Boolean, nullable=False, default=False)
    business_interests_count = Column(Integer, nullable=False, default=0)
    is_deleted = Column(Boolean, nullable=False, default=False)
    created = Column(DateTime, nullable=False, server_default=func.now())
    updated = Column(DateTime, nullable=False, server_default=func.now(), onupdate=func.now())
    
    # Relationships
    statement = relationship("Statement", foreign_keys=[statement_id])


class CTOSPTPTNStatus(Base):
    """PTPTN (education loan) status"""
    __tablename__ = "ctos_ptptn_status"
    
    ptptn_status_id = Column(Integer, primary_key=True, index=True)
    statement_id = Column(Integer, ForeignKey("statement.statement_id"), nullable=False, unique=True)
    number_of_ptptn_loans = Column(Integer, nullable=False, default=0)
    local_lenders_count = Column(Integer, nullable=False, default=0)
    foreign_lenders_count = Column(Integer, nullable=False, default=0)
    is_deleted = Column(Boolean, nullable=False, default=False)
    created = Column(DateTime, nullable=False, server_default=func.now())
    updated = Column(DateTime, nullable=False, server_default=func.now(), onupdate=func.now())
    
    # Relationships
    statement = relationship("Statement", foreign_keys=[statement_id])


class AIAnalysis(Base):
    __tablename__ = "ai_analysis"
    
    analysis_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("user.user_id"), nullable=False)
    statement_id = Column(Integer, ForeignKey("statement.statement_id"), nullable=True)
    analysis_content = Column(JSON, nullable=False)
    analysis_type = Column(String, nullable=False)
    is_deleted = Column(Boolean, nullable=False, default=False)
    created = Column(DateTime, nullable=False, server_default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="ai_analyses")
    statement = relationship("Statement", back_populates="ai_analyses")


class UserCreditCard(Base):
    __tablename__ = "user_credit_card"
    
    card_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("user.user_id"), nullable=False)
    card_number = Column(String, nullable=False)
    card_name = Column(String, nullable=False)
    bank_name = Column(String, nullable=False)
    card_brand = Column(String, nullable=False)
    expiry_month = Column(Integer, nullable=False)
    expiry_year = Column(Integer, nullable=False)
    credit_limit = Column(Float, nullable=False)
    annual_fee = Column(Float, nullable=False, default=0.00)
    next_payment_amount = Column(Float, nullable=True)
    next_payment_date = Column(Date, nullable=True)
    benefits = Column(JSON, nullable=False, default={})
    current_balance = Column(Float, nullable=False, default=0.00)
    is_deleted = Column(Boolean, nullable=False, default=False)
    created = Column(DateTime, nullable=False, server_default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="user_credit_cards")
    accounts = relationship("Account", foreign_keys=[Account.card_id], overlaps="card")
    debt_payments = relationship("Expense", foreign_keys=[Expense.card_id], overlaps="card")
    terms_history = relationship("UserCreditCardTermsHistory", back_populates="card")

class UserCreditCardTermsHistory(Base):
    __tablename__ = "user_credit_card_terms_history"
    
    term_history_id = Column(Integer, primary_key=True, index=True)
    card_id = Column(Integer, ForeignKey("user_credit_card.card_id"), nullable=False)
    effective_date = Column(Date, nullable=False)
    interest_rate = Column(Float, nullable=False)
    minimum_payment = Column(Float, nullable=True)
    is_deleted = Column(Boolean, nullable=False, default=False)  # Soft delete flag
    created = Column(DateTime, nullable=False, server_default=func.now())
    
    # Relationships
    card = relationship("UserCreditCard", back_populates="terms_history")


class MarketCreditCard(Base):
    __tablename__ = "market_credit_card"
    
    card_id = Column(Integer, primary_key=True, index=True)
    card_name = Column(String, nullable=False)
    bank_name = Column(String, nullable=False)
    card_brand = Column(String, nullable=False)
    annual_fee = Column(Float, nullable=False, default=0.00)
    eligibility_criteria = Column(JSON, nullable=False, default={})
    benefits = Column(JSON, nullable=False, default={})
    is_deleted = Column(Boolean, nullable=False, default=False)  # Soft delete flag
    created = Column(DateTime, nullable=False, server_default=func.now())
    updated = Column(DateTime, nullable=False, server_default=func.now(), onupdate=func.now())



class Goal(Base):
    __tablename__ = "goal"
    
    goal_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("user.user_id"), nullable=False)
    goal_name = Column(String, nullable=False)
    description = Column(String, nullable=False)
    category = Column(String, nullable=False)
    priority = Column(String, nullable=False)
    target_amount = Column(Float, nullable=False)
    current_amount = Column(Float, nullable=False, default=0.00)
    target_date = Column(Date, nullable=True)
    is_deleted = Column(Boolean, nullable=False, default=False)  # Soft delete flag
    created_at = Column(DateTime, nullable=False, server_default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="goals")


class Budget(Base):
    __tablename__ = "budget"
    
    budget_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("user.user_id"), nullable=False)
    name = Column(String, nullable=False)
    limit_amount = Column(Float, nullable=False)
    category = Column(String, nullable=False)
    period_start = Column(Date, nullable=False)
    period_end = Column(Date, nullable=False)
    alert_threshold = Column(Float, nullable=False, default=0.8)
    is_deleted = Column(Boolean, nullable=False, default=False)  # Soft delete flag
    created = Column(DateTime, nullable=False, server_default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="budgets")


class ChatConversation(Base):
    __tablename__ = "chat_conversation"
    
    conversation_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("user.user_id"), nullable=False)
    title = Column(String, nullable=True)  # Auto-generated from first message
    is_deleted = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime, nullable=False, server_default=func.now())
    updated_at = Column(DateTime, nullable=False, server_default=func.now(), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="conversations")
    messages = relationship("ChatMessage", back_populates="conversation", order_by="ChatMessage.created_at")


class ChatMessage(Base):
    __tablename__ = "chat_message"
    
    message_id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("chat_conversation.conversation_id"), nullable=False)
    role = Column(String, nullable=False)  # "user" or "assistant"
    content = Column(Text, nullable=False)
    metadata_json = Column(JSON, nullable=True)  # Store structured data, action results, etc.
    token_count = Column(Integer, nullable=True)  # Token count for this message
    created_at = Column(DateTime, nullable=False, server_default=func.now())
    
    # Relationships
    conversation = relationship("ChatConversation", back_populates="messages")


class ContextSummary(Base):
    __tablename__ = "context_summary"
    
    summary_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("user.user_id"), nullable=False)
    summary_content = Column(Text, nullable=False)  # JSON or text summary
    summary_type = Column(String, nullable=False)  # "financial_snapshot", "conversation_summary", etc.
    data_snapshot_date = Column(Date, nullable=False)  # Date when data was captured
    created_at = Column(DateTime, nullable=False, server_default=func.now())
    expires_at = Column(DateTime, nullable=True)  # When summary expires
    
    # Relationships
    user = relationship("User", back_populates="context_summaries")


class UserEmbeddingCache(Base):
    __tablename__ = "user_embedding_cache"
    
    cache_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("user.user_id"), nullable=False)
    entity_type = Column(String, nullable=False)  # "transaction", "goal", "budget", etc.
    entity_id = Column(Integer, nullable=False)  # ID of the entity
    embedding_vector = Column(ARRAY(Float), nullable=True)  # Vector embedding (if using pgvector)
    embedding_text = Column(Text, nullable=False)  # Text that was embedded
    metadata_json = Column(JSON, nullable=True)  # Additional metadata
    created_at = Column(DateTime, nullable=False, server_default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="embedding_caches")
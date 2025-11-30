"""
Chat API Router
Endpoints for AI chat functionality with RAG
"""
from fastapi import APIRouter, Depends, HTTPException, status, Query, UploadFile, File, Form
from sqlalchemy.orm import Session
from typing import List, Optional
import models
import schemas
from database import get_db
from routers.utils import get_current_user, upload_file_to_s3
from services.gemini_service import GeminiService
from services.rag_service import RAGService
from services.pii_masking import PIIMaskingService
from services.context_summarizer import ContextSummarizer
from services.conversation_manager import ConversationManager
from services.action_executor import ActionExecutor
from services.search_service import SearchService
from routers.statements import detect_statement_type
from services.processors import process_statement_pdf
from sqlalchemy import func
import hashlib
import os
import json
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize Gemini service (singleton pattern)
_gemini_service: Optional[GeminiService] = None

def get_gemini_service() -> GeminiService:
    """Get or create Gemini service instance."""
    global _gemini_service
    if _gemini_service is None:
        _gemini_service = GeminiService()
    return _gemini_service

@router.post("/conversations", response_model=schemas.ChatConversationResponse)
async def create_conversation(
    conversation_data: schemas.ChatConversationCreate,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new conversation."""
    gemini_service = get_gemini_service()
    context_summarizer = ContextSummarizer(db, gemini_service)
    conversation_manager = ConversationManager(db, gemini_service, context_summarizer)
    
    conversation = conversation_manager.create_conversation(
        user_id=current_user.user_id,
        title=conversation_data.title
    )
    
    return conversation

@router.get("/conversations", response_model=schemas.ChatConversationListResponse)
async def list_conversations(
    skip: int = 0,
    limit: int = 50,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get list of user's conversations."""
    gemini_service = get_gemini_service()
    context_summarizer = ContextSummarizer(db, gemini_service)
    conversation_manager = ConversationManager(db, gemini_service, context_summarizer)
    
    conversations = conversation_manager.get_user_conversations(
        user_id=current_user.user_id,
        limit=limit,
        skip=skip
    )
    
    # Add message count to each conversation
    conversation_responses = []
    for conv in conversations:
        messages = conversation_manager.get_conversation_messages(conv.conversation_id, current_user.user_id)
        conv_dict = {
            **conv.__dict__,
            "message_count": len(messages)
        }
        conversation_responses.append(schemas.ChatConversationResponse(**conv_dict))
    
    return {
        "conversations": conversation_responses,
        "total": len(conversation_responses)
    }

@router.get("/conversations/{conversation_id}", response_model=schemas.ChatConversationResponse)
async def get_conversation(
    conversation_id: int,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get conversation details."""
    gemini_service = get_gemini_service()
    context_summarizer = ContextSummarizer(db, gemini_service)
    conversation_manager = ConversationManager(db, gemini_service, context_summarizer)
    
    conversation = conversation_manager.get_conversation(conversation_id, current_user.user_id)
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    
    messages = conversation_manager.get_conversation_messages(conversation_id, current_user.user_id)
    conv_dict = {
        **conversation.__dict__,
        "message_count": len(messages)
    }
    
    return schemas.ChatConversationResponse(**conv_dict)

@router.patch("/conversations/{conversation_id}")
async def update_conversation(
    conversation_id: int,
    title: str = Query(..., description="New conversation title"),
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update conversation title (rename)."""
    gemini_service = get_gemini_service()
    context_summarizer = ContextSummarizer(db, gemini_service)
    conversation_manager = ConversationManager(db, gemini_service, context_summarizer)
    
    conversation = conversation_manager.get_conversation(conversation_id, current_user.user_id)
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    
    conversation.title = title
    db.commit()
    db.refresh(conversation)
    
    messages = conversation_manager.get_conversation_messages(conversation_id, current_user.user_id)
    conv_dict = {
        **conversation.__dict__,
        "message_count": len(messages)
    }
    
    return schemas.ChatConversationResponse(**conv_dict)

@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: int,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete a conversation (soft delete)."""
    gemini_service = get_gemini_service()
    context_summarizer = ContextSummarizer(db, gemini_service)
    conversation_manager = ConversationManager(db, gemini_service, context_summarizer)
    
    deleted = conversation_manager.delete_conversation(conversation_id, current_user.user_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    
    return {"message": "Conversation deleted successfully"}

@router.get("/search/messages") # currently not applied
async def search_messages(
    q: str,
    conversation_id: Optional[int] = None,
    role: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Full-text search over chat messages."""
    ss = SearchService(db)
    results = ss.search_messages(
        user_id=current_user.user_id,
        query=q,
        conversation_id=conversation_id,
        role=role,
        start_iso=start,
        end_iso=end,
        limit=limit,
        offset=offset,
    )
    return {"results": results, "count": len(results)}

@router.get("/search/conversations") # currently not applied
async def search_conversations(
    q: str = "",
    limit: int = 20,
    offset: int = 0,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    ss = SearchService(db)
    results = ss.search_conversations(
        user_id=current_user.user_id,
        query=q,
        limit=limit,
        offset=offset,
    )
    return {"results": results, "count": len(results)}

@router.get("/export/{conversation_id}") # currently not applied 
async def export_conversation(
    conversation_id: int,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    ss = SearchService(db)
    data = ss.export_conversation(current_user.user_id, conversation_id)
    return {"conversation_id": conversation_id, "messages": data}

@router.patch("/messages/{message_id}") # currently not applied 
async def edit_message(
    message_id: int,
    content: str,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Verify ownership via conversation
    msg = db.query(models.ChatMessage).filter(models.ChatMessage.message_id == message_id).first()
    if not msg:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Message not found")
    conv = db.query(models.ChatConversation).filter(models.ChatConversation.conversation_id == msg.conversation_id).first()
    if not conv or conv.user_id != current_user.user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not allowed")
    msg.content = content
    db.commit()
    return {"message": "Updated"}

@router.delete("/messages/{message_id}") # currently not applied 
async def delete_message(
    message_id: int,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    msg = db.query(models.ChatMessage).filter(models.ChatMessage.message_id == message_id).first()
    if not msg:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Message not found")
    conv = db.query(models.ChatConversation).filter(models.ChatConversation.conversation_id == msg.conversation_id).first()
    if not conv or conv.user_id != current_user.user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not allowed")
    db.delete(msg)
    db.commit()
    return {"message": "Deleted"}

@router.get("/conversations/{conversation_id}/messages", response_model=List[schemas.ChatMessageResponse])
async def get_messages(
    conversation_id: int,
    limit: Optional[int] = None,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get messages for a conversation."""
    gemini_service = get_gemini_service()
    context_summarizer = ContextSummarizer(db, gemini_service)
    conversation_manager = ConversationManager(db, gemini_service, context_summarizer)
    
    messages = conversation_manager.get_conversation_messages(
        conversation_id,
        current_user.user_id,
        limit=limit
    )
    
    # Convert to response format with metadata field
    response_messages = [
        schemas.ChatMessageResponse(
            **{**msg.__dict__, "metadata": msg.metadata_json}
        )
        for msg in messages
    ]
    
    return response_messages


@router.post("/conversations/{conversation_id}/messages", response_model=schemas.ChatSendMessageResponse)
async def send_message(
    conversation_id: int,
    message: str = Form(...),
    files: List[UploadFile] = File(default=[]),
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Send a message in a specific conversation. Supports file uploads."""
    logger.info(f"Received message in conversation {conversation_id}, files: {len(files) if files else 0}")
    if files:
        for f in files:
            logger.info(f"  - File: {f.filename}, size: {f.size if hasattr(f, 'size') else 'unknown'}")
    request = schemas.ChatSendMessageRequest(message=message)
    return await _process_chat_message(conversation_id, request, current_user, db, files=files)

@router.post("/messages", response_model=schemas.ChatSendMessageResponse)
async def send_message_simple(
    message: str = Form(...),
    files: List[UploadFile] = File(default=[]),
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Send a message (creates conversation if needed). Supports file uploads."""
    logger.info(f"Received message (new conversation), files: {len(files) if files else 0}")
    if files:
        for f in files:
            logger.info(f"  - File: {f.filename}, size: {f.size if hasattr(f, 'size') else 'unknown'}")
    request = schemas.ChatSendMessageRequest(message=message)
    return await _process_chat_message(None, request, current_user, db, files=files)

async def _process_uploaded_file(
    file: UploadFile,
    user_id: int,
    db: Session
) -> dict:
    """
    Process an uploaded file using preview-first workflow:
    1. Upload to S3
    2. Create statement record
    3. Extract transactions with AI (cached)
    4. Return preview info WITHOUT auto-saving transactions

    AI will then ask user to confirm before importing transactions.
    Returns a dict with statement info and extraction preview.
    """
    try:
        # Validate file type
        allowed_extensions = {".pdf", ".jpg", ".jpeg", ".png"}
        file_ext = os.path.splitext(file.filename)[1].lower()

        if file_ext not in allowed_extensions:
            return {
                "success": False,
                "error": f"File type {file_ext} not allowed. Allowed: {', '.join(allowed_extensions)}",
                "filename": file.filename
            }

        # Validate file size (max 10MB)
        file.file.seek(0, 2)  # Seek to end
        file_size = file.file.tell()
        file.file.seek(0)  # Reset to beginning

        max_size = 10 * 1024 * 1024  # 10MB
        if file_size > max_size:
            return {
                "success": False,
                "error": f"File too large. Max size: {max_size / (1024*1024)}MB",
                "filename": file.filename
            }

        # Auto-detect statement type
        statement_type = detect_statement_type(file.filename)

        # Read file contents first (before S3 upload consumes the stream)
        file.file.seek(0)
        file_contents = await file.read()

        # Calculate file hash from contents
        file_hash = hashlib.sha256(file_contents).hexdigest()

        # Upload to S3 using the bytes we already read
        # We need to create a temporary file-like object for upload_file_to_s3
        from io import BytesIO
        import tempfile

        # Create a new UploadFile-like object from the bytes
        file_for_upload = UploadFile(
            filename=file.filename,
            file=BytesIO(file_contents),
            size=len(file_contents),
            headers=file.headers
        )

        statement_url, _ = await upload_file_to_s3(
            file=file_for_upload, user_id=user_id, folder="statements"
        )

        # Check for duplicate (skip for now, allow duplicates in chat)
        # Create database record
        db_statement = models.Statement(
            user_id=user_id,
            statement_type=statement_type,
            statement_url=statement_url,
            file_hash=file_hash,
            display_name=file.filename,
            period_start=None,
            period_end=None,
            is_deleted=False,
            processing_status='pending'
        )
        db.add(db_statement)
        db.commit()
        db.refresh(db_statement)

        # Extract transactions with AI for PREVIEW ONLY (don't save to database yet)
        db_statement.processing_status = 'extracting'
        db.commit()

        try:
            result = process_statement_pdf(file_contents)

            if not result.get('success'):
                db_statement.processing_status = 'failed'
                db_statement.processing_error = "Failed to extract transactions from statement"
                db_statement.last_processed = datetime.now(timezone.utc)
                db.commit()
                return {
                    "success": False,
                    "error": "Failed to extract transactions from statement",
                    "filename": file.filename,
                    "statement_id": db_statement.statement_id
                }

            # Cache extraction result (for fast preview later)
            db_statement.extracted_data = result
            db_statement.processing_status = 'extracted'  # ✅ Extracted but NOT imported yet
            db_statement.processing_error = None
            db_statement.last_processed = datetime.now(timezone.utc)

            # Update period dates
            if result.get('statement_period'):
                period = result['statement_period']
                if period.get('start_date'):
                    db_statement.period_start = datetime.strptime(period['start_date'], '%Y-%m-%d').date()
                if period.get('end_date'):
                    db_statement.period_end = datetime.strptime(period['end_date'], '%Y-%m-%d').date()

            db.commit()

            # ✅ STOP HERE - Return preview info without creating transactions
            # AI will show this to user and ask for confirmation
            transaction_count = len(result.get('transactions', []))
            credit_count = len([t for t in result.get('transactions', []) if t.get('type') == 'credit'])
            debit_count = len([t for t in result.get('transactions', []) if t.get('type') == 'debit'])

            return {
                "success": True,
                "statement_id": db_statement.statement_id,
                "filename": file.filename,
                "statement_type": statement_type,
                "preview_mode": True,  # ✅ Indicates this is a preview, not imported yet
                "transactions_count": transaction_count,
                "credit_count": credit_count,
                "debit_count": debit_count,
                "period_start": db_statement.period_start.isoformat() if db_statement.period_start else None,
                "period_end": db_statement.period_end.isoformat() if db_statement.period_end else None,
                "account_info": result.get('account_info'),
                "opening_balance": result.get('opening_balance'),
                "closing_balance": result.get('closing_balance'),
                "processing_status": db_statement.processing_status,
                "message": "Statement extracted successfully. Transactions ready for review."
            }

        except Exception as e:
            logger.error(f"Error processing statement {db_statement.statement_id}: {e}", exc_info=True)
            db_statement.processing_status = 'failed'
            db_statement.processing_error = str(e)
            db_statement.last_processed = datetime.now(timezone.utc)
            db.commit()
            return {
                "success": False,
                "error": f"Processing failed: {str(e)}",
                "filename": file.filename,
                "statement_id": db_statement.statement_id
            }

    except Exception as e:
        logger.error(f"Error uploading file {file.filename}: {e}", exc_info=True)
        return {
            "success": False,
            "error": f"Upload failed: {str(e)}",
            "filename": file.filename
        }

async def _process_chat_message(
    conversation_id: Optional[int],
    request: schemas.ChatSendMessageRequest,
    current_user: models.User,
    db: Session,
    files: Optional[List[UploadFile]] = None
) -> schemas.ChatSendMessageResponse:
    """
    Process a chat message and generate AI response.
    Shared logic for both send_message endpoints.
    """
    gemini_service = get_gemini_service()
    context_summarizer = ContextSummarizer(db, gemini_service)
    conversation_manager = ConversationManager(db, gemini_service, context_summarizer)
    rag_service = RAGService(db)
    pii_masker = PIIMaskingService(
        user_first_name=current_user.first_name,
        user_last_name=current_user.last_name
    )
    action_executor = ActionExecutor(db, current_user.user_id)
    
    # Process uploaded files first (if any)
    processed_files = []
    if files and len(files) > 0:
        logger.info(f"Processing {len(files)} uploaded file(s)")
        for file in files:
            if file and file.filename:  # Only process if file has a name
                logger.info(f"Processing file: {file.filename}")
                result = await _process_uploaded_file(file, current_user.user_id, db)
                processed_files.append(result)
                logger.info(f"File processing result: success={result.get('success')}, filename={result.get('filename')}")
            else:
                logger.warning(f"Skipping file without filename: {file}")
    else:
        logger.info("No files to process")
    
    # Use conversation_id from path or request
    actual_conversation_id = conversation_id or request.conversation_id
    
    # Get or create conversation
    conversation = None
    if actual_conversation_id:
        conversation = conversation_manager.get_conversation(actual_conversation_id, current_user.user_id)
    
    if not conversation:
        # Create new conversation if not found
        conversation = conversation_manager.create_conversation(
            current_user.user_id,
            title=request.message[:50] if len(request.message) > 50 else request.message
        )
    
    # Build message content with file references
    message_content = request.message
    if processed_files:
        file_summaries = []
        for file_result in processed_files:
            if file_result.get('success'):
                summary = f"Uploaded {file_result['filename']}"
                if file_result.get('transactions_count', 0) > 0:
                    summary += f" - {file_result['transactions_count']} transactions extracted"
                if file_result.get('period_start') and file_result.get('period_end'):
                    summary += f" (Period: {file_result['period_start']} to {file_result['period_end']})"
                file_summaries.append(summary)
            else:
                file_summaries.append(f"Failed to process {file_result.get('filename', 'file')}: {file_result.get('error', 'Unknown error')}")
        
        if file_summaries:
            message_content += "\n\n[Files processed: " + "; ".join(file_summaries) + "]"
    
    # Save user message
    user_message = conversation_manager.add_message(
        conversation_id=conversation.conversation_id,
        role="user",
        content=message_content
    )
    
    try:
        # Prepare conversation context
        conv_context = await conversation_manager.prepare_conversation_context(
            conversation.conversation_id,
            current_user.user_id
        )
        
        # Get financial context
        financial_data = rag_service.get_financial_summary(current_user.user_id)
        
        # Get or generate cached summary
        summary_obj = await context_summarizer.get_or_generate_summary(
            current_user.user_id,
            force_refresh=False
        )
        
        # Format context for LLM
        financial_context_text = rag_service.format_context_for_llm(financial_data)
        
        # Mask PII from financial context
        masked_context = pii_masker.mask_financial_context(financial_data)
        masked_context_text = rag_service.format_context_for_llm(masked_context)
        
        # Build system instruction with Markdown formatting guidance
        system_instruction = """You are RayyAI, a professional, trustworthy, and knowledgeable financial assistant.
You help users manage their finances, analyze spending patterns, create budgets, track goals, and make informed financial decisions.

Personality Guidelines:
- Be professional, clear, and trustworthy in all communications
- Provide accurate, actionable financial advice with sound reasoning
- Use a respectful and supportive tone when discussing financial matters
- Remain objective and fact-based in your analysis
- Do not use emojis or emoticons in your responses
- Acknowledge user progress with professional encouragement
- Present financial concerns or issues in a constructive, solution-focused manner

Response format requirements (CRITICAL - must follow exactly):
- Always respond in GitHub‑flavored Markdown format.
- Start with a clear title line: `# <Concise Title>`. Titles and subtitles MUST use bold formatting.
- Use `##` section headings (e.g., "## Summary", "## Key Insights", "## Recommendations", "## Next Steps"). All headings MUST be bold.
- NEVER write plain text section headers. ALWAYS use `##` for headings.
- Under each heading (##), ALL content MUST be formatted as bullet points using `- ` or numbered lists using `1. ` prefix.
- Body text under subtitles MUST use light font weight (not bold).
- After every subtitle (## heading), insert a blank line, then list the content as bullets or numbered items.
- Example of CORRECT format:
  ## Summary
  
  - Total Balance: RM0.00
  - Income: RM0.00
  - Expenses: RM0.00
  
  ## Key Insights
  
  1. First insight here
  2. Second insight here
- Example of WRONG format (DO NOT USE):
  Summary
  Total Balance: RM0.00
  Income: RM0.00
- Use bullet lists with `- ` or numbered lists with `1. ` and indent sub-items by two spaces.
- Insert a blank line before and after every list block.
- Insert a blank line after every subtitle (## heading) before the content.
- Keep each bullet to a single line (no hard wraps inside bullets).
- Keep paragraphs short and scannable; use bold for key terms in body text.
- Tables are allowed when listing comparable items.
- If you list multiple items, ALWAYS use bullets or numbered lists - never plain text lines.

You have access to the user's financial data including:
- Account balances and transactions
- Spending patterns and categories
- Active budgets and their status
- Financial goals and progress
- Credit card information (balances, limits, utilization, payment dates)

When users upload statement files through the chat:
- Files are uploaded to AWS S3 and extracted by AI for PREVIEW ONLY (transactions are NOT automatically saved)
- You will receive preview information including: transaction count (credit/debit breakdown), period dates, account info, and balances
- ALWAYS present the preview to the user and ask what they want to do:
  • Show: Filename, account info, statement period, transaction count (X incomes, Y expenses), opening/closing balance
  • Example: "I've processed your bank statement successfully! Here's what I found:

    📄 Statement: maybank_dec2024.pdf
    🏦 Account: Maybank Savings (****1234)
    📅 Period: Dec 1-31, 2024
    💰 Balance: RM5,234.50 → RM3,120.80
    📊 Transactions: 42 total (3 incomes, 39 expenses)

    What would you like me to do?
    1. **Import** these transactions into your account
    2. **Analyze** the statement without importing (I'll provide insights on spending patterns, categories, trends, etc.)

    Just let me know!"

- Based on user's response:
  • If they want to IMPORT (e.g., "import", "save these transactions", "add to my account"):
    → IMPORTANT: DO NOT execute any import action in the chat
    → Instead, direct the user to the Upload Statement page to review and confirm the import

    → When user responds with "Import" to your preview:
      - Show a summary of what they'll be reviewing:
        * Number of transactions (X incomes, Y expenses)
        * Account name and closing balance
        * Period covered
      - Provide a clickable link/button to the Upload Statement page
      - Use this format: "**[Click here to review and import transactions](/transactions/upload?statement_id=XXX)**"
      - Replace XXX with the actual statement_id from the preview data

    → Example response:
      "Great! I've prepared your transactions for import. Here's what you'll be reviewing:

      📊 **Transaction Summary:**
      • Statement: maybank_dec2024.pdf
      • Account: Maybank Savings (****1234)
      • Period: Dec 1-31, 2024
      • Transactions: 42 total (3 incomes, 39 expenses)
      • Balance: RM5,234.50 → RM3,120.80

      **[Click here to review and import these transactions](/transactions/upload?statement_id=42)**

      You'll be able to review each transaction and make any necessary edits before importing."

    → CRITICAL: Never execute confirm_statement_import action from chat - always redirect to Upload Statement page

  • If they want to ANALYZE (e.g., "analyze", "just analyze", "show insights", "don't import"):
    → Provide detailed analysis of the extracted transactions WITHOUT importing
    → Analyze spending patterns, top categories, unusual transactions, trends
    → Suggest budgets or savings opportunities based on the data
    → Do NOT execute confirm_statement_import action

- If user's intent is unclear, ask them to clarify
- For imports: Always redirect to Upload Statement page with clickable link - NEVER execute import in chat
- For analysis: Provide insights directly in chat without importing
- If processing fails, inform the user about the error in a helpful way

You can execute actions such as:
- Creating, updating, or deleting budgets
- Creating, updating, or deleting financial goals
- Adding, updating, or removing credit cards
- Analyzing credit card utilization and payment schedules
- Categorizing transactions
- Creating expense or income records

AI-POWERED SUGGESTIONS:
You have access to intelligent budget and goal suggestions that are automatically generated based on:
- Historical spending patterns (for budget suggestions)
- Income/expense ratios and financial health metrics (for goal suggestions)
- Existing budgets and goals (to avoid duplicates)
These suggestions will appear in the [AI-Generated Suggestions] context section and include:
- Recommended budget amounts with justifications
- Suggested financial goals (emergency fund, savings, debt payoff, retirement)
- Reasoning behind each suggestion
Use these suggestions to proactively recommend budgets and goals to users when appropriate.

IMPORTANT ACTION EXECUTION FLOW:
1. When a user requests an action (e.g., "Set a budget for groceries"), DO NOT execute immediately
2. IMMEDIATELY on the FIRST response, provide ALL details of what you'll create/update in a structured format with ALL parameters filled in
3. CRITICAL: Do NOT give vague suggestions first - your FIRST response must include complete details (name, amount, category, priority, dates, etc.)
4. ALWAYS show the user the exact parameters that will be used in a clear, readable format
5. After showing complete details, ask for explicit confirmation: "Would you like me to proceed with this?" or "Shall I create this for you?"
6. Only include the <action> block in your response AFTER the user has confirmed (e.g., "yes", "proceed", "confirm", "go ahead")
7. If the user asks for information or analysis (not requesting an action), provide insights without needing confirmation

WRONG (Do NOT do this):
User: "Create a budget for food"
You: "I can help you create a food budget. Would you like me to proceed?" ← TOO VAGUE

RIGHT (Do this instead):
User: "Create a budget for food"
You: "Based on your spending history, I can create a monthly food budget with these details:
  • Budget Name: Monthly Food Budget
  • Category: Food
  • Limit: RM500
  • Period: January 1-31, 2025
  • Alert Threshold: 80%

  💡 Your average food spending is RM450/month.

  Shall I proceed?" ← COMPLETE DETAILS ON FIRST RESPONSE

IMPORTANT: When presenting actions for confirmation, format the details clearly so users can verify:
- For GOALS: Show goal name, target amount, category, priority, target date (if applicable), and brief description
- For BUDGETS: Show budget name, category, limit amount, period (start/end dates), alert threshold, AND provide personalized reasoning based on their spending history (e.g., "Based on your average RM450/month food spending, this RM500 budget gives you a comfortable 11% buffer")
- For CREDIT CARDS: Show bank name, card type, credit limit, statement date, payment due date
- For TRANSACTIONS: Show amount, category, type (expense/income), date, merchant/description

BUDGET RECOMMENDATIONS - Use User's Financial Context:
CRITICAL: When users request budget creation, provide COMPLETE budget details in your FIRST response (not a vague "I can help" message).

When analyzing and proposing budgets:
- Reference their actual historical spending in that category (e.g., "Your average monthly food spending is RM450")
- Suggest realistic budget amounts based on their spending patterns (not arbitrary round numbers)
- If they're overspending, suggest a gradual reduction path (e.g., "Your current RM800/month dining spend could be reduced to RM600 as a first step")
- If they have healthy spending, acknowledge it and suggest maintaining current levels
- Always explain WHY you're recommending a specific amount based on their data
- Set alert thresholds strategically (80% for flexible categories like Food, 90% for fixed categories like Housing)
- ALWAYS include: Budget Name, Category, Limit Amount, Period dates (start/end), Alert Threshold in your FIRST response

Example flows (showing complete details on FIRST response):
- User: "Set a RM500 budget for groceries" → You (FIRST RESPONSE with ALL details): "Based on your spending history, I can create a monthly food budget with these details:
  • Budget Name: Monthly Food Budget
  • Category: Food
  • Limit: RM500
  • Period: January 1-31, 2025
  • Alert Threshold: 80% (you'll be notified when spending reaches RM400)

  💡 Context: Your average food spending over the last 3 months is RM450, so this RM500 budget gives you a comfortable 11% buffer while keeping you disciplined.

  Shall I proceed?" → User: "Yes" → You (SECOND RESPONSE): [include action block and confirmation]

- User: "Help me create a food budget" → You (FIRST RESPONSE with ALL details): "I'll create a monthly food budget based on your spending patterns:
  • Budget Name: Monthly Food Budget
  • Category: Food
  • Limit: RM500
  • Period: January 1-31, 2025
  • Alert Threshold: 80%

  💡 Your average food spending is RM450/month, so this gives you a 11% buffer.

  Would you like me to create this budget?" → User: "Yes" → You (SECOND RESPONSE): [include action block and confirmation]

- User: "Help me save RM4500 for a laptop in 6 months" → You (FIRST RESPONSE with ALL details): "I'll create a savings goal with these details:
  • Goal Name: Laptop Savings
  • Target Amount: RM4500
  • Category: Other
  • Priority: medium
  • Target Date: July 1, 2025 (6 months from today)
  • Description: Save RM4500 for new laptop in 6 months

  Would you like me to create this goal for you?" → User: "Yes" → You (SECOND RESPONSE): [include action block and confirmation]

- User uploads statement, You show preview with two options, User: "Import" → You: "Great! Here's what you'll be reviewing:
  • Statement: maybank_dec2024.pdf
  • Account: Maybank Savings (****1234)
  • Period: Dec 1-31, 2024
  • Transactions: 42 total (3 incomes, 39 expenses)
  • Balance: RM5,234.50 → RM3,120.80

  **[Click here to review and import these transactions](/transactions/upload?statement_id=42)**" → User clicks link → Navigate to Upload Statement page

- User: "What's my credit utilization?" → You: [provide analysis directly, no confirmation needed]

When executing actions, use the following exact format (but DO NOT show this to the user - it will be automatically extracted):
<action>
{
  "action": "action_name",
  "parameters": {
    "param1": "value1",
    "param2": "value2"
  }
}
</action>

AVAILABLE ACTIONS AND REQUIRED FIELDS:

1. CREATE GOAL (create_goal):
Required fields: goal_name, description, category, priority, target_amount
Optional fields: current_amount, target_date
Categories: "Emergency Fund", "Vacation", "Car Purchase", "Home Down Payment", "Education", "Retirement", "Investment", "Other"
Priorities: "low", "medium", "high"
Example:
<action>
{
  "action": "create_goal",
  "parameters": {
    "goal_name": "Laptop Savings",
    "description": "Save RM4500 for new laptop in 6 months",
    "category": "Other",
    "priority": "medium",
    "target_amount": 4500,
    "current_amount": 0,
    "target_date": "2025-07-01"
  }
}
</action>

2. UPDATE GOAL (update_goal):
Required fields: goal_id
Optional fields: goal_name, description, category, priority, target_amount, current_amount, target_date, status
Status values: "active", "completed", "cancelled"
Example:
<action>
{
  "action": "update_goal",
  "parameters": {
    "goal_id": 123,
    "current_amount": 1500,
    "status": "active"
  }
}
</action>

3. DELETE GOAL (delete_goal):
Required fields: goal_id
Example:
<action>
{
  "action": "delete_goal",
  "parameters": {
    "goal_id": 123
  }
}
</action>

4. CREATE BUDGET (create_budget):
Required fields: name, limit_amount, category, period_start, period_end, alert_threshold
Optional fields: none (all fields required)
Categories: "Housing", "Food", "Transportation", "Entertainment", "Utilities", "Shopping", "Health & Fitness", "Travel", "Education", "Others"
Alert threshold: 0-100 (percentage, e.g., 80 means alert at 80% of limit)
Example:
<action>
{
  "action": "create_budget",
  "parameters": {
    "name": "Monthly Food Budget",
    "limit_amount": 500,
    "category": "Food",
    "period_start": "2025-01-01",
    "period_end": "2025-01-31",
    "alert_threshold": 80
  }
}
</action>

5. UPDATE BUDGET (update_budget):
Required fields: budget_id
Optional fields: name, limit_amount, category, period_start, period_end, alert_threshold, is_active
Example:
<action>
{
  "action": "update_budget",
  "parameters": {
    "budget_id": 456,
    "limit_amount": 600,
    "alert_threshold": 0.9
  }
}
</action>

6. DELETE BUDGET (delete_budget):
Required fields: budget_id
Example:
<action>
{
  "action": "delete_budget",
  "parameters": {
    "budget_id": 456
  }
}
</action>

7. CREATE CREDIT CARD (create_credit_card):
Required fields: bank_name, card_type, credit_limit
Optional fields: card_last_four, statement_date, payment_due_date, current_balance, minimum_payment
Example:
<action>
{
  "action": "create_credit_card",
  "parameters": {
    "bank_name": "Maybank",
    "card_type": "Visa Platinum",
    "credit_limit": 10000,
    "card_last_four": "1234",
    "statement_date": 5,
    "payment_due_date": 20,
    "current_balance": 0
  }
}
</action>

8. UPDATE CREDIT CARD (update_credit_card):
Required fields: card_id
Optional fields: bank_name, card_type, credit_limit, card_last_four, statement_date, payment_due_date, current_balance, minimum_payment, is_active
Example:
<action>
{
  "action": "update_credit_card",
  "parameters": {
    "card_id": 789,
    "current_balance": 2500,
    "minimum_payment": 250
  }
}
</action>

9. DELETE CREDIT CARD (delete_credit_card):
Required fields: card_id
Example:
<action>
{
  "action": "delete_credit_card",
  "parameters": {
    "card_id": 789
  }
}
</action>

10. CATEGORIZE TRANSACTION (categorize_transaction):
Required fields: transaction_id, category
Optional fields: subcategory, notes
Categories: must match valid transaction categories
Example:
<action>
{
  "action": "categorize_transaction",
  "parameters": {
    "transaction_id": 12345,
    "category": "Groceries",
    "subcategory": "Supermarket"
  }
}
</action>

11. CREATE TRANSACTION (create_transaction):
Required fields: amount, category, transaction_type, transaction_date, description
Optional fields: subcategory, merchant_name, payment_method, notes
Transaction types: "expense", "income"
Example:
<action>
{
  "action": "create_transaction",
  "parameters": {
    "amount": 50.00,
    "category": "Dining",
    "transaction_type": "expense",
    "transaction_date": "2025-01-15",
    "description": "Lunch at cafe",
    "merchant_name": "Starbucks",
    "payment_method": "credit_card"
  }
}
</action>

12. CONFIRM STATEMENT IMPORT (confirm_statement_import):
Required fields: statement_id
This action imports transactions from a previously uploaded and extracted statement into the user's account.
IMPORTANT: Only use this action AFTER user confirms the preview. Never execute without confirmation.
Example:
<action>
{
  "action": "confirm_statement_import",
  "parameters": {
    "statement_id": 42
  }
}
</action>

IMPORTANT RULES FOR ACTIONS:
- ALWAYS include ALL required fields for the action type
- Use correct data types (numbers for amounts, strings for names, dates in YYYY-MM-DD format)
- For goal categories, use ONLY: "Emergency Fund", "Vacation", "Car Purchase", "Home Down Payment", "Education", "Retirement", "Investment", "Other"
- For goal priorities, use ONLY: "low", "medium", "high"
- For budget categories, use ONLY: "Housing", "Food", "Transportation", "Entertainment", "Utilities", "Shopping", "Health & Fitness", "Travel", "Education", "Others"
- For budget alert_threshold, use 0-100 (percentage value, NOT decimal 0.0-1.0)
- For dates, always use YYYY-MM-DD format (e.g., "2025-01-15")
- Never create goals or budgets with missing required fields - if user doesn't provide info, ask them first
- When inferring values (like category or priority), choose the most reasonable option based on context
- Categories are case-sensitive and must match exactly (e.g., "Emergency Fund" not "emergency_fund", "Food" not "food" or "Groceries")
- When user says "groceries", map to "Food" category; when user says "gas" or "petrol", map to "Transportation"

IMPORTANT: Never display action blocks, code examples, or the action template in your response. Action blocks are internal commands and will be automatically processed. Only show natural language explanations of what actions you're taking (e.g., "I'll set up a budget alert for you" instead of showing the action code).

Be clear, professional, and provide actionable insights based on data. Maintain credibility and trustworthiness as a financial advisor at all times."""
        
        # Build messages with context
        messages = []
        
        # Add conversation summary if exists
        if conv_context.get("summary"):
            messages.append({
                "role": "system",
                "content": f"[Previous conversation summary]: {conv_context['summary']}"
            })
        
        # Add financial summary
        if summary_obj:
            messages.append({
                "role": "system",
                "content": f"[Financial Summary]: {summary_obj.summary_content}"
            })
        
        # Add recent financial context
        messages.append({
            "role": "system",
            "content": f"[Current Financial Status]:\n{masked_context_text}"
        })

        # Add intelligent budget and goal suggestions
        suggestions = rag_service.get_budget_goal_suggestions(current_user.user_id)
        if suggestions.get("budget_suggestions") or suggestions.get("goal_suggestions"):
            suggestions_text = rag_service.format_suggestions_for_llm(suggestions)
            messages.append({
                "role": "system",
                "content": f"[AI-Generated Suggestions]:\n{suggestions_text}"
            })
        
        # Add statement processing results if files were uploaded OR if there are pending statements in recent messages
        pending_statements = []

        # Check if files were just uploaded
        if processed_files:
            successful_files = [f for f in processed_files if f.get('success')]
            pending_statements.extend(successful_files)

        # Also check for pending statements in recent conversation messages (last 5 messages)
        if not processed_files:
            recent_messages = conv_context.get("messages", [])[-5:] if conv_context.get("messages") else []
            for msg in reversed(recent_messages):  # Check most recent first
                if msg.get("role") == "assistant":
                    # Check if this message has pending_statements in its content
                    # We need to fetch the actual message from DB to get metadata
                    pass

            # Fetch recent assistant messages with metadata
            recent_db_messages = db.query(models.ChatMessage).filter(
                models.ChatMessage.conversation_id == conversation.conversation_id,
                models.ChatMessage.role == "assistant"
            ).order_by(models.ChatMessage.created_at.desc()).limit(5).all()

            for db_msg in recent_db_messages:
                if db_msg.metadata_json and "pending_statements" in db_msg.metadata_json:
                    # Check if statements are still pending (not imported yet)
                    for stmt_info in db_msg.metadata_json["pending_statements"]:
                        stmt_id = stmt_info.get("statement_id")
                        stmt = db.query(models.Statement).filter(
                            models.Statement.statement_id == stmt_id,
                            models.Statement.processing_status == 'extracted'  # Still pending
                        ).first()
                        if stmt:
                            # Convert metadata format to match processed_files format
                            pending_statements.append({
                                "success": True,
                                "preview_mode": True,
                                **stmt_info
                            })
                    break  # Only use the most recent message with pending statements

        if pending_statements:
            statement_context = "[Recently Processed Statements - PREVIEW MODE]:\n"
            statement_context += "NOTE: These statements have been extracted but NOT imported yet. Ask user what they want to do (import or analyze).\n"
            successful_files = pending_statements
            for file_result in successful_files:
                statement_context += f"\n- File: {file_result.get('filename', 'Unknown')}\n"
                statement_context += f"  Statement ID: {file_result.get('statement_id')} (use this for confirm_statement_import)\n"
                statement_context += f"  Type: {file_result.get('statement_type', 'Unknown')}\n"
                statement_context += f"  Status: {'PREVIEW - Not imported yet' if file_result.get('preview_mode') else 'Imported'}\n"
                if file_result.get('transactions_count', 0) > 0:
                    credit_count = file_result.get('credit_count', 0)
                    debit_count = file_result.get('debit_count', 0)
                    statement_context += f"  Transactions: {file_result.get('transactions_count')} total ({credit_count} incomes, {debit_count} expenses)\n"
                if file_result.get('period_start') and file_result.get('period_end'):
                    statement_context += f"  Period: {file_result.get('period_start')} to {file_result.get('period_end')}\n"
                if file_result.get('opening_balance') is not None and file_result.get('closing_balance') is not None:
                    statement_context += f"  Balance: RM{file_result.get('opening_balance'):.2f} → RM{file_result.get('closing_balance'):.2f}\n"
                if file_result.get('account_info'):
                    acc_info = file_result.get('account_info', {})
                    if acc_info.get('bank_name'):
                        statement_context += f"  Bank: {acc_info.get('bank_name')}\n"
                    if acc_info.get('account_name'):
                        statement_context += f"  Account: {acc_info.get('account_name')}\n"
                    if acc_info.get('account_number'):
                        # Mask account number (show last 4 digits only)
                        acc_no = str(acc_info.get('account_number'))
                        masked_no = '****' + acc_no[-4:] if len(acc_no) > 4 else acc_no
                        statement_context += f"  Account Number: {masked_no}\n"

                # Fetch and include transaction details for analysis
                statement_id = file_result.get('statement_id')
                if statement_id and file_result.get('preview_mode'):
                    statement = db.query(models.Statement).filter(
                        models.Statement.statement_id == statement_id
                    ).first()

                    if statement and statement.extracted_data:
                        transactions = statement.extracted_data.get('transactions', [])
                        if transactions:
                            statement_context += f"\n  Transaction Details (for analysis):\n"

                            # Group by category for summary
                            from collections import defaultdict
                            category_totals = defaultdict(lambda: {'count': 0, 'total': 0})

                            # Limit to top transactions by amount and show category breakdown
                            sorted_txns = sorted(transactions, key=lambda x: abs(x.get('amount', 0)), reverse=True)

                            for txn in transactions:
                                cat = txn.get('category', 'Other')
                                amount = abs(txn.get('amount', 0))
                                category_totals[cat]['count'] += 1
                                category_totals[cat]['total'] += amount

                            # Show category breakdown
                            statement_context += f"  Category Breakdown:\n"
                            for cat, data in sorted(category_totals.items(), key=lambda x: x[1]['total'], reverse=True):
                                statement_context += f"    - {cat}: RM{data['total']:.2f} ({data['count']} transactions)\n"

                            # Show top 10 largest transactions for analysis
                            statement_context += f"\n  Top 10 Largest Transactions:\n"
                            for i, txn in enumerate(sorted_txns[:10], 1):
                                txn_type = "Income" if txn.get('type') == 'credit' else "Expense"
                                amount = abs(txn.get('amount', 0))
                                desc = txn.get('description', 'Unknown')[:50]
                                cat = txn.get('category', 'Other')
                                date = txn.get('date', 'Unknown')
                                statement_context += f"    {i}. [{txn_type}] RM{amount:.2f} - {desc} ({cat}) on {date}\n"

            # Only check for failed files if we're processing new files (not pulling from history)
            if processed_files:
                failed_files = [f for f in processed_files if not f.get('success')]
                if failed_files:
                    statement_context += "\nFailed Files:\n"
                    for file_result in failed_files:
                        statement_context += f"- {file_result.get('filename', 'Unknown')}: {file_result.get('error', 'Unknown error')}\n"

            messages.append({
                "role": "system",
                "content": statement_context
            })
        
        # Add conversation history
        messages.extend(conv_context["messages"])
        
        # Add current user message
        messages.append({
            "role": "user",
            "content": message_content
        })
        
        # Generate AI response using gemini-2.5-pro for chat
        ai_response = await gemini_service.generate_response(
            system_instruction=system_instruction,
            messages=messages,
            temperature=0.7,
            max_output_tokens=4000,
            model_override="gemini-2.5-pro"
        )
        
        response_content = ai_response.get("content", "")
        
        # Ensure markdown has a title and basic structure if model omitted it
        def _ensure_markdown_structure(text: str) -> str:
            stripped = (text or "").lstrip()
            if not stripped.startswith("# ") and not stripped.startswith("## "):
                # Prepend a default title and wrap existing text under a section
                title = "# RayyAI Response 📊\n\n"
                body = "## Summary\n" + (text or "")
                return title + body
            return text
        
        response_content = _ensure_markdown_structure(response_content)

        # Normalize spacing between headings and list blocks for cleaner rendering
        def _normalize_markdown_spacing(text: str) -> str:
            lines = (text or "").splitlines()
            # Ensure blank line before headings (except at start)
            tmp: list[str] = []
            for i, ln in enumerate(lines):
                if (ln.startswith("# ") or ln.startswith("## ")) and tmp and tmp[-1].strip() != "":
                    tmp.append("")
                tmp.append(ln.rstrip())
            # Ensure blank lines around list blocks
            out: list[str] = []
            in_list = False
            for ln in tmp:
                is_bullet = ln.lstrip().startswith("- ")
                if is_bullet and not in_list:
                    if out and out[-1].strip() != "":
                        out.append("")
                    in_list = True
                if not is_bullet and in_list:
                    if out and out[-1].strip() != "":
                        out.append("")
                    in_list = False
                out.append(ln)
            return "\n".join(out).strip() + "\n"

        response_content = _normalize_markdown_spacing(response_content)

        # Enforce bullet formatting for common sections and plain lists
        def _enforce_bullets(text: str) -> str:
            lines = (text or "").splitlines()
            out: list[str] = []
            in_list_mode = False
            last_was_blank = True
            last_was_heading = False
            consecutive_items = 0
            
            for i, ln in enumerate(lines):
                stripped = ln.strip()
                is_heading = stripped.startswith("# ") or stripped.startswith("## ")
                is_code_fence = stripped.startswith("```")
                is_action_open = stripped.startswith("<action>")
                is_action_close = stripped.startswith("</action>")
                is_already_bullet = stripped.startswith("- ") or stripped.startswith("* ") or stripped.startswith("1. ")
                
                # Handle headings - enable list mode for content after headings
                if is_heading:
                    in_list_mode = False  # Reset, will enable on next non-blank line
                    last_was_heading = True
                    if out and out[-1].strip() != "":
                        out.append("")
                    out.append(ln.rstrip())
                    last_was_blank = False
                    consecutive_items = 0
                    continue

                # Preserve code fences verbatim
                if is_code_fence:
                    in_list_mode = False
                    out.append(ln)
                    last_was_blank = False
                    last_was_heading = False
                    consecutive_items = 0
                    continue

                # Wrap <action> blocks in xml code fences for clear rendering
                if is_action_open:
                    in_list_mode = False
                    if out and out[-1].strip() != "":
                        out.append("")
                    out.append("```xml")
                    out.append(ln)
                    last_was_blank = False
                    last_was_heading = False
                    consecutive_items = 0
                    continue
                if is_action_close:
                    in_list_mode = False
                    out.append(ln)
                    out.append("```")
                    out.append("")
                    last_was_blank = False
                    last_was_heading = False
                    consecutive_items = 0
                    continue

                # Blank line handling
                if stripped == "":
                    if not last_was_blank:
                        out.append("")
                    last_was_blank = True
                    last_was_heading = False
                    in_list_mode = False
                    consecutive_items = 0
                    continue

                # Detect plain text section headers (like "Summary", "Key Insights" without ##)
                is_plain_section_header = (
                    not is_heading and
                    not is_already_bullet and
                    stripped and
                    len(stripped) < 50 and
                    (stripped[0].isupper() if stripped else False) and
                    stripped.lower() in {
                        "summary", "key insights", "recommendations", "next steps",
                        "account management", "budgeting", "financial goals", 
                        "spending analysis", "credit card management", "financial tracking",
                        "goal setting", "reporting", "actions i can execute",
                        "proposed budget", "gamification features"
                    }
                )
                
                if is_plain_section_header:
                    # Convert to markdown heading
                    out.append(f"## {stripped}")
                    last_was_heading = True
                    in_list_mode = False
                    consecutive_items = 0
                    last_was_blank = False
                    continue

                # Enable list mode after heading if we see list-like content
                if (last_was_heading or is_plain_section_header) and not is_already_bullet:
                    # Check if this looks like a list item
                    looks_like_list = (
                        ":" in stripped or  # key: value
                        (stripped and len(stripped) > 5 and not stripped[0].isupper()) or  # Descriptive text, not sentence case
                        (stripped and stripped[0].isupper() and ":" in stripped)  # Capitalized with colon
                    )
                    if looks_like_list:
                        in_list_mode = True

                # Decide if should be bulleted
                should_bullet = False
                if not is_already_bullet and stripped:
                    # Always bullet if we're in list mode
                    if in_list_mode:
                        should_bullet = True
                    # Also bullet if it looks like a list item
                    elif ":" in stripped and not stripped.startswith("http"):
                        should_bullet = True
                    # Or if previous line was bulleted and this line is similar format
                    elif consecutive_items > 0 and len(stripped) > 5:
                        should_bullet = True
                    # Or if we just had a heading and this line contains a colon or looks like list content
                    elif last_was_heading and (":" in stripped or len(stripped) > 10):
                        should_bullet = True
                        in_list_mode = True

                if should_bullet and not is_already_bullet:
                    if out and out[-1].strip() != "":
                        # Ensure blank line before list if needed
                        prev_line = out[-1].strip()
                        if not prev_line.startswith("- ") and not prev_line.startswith("#"):
                            out.append("")
                    out.append(f"- {stripped}")
                    in_list_mode = True
                    consecutive_items += 1
                    last_was_blank = False
                    last_was_heading = False
                else:
                    if is_already_bullet:
                        consecutive_items += 1
                    else:
                        consecutive_items = 0
                        in_list_mode = False
                    out.append(ln.rstrip())
                    last_was_blank = False
                    last_was_heading = False

            # Close with a trailing newline
            text2 = "\n".join(out).strip() + "\n"
            
            # Ensure blank lines around list blocks
            return _normalize_markdown_spacing(text2)

        response_content = _enforce_bullets(response_content)
        token_count = ai_response.get("token_count", 0)
        
        # Parse and execute actions FIRST (from original response before any modifications)
        actions_executed = []
        original_response = ai_response.get("content", "")
        parsed_actions = action_executor.parse_action_request(original_response)
        
        for action in parsed_actions:
            try:
                result = await action_executor.execute_action(action)
                actions_executed.append(result)

                # Add action confirmation to response (natural language, not code)
                if result.get("success"):
                    # Append an Actions Executed section in markdown
                    action_message = result.get('message', 'Action completed successfully')
                    if "## ✅ Actions Executed" not in response_content:
                        response_content += f"\n\n## ✅ Actions Executed\n\n"
                    response_content += f"- {action_message}\n"
                else:
                    # Action failed - show error to user and override optimistic response
                    action_type = action.get('action', 'action')
                    error_msg = result.get('error', 'Unknown error occurred')

                    # Replace the optimistic response with error feedback
                    response_content = f"""# ✗ Action Failed

I tried to execute the requested action, but encountered an error:

**Action:** {action_type}
**Error:** {error_msg}

Please check the details and try again. If you need help, let me know!"""

            except Exception as e:
                logger.error(f"Error executing action: {e}")
                actions_executed.append({
                    "success": False,
                    "error": str(e)
                })

                # Show exception error to user
                action_type = action.get('action', 'action')
                response_content = f"""# ✗ Action Failed

I tried to execute the requested action, but encountered an unexpected error:

**Action:** {action_type}
**Error:** {str(e)}

This might be a temporary issue. Please try again or contact support if the problem persists."""
        
        # Remove action blocks from response before showing to user (AFTER parsing and executing)
        def _remove_action_blocks(text: str) -> str:
            """Remove <action>...</action> blocks and their content from the response."""
            import re
            # Remove action blocks and any surrounding code fences
            text = re.sub(r'```xml\s*<action>.*?</action>\s*```', '', text, flags=re.DOTALL)
            text = re.sub(r'<action>.*?</action>', '', text, flags=re.DOTALL)
            # Clean up any remaining code fence artifacts
            text = re.sub(r'```xml\s*```', '', text, flags=re.DOTALL)
            # Remove multiple blank lines
            text = re.sub(r'\n{3,}', '\n\n', text)
            return text.strip()
        
        response_content = _remove_action_blocks(response_content)
        
        # Save assistant response
        # Include statement info in metadata if files were processed
        message_metadata = None
        if actions_executed or processed_files:
            message_metadata = {
                "actions_executed": actions_executed,
                "token_count": token_count,
                "usage_metadata": ai_response.get("usage_metadata")
            }
            # Add processed statement info for context persistence
            if processed_files:
                successful_files = [f for f in processed_files if f.get('success') and f.get('preview_mode')]
                if successful_files:
                    message_metadata["pending_statements"] = [
                        {
                            "statement_id": f.get('statement_id'),
                            "filename": f.get('filename'),
                            "transactions_count": f.get('transactions_count'),
                            "credit_count": f.get('credit_count'),
                            "debit_count": f.get('debit_count'),
                            "period_start": f.get('period_start'),
                            "period_end": f.get('period_end')
                        }
                        for f in successful_files
                    ]

        assistant_message = conversation_manager.add_message(
            conversation_id=conversation.conversation_id,
            role="assistant",
            content=response_content,
            metadata=message_metadata,
            token_count=token_count
        )
        
        # Update conversation with message count
        messages_all = conversation_manager.get_conversation_messages(
            conversation.conversation_id,
            current_user.user_id
        )
        conv_dict = {
            **conversation.__dict__,
            "message_count": len(messages_all)
        }
        
        return {
            "message": schemas.ChatMessageResponse(
                **{**user_message.__dict__, "metadata": user_message.metadata_json}
            ),
            "assistant_response": schemas.ChatMessageResponse(
                **{**assistant_message.__dict__, "metadata": assistant_message.metadata_json}
            ),
            "conversation": schemas.ChatConversationResponse(**conv_dict),
            "actions_executed": actions_executed if actions_executed else None
        }
    
    except Exception as e:
        logger.error(f"Error processing chat message: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing message: {str(e)}"
        )

@router.post("/context/refresh")
async def refresh_context(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Manually refresh user's financial context cache."""
    gemini_service = get_gemini_service()
    context_summarizer = ContextSummarizer(db, gemini_service)
    
    summary = await context_summarizer.get_or_generate_summary(
        current_user.user_id,
        summary_type="financial_snapshot",
        force_refresh=True
    )
    
    return {
        "message": "Context refreshed successfully",
        "summary_id": summary.summary_id
    }

@router.post("/context/summarize")
async def summarize_context(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Trigger context summarization."""
    gemini_service = get_gemini_service()
    context_summarizer = ContextSummarizer(db, gemini_service)
    
    summary = await context_summarizer.generate_financial_summary(current_user.user_id)
    
    return {
        "message": "Context summarized successfully",
        "summary": summary
    }


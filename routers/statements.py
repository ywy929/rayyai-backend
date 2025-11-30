from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Query, Response
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List
from datetime import date, datetime, timezone, timedelta
import os
import logging
import hashlib
import httpx
from pathlib import Path
import models
from schemas import StatementResponse
from database import get_db
from routers.utils import (
    get_current_user, get_current_user_optional, map_account_type, verify_token,
    upload_file_to_s3, s3_client, S3_BUCKET_NAME, AWS_REGION,
    UPLOAD_DIR, STATEMENTS_DIR, CTOS_DIR, BASE_URL
)
from dotenv import load_dotenv
from services.processors import process_statement_pdf, process_ctos_pdf
from routers.transactions import infer_expense_type

load_dotenv()
router = APIRouter()
logger = logging.getLogger(__name__)


def update_credit_card_balance(db: Session, account_id: int, user_id: int, closing_balance: float = None):
    """
    Update credit card balance from the extracted statement balance.
    This is called after processing statements to update the credit card balance.

    Args:
        db: Database session
        account_id: Account ID linked to the credit card
        user_id: User ID for security check
        closing_balance: The extracted closing balance from the statement
    """
    try:
        logger.info(f"[update_credit_card_balance] Called with account_id={account_id}, user_id={user_id}, closing_balance={closing_balance}")

        # Get the account
        account = db.query(models.Account).filter(
            models.Account.account_id == account_id,
            models.Account.user_id == user_id,
            models.Account.is_deleted == False
        ).first()

        if not account:
            logger.info(f"[update_credit_card_balance] Account {account_id} not found or deleted")
            return

        if not account.card_id:
            logger.info(f"[update_credit_card_balance] Account {account_id} is not linked to a credit card (card_id is None)")
            return

        logger.info(f"[update_credit_card_balance] Account {account_id} is linked to credit card {account.card_id}")

        # Get the credit card
        credit_card = db.query(models.UserCreditCard).filter(
            models.UserCreditCard.card_id == account.card_id,
            models.UserCreditCard.user_id == user_id,
            models.UserCreditCard.is_deleted == False
        ).first()

        if not credit_card:
            logger.warning(f"[update_credit_card_balance] Credit card {account.card_id} not found or deleted")
            return

        # Update credit card balance from extracted closing balance
        if closing_balance is not None:
            old_balance = credit_card.current_balance
            credit_card.current_balance = abs(closing_balance)
            db.commit()
            logger.info(f"[update_credit_card_balance] ✓ Updated credit card {credit_card.card_id} balance from RM{old_balance:,.2f} to RM{credit_card.current_balance:,.2f}")
        else:
            logger.info(f"[update_credit_card_balance] Closing balance is None, skipping update")

    except Exception as e:
        logger.error(f"[update_credit_card_balance] Error updating credit card balance: {str(e)}", exc_info=True)
        # Don't raise - this is a non-critical update


def update_user_from_extracted_info(user: models.User, user_info: dict, db: Session):
    """
    Update user profile with extracted information from statements.
    Only updates fields that are currently missing or empty.
    
    Args:
        user: User model instance
        user_info: Dictionary with extracted user information
        db: Database session
    """
    if not user_info:
        return
    
    updated = False
    
    # Update first_name if missing and extracted
    if user_info.get('first_name') and (not user.first_name or user.first_name.strip() == ''):
        user.first_name = user_info['first_name'].strip()
        updated = True
        logger.info(f"Updated first_name from statement: {user.first_name}")
    
    # Update last_name if missing and extracted
    if user_info.get('last_name') and (not user.last_name or user.last_name.strip() == ''):
        user.last_name = user_info['last_name'].strip()
        updated = True
        logger.info(f"Updated last_name from statement: {user.last_name}")
    
    # Update DOB if missing and extracted
    if user_info.get('date_of_birth'):
        try:
            extracted_dob = datetime.strptime(user_info['date_of_birth'], '%Y-%m-%d').date()
            # Only update if current DOB is missing or default
            if not user.dob or user.dob.year == 1900:  # Common default date
                user.dob = extracted_dob
                updated = True
                logger.info(f"Updated DOB from statement: {user.dob}")
        except (ValueError, TypeError) as e:
            logger.warning(f"Invalid date_of_birth format in extracted data: {user_info.get('date_of_birth')}")
    
    # Note: We don't update gender, address, email automatically as these might be sensitive
    # and the user may have intentionally set different values
    
    if updated:
        user.updated = datetime.now(timezone.utc)
        db.commit()
        logger.info("User profile updated with extracted information from statement")


# Helper function to detect statement type from filename
def detect_statement_type(filename: str, content_preview: bytes = None) -> str:
    """
    Auto-detect statement type from filename using heuristics.
    
    Args:
        filename: The filename of the uploaded file
        content_preview: Optional preview of file content (not used currently, reserved for future AI detection)
    
    Returns:
        Statement type: "bank", "credit_card", "ewallet", or "receipt"
    """
    if not filename:
        return "bank"  # Default to bank
    
    filename_lower = filename.lower()
    
    # Check for credit card keywords
    credit_keywords = ["credit", "card", "visa", "mastercard", "amex", "american express", "cc", "creditcard"]
    if any(keyword in filename_lower for keyword in credit_keywords):
        return "credit_card"
    
    # Check for e-wallet keywords
    ewallet_keywords = ["ewallet", "e-wallet", "wallet", "grab", "touchngo", "touch n go", "boost", "bigpay", "fave", "shopee pay", "shopee pay"]
    if any(keyword in filename_lower for keyword in ewallet_keywords):
        return "ewallet"
    
    # Check for receipt keywords
    receipt_keywords = ["receipt", "invoice", "bill"]
    if any(keyword in filename_lower for keyword in receipt_keywords):
        return "receipt"
    
    # Default to bank statement
    return "bank"


# Helper function to generate descriptive display name
def generate_display_name(statement_type: str, account_info: dict = None, period_start: date = None, period_end: date = None, original_filename: str = None) -> str:
    """
    Generate a user-friendly display name for statements

    Format: {Bank}_{Type}_{Period}.pdf
    Example: Maybank_Bank_Statement_Jan2025.pdf

    Fallbacks to original filename if extraction data is not available
    """
    try:
        # Extract bank name if available
        bank_name = None
        if account_info:
            # Try explicit bank_name or bank fields first
            bank_name = account_info.get('bank_name') or account_info.get('bank')

            # If not found, try to extract from account_name
            # e.g., "Maybank Visa Platinum" -> "Maybank"
            if not bank_name and account_info.get('account_name'):
                account_name = account_info.get('account_name', '')
                # Common Malaysian banks
                banks = ['Maybank', 'CIMB', 'Public Bank', 'RHB', 'Hong Leong', 'AmBank',
                         'Bank Islam', 'OCBC', 'UOB', 'Standard Chartered', 'HSBC',
                         'Affin Bank', 'Bank Rakyat', 'Alliance Bank', 'Bank Muamalat']

                for bank in banks:
                    if bank.lower() in account_name.lower():
                        bank_name = bank
                        break

        # Format statement type
        type_label = statement_type.replace('_', ' ').title()

        # Format period
        period_str = None
        if period_start and period_end:
            start_month = period_start.strftime('%b')
            end_month = period_end.strftime('%b')
            year = period_end.strftime('%Y')

            if start_month == end_month:
                # Same month: "Jan2025"
                period_str = f"{start_month}{year}"
            else:
                # Different months: "Jan-Feb2025"
                period_str = f"{start_month}-{end_month}{year}"

        # Build display name
        parts = []
        if bank_name:
            parts.append(bank_name.replace(' ', '_'))
        parts.append(type_label.replace(' ', '_'))
        if period_str:
            parts.append(period_str)

        if parts:
            return '_'.join(parts) + '.pdf'

        # Fallback to original filename
        if original_filename:
            return original_filename

        # Last resort: generic name
        return f"{type_label}_{datetime.now().strftime('%d%b%Y')}.pdf"

    except Exception as e:
        logger.error(f"Error generating display name: {e}")
        return original_filename or f"Statement_{datetime.now().strftime('%d%b%Y')}.pdf"


# Helper function to upload file locally
async def upload_file_local(
    file: UploadFile, user_id: int, folder: str, prefix: str = ""
) -> tuple[str, str]:
    """
    Upload file to local storage and return the file path and SHA-256 hash.

    Args:
        file: The uploaded file
        user_id: Current user's ID
        folder: Folder name (statements/ctos)
        prefix: Optional prefix for filename (e.g., "CTOS_")

    Returns:
        Tuple of (File URL/path, SHA-256 hash)
    """
    try:
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Handle case where filename might be None
        filename = file.filename or "uploaded_file"
        if not isinstance(filename, str):
            filename = str(filename)
        file_extension = os.path.splitext(filename)[1] or ".pdf"  # Default to .pdf if no extension
        
        # Create user-specific directory
        folder_path = STATEMENTS_DIR if folder == "statements" else CTOS_DIR
        user_dir = os.path.join(folder_path, str(user_id))
        Path(user_dir).mkdir(parents=True, exist_ok=True)
        
        # Generate unique filename
        unique_filename = f"{prefix}{timestamp}{file_extension}"
        file_path = os.path.join(user_dir, unique_filename)

        # Read file contents
        contents = await file.read()
        
        if not contents:
            raise HTTPException(status_code=400, detail="File is empty")

        # Compute SHA-256 hash for duplicate detection
        file_hash = hashlib.sha256(contents).hexdigest()

        # Save file to local storage
        with open(file_path, "wb") as f:
            f.write(contents)

        # Generate URL for accessing the file
        # Use relative path that can be served by FastAPI static files or a route
        relative_path = f"/files/{folder}/{user_id}/{unique_filename}"
        url = f"{BASE_URL}{relative_path}"

        logger.info(f"File uploaded successfully to: {file_path}")
        return url, file_hash

    except Exception as e:
        logger.error(f"File upload error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"File upload error: {str(e)}")


# ============================================
# STANDARD STATEMENT UPLOADS
# ============================================


@router.get("/statement", response_model=List[StatementResponse])
async def get_statements(
    db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)
):
    """Get all non-deleted statements (excluding CTOS)"""
    statements = (
        db.query(models.Statement)
        .filter(
            models.Statement.user_id == current_user.user_id,
            models.Statement.statement_type.notin_(["CTOS"]),
            models.Statement.is_deleted == False,  # ✅ Only non-deleted
        )
        .order_by(models.Statement.date_uploaded.desc())  # ✅ Latest first
        .all()
    )
    return statements


@router.post(
    "/statement",
    response_model=StatementResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_statement(
    file: UploadFile = File(...),
    statement_type: str = Query(
        ..., description="Type: bank, credit_card, ewallet, receipt"
    ),
    force_upload: bool = Query(False, description="Force upload even if duplicate detected"),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    """
    Upload financial statements to local storage for AI extraction.
    Supports: bank statements, credit card statements, e-wallet statements, receipts
    AI will extract: transactions, period dates, account details
    """
    # Validate file type
    allowed_extensions = {".pdf", ".jpg", ".jpeg", ".png"}
    # Handle case where filename might be None
    filename = file.filename or "uploaded_file"
    file_ext = os.path.splitext(filename)[1].lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"File type {file_ext} not allowed. Allowed: {', '.join(allowed_extensions)}",
        )

    # Validate statement type
    valid_types = ["bank", "credit_card", "ewallet", "receipt"]
    if statement_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid statement_type. Must be one of: {', '.join(valid_types)}",
        )

    # Validate file size (max 10MB)
    file.file.seek(0, 2)  # Seek to end
    file_size = file.file.tell()
    file.file.seek(0)  # Reset to beginning

    max_size = 10 * 1024 * 1024  # 10MB
    if file_size > max_size:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Max size: {max_size / (1024*1024)}MB",
        )

    # Upload to local storage and get file hash
    statement_url, file_hash = await upload_file_local(
        file=file, user_id=current_user.user_id, folder="statements"
    )

    # Check for duplicate file uploads using SHA-256 hash (unless force_upload is True)
    if not force_upload:
        existing_statement = db.query(models.Statement).filter(
            models.Statement.user_id == current_user.user_id,
            models.Statement.file_hash == file_hash,
            models.Statement.is_deleted == False
        ).first()

        if existing_statement:
            # Return detailed duplicate information for frontend dialog
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "message": f"Duplicate file detected. This file was already uploaded on {existing_statement.date_uploaded.strftime('%Y-%m-%d %H:%M:%S')}",
                    "duplicate_statement": {
                        "statement_id": existing_statement.statement_id,
                        "statement_type": existing_statement.statement_type,
                        "date_uploaded": existing_statement.date_uploaded.isoformat(),
                        "period_start": existing_statement.period_start.isoformat() if existing_statement.period_start else None,
                        "period_end": existing_statement.period_end.isoformat() if existing_statement.period_end else None,
                        "credit_score": existing_statement.credit_score
                    }
                }
            )

    # Create database record (period dates will be populated by AI)
    db_statement = models.Statement(
        user_id=current_user.user_id,
        statement_type=statement_type,
        statement_url=statement_url,
        file_hash=file_hash,
        display_name=filename,  # Use original filename initially, will be updated after extraction
        period_start=None,  # Will be extracted by AI
        period_end=None,  # Will be extracted by AI
        is_deleted=False,  # ✅ Explicitly set to False
    )
    db.add(db_statement)
    db.commit()
    db.refresh(db_statement)

    # TODO: Trigger AI processing to extract transactions and period dates
    # from background_tasks import process_statement_with_ai
    # process_statement_with_ai.delay(db_statement.statement_id, statement_url)

    return db_statement


@router.delete("/statement/{statement_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_statement(
    statement_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    """Soft delete a statement (marks as deleted, doesn't remove from database)"""
    statement = (
        db.query(models.Statement)
        .filter(
            models.Statement.statement_id == statement_id,
            models.Statement.user_id == current_user.user_id,
            models.Statement.is_deleted == False,
        )
        .first()
    )

    if not statement:
        raise HTTPException(
            status_code=404, detail="Statement not found or already deleted"
        )

    # ✅ Soft delete - just mark as deleted
    statement.is_deleted = True
    statement.deleted_at = datetime.now(timezone.utc)  # Optional: track when deleted

    db.commit()

    return None


@router.patch("/statement/{statement_id}/restore", status_code=status.HTTP_200_OK)
async def restore_statement(
    statement_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    """Restore a soft-deleted statement"""
    statement = (
        db.query(models.Statement)
        .filter(
            models.Statement.statement_id == statement_id,
            models.Statement.user_id == current_user.user_id,
            models.Statement.is_deleted == True,
        )
        .first()
    )

    if not statement:
        raise HTTPException(status_code=404, detail="Deleted statement not found")

    # ✅ Restore the statement
    statement.is_deleted = False
    statement.deleted_at = None

    db.commit()
    db.refresh(statement)

    return {"message": "Statement restored successfully", "statement_id": statement_id}


@router.get("/statement/deleted", response_model=List[StatementResponse])
async def get_deleted_statements(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    """Get all soft-deleted statements for recovery"""
    statements = (
        db.query(models.Statement)
        .filter(
            models.Statement.user_id == current_user.user_id,
            models.Statement.statement_type.notin_(["CTOS"]),
            models.Statement.is_deleted == True,
        )
        .order_by(models.Statement.deleted_at.desc())
        .all()
    )
    return statements


@router.get("/statements/{statement_id}/transactions")
async def get_statement_transactions(
    statement_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    """Retrieve all expenses/income sourced from a specific statement file"""
    statement = (
        db.query(models.Statement)
        .filter(
            models.Statement.statement_id == statement_id,
            models.Statement.user_id == current_user.user_id,
            models.Statement.is_deleted == False,
        )
        .first()
    )

    if not statement:
        raise HTTPException(status_code=404, detail="Statement not found")

    incomes = (
        db.query(models.Income)
        .filter(
            models.Income.statement_id == statement_id,
            models.Income.user_id == current_user.user_id,
        )
        .all()
    )

    expenses = (
        db.query(models.Expense)
        .filter(
            models.Expense.statement_id == statement_id,
            models.Expense.user_id == current_user.user_id,
        )
        .all()
    )

    return {"statement_id": statement_id, "incomes": incomes, "expenses": expenses}


# ============================================
# CTOS STATEMENT UPLOADS
# ============================================


@router.get("/ctosstatement", response_model=List[StatementResponse])
async def get_ctos_statements(
    db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)
):
    """Get all non-deleted CTOS statements"""
    statements = (
        db.query(models.Statement)
        .filter(
            models.Statement.user_id == current_user.user_id,
            models.Statement.statement_type == "CTOS",
            models.Statement.is_deleted == False,
        )
        .order_by(models.Statement.date_uploaded.desc())
        .all()
    )
    return statements


@router.post(
    "/ctosstatement",
    response_model=StatementResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_ctos_statement(
    file: UploadFile = File(...),
    force_upload: bool = Query(False, description="Force upload even if duplicate detected"),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    """Upload CTOS credit report - AI will extract credit score and period dates"""
    # Validate file type
    allowed_extensions = {".pdf", ".jpg", ".jpeg", ".png"}
    # Handle case where filename might be None
    filename = file.filename or "uploaded_file"
    file_ext = os.path.splitext(filename)[1].lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400, detail=f"File type {file_ext} not allowed for CTOS"
        )

    # Validate file size (max 10MB)
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)

    if file_size > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Max size: 10MB")

    # Upload to local storage and get file hash
    statement_url, file_hash = await upload_file_local(
        file=file, user_id=current_user.user_id, folder="ctos", prefix="CTOS_"
    )

    # Check for duplicate file uploads using SHA-256 hash (unless force_upload is True)
    if not force_upload:
        existing_statement = db.query(models.Statement).filter(
            models.Statement.user_id == current_user.user_id,
            models.Statement.file_hash == file_hash,
            models.Statement.is_deleted == False
        ).first()

        if existing_statement:
            # Return detailed duplicate information for frontend dialog
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "message": f"Duplicate CTOS file detected. This file was already uploaded on {existing_statement.date_uploaded.strftime('%Y-%m-%d %H:%M:%S')}",
                    "duplicate_statement": {
                        "statement_id": existing_statement.statement_id,
                        "statement_type": existing_statement.statement_type,
                        "date_uploaded": existing_statement.date_uploaded.isoformat(),
                        "period_start": existing_statement.period_start.isoformat() if existing_statement.period_start else None,
                        "period_end": existing_statement.period_end.isoformat() if existing_statement.period_end else None,
                        "credit_score": existing_statement.credit_score
                    }
                }
            )

    # Create database record with CTOS type (AI will populate credit_score and dates)
    db_statement = models.Statement(
        user_id=current_user.user_id,
        statement_type="CTOS",
        statement_url=statement_url,
        file_hash=file_hash,
        display_name=filename,  # Use original filename
        period_start=None,
        period_end=None,
        credit_score=None,
        is_deleted=False,
    )
    db.add(db_statement)
    db.commit()
    db.refresh(db_statement)

    # Trigger AI processing to extract credit score and analysis
    try:
        logger.info(f"Starting AI extraction for CTOS statement {db_statement.statement_id}")
        
        # Update status to extracting
        db_statement.processing_status = 'extracting'
        db.commit()
        
        # Read PDF from local storage
        statement_url = db_statement.statement_url
        if statement_url.startswith(BASE_URL):
            relative_path = statement_url.replace(f"{BASE_URL}/files/", "")
            file_path = os.path.join(UPLOAD_DIR, relative_path)
        elif statement_url.startswith("/files/"):
            file_path = os.path.join(UPLOAD_DIR, statement_url.replace("/files/", ""))
        else:
            file_path = statement_url
        
        if not os.path.exists(file_path):
            logger.warning(f"CTOS file not found at {file_path}, skipping AI extraction")
            db_statement.processing_status = 'failed'
            db_statement.processing_error = "File not found for processing"
            db.commit()
        else:
            # Read PDF bytes
            with open(file_path, "rb") as f:
                pdf_bytes = f.read()
            
            # Process CTOS PDF with AI
            result = process_ctos_pdf(pdf_bytes)
            
            if result.get('success'):
                # Update statement with extracted data (legacy fields for backward compatibility)
                if result.get('credit_score') is not None:
                    db_statement.credit_score = result['credit_score']
                
                if result.get('score_text'):
                    db_statement.score_text = result['score_text']
                
                # Parse and set period dates
                if result.get('period_start'):
                    try:
                        db_statement.period_start = datetime.strptime(result['period_start'], '%Y-%m-%d').date()
                    except ValueError:
                        logger.warning(f"Invalid period_start date format: {result['period_start']}")
                
                if result.get('period_end'):
                    try:
                        db_statement.period_end = datetime.strptime(result['period_end'], '%Y-%m-%d').date()
                    except ValueError:
                        logger.warning(f"Invalid period_end date format: {result['period_end']}")
                
                # Store full extracted data in JSON for reference
                db_statement.extracted_data = {
                    "report_date": result.get('report_date'),
                    "personal_info": result.get('personal_info'),
                    "ctos_score": result.get('ctos_score'),
                    "legal_records": result.get('legal_records'),
                    "credit_facility_summary": result.get('credit_facility_summary'),
                    "credit_facilities": result.get('credit_facilities', []),
                    "credit_utilisation": result.get('credit_utilisation'),
                    "loan_applications": result.get('loan_applications', []),
                    "employment_info": result.get('employment_info'),
                    "ptptn_status": result.get('ptptn_status'),
                }
                
                # Save detailed CTOS data to dedicated database models
                try:
                    save_ctos_detailed_data(db_statement.statement_id, result, db)
                except Exception as e:
                    logger.error(f"Error saving detailed CTOS data: {str(e)}", exc_info=True)
                    # Don't fail the extraction, just log the error
                
                # Optionally update user profile with extracted user info (only if fields are missing)
                if result.get('personal_info'):
                    personal_info = result['personal_info']
                    # Map personal_info to user_info format for backward compatibility
                    user_info = {
                        "full_name": personal_info.get('full_name'),
                        "ic_number": personal_info.get('ic_nric'),
                        "date_of_birth": personal_info.get('date_of_birth'),
                        "address": personal_info.get('address_line1')
                    }
                    update_user_from_extracted_info(current_user, user_info, db)
                
                db_statement.processing_status = 'extracted'
                db_statement.last_processed = datetime.now(timezone.utc)
                logger.info(f"Successfully extracted CTOS data: score={result.get('credit_score')}, period={result.get('period_start')} to {result.get('period_end')}")
            else:
                # Extraction failed
                db_statement.processing_status = 'failed'
                db_statement.processing_error = result.get('error', 'Unknown error during extraction')
                logger.error(f"CTOS extraction failed: {result.get('error')}")
            
            db.commit()
            db.refresh(db_statement)
    
    except Exception as e:
        logger.error(f"Error during CTOS AI extraction: {str(e)}")
        # Don't fail the upload, just mark as failed
        db_statement.processing_status = 'failed'
        db_statement.processing_error = f"Error during extraction: {str(e)}"
        db.commit()

    return db_statement


@router.get("/ctosstatement/{statement_id}", response_model=StatementResponse)
async def get_ctos_statement(
    statement_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    """Retrieve details for a single CTOS statement upload"""
    statement = (
        db.query(models.Statement)
        .filter(
            models.Statement.statement_id == statement_id,
            models.Statement.user_id == current_user.user_id,
            models.Statement.statement_type == "CTOS",
            models.Statement.is_deleted == False,
        )
        .first()
    )

    if not statement:
        raise HTTPException(status_code=404, detail="CTOS statement not found")

    return statement


@router.delete("/ctosstatement/{statement_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_ctos_statement(
    statement_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    """Soft delete a CTOS statement"""
    statement = (
        db.query(models.Statement)
        .filter(
            models.Statement.statement_id == statement_id,
            models.Statement.user_id == current_user.user_id,
            models.Statement.statement_type == "CTOS",
            models.Statement.is_deleted == False,
        )
        .first()
    )

    if not statement:
        raise HTTPException(
            status_code=404, detail="CTOS statement not found or already deleted"
        )

    # ✅ Soft delete
    statement.is_deleted = True
    statement.deleted_at = datetime.now(timezone.utc)

    db.commit()

    return None


@router.patch("/ctosstatement/{statement_id}/restore", status_code=status.HTTP_200_OK)
async def restore_ctos_statement(
    statement_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    """Restore a soft-deleted CTOS statement"""
    statement = (
        db.query(models.Statement)
        .filter(
            models.Statement.statement_id == statement_id,
            models.Statement.user_id == current_user.user_id,
            models.Statement.statement_type == "CTOS",
            models.Statement.is_deleted == True,
        )
        .first()
    )

    if not statement:
        raise HTTPException(status_code=404, detail="Deleted CTOS statement not found")

    statement.is_deleted = False
    statement.deleted_at = None

    db.commit()
    db.refresh(statement)

    return {
        "message": "CTOS statement restored successfully",
        "statement_id": statement_id,
    }


def save_ctos_detailed_data(statement_id: int, result: dict, db: Session):
    """
    Save detailed CTOS data to database models
    """
    try:
        # 1. Save Personal Info
        if result.get('personal_info'):
            personal_info = result['personal_info']
            # Check if exists, update or create
            existing = db.query(models.CTOSPersonalInfo).filter(
                models.CTOSPersonalInfo.statement_id == statement_id
            ).first()
            
            if existing:
                for key, value in personal_info.items():
                    if value is not None:
                        if key == 'date_of_birth' and isinstance(value, str):
                            try:
                                setattr(existing, key, datetime.strptime(value, '%Y-%m-%d').date())
                            except ValueError:
                                pass
                        else:
                            setattr(existing, key, value)
            else:
                dob = None
                if personal_info.get('date_of_birth'):
                    try:
                        dob = datetime.strptime(personal_info['date_of_birth'], '%Y-%m-%d').date()
                    except (ValueError, TypeError):
                        pass
                
                db_personal_info = models.CTOSPersonalInfo(
                    statement_id=statement_id,
                    full_name=personal_info.get('full_name'),
                    ic_nric=personal_info.get('ic_nric'),
                    date_of_birth=dob,
                    nationality=personal_info.get('nationality', 'Malaysia'),
                    address_line1=personal_info.get('address_line1'),
                    address_line2=personal_info.get('address_line2'),
                )
                db.add(db_personal_info)
        
        # 2. Save CTOS Score
        if result.get('ctos_score'):
            score_data = result['ctos_score']
            existing = db.query(models.CTOSScore).filter(
                models.CTOSScore.statement_id == statement_id
            ).first()
            
            if existing:
                existing.ctos_score = score_data.get('score')
                existing.score_text = score_data.get('score_text')
                existing.risk_factors = score_data.get('risk_factors')
            else:
                db_score = models.CTOSScore(
                    statement_id=statement_id,
                    ctos_score=score_data.get('score'),
                    score_text=score_data.get('score_text'),
                    risk_factors=score_data.get('risk_factors')
                )
                db.add(db_score)
        
        # 3. Save Legal Records
        if result.get('legal_records'):
            legal_data = result['legal_records']
            existing = db.query(models.CTOSLegalRecords).filter(
                models.CTOSLegalRecords.statement_id == statement_id
            ).first()
            
            if existing:
                for key, value in legal_data.items():
                    if value is not None:
                        setattr(existing, key, value)
            else:
                db_legal = models.CTOSLegalRecords(
                    statement_id=statement_id,
                    is_bankrupt=legal_data.get('is_bankrupt', False),
                    legal_records_personal_24m=legal_data.get('legal_records_personal_24m', 0),
                    legal_records_non_personal_24m=legal_data.get('legal_records_non_personal_24m', 0),
                    has_special_attention_accounts=legal_data.get('has_special_attention_accounts', False),
                    has_trade_referee_listing=legal_data.get('has_trade_referee_listing', False)
                )
                db.add(db_legal)
        
        # 4. Save Credit Facility Summary
        if result.get('credit_facility_summary'):
            summary_data = result['credit_facility_summary']
            existing = db.query(models.CTOSCreditFacilitySummary).filter(
                models.CTOSCreditFacilitySummary.statement_id == statement_id
            ).first()
            
            if existing:
                for key, value in summary_data.items():
                    if value is not None:
                        setattr(existing, key, value)
            else:
                db_summary = models.CTOSCreditFacilitySummary(
                    statement_id=statement_id,
                    total_outstanding_balance=summary_data.get('total_outstanding_balance'),
                    total_credit_limit=summary_data.get('total_credit_limit'),
                    credit_applications_12m_total=summary_data.get('credit_applications_12m_total', 0),
                    credit_applications_12m_approved=summary_data.get('credit_applications_12m_approved', 0),
                    credit_applications_12m_pending=summary_data.get('credit_applications_12m_pending', 0)
                )
                db.add(db_summary)
        
        # 5. Save Credit Facilities (delete old ones first, then add new)
        if result.get('credit_facilities'):
            # Delete existing facilities for this statement
            db.query(models.CTOSCreditFacility).filter(
                models.CTOSCreditFacility.statement_id == statement_id
            ).delete()
            
            for facility_data in result['credit_facilities']:
                db_facility = models.CTOSCreditFacility(
                    statement_id=statement_id,
                    facility_number=facility_data.get('facility_number'),
                    facility_type=facility_data.get('facility_type'),
                    facility_name=facility_data.get('facility_name'),
                    bank_name=facility_data.get('bank_name'),
                    account_number=facility_data.get('account_number'),
                    account_name=facility_data.get('account_name'),
                    credit_limit=facility_data.get('credit_limit'),
                    outstanding_balance=facility_data.get('outstanding_balance'),
                    collateral_type=facility_data.get('collateral_type'),
                    collateral_code=facility_data.get('collateral_code'),
                    conduct_12m=facility_data.get('conduct_12m')
                )
                db.add(db_facility)
        
        # 6. Save Credit Utilisation
        if result.get('credit_utilisation'):
            util_data = result['credit_utilisation']
            existing = db.query(models.CTOSCreditUtilisation).filter(
                models.CTOSCreditUtilisation.statement_id == statement_id
            ).first()
            
            if existing:
                for key, value in util_data.items():
                    if value is not None:
                        if key == 'earliest_known_facility_date' and isinstance(value, str):
                            try:
                                setattr(existing, key, datetime.strptime(value, '%Y-%m-%d').date())
                            except ValueError:
                                pass
                        else:
                            setattr(existing, key, value)
            else:
                earliest_date = None
                if util_data.get('earliest_known_facility_date'):
                    try:
                        earliest_date = datetime.strptime(util_data['earliest_known_facility_date'], '%Y-%m-%d').date()
                    except (ValueError, TypeError):
                        pass
                
                db_util = models.CTOSCreditUtilisation(
                    statement_id=statement_id,
                    earliest_known_facility_date=earliest_date,
                    total_outstanding=util_data.get('total_outstanding'),
                    outstanding_percentage_of_limit=util_data.get('outstanding_percentage_of_limit'),
                    number_of_unsecured_facilities=util_data.get('number_of_unsecured_facilities', 0),
                    number_of_secured_facilities=util_data.get('number_of_secured_facilities', 0),
                    avg_utilisation_credit_card_6m=util_data.get('avg_utilisation_credit_card_6m'),
                    avg_utilisation_revolving_6m=util_data.get('avg_utilisation_revolving_6m')
                )
                db.add(db_util)
        
        # 7. Save Loan Applications (delete old ones first, then add new)
        if result.get('loan_applications'):
            # Delete existing applications for this statement
            db.query(models.CTOSLoanApplication).filter(
                models.CTOSLoanApplication.statement_id == statement_id
            ).delete()
            
            for app_data in result['loan_applications']:
                app_date = None
                if app_data.get('application_date'):
                    try:
                        app_date = datetime.strptime(app_data['application_date'], '%Y-%m-%d').date()
                    except (ValueError, TypeError):
                        pass
                
                db_app = models.CTOSLoanApplication(
                    statement_id=statement_id,
                    application_date=app_date,
                    application_type=app_data.get('application_type'),
                    amount=app_data.get('amount'),
                    status=app_data.get('status'),
                    lender_name=app_data.get('lender_name')
                )
                db.add(db_app)
        
        # 8. Save Employment Info
        if result.get('employment_info'):
            emp_data = result['employment_info']
            existing = db.query(models.CTOSEmploymentInfo).filter(
                models.CTOSEmploymentInfo.statement_id == statement_id
            ).first()
            
            if existing:
                for key, value in emp_data.items():
                    if value is not None:
                        setattr(existing, key, value)
            else:
                db_emp = models.CTOSEmploymentInfo(
                    statement_id=statement_id,
                    has_directorships=emp_data.get('has_directorships', False),
                    directorships_count=emp_data.get('directorships_count', 0),
                    has_business_interests=emp_data.get('has_business_interests', False),
                    business_interests_count=emp_data.get('business_interests_count', 0)
                )
                db.add(db_emp)
        
        # 9. Save PTPTN Status
        if result.get('ptptn_status'):
            ptptn_data = result['ptptn_status']
            existing = db.query(models.CTOSPTPTNStatus).filter(
                models.CTOSPTPTNStatus.statement_id == statement_id
            ).first()
            
            if existing:
                for key, value in ptptn_data.items():
                    if value is not None:
                        setattr(existing, key, value)
            else:
                db_ptptn = models.CTOSPTPTNStatus(
                    statement_id=statement_id,
                    number_of_ptptn_loans=ptptn_data.get('number_of_ptptn_loans', 0),
                    local_lenders_count=ptptn_data.get('local_lenders_count', 0),
                    foreign_lenders_count=ptptn_data.get('foreign_lenders_count', 0)
                )
                db.add(db_ptptn)
        
        db.commit()
        logger.info(f"Successfully saved detailed CTOS data for statement {statement_id}")
        
    except Exception as e:
        logger.error(f"Error saving CTOS detailed data: {str(e)}", exc_info=True)
        db.rollback()
        raise


@router.post("/ctosstatement/process/{statement_id}")
def process_ctos_statement(
    statement_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """
    Process a CTOS credit report to extract credit score and information using AI
    
    This endpoint:
    1. Downloads the CTOS PDF from local storage
    2. Extracts credit score, period dates, and additional info using Gemini Vision AI
    3. Updates statement with extracted data
    
    Parameters:
    - statement_id: ID of the CTOS statement to process
    
    Note: Processing may take 10-30 seconds
    """
    
    # Get statement from database
    statement = db.query(models.Statement).filter(
        models.Statement.statement_id == statement_id,
        models.Statement.user_id == current_user.user_id,
        models.Statement.statement_type == "CTOS",
        models.Statement.is_deleted == False
    ).first()
    
    if not statement:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="CTOS statement not found"
        )
    
    # Prevent concurrent processing
    if statement.processing_status == 'extracting':
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="CTOS statement is currently being processed. Please wait for extraction to complete."
        )
    
    try:
        # Update status to extracting
        statement.processing_status = 'extracting'
        statement.processing_error = None
        db.commit()
        
        # Read PDF from local storage
        statement_url = statement.statement_url
        if statement_url.startswith(BASE_URL):
            relative_path = statement_url.replace(f"{BASE_URL}/files/", "")
            file_path = os.path.join(UPLOAD_DIR, relative_path)
        elif statement_url.startswith("/files/"):
            file_path = os.path.join(UPLOAD_DIR, statement_url.replace("/files/", ""))
        else:
            file_path = statement_url
        
        if not os.path.exists(file_path):
            raise HTTPException(
                status_code=404,
                detail=f"CTOS file not found: {file_path}"
            )
        
        # Read PDF bytes
        with open(file_path, "rb") as f:
            pdf_bytes = f.read()
        
        # Process CTOS PDF with AI
        result = process_ctos_pdf(pdf_bytes)
        
        if result.get('success'):
            # Update statement with extracted data (legacy fields for backward compatibility)
            if result.get('credit_score') is not None:
                statement.credit_score = result['credit_score']
            
            if result.get('score_text'):
                statement.score_text = result['score_text']
            
            # Parse and set period dates
            if result.get('period_start'):
                try:
                    statement.period_start = datetime.strptime(result['period_start'], '%Y-%m-%d').date()
                except ValueError:
                    logger.warning(f"Invalid period_start date format: {result['period_start']}")
            
            if result.get('period_end'):
                try:
                    statement.period_end = datetime.strptime(result['period_end'], '%Y-%m-%d').date()
                except ValueError:
                    logger.warning(f"Invalid period_end date format: {result['period_end']}")
            
            # Store full extracted data in JSON for reference
            statement.extracted_data = {
                "report_date": result.get('report_date'),
                "personal_info": result.get('personal_info'),
                "ctos_score": result.get('ctos_score'),
                "legal_records": result.get('legal_records'),
                "credit_facility_summary": result.get('credit_facility_summary'),
                "credit_facilities": result.get('credit_facilities', []),
                "credit_utilisation": result.get('credit_utilisation'),
                "loan_applications": result.get('loan_applications', []),
                "employment_info": result.get('employment_info'),
                "ptptn_status": result.get('ptptn_status'),
            }
            
            # Save detailed CTOS data to dedicated database models
            save_ctos_detailed_data(statement_id, result, db)
            
            # Optionally update user profile with extracted user info (only if fields are missing)
            if result.get('personal_info'):
                personal_info = result['personal_info']
                # Map personal_info to user_info format for backward compatibility
                user_info = {
                    "full_name": personal_info.get('full_name'),
                    "ic_number": personal_info.get('ic_nric'),
                    "date_of_birth": personal_info.get('date_of_birth'),
                    "address": personal_info.get('address_line1')
                }
                update_user_from_extracted_info(current_user, user_info, db)
            
            statement.processing_status = 'extracted'
            statement.processing_error = None
            statement.last_processed = datetime.now(timezone.utc)
            logger.info(f"Successfully extracted CTOS data: score={result.get('credit_score')}, period={result.get('period_start')} to {result.get('period_end')}")
        else:
            # Extraction failed
            statement.processing_status = 'failed'
            statement.processing_error = result.get('error', 'Unknown error during extraction')
            logger.error(f"CTOS extraction failed: {result.get('error')}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to extract CTOS data: {result.get('error')}"
            )
        
        db.commit()
        db.refresh(statement)
        
        return statement
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during CTOS processing: {str(e)}", exc_info=True)
        statement.processing_status = 'failed'
        statement.processing_error = f"Error during processing: {str(e)}"
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process CTOS statement: {str(e)}"
        )


# ============================================

# ===========================
# Statement Processing Endpoints
# ===========================

@router.post("/statement/process/{statement_id}")
def process_statement(
    statement_id: int,
    force_reimport: bool = False,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """
    Process a bank statement to extract transactions using AI

    This endpoint:
    1. Downloads the statement PDF from local storage
    2. Extracts transactions using Gemini Vision AI with improved location extraction
    3. Creates Income/Expense records in database
    4. Updates statement with period dates and status

    Parameters:
    - statement_id: ID of the statement to process
    - force_reimport: If True, deletes existing transactions and re-imports from scratch.
                      Useful for rescanning statements with improved AI extraction.

    Note: Processing may take 30-120 seconds for multi-page statements
    """

    # Get statement from database
    statement = db.query(models.Statement).filter(
        models.Statement.statement_id == statement_id,
        models.Statement.user_id == current_user.user_id,
        models.Statement.is_deleted == False
    ).first()

    if not statement:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Statement not found"
        )

    # Only process bank/credit/ewallet statements
    if statement.statement_type.lower() not in ['bank', 'credit_card', 'ewallet']:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot process {statement.statement_type} statements. Only bank, credit_card, and ewallet are supported."
        )

    # Prevent concurrent processing
    if statement.processing_status == 'extracting':
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Statement is currently being processed. Please wait for extraction to complete."
        )

    try:
        # Check if we have cached extraction data
        if statement.extracted_data:
            # Use cached data (no need to call Gemini again!)
            result = statement.extracted_data
            logger.info(f"Using cached extraction data for statement {statement_id}")
        else:
            # No cache - need to extract with Gemini
            logger.info(f"No cache found. Extracting with Gemini for statement {statement_id}")

            # Update status to extracting
            statement.processing_status = 'extracting'
            db.commit()

            # Extract file path from URL
            statement_url = statement.statement_url
            
            # Convert URL to local file path
            if statement_url.startswith(BASE_URL):
                relative_path = statement_url.replace(f"{BASE_URL}/files/", "")
                file_path = os.path.join(UPLOAD_DIR, relative_path)
            elif statement_url.startswith("/files/"):
                file_path = os.path.join(UPLOAD_DIR, statement_url.replace("/files/", ""))
            else:
                file_path = statement_url
            
            # Read PDF from local storage
            if not os.path.exists(file_path):
                raise HTTPException(
                    status_code=404,
                    detail=f"Statement file not found: {file_path}"
                )
            
            with open(file_path, "rb") as f:
                pdf_bytes = f.read()

            # Process the statement with AI
            result = process_statement_pdf(pdf_bytes)

            if not result['success']:
                # Update status to failed
                statement.processing_status = 'failed'
                statement.processing_error = "Failed to extract transactions from statement"
                statement.last_processed = datetime.now()
                db.commit()

                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to extract transactions from statement"
                )

            # Cache the extraction result (including user_info)
            statement.extracted_data = result
            statement.processing_status = 'extracted'
            statement.processing_error = None
            statement.last_processed = datetime.now()
            db.commit()
            
            # Optionally update user profile with extracted user info (only if fields are missing)
            if result.get('user_info'):
                user_info = result['user_info']
                update_user_from_extracted_info(current_user, user_info, db)

        # Update statement with extracted period dates
        if result.get('statement_period'):
            period = result['statement_period']
            logger.info(f"Extracted statement_period: {period}")

            if period.get('start_date'):
                statement.period_start = datetime.strptime(
                    period['start_date'],
                    '%Y-%m-%d'
                ).date()
                logger.info(f"Set period_start to: {statement.period_start}")

            if period.get('end_date'):
                statement.period_end = datetime.strptime(
                    period['end_date'],
                    '%Y-%m-%d'
                ).date()
                logger.info(f"Set period_end to: {statement.period_end}")
            else:
                logger.warning(f"end_date not extracted by AI (kept existing value: {statement.period_end})")

            # Commit period dates immediately to ensure they're saved
            db.commit()
            logger.info("Committed period dates to database")

        # Auto-create account from statement if not exists
        target_account = None

        if result.get('account_info'):
            account_info = result['account_info']

            # Check if account already exists by account number
            if account_info.get('account_number'):
                existing_account = db.query(models.Account).filter(
                    models.Account.user_id == current_user.user_id,
                    models.Account.account_no == account_info['account_number'],
                    models.Account.is_deleted == False
                ).first()

                if existing_account:
                    target_account = existing_account
                    # Use existing account_name instead of AI-generated one to maintain consistency
                    account_info['account_name'] = existing_account.account_name
                    logger.info(f"Found existing account, using stored name: {existing_account.account_name}")

            # If no account exists, create new one
            if not target_account:
                # Map extracted account type to standard type + subtype
                extracted_type = account_info.get('account_type', '')
                standard_type, subtype = map_account_type(extracted_type)

                # Use AI-generated account_name if provided, otherwise generate default
                account_name = account_info.get('account_name') or f"{extracted_type} Account"

                # Create new account
                new_account = models.Account(
                    user_id=current_user.user_id,
                    account_no=account_info.get('account_number', ''),
                    account_name=account_name,
                    account_type=standard_type,
                    account_subtype=subtype,
                    is_deleted=False
                )
                db.add(new_account)
                db.flush()  # Get the account_id
                target_account = new_account
                logger.info(f"Created new account with name: {account_name}")

        # Update account balance and create balance snapshot
        logger.info(f"Checking balance update: target_account={target_account is not None}, closing_balance={result.get('closing_balance')}")

        if target_account and result.get('closing_balance') is not None:
            closing_balance = result['closing_balance']

            # Update current balance on account
            target_account.account_balance = closing_balance
            logger.info(f"Updated account balance to: {closing_balance} for account_id={target_account.account_id}")

            # Create balance snapshot if we have statement period end date
            if statement.period_end:
                # Check if snapshot already exists for this date
                existing_snapshot = db.query(models.AccountBalanceSnapshot).filter(
                    models.AccountBalanceSnapshot.account_id == target_account.account_id,
                    models.AccountBalanceSnapshot.snapshot_date == statement.period_end,
                    models.AccountBalanceSnapshot.is_deleted == False
                ).first()

                if existing_snapshot:
                    # Update existing snapshot
                    existing_snapshot.closing_balance = closing_balance
                    logger.info(f"Updated existing balance snapshot for {statement.period_end}: {closing_balance}")
                else:
                    # Create new snapshot
                    snapshot = models.AccountBalanceSnapshot(
                        account_id=target_account.account_id,
                        snapshot_date=statement.period_end,
                        closing_balance=closing_balance,
                        is_deleted=False
                    )
                    db.add(snapshot)
                    logger.info(f"Created balance snapshot for {statement.period_end}: {closing_balance}")

            db.commit()

        logger.info("Balance update completed, now checking for existing transactions...")

        # Check if statement was already processed (prevent double-import)
        existing_transaction_count = 0

        # Count existing transactions linked to this statement
        logger.info(f"Querying for existing incomes with statement_id={statement.statement_id}")
        existing_incomes = db.query(models.Income).filter(
            models.Income.statement_id == statement.statement_id,
            models.Income.is_deleted == False
        ).count()

        existing_expenses = db.query(models.Expense).filter(
            models.Expense.statement_id == statement.statement_id,
            models.Expense.is_deleted == False
        ).count()

        existing_transaction_count = existing_incomes + existing_expenses
        logger.info(f"Found {existing_transaction_count} existing transactions")

        # Skip duplicate check when using cached data - the frontend already imported transactions
        # Only enforce this check when doing a fresh extraction (not using cache)
        using_cache = statement.extracted_data is not None

        if existing_transaction_count > 0 and not force_reimport and not using_cache:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Statement already processed. {existing_transaction_count} transactions already imported from this statement. Use force_reimport=true to delete and re-import, or delete transactions manually first."
            )
        elif existing_transaction_count > 0:
            logger.info(f"Statement has {existing_transaction_count} transactions already imported. Skipping transaction creation, proceeding with balance reconciliation.")

        # If force_reimport is true, delete existing transactions
        if force_reimport and existing_transaction_count > 0:
            # Soft delete existing transactions
            db.query(models.Income).filter(
                models.Income.statement_id == statement.statement_id,
                models.Income.is_deleted == False
            ).update({"is_deleted": True})

            db.query(models.Expense).filter(
                models.Expense.statement_id == statement.statement_id,
                models.Expense.is_deleted == False
            ).update({"is_deleted": True})

            db.commit()
            logger.info(f"Force re-import: Deleted {existing_transaction_count} existing transactions from statement {statement_id}")

        # Create Income/Expense/Transfer records from transactions
        created_incomes = 0
        created_expenses = 0
        created_transfers = 0
        skipped = 0
        duplicates_removed = 0
        neutralized_transfers = 0

        # Helper function to find and remove duplicate transactions
        def remove_duplicate_transaction(amount, date, description, account_id, reference_no, is_income):
            """
            Find and remove duplicate transactions with similar attributes.
            Returns True if a duplicate was found and removed, False otherwise.
            Uses multiple matching strategies:
            1. Reference number match (strongest indicator)
            2. Amount + date + description match
            3. Amount + date + account match (if accounts match)
            """
            # Normalize description for comparison (take first 50 chars, lowercase, strip)
            desc_normalized = description[:50].lower().strip() if description else ""
            
            # Allow 1 day tolerance for date matching (statements might have slight date variations)
            date_tolerance = timedelta(days=1)
            date_start = date - date_tolerance
            date_end = date + date_tolerance
            
            if is_income:
                # First check: If reference number exists, check for exact match
                if reference_no:
                    existing_by_ref = db.query(models.Income).filter(
                        models.Income.user_id == current_user.user_id,
                        models.Income.reference_no == reference_no,
                        models.Income.is_deleted == False
                    ).first()
                    if existing_by_ref:
                        # Soft delete the duplicate
                        existing_by_ref.is_deleted = True
                        logger.info(f"Removed duplicate income by reference: {existing_by_ref.description} - {existing_by_ref.amount} on {existing_by_ref.date_received}")
                        return True
                
                # Second check: Amount + date + description match
                existing = db.query(models.Income).filter(
                    models.Income.user_id == current_user.user_id,
                    models.Income.amount == amount,
                    models.Income.date_received >= date_start,
                    models.Income.date_received <= date_end,
                    models.Income.is_deleted == False
                ).all()
                
                # Check if description matches (fuzzy match on first 50 chars)
                for existing_income in existing:
                    existing_desc = (existing_income.description or "")[:50].lower().strip()
                    should_remove = False
                    
                    if desc_normalized and existing_desc:
                        # Check if descriptions are similar (substring match)
                        if desc_normalized in existing_desc or existing_desc in desc_normalized:
                            should_remove = True
                    
                    # Also check if account matches (if both have accounts) - strong indicator
                    if account_id and existing_income.account_id == account_id:
                        should_remove = True
                    
                    if should_remove:
                        # Soft delete the duplicate
                        existing_income.is_deleted = True
                        logger.info(f"Removed duplicate income: {existing_income.description} - {existing_income.amount} on {existing_income.date_received}")
                        return True
            else:
                # First check: If reference number exists, check for exact match
                if reference_no:
                    existing_by_ref = db.query(models.Expense).filter(
                        models.Expense.user_id == current_user.user_id,
                        models.Expense.reference_no == reference_no,
                        models.Expense.is_deleted == False
                    ).first()
                    if existing_by_ref:
                        # Soft delete the duplicate
                        existing_by_ref.is_deleted = True
                        logger.info(f"Removed duplicate expense by reference: {existing_by_ref.description} - {existing_by_ref.amount} on {existing_by_ref.date_spent}")
                        return True
                
                # Second check: Amount + date + description match
                existing = db.query(models.Expense).filter(
                    models.Expense.user_id == current_user.user_id,
                    models.Expense.amount == amount,
                    models.Expense.date_spent >= date_start,
                    models.Expense.date_spent <= date_end,
                    models.Expense.is_deleted == False
                ).all()
                
                # Check if description matches (fuzzy match on first 50 chars)
                for existing_expense in existing:
                    existing_desc = (existing_expense.description or "")[:50].lower().strip()
                    should_remove = False
                    
                    if desc_normalized and existing_desc:
                        # Check if descriptions are similar (substring match)
                        if desc_normalized in existing_desc or existing_desc in desc_normalized:
                            should_remove = True
                    
                    # Also check if account matches (if both have accounts) - strong indicator
                    if account_id and existing_expense.account_id == account_id:
                        should_remove = True
                    
                    if should_remove:
                        # Soft delete the duplicate
                        existing_expense.is_deleted = True
                        logger.info(f"Removed duplicate expense: {existing_expense.description} - {existing_expense.amount} on {existing_expense.date_spent}")
                        return True
            
            return False

        # Helper function to check if transfer should be neutralized (not counted as expense/income)
        def should_neutralize_transfer(txn: dict, user_accounts: list) -> bool:
            """
            Determine if a transfer transaction should be neutralized (not counted as expense/income).
            Returns True if transfer is to own account/savings, False if to another person.
            
            Args:
                txn: Transaction dictionary with description, transfer_type, etc.
                user_accounts: List of user's account objects with account_name, account_no
            """
            description = (txn.get('description') or '').lower()
            transfer_type = txn.get('transfer_type')
            
            # If AI classified it as intra_person, neutralize it
            if transfer_type == 'intra_person':
                logger.info(f"Neutralizing intra-person transfer (AI classified): {txn.get('description', 'Unknown')}")
                return True
            
            # If AI classified it as inter_person, don't neutralize
            if transfer_type == 'inter_person':
                return False
            
            # Fallback: Use keyword matching if AI didn't classify
            # Check for intra-person transfer keywords
            intra_person_keywords = [
                'savings', 'saving', 'own account', 'self transfer', 'internal transfer',
                'tabung', 'asb', 'sspni', 'ssp1m', 'stash', 'goal transfer', 'auto-save',
                'autosave', 'auto save', 'standing instruction', 'top up', 'top-up', 'topup',
                'cash deposit', 'deposit to', 'rainy day', 'investment account', 'duitnow to self'
            ]
            
            # Check if description contains intra-person keywords
            if any(keyword in description for keyword in intra_person_keywords):
                logger.info(f"Neutralizing intra-person transfer (keyword match): {txn.get('description', 'Unknown')}")
                return True
            
            # Check if transfer is to one of user's own accounts
            # Extract account number/name from description if possible
            if user_accounts:
                for account in user_accounts:
                    account_name = (account.account_name or '').lower()
                    account_no = (account.account_no or '').lower()
                    
                    # Check if description mentions user's account name or number
                    if account_name and account_name in description:
                        logger.info(f"Neutralizing transfer to own account (account name match): {txn.get('description', 'Unknown')}")
                        return True
                    if account_no and account_no in description:
                        logger.info(f"Neutralizing transfer to own account (account number match): {txn.get('description', 'Unknown')}")
                        return True
            
            # Default: if it's a transfer but we can't determine, don't neutralize (treat as expense/income)
            # This is conservative - better to count uncertain transfers than miss real expenses
            return False

        # Get user's accounts for transfer checking
        user_accounts = db.query(models.Account).filter(
            models.Account.user_id == current_user.user_id
        ).all()

        # Only create transactions if:
        # 1. Extraction data contains them
        # 2. No existing transactions (prevent duplicates)
        # 3. OR force_reimport is true (re-import after deletion)
        should_create_transactions = (
            'transactions' in result
            and result['transactions']
            and (existing_transaction_count == 0 or force_reimport)
        )

        if should_create_transactions:
            logger.info(f"Creating transactions from extraction data ({len(result['transactions'])} transactions)")
            for txn in result['transactions']:
                # Validate transaction
                if not txn.get('date') or not txn.get('amount') or not txn.get('description'):
                    skipped += 1
                    continue

                # Parse transaction date
                try:
                    txn_date = datetime.strptime(txn['date'], '%Y-%m-%d').date()
                except:
                    skipped += 1
                    continue

                amount = abs(txn['amount'])
                description = txn['description'][:255]  # Limit length
                account_id = target_account.account_id if target_account else None
                reference_no = txn.get('reference', '')

                # Check if this is a transfer that should be neutralized
                # First check if AI classified it as intra_person transfer
                transfer_type = txn.get('transfer_type')
                is_intra_person = transfer_type == 'intra_person'
                should_neutralize = is_intra_person or should_neutralize_transfer(txn, user_accounts)
                
                if should_neutralize:
                    neutralized_transfers += 1
                    # Ensure transfer_type is set correctly
                    if not transfer_type:
                        transfer_type = 'intra_person'
                    
                    # Force category to "Transfer" for intra_person transfers (not "Other")
                    # If AI categorized it as "Other" but it's an intra_person transfer, override to "Transfer"
                    category = txn.get('category', 'Transfer')
                    if transfer_type == 'intra_person' and category == 'Other':
                        category = 'Transfer'
                        logger.info(f"Overriding category from 'Other' to 'Transfer' for intra_person transfer: {description}")
                    
                    # Create Transfer record instead of expense/income
                    transfer = models.Transfer(
                        user_id=current_user.user_id,
                        account_id=account_id,
                        statement_id=statement.statement_id,
                        amount=amount,
                        description=description,
                        category=category,  # "Transfer" for intra_person, not "Other"
                        transfer_type=transfer_type,
                        date_transferred=txn_date,
                        recipient_account_name=None,  # Could be extracted from description if available
                        recipient_account_no=None,  # Could be extracted from description if available
                        reference_no=reference_no,
                        is_deleted=False,
                        created=datetime.now(timezone.utc)
                    )
                    db.add(transfer)
                    created_transfers += 1
                    logger.info(f"Created transfer record (neutralized): {description} - {amount} on {txn_date} (type: {transfer_type}, category: {category})")
                    continue

                # Determine if income or expense
                if txn['type'] == 'credit' and txn['amount'] > 0:
                    # Check for and remove duplicate before creating
                    if remove_duplicate_transaction(amount, txn_date, description, account_id, reference_no, is_income=True):
                        duplicates_removed += 1
                        logger.info(f"Removed duplicate income, creating new from statement: {description} - {amount} on {txn_date} (ref: {reference_no})")
                    
                    # Create Income record
                    income = models.Income(
                        user_id=current_user.user_id,
                        account_id=account_id,
                        statement_id=statement.statement_id,
                        amount=amount,
                        description=description,
                        category=txn.get('category', 'Other'),
                        date_received=txn_date,
                        payer=txn.get('payer', ''),
                        reference_no=txn.get('reference', ''),
                        is_deleted=False,
                        created=datetime.now(timezone.utc)
                    )
                    db.add(income)
                    created_incomes += 1

                elif txn['type'] == 'debit' and txn['amount'] < 0:
                    # Check for and remove duplicate before creating
                    if remove_duplicate_transaction(amount, txn_date, description, account_id, reference_no, is_income=False):
                        duplicates_removed += 1
                        logger.info(f"Removed duplicate expense, creating new from statement: {description} - {amount} on {txn_date} (ref: {reference_no})")
                    
                    # Infer expense_type (wants vs needs) based on category, description, and amount
                    # Note: inter_person transfers (transfers to others) are expenses and should be categorized
                    # Only intra_person transfers are neutralized (handled above)
                    inferred_expense_type = infer_expense_type(
                        category=txn.get('category', 'Other'),
                        description=description,
                        amount=abs(amount)  # Use absolute value for categorization
                    )
                    # Default to 'needs' if inference returns None
                    # This should only happen for actual transfers (which are handled above) or edge cases
                    expense_type = inferred_expense_type if inferred_expense_type else 'needs'
                    
                    # Create Expense record
                    expense = models.Expense(
                        user_id=current_user.user_id,
                        account_id=account_id,
                        statement_id=statement.statement_id,
                        amount=amount,
                        description=description,
                        category=txn.get('category', 'Other'),
                        expense_type=expense_type,
                        date_spent=txn_date,
                        seller=txn.get('seller', ''),
                        location=txn.get('location', ''),
                        reference_no=txn.get('reference', ''),
                        tax_amount=0.0,
                        tax_deductible=False,
                        is_reimbursable=False,
                        is_deleted=False,
                        created=datetime.now(timezone.utc)
                    )
                    db.add(expense)
                    created_expenses += 1
        else:
            logger.info(f"Skipping transaction creation - {existing_transaction_count} transactions already exist from this statement")

        # Update processing status to imported
        statement.processing_status = 'imported'
        statement.processing_error = None

        # Balance reconciliation check
        reconciliation_info = None
        try:
            logger.info("Starting balance reconciliation...")
            if result.get('opening_balance') is not None and result.get('closing_balance') is not None:
                opening_balance = result['opening_balance']
                closing_balance = result['closing_balance']
                logger.info(f"Balances from extraction: opening={opening_balance}, closing={closing_balance}")

                # Calculate total income and expense from DATABASE transactions (not extraction data)
                # This allows reconciliation to work even when called after manual import
                total_income = db.query(func.sum(models.Income.amount)).filter(
                    models.Income.statement_id == statement_id,
                    models.Income.is_deleted == False
                ).scalar() or 0.0

                total_expenses = db.query(func.sum(models.Expense.amount)).filter(
                    models.Expense.statement_id == statement_id,
                    models.Expense.is_deleted == False
                ).scalar() or 0.0

                logger.info(f"Totals from database: income={total_income}, expenses={total_expenses}")

                # Calculate expected closing balance
                calculated_balance = opening_balance + total_income - total_expenses

                # Calculate difference (tolerance of 0.01 for rounding)
                difference = closing_balance - calculated_balance
                matches = abs(difference) <= 0.01

                reconciliation_info = {
                    "extracted_opening_balance": opening_balance,
                    "extracted_closing_balance": closing_balance,
                    "calculated_closing_balance": round(calculated_balance, 2),
                    "total_income": round(total_income, 2),
                    "total_expenses": round(total_expenses, 2),
                    "difference": round(difference, 2),
                    "matches": matches
                }

                # Log reconciliation results
                logger.info(f"Balance reconciliation: Opening={opening_balance}, Income={total_income:.2f}, Expenses={total_expenses:.2f}")
                logger.info(f"Calculated closing balance: {calculated_balance:.2f} vs Extracted: {closing_balance} (Diff: {difference:.2f})")

                if not matches:
                    logger.warning(f"⚠️  BALANCE MISMATCH! Difference of {difference:.2f} detected.")
                    logger.warning(f"Possible causes: Missing transactions, incorrect amounts, or statement errors")
                    # Temporarily disabled: Don't show warning to user, but keep logging
                    # reconciliation_info["warning"] = f"Balance discrepancy of {difference:.2f} detected. Some transactions may be missing or have incorrect amounts."
                else:
                    logger.info("✓ Balance reconciliation successful - transactions match statement balance")
            else:
                logger.info("Skipping reconciliation - no balance data available")
        except Exception as recon_error:
            logger.error(f"Error during reconciliation: {str(recon_error)}", exc_info=True)
            # Don't fail the entire request if reconciliation fails
            reconciliation_info = None

        # Update credit card information if this is a credit card statement
        if target_account and target_account.card_id:
            try:
                logger.info(f"Updating credit card info for card_id={target_account.card_id}")
                credit_card = db.query(models.UserCreditCard).filter(
                    models.UserCreditCard.card_id == target_account.card_id,
                    models.UserCreditCard.is_deleted == False
                ).first()

                if credit_card:
                    # Update current balance from statement closing balance
                    if result.get('closing_balance') is not None:
                        credit_card.current_balance = abs(result['closing_balance'])
                        logger.info(f"Updated credit card balance to {credit_card.current_balance}")

                    # Extract and update payment due date from credit card terms
                    credit_card_terms = result.get('credit_card_terms', {})
                    if credit_card_terms:
                        # Try to get payment due date
                        payment_due_date_str = credit_card_terms.get('payment_due_date') or credit_card_terms.get('due_date')
                        if payment_due_date_str:
                            try:
                                # Parse date string - try multiple formats
                                for date_format in ['%Y-%m-%d', '%d/%m/%Y', '%d-%m-%Y']:
                                    try:
                                        payment_due_date = datetime.strptime(payment_due_date_str, date_format).date()
                                        credit_card.next_payment_date = payment_due_date
                                        logger.info(f"Updated next payment date to {payment_due_date}")
                                        break
                                    except ValueError:
                                        continue
                            except Exception as date_error:
                                logger.warning(f"Could not parse payment due date: {payment_due_date_str} - {date_error}")

                        # Try to get minimum payment amount
                        min_payment_str = credit_card_terms.get('minimum_payment') or credit_card_terms.get('minimum_payment_amount')
                        if min_payment_str:
                            try:
                                # Parse numeric value (handle strings like "RM 50.00" or "50.00")
                                min_payment_clean = str(min_payment_str).replace('RM', '').replace(',', '').strip()
                                min_payment = abs(float(min_payment_clean))
                                credit_card.next_payment_amount = min_payment
                                logger.info(f"Updated next payment amount to {min_payment}")
                            except Exception as amount_error:
                                logger.warning(f"Could not parse minimum payment: {min_payment_str} - {amount_error}")

                    # If no terms data, try to get from credit card summary
                    credit_card_summary = result.get('credit_card_summary', {})
                    if credit_card_summary and not credit_card.next_payment_amount:
                        total_due_str = credit_card_summary.get('total_amount_due') or credit_card_summary.get('outstanding_balance')
                        if total_due_str:
                            try:
                                total_due_clean = str(total_due_str).replace('RM', '').replace(',', '').strip()
                                total_due = abs(float(total_due_clean))
                                credit_card.next_payment_amount = total_due
                                logger.info(f"Updated next payment amount from summary to {total_due}")
                            except Exception as amount_error:
                                logger.warning(f"Could not parse total due: {total_due_str} - {amount_error}")

                    logger.info(f"Credit card {credit_card.card_id} updated successfully")
                else:
                    logger.warning(f"Credit card with card_id={target_account.card_id} not found")
            except Exception as card_error:
                logger.error(f"Error updating credit card: {str(card_error)}", exc_info=True)
                # Don't fail the entire request if card update fails
        elif target_account:
            # For non-credit-card statements, check if account is linked to a credit card
            logger.info(f"Checking if account {target_account.account_id} is linked to a credit card")
            closing_balance = result.get('closing_balance') if result else None
            logger.info(f"Extracted closing_balance: {closing_balance}")
            update_credit_card_balance(db, target_account.account_id, current_user.user_id, closing_balance)

        # Commit all changes
        db.commit()
        db.refresh(statement)

        # Prepare account info for response
        account_response = None
        if target_account:
            account_response = {
                "account_id": target_account.account_id,
                "account_name": target_account.account_name,
                "account_number": target_account.account_no,
                "account_type": target_account.account_type,
                "account_subtype": target_account.account_subtype,
                "was_auto_created": target_account.account_id not in [acc.account_id for acc in current_user.accounts]
            }

        response_data = {
            "success": True,
            "message": "Statement processed successfully",
            "statement_id": statement_id,
            "processing_status": statement.processing_status,
            "summary": {
                "total_pages": result.get('total_pages', 0),
                "total_transactions_extracted": result.get('total_transactions', 0),
                "incomes_created": created_incomes,
                "expenses_created": created_expenses,
                "transfers_created": created_transfers,
                "duplicates_removed": duplicates_removed,
                "neutralized_transfers": neutralized_transfers,
                "skipped": skipped,
                "errors": result.get('errors')
            },
            "statement_period": result.get('statement_period'),
            "account": account_response,
            "opening_balance": result.get('opening_balance'),
            "closing_balance": result.get('closing_balance')
        }

        # Add reconciliation info if available
        if reconciliation_info:
            response_data["reconciliation"] = reconciliation_info

        # Include credit card terms & summary when available
        if result.get("credit_card_terms"):
            response_data["credit_card_terms"] = result["credit_card_terms"]
            if result.get("credit_card_summary"):
                response_data["credit_card_summary"] = result["credit_card_summary"]

        return response_data

    except HTTPException:
        raise
    except Exception as e:
        # Log the full error with traceback
        logger.error(f"CRITICAL ERROR in process_statement: {str(e)}", exc_info=True)

        # Update status to failed
        statement.processing_status = 'failed'
        statement.processing_error = str(e)
        statement.last_processed = datetime.now()
        db.commit()

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process statement: {str(e)}"
        )

@router.post("/statement/preview/{statement_id}")
def preview_statement(
    statement_id: int,
    force_refresh: bool = Query(False, description="Force re-extraction even if cached data exists"),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """
    Preview extracted transactions from a bank statement WITHOUT saving to database

    This endpoint:
    1. Checks if cached extraction data exists in database
    2. If cached and not force_refresh: returns cached data (FAST, no Gemini API call)
    3. If no cache or force_refresh: downloads PDF and extracts with Gemini AI
    4. Caches the extraction result for future requests
    5. Does NOT create income/expense records

    Parameters:
    - force_refresh: Set to true to bypass cache and re-extract with Gemini

    Benefits of caching:
    - Faster response times (no file download or AI processing)
    - Reduced Gemini API costs
    - Consistent results when viewing multiple times
    """

    # Get statement from database
    statement = db.query(models.Statement).filter(
        models.Statement.statement_id == statement_id,
        models.Statement.user_id == current_user.user_id,
        models.Statement.is_deleted == False
    ).first()

    if not statement:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Statement not found"
        )

    # Only process bank/credit/ewallet statements
    if statement.statement_type.lower() not in ['bank', 'credit_card', 'ewallet']:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot process {statement.statement_type} statements. Only bank, credit_card, and ewallet are supported."
        )

    # Check if we have cached extraction data
    if statement.extracted_data and not force_refresh:
        # Return cached data (no Gemini API call needed!)
        cached_result = statement.extracted_data
        return {
            "success": True,
            "statement_id": statement_id,
            "statement_name": statement.statement_url.split('/')[-1],
            "statement_type": statement.statement_type,
            "summary": {
                "total_pages": cached_result.get('total_pages'),
                "total_transactions": cached_result.get('total_transactions'),
                "errors": cached_result.get('errors')
            },
            "statement_period": cached_result.get('statement_period'),
            "account_info": cached_result.get('account_info'),
            "opening_balance": cached_result.get('opening_balance'),
            "closing_balance": cached_result.get('closing_balance'),
            "credit_card_terms": cached_result.get('credit_card_terms'),
            "credit_card_summary": cached_result.get('credit_card_summary'),
            "transactions": cached_result.get('transactions', []),
            "message": "Returning cached extraction data. Set force_refresh=true to re-extract.",
            "cached": True,
            "last_processed": statement.last_processed.isoformat() if statement.last_processed else None
        }

    # No cache or force refresh - need to extract with Gemini
    try:
        # Update status to extracting
        statement.processing_status = 'extracting'
        db.commit()

        # Extract file path from URL
        statement_url = statement.statement_url
        
        # Convert URL to local file path
        if statement_url.startswith(BASE_URL):
            relative_path = statement_url.replace(f"{BASE_URL}/files/", "")
            file_path = os.path.join(UPLOAD_DIR, relative_path)
        elif statement_url.startswith("/files/"):
            file_path = os.path.join(UPLOAD_DIR, statement_url.replace("/files/", ""))
        else:
            file_path = statement_url
        
        # Read PDF from local storage
        if not os.path.exists(file_path):
            raise HTTPException(
                status_code=404,
                detail=f"Statement file not found: {file_path}"
            )
        
        with open(file_path, "rb") as f:
            pdf_bytes = f.read()

        # Process the statement with AI (extract only, don't save)
        result = process_statement_pdf(pdf_bytes)

        if not result['success']:
            # Update status to failed
            statement.processing_status = 'failed'
            statement.processing_error = "Failed to extract transactions from statement"
            statement.last_processed = datetime.now()
            db.commit()

            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to extract transactions from statement"
            )

        # Cache the extraction result in database
        statement.extracted_data = result
        statement.processing_status = 'extracted'
        statement.processing_error = None
        statement.last_processed = datetime.now()

        # Update statement with extracted period dates
        if result.get('statement_period'):
            period = result['statement_period']
            logger.info(f"Extracted statement_period: {period}")

            if period.get('start_date'):
                statement.period_start = datetime.strptime(
                    period['start_date'],
                    '%Y-%m-%d'
                ).date()
                logger.info(f"Set period_start to: {statement.period_start}")

            if period.get('end_date'):
                statement.period_end = datetime.strptime(
                    period['end_date'],
                    '%Y-%m-%d'
                ).date()
                logger.info(f"Set period_end to: {statement.period_end}")
            else:
                logger.warning(f"end_date not extracted by AI (kept existing value: {statement.period_end})")

        # Generate descriptive display name from extracted data
        # Always update display name after extraction to use AI-extracted bank name and period
        account_info = result.get('account_info')

        # If account exists, use stored account_name instead of AI-generated one
        if account_info and account_info.get('account_number'):
            existing_account = db.query(models.Account).filter(
                models.Account.user_id == current_user.user_id,
                models.Account.account_no == account_info['account_number'],
                models.Account.is_deleted == False
            ).first()

            if existing_account:
                account_info['account_name'] = existing_account.account_name
                logger.info(f"Using existing account name: {existing_account.account_name}")

        logger.info(f"Account info from extraction: {account_info}")
        logger.info(f"Period: {statement.period_start} to {statement.period_end}")

        statement.display_name = generate_display_name(
            statement_type=statement.statement_type,
            account_info=account_info,
            period_start=statement.period_start,
            period_end=statement.period_end,
            original_filename=statement.display_name or statement_url.split('/')[-1]
        )
        logger.info(f"Generated display_name: {statement.display_name}")

        db.commit()
        logger.info("Committed period dates and display name to database")

        # Return the extracted data
        return {
            "success": True,
            "statement_id": statement_id,
            "statement_name": statement_url.split('/')[-1],
            "statement_type": statement.statement_type,
            "summary": {
                "total_pages": result['total_pages'],
                "total_transactions": result['total_transactions'],
                "errors": result.get('errors')
            },
            "statement_period": result.get('statement_period'),
            "account_info": result.get('account_info'),
            "opening_balance": result.get('opening_balance'),
            "closing_balance": result.get('closing_balance'),
            "credit_card_terms": result.get('credit_card_terms'),
            "credit_card_summary": result.get('credit_card_summary'),
            "transactions": result.get('transactions', []),
            "message": "Transactions extracted successfully and cached. Review and use /process endpoint to import.",
            "cached": False,
            "last_processed": statement.last_processed.isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        # Update status to failed
        statement.processing_status = 'failed'
        statement.processing_error = str(e)
        statement.last_processed = datetime.now()
        db.commit()

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to preview statement: {str(e)}"
        )

@router.post("/statement/rescan/{statement_id}")
def rescan_statement(
    statement_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """
    Rescan/Reprocess a previously processed statement with improved AI extraction

    This is a convenience endpoint that:
    1. Automatically deletes previously imported transactions from this statement
    2. Re-extracts transactions using the latest AI prompt with improved location extraction
    3. Imports the newly extracted transactions

    Use this when:
    - You want to take advantage of improved location extraction
    - The original extraction had errors or missing data
    - You've updated the AI prompt and want to re-extract

    This is equivalent to calling /process with force_reimport=true

    Note: This will delete all existing income/expense records associated with this statement
    before re-importing. Use with caution!
    """

    # Call the process_statement function with force_reimport=True
    return process_statement(
        statement_id=statement_id,
        force_reimport=True,
        db=db,
        current_user=current_user
    )


@router.get("/statement/{statement_id}/view")
async def view_statement_pdf(
    statement_id: int,
    token: str = Query(None, description="JWT token for authentication (alternative to Authorization header)"),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user_optional),
):
    """
    Serve statement PDF file from local storage with proper headers for viewing in browser.
    This endpoint serves the local file to avoid CORS issues.
    """
    # Authenticate user - try from query parameter if not from header
    user = current_user
    if not user and token:
        try:
            payload = verify_token(token)
            user_id = payload.get("user_id")
            if user_id:
                user = db.query(models.User).filter(
                    models.User.user_id == user_id,
                    models.User.is_deleted == False
                ).first()
        except HTTPException:
            pass
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    # Get statement and verify ownership
    statement = (
        db.query(models.Statement)
        .filter(
            models.Statement.statement_id == statement_id,
            models.Statement.user_id == user.user_id,
            models.Statement.is_deleted == False,
        )
        .first()
    )

    if not statement:
        raise HTTPException(status_code=404, detail="Statement not found")

    try:
        statement_url = statement.statement_url
        filename = statement.display_name or statement_url.split("/")[-1] or "statement.pdf"
        
        # Convert URL to local file path
        file_content = None
        content_type = 'application/pdf'
        
        if statement_url.startswith(BASE_URL) or statement_url.startswith("/files/"):
            # Local file storage URL
            if statement_url.startswith(BASE_URL):
                relative_path = statement_url.replace(f"{BASE_URL}/files/", "")
            else:
                relative_path = statement_url.replace("/files/", "")
            
            file_path = os.path.join(UPLOAD_DIR, relative_path)
            
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    file_content = f.read()
            else:
                raise HTTPException(
                    status_code=404,
                    detail=f"File not found: {file_path}"
                )
        elif statement_url.startswith(('http://', 'https://')):
            # Handle external HTTP/HTTPS URLs (for migration/backward compatibility)
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    http_response = await client.get(statement_url)
                    http_response.raise_for_status()
                    file_content = http_response.content
                    content_type = http_response.headers.get('Content-Type', 'application/pdf')
            except httpx.HTTPError as e:
                logger.error(f"Error fetching file from URL: {e}")
                raise HTTPException(
                    status_code=404,
                    detail=f"Failed to fetch file from URL: {statement_url}"
                )
        else:
            # Assume it's a local file path
            if os.path.exists(statement_url):
                with open(statement_url, 'rb') as f:
                    file_content = f.read()
            else:
                raise HTTPException(
                    status_code=404,
                    detail=f"File not found at path: {statement_url}"
                )

        if not file_content:
            raise HTTPException(
                status_code=404,
                detail="File content is empty"
            )

        # Return file with proper headers for PDF viewing
        return Response(
            content=file_content,
            media_type=content_type,
            headers={
                "Content-Disposition": f'inline; filename="{filename}"',
                "X-Content-Type-Options": "nosniff",
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving statement PDF: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to serve statement file: {str(e)}"
    )

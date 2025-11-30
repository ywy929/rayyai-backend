from fastapi import Depends, HTTPException, status, Header, UploadFile
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from sqlalchemy import func
from jose import JWTError, jwt
from datetime import datetime, timedelta, timezone
from typing import Optional
from pathlib import Path
import hashlib
import logging
import boto3
from botocore.exceptions import ClientError
import models
from database import get_db
from dotenv import load_dotenv
import os
load_dotenv()

logger = logging.getLogger(__name__)
# JWT Configuration
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES"))

# Security scheme
# Use auto_error=False to handle missing/invalid tokens ourselves
security = HTTPBearer(auto_error=False)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """
    Create a JWT access token.
    
    Args:
        data: Dictionary containing user data to encode in token
        expires_delta: Optional custom expiration time
    
    Returns:
        Encoded JWT token string
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str) -> dict:
    """
    Verify and decode a JWT token.
    
    Args:
        token: JWT token string
    
    Returns:
        Decoded token payload
    
    Raises:
        HTTPException: If token is invalid or expired
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Session = Depends(get_db)
) -> models.User:
    """
    Dependency for getting the current authenticated user from JWT token.
    
    This extracts the JWT token from the Authorization header,
    validates it, and returns the corresponding user from the database.
    
    Args:
        credentials: HTTP Bearer token from Authorization header (optional when auto_error=False)
        db: Database session
    
    Returns:
        User model instance
    
    Raises:
        HTTPException: If token is invalid or user not found
    """
    # Check if credentials were provided
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Extract token from credentials
    token = credentials.credentials
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Verify and decode token
    payload = verify_token(token)
    
    # Get user_id from token payload
    user_id: int = payload.get("user_id")
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Query user from database
    user = db.query(models.User).filter(
        models.User.user_id == user_id,
        models.User.is_deleted == False
    ).first()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return user

async def get_current_user_optional(
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db)
) -> Optional[models.User]:
    """
    Optional authentication dependency.
    Returns user if authenticated, None if not.
    
    Useful for endpoints that work differently for authenticated vs anonymous users.
    """
    if not authorization or not authorization.startswith("Bearer "):
        return None
    
    try:
        token = authorization.replace("Bearer ", "")
        payload = verify_token(token)
        user_id = payload.get("user_id")
        
        if user_id:
            user = db.query(models.User).filter(
                models.User.user_id == user_id,
                models.User.is_deleted == False
            ).first()
            return user
    except HTTPException:
        return None
    
    return None

def calculate_account_balance(db: Session, account_id: int) -> float:
    """
    Calculate current balance using latest snapshot + transactions since snapshot.
    Falls back to full calculation if no snapshot exists.
    
    This is much faster for accounts with many transactions!
    """
    # Validate account exists and is not deleted
    account = db.query(models.Account).filter(
        models.Account.account_id == account_id,
        models.Account.is_deleted == False
    ).first()
    
    if not account:
        raise ValueError(f"Account {account_id} not found or is deleted")
    
    # Get latest snapshot
    latest_snapshot = db.query(models.AccountBalanceSnapshot).filter(
        models.AccountBalanceSnapshot.account_id == account_id,
        models.AccountBalanceSnapshot.is_deleted == False
    ).order_by(models.AccountBalanceSnapshot.snapshot_date.desc()).first()
    
    if not latest_snapshot:
        # No snapshot exists, calculate from all transactions
        return _calculate_from_all_transactions(db, account_id)
    
    # Calculate transactions AFTER the snapshot date
    income_after = db.query(func.sum(models.Income.amount)).filter(
        models.Income.account_id == account_id,
        models.Income.is_deleted == False,
        models.Income.date_received > latest_snapshot.snapshot_date
    ).scalar() or 0.0
    
    expense_after = db.query(func.sum(models.Expense.amount)).filter(
        models.Expense.account_id == account_id,
        models.Expense.is_deleted == False,
        models.Expense.date_spent > latest_snapshot.snapshot_date
    ).scalar() or 0.0
    
    # Snapshot balance + new income - new expenses
    return latest_snapshot.closing_balance + income_after - expense_after


def _calculate_from_all_transactions(db: Session, account_id: int) -> float:
    """
    Fallback: Calculate balance from all transactions.
    Used when no snapshot exists.
    """
    total_income = db.query(func.sum(models.Income.amount)).filter(
        models.Income.account_id == account_id,
        models.Income.is_deleted == False
    ).scalar() or 0.0
    
    total_expense = db.query(func.sum(models.Expense.amount)).filter(
        models.Expense.account_id == account_id,
        models.Expense.is_deleted == False
    ).scalar() or 0.0

    return total_income - total_expense


def map_account_type(extracted_type: str) -> tuple[str, str]:
    """
    Maps extracted account type from AI to standard enum type + subtype.

    Args:
        extracted_type: Account type string extracted from statement (e.g., "Maybank Savings Account-i")

    Returns:
        Tuple of (standard_type, subtype)
        - standard_type: One of the AccountTypeEnum values
        - subtype: Specific variant or the original extracted_type

    Examples:
        "Maybank Savings Account-i" → ("savings", "Islamic Savings Account")
        "CIMB Visa Platinum" → ("credit", "Platinum Credit Card")
        "Touch n Go eWallet" → ("ewallet", "Touch n Go eWallet")
    """
    if not extracted_type:
        return ("savings", None)

    extracted_lower = extracted_type.lower()

    # Savings account variants
    if any(keyword in extracted_lower for keyword in ['savings', 'simpanan', 'tabungan', 'saving']):
        if any(keyword in extracted_lower for keyword in ['islamic', '-i ', 'shariah', 'syariah']):
            return ('savings', 'Islamic Savings Account')
        elif any(keyword in extracted_lower for keyword in ['junior', 'kid', 'child']):
            return ('savings', 'Junior Savings Account')
        elif any(keyword in extracted_lower for keyword in ['premier', 'premium', 'privilege']):
            return ('savings', 'Premier Savings Account')
        else:
            return ('savings', extracted_type)

    # Current/Checking account variants
    elif any(keyword in extracted_lower for keyword in ['current', 'checking', 'semasa', 'chequing']):
        if any(keyword in extracted_lower for keyword in ['islamic', '-i ', 'shariah', 'syariah']):
            return ('current', 'Islamic Current Account')
        else:
            return ('current', extracted_type)

    # Credit card variants
    elif any(keyword in extracted_lower for keyword in ['credit card', 'visa', 'mastercard', 'amex', 'american express', 'credit-card']):
        # Extract card tier if present
        if 'world elite' in extracted_lower:
            return ('credit', 'World Elite Credit Card')
        elif 'signature' in extracted_lower:
            return ('credit', 'Signature Credit Card')
        elif 'platinum' in extracted_lower:
            return ('credit', 'Platinum Credit Card')
        elif 'gold' in extracted_lower:
            return ('credit', 'Gold Credit Card')
        elif 'classic' in extracted_lower:
            return ('credit', 'Classic Credit Card')
        else:
            return ('credit', extracted_type)

    # E-wallet variants
    elif any(keyword in extracted_lower for keyword in ['touch n go', 'tng', 'grabpay', 'grab pay', 'boost', 'shopeepay', 'shopee pay', 'ewallet', 'e-wallet', 'wallet', 'mayabank', 'gcash']):
        return ('ewallet', extracted_type)

    # Investment accounts
    elif any(keyword in extracted_lower for keyword in ['investment', 'brokerage', 'trading', 'asb', 'asn', 'unit trust', 'amanah saham', 'mutual fund']):
        return ('investment', extracted_type)

    # Cash
    elif any(keyword in extracted_lower for keyword in ['cash', 'tunai', 'physical cash']):
        return ('cash', 'Cash')

    # Default to savings if unclear (most common account type)
    else:
        return ('savings', extracted_type)


# ============ S3 CONFIGURATION ============
# Local file storage configuration
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
STATEMENTS_DIR = os.path.join(UPLOAD_DIR, "statements")
CTOS_DIR = os.path.join(UPLOAD_DIR, "ctos")

# Create upload directories if they don't exist
Path(STATEMENTS_DIR).mkdir(parents=True, exist_ok=True)
Path(CTOS_DIR).mkdir(parents=True, exist_ok=True)

# Base URL for serving files (for local development)
BASE_URL = os.getenv("BASE_URL", "https://rayyai-api-service-838873798405.us-central1.run.app/")

# S3 Configuration
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-1")

# Initialize S3 client if credentials are available
s3_client = None
if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY and S3_BUCKET_NAME:
    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION
        )
        logger.info(f"S3 client initialized successfully for bucket: {S3_BUCKET_NAME}")
    except Exception as e:
        logger.error(f"Failed to initialize S3 client: {e}")
        s3_client = None
else:
    logger.warning("S3 credentials not configured. Files will be stored locally.")


async def upload_file_to_s3(
    file: UploadFile, user_id: int, folder: str, prefix: str = ""
) -> tuple[str, str]:
    """
    Upload file to AWS S3 and return the file URL and SHA-256 hash.
    Falls back to local storage if S3 is not configured.

    Args:
        file: The uploaded file
        user_id: Current user's ID
        folder: Folder name (statements/ctos)
        prefix: Optional prefix for filename (e.g., "CTOS_")

    Returns:
        Tuple of (File URL, SHA-256 hash)
    """
    try:
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = file.filename or "uploaded_file"
        if not isinstance(filename, str):
            filename = str(filename)
        file_extension = os.path.splitext(filename)[1] or ".pdf"

        # Generate S3 key (path in bucket)
        unique_filename = f"{prefix}{timestamp}{file_extension}"
        s3_key = f"{folder}/{user_id}/{unique_filename}"

        # Read file contents
        contents = await file.read()

        if not contents:
            raise HTTPException(status_code=400, detail="File is empty")

        # Compute SHA-256 hash for duplicate detection
        file_hash = hashlib.sha256(contents).hexdigest()

        # Upload to S3 if configured, otherwise fall back to local storage
        if s3_client and S3_BUCKET_NAME:
            try:
                # Upload to S3
                s3_client.put_object(
                    Bucket=S3_BUCKET_NAME,
                    Key=s3_key,
                    Body=contents,
                    ContentType=file.content_type or 'application/pdf',
                    Metadata={
                        'user_id': str(user_id),
                        'original_filename': filename,
                        'upload_timestamp': timestamp
                    }
                )

                # Generate S3 URL
                s3_url = f"https://{S3_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{s3_key}"

                logger.info(f"File uploaded successfully to S3: {s3_url}")
                return s3_url, file_hash

            except ClientError as e:
                logger.error(f"S3 upload failed, falling back to local storage: {e}")
                # Fall through to local storage

        # Fall back to local storage if S3 fails or is not configured
        logger.info("Using local storage for file upload")

        # Create user-specific directory
        folder_path = STATEMENTS_DIR if folder == "statements" else CTOS_DIR
        user_dir = os.path.join(folder_path, str(user_id))
        Path(user_dir).mkdir(parents=True, exist_ok=True)

        # Save file to local storage
        file_path = os.path.join(user_dir, unique_filename)
        with open(file_path, "wb") as f:
            f.write(contents)

        # Generate local URL
        relative_path = f"/files/{folder}/{user_id}/{unique_filename}"
        url = f"{BASE_URL}{relative_path}"

        logger.info(f"File uploaded successfully to local storage: {file_path}")
        return url, file_hash

    except Exception as e:
        logger.error(f"File upload error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"File upload error: {str(e)}")
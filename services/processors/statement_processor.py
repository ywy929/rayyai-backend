"""
Bank Statement Processor
Extracts transaction data from bank statements using Gemini Vision AI
Multi-page PDF processing with async support
"""

from fastapi import HTTPException, status
from PIL import Image
import io
import os
import json
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai
import fitz  # PyMuPDF - no poppler required!
from typing import List, Dict, Any, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configure Gemini AI
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Categories for expense categorization (same as scanner.py)
# Includes both English and Bahasa Malaysia keywords for better categorization
EXPENSE_CATEGORIES = {
    "Housing": ["rent", "rental", "housing", "accommodation", "apartment", "condo", "house", "lease", "sewa", "rumah", "sewa rumah", "bilik", "apartment", "kondo"],
    "Groceries": ["grocery", "supermarket", "market", "fresh", "vegetables", "fruits", "familymart", "family mart", "7-eleven", "tesco", "giant", "aeon", "pasar", "pasar malam", "pasar pagi", "kedai runcit", "mini market", "99 speedmart", "kk mart", "mydin", "sayur", "buah", "basah"],
    "Dining": [
        # English keywords
        "restaurant", "cafe", "coffee", "starbucks", "mcdonald", "kfc", "dining", "food court", "pizza", "burger", "sushi", "food", "eat", "lunch", "dinner", "breakfast", "brunch",
        # Bahasa Malaysia keywords
        "warung", "kedai", "restoran", "mamak", "kopitiam", "gerai", "stall", "nasi", "mee", "roti", "teh", "kopi", "makan", "minum", "buka puasa", "sahur",
        # Malaysian food places
        "old town", "secret recipe", "pappa rich", "nandos", "domino", "pizza hut", "subway", "marrybrown", "ramly", "ayam", "ikan", "daging",
        # Food delivery
        "grab food", "grabfood", "foodpanda", "deliveroo", "food delivery"
    ],
    "Transportation": ["fuel", "petrol", "gas", "taxi", "grab", "uber", "parking", "toll", "shell", "petronas", "bhp", "minyak", "bensin", "letak kereta", "tol", "lrt", "mrt", "ktm", "bas", "teksi", "kereta", "motor"],
    "Shopping": ["mall", "store", "shop", "clothing", "fashion", "apparel", "uniqlo", "h&m", "zara", "kedai", "butik", "pasaraya", "beli", "belian", "pembelian"],
    "Entertainment": ["cinema", "movie", "game", "entertainment", "theme park", "wayang", "pawagam", "gsc", "tgv", "mbo", "hiburan", "permainan"],
    "Healthcare": ["clinic", "hospital", "pharmacy", "medical", "doctor", "health", "guardian", "watsons", "klinik", "hospital", "farmasi", "doktor", "ubat", "kesihatan", "rawatan"],
    "Bills & Utilities": ["electric", "water", "internet", "phone", "bill", "utility", "telco", "bil", "elektrik", "air", "internet", "telefon", "tnb", "syabas", "tm", "unifi", "maxis", "celcom", "digi", "astro"],
    "Education": ["school", "university", "course", "book", "education", "tuition", "sekolah", "universiti", "kuliah", "buku", "pendidikan", "yuran", "tuisyen"],
    "Travel": ["hotel", "flight", "airbnb", "booking", "travel", "tourism", "hotel", "penerbangan", "kapal terbang", "perjalanan", "pelancongan", "cuti"],
    "Insurance": ["insurance", "takaful", "policy", "insurans", "takaful", "polis"],
    "Personal Care": ["salon", "spa", "beauty", "gym", "fitness", "salun", "kecantikan", "gim", "kecergasan", "grooming", "rambut", "facial"],
}

def guess_category(description: str) -> str:
    """Guess expense category based on transaction description"""
    desc_lower = description.lower()

    for category, keywords in EXPENSE_CATEGORIES.items():
        if any(keyword in desc_lower for keyword in keywords):
            return category

    return "Other"


def detect_card_brand(account_name: str = None, account_type: str = None, card_brand: str = None) -> str:
    """
    Detect credit card brand from account name, type, or explicit card_brand field.
    Returns: Visa, Mastercard, American Express, JCB, UnionPay
    Defaults to Visa if brand cannot be determined (most common in Malaysia)
    """
    # If card_brand is already provided and valid, use it
    if card_brand:
        brand_lower = card_brand.lower().strip()
        if "visa" in brand_lower:
            return "Visa"
        elif "master" in brand_lower:
            return "Mastercard"
        elif "amex" in brand_lower or "american express" in brand_lower:
            return "American Express"
        elif "jcb" in brand_lower:
            return "JCB"
        elif "union" in brand_lower:
            return "UnionPay"

    # Check account_name and account_type
    combined_text = " ".join(filter(None, [account_name or "", account_type or ""])).lower()

    if "visa" in combined_text:
        return "Visa"
    elif "master" in combined_text:
        return "Mastercard"
    elif "amex" in combined_text or "american express" in combined_text:
        return "American Express"
    elif "jcb" in combined_text:
        return "JCB"
    elif "union" in combined_text:
        return "UnionPay"

    # Default to Visa (most common in Malaysia)
    return "Visa"


def parse_numeric_value(value: Any) -> Optional[float]:
    """
    Convert various numeric string formats (including CR/DR suffixes) to float.
    Returns None when conversion is not possible.
    """
    if value is None:
        return None

    if isinstance(value, (int, float)):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return None

        # Handle CR/DR suffix
        sign = 1
        upper_cleaned = cleaned.upper()
        if upper_cleaned.endswith("CR"):
            cleaned = cleaned[:-2].strip()
        elif upper_cleaned.endswith("DR"):
            cleaned = cleaned[:-2].strip()
            sign = -1

        # Remove currency symbols and commas
        cleaned = (
            cleaned.replace("RM", "")
            .replace("MYR", "")
            .replace(",", "")
            .replace(" ", "")
        )

        # Handle parentheses for negatives e.g., (123.45)
        if cleaned.startswith("(") and cleaned.endswith(")"):
            cleaned = cleaned[1:-1]
            sign *= -1

        if cleaned in ("", "-", "--"):
            return None

        try:
            return float(cleaned) * sign
        except ValueError:
            return None

    return None


def build_credit_card_summary(credit_terms: Dict[str, Any], closing_balance: Optional[Any]) -> Dict[str, Optional[float]]:
    """
    Build a normalized summary of key credit card values (limit, outstanding, balance).
    Provides sensible fallbacks for missing fields.
    """
    summary = {}

    credit_limit = parse_numeric_value(credit_terms.get("credit_limit"))
    available_credit = parse_numeric_value(credit_terms.get("available_credit"))
    current_balance = parse_numeric_value(credit_terms.get("current_balance"))
    total_amount_due = parse_numeric_value(credit_terms.get("total_amount_due"))
    minimum_payment = parse_numeric_value(credit_terms.get("minimum_payment"))

    closing_balance_value = parse_numeric_value(closing_balance)

    if current_balance is None:
        current_balance = closing_balance_value

    outstanding_balance = (
        parse_numeric_value(credit_terms.get("outstanding_balance"))
        or total_amount_due
        or current_balance
        or closing_balance_value
    )

    if available_credit is None and credit_limit is not None and current_balance is not None:
        available_credit = max(credit_limit - current_balance, 0)

    summary["credit_limit"] = credit_limit
    summary["available_credit"] = available_credit
    summary["current_balance"] = current_balance
    summary["outstanding_balance"] = outstanding_balance
    summary["total_amount_due"] = total_amount_due or outstanding_balance
    summary["minimum_payment"] = minimum_payment

    return summary

def convert_pdf_to_images(pdf_bytes: bytes, dpi: int = 200) -> List[Image.Image]:
    """
    Convert PDF to list of PIL Images using PyMuPDF (no poppler required!)

    Args:
        pdf_bytes: PDF file as bytes
        dpi: Resolution for conversion (200 is good balance of quality/speed)

    Returns:
        List of PIL Image objects, one per page
    """
    try:
        logger.info(f"Converting PDF to images using PyMuPDF (DPI: {dpi})")

        # Open PDF from bytes
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        images = []

        # Calculate zoom factor from DPI (default 72 DPI)
        zoom = dpi / 72
        matrix = fitz.Matrix(zoom, zoom)

        # Convert each page to image
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]

            # Render page to pixmap (image)
            pix = page.get_pixmap(matrix=matrix)

            # Convert pixmap to PIL Image
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)

        pdf_document.close()

        logger.info(f"Successfully converted PDF to {len(images)} images")
        return images
    except Exception as e:
        logger.error(f"Failed to convert PDF to images: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid PDF file: {str(e)}"
        )

def image_to_bytes(image: Image.Image) -> bytes:
    """Convert PIL Image to JPEG bytes for Gemini API"""
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Convert to bytes
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG', quality=85)
    return img_byte_arr.getvalue()

def extract_transactions_from_image(image_bytes: bytes, page_number: int) -> Dict[str, Any]:
    """
    Extract transaction data from a single statement page using Gemini Vision AI

    Args:
        image_bytes: Image as JPEG bytes
        page_number: Page number for logging

    Returns:
        Dictionary containing extracted data
    """
    try:
        if not GEMINI_API_KEY:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Gemini API key not configured"
            )

        logger.info(f"Processing page {page_number} with Gemini Vision AI")

        # Initialize Gemini model
        model = genai.GenerativeModel('gemini-2.0-flash')

        # Comprehensive prompt for bank statement extraction with AI categorization
        prompt = """
        Analyze this financial statement page and extract ALL transaction data in JSON format.

        IMPORTANT RULES:
        1. Extract EVERY transaction visible on this page
        2. DO NOT include 'Opening Balance', 'Closing Balance', 'Previous Balance', 'PREVIOUS BALANCE', or similar entries as transactions
           - Balance entries may appear AS ROWS within the transaction table (often as the first or last row)
           - When you encounter a balance row in the transaction table:
             a) Extract the amount from that row and assign it to opening_balance or closing_balance
             b) DO NOT add that row to the transactions array
           - Common indicators of balance rows in transaction tables:
             * Transaction description contains: "PREVIOUS BALANCE", "OPENING BALANCE", "CLOSING BALANCE", "BALANCE B/F", "BALANCE C/F", "Previous Balance", "Opening Balance"
             * Date fields may be empty, show "()", "---", "-", or use placeholder dates
           - Balance rows should ONLY go in the 'balances' section, NOT in transactions array
        3. For amounts, determine if it's a DEBIT (expense/withdrawal) or CREDIT (income/deposit)
        4. Debit transactions should have NEGATIVE amounts
        5. Credit transactions should have POSITIVE amounts
        6. **CRITICAL FOR CREDIT CARD STATEMENTS**: 
           - Credit card statements typically do NOT show "DR" explicitly
           - If an amount does NOT have "CR" notation, it should be treated as DR (debit/negative)
           - Example: "179.00" without CR → treat as DR, amount should be -179.00
           - Only amounts with "CR" are credits (positive)
           - This applies to BOTH transaction amounts AND balance amounts
        7. Categorize each transaction based on the description
        8. Look for statement period dates (usually at top of page)
        9. Extract account information if visible
        10. CRITICAL: Determine the CORRECT account type based on the statement header/title
        11. For CREDIT CARD statements, extract additional credit card terms (interest rate, credit limit, payment info)
        12. **BALANCE EXTRACTION WITH CR/DR NOTATION**:
            - Malaysian bank statements often show "CR" (credit) or "DR" (debit) after balance amounts
            - "Previous Statement: 0.00" or "Previous Statement Balance: 1000.00 CR" → opening_balance
            - "Statement Balance: 112.01 CR" or "Closing Balance: 500.00 DR" → closing_balance
            - **CRITICAL FOR CREDIT CARD STATEMENTS**: 
              * If a balance amount does NOT have "CR", treat it as DR (negative)
              * Example: "Closing Balance: 179.00" (no CR) → closing_balance = -179.00
              * Only balances with "CR" are positive
            - **CRITICAL FOR TABLE ROWS**: If you see a transaction table row with description "PREVIOUS BALANCE" or "OPENING BALANCE":
              * This is NOT a transaction - it's the opening balance!
              * Extract the amount from that row as opening_balance
              * The amount may be in Amount, Debit, or Credit column
              * Example: | () | () | PREVIOUS BALANCE | 0.0 | means opening_balance = 0.0
              * For credit card statements: If amount has no CR, treat as negative
            - CR means credit (positive), DR means debit (negative/overdraft)
            - For credit card statements: No CR notation = DR (negative) by default
            - ALWAYS remove CR/DR text and return ONLY the numeric value with correct sign
        13. Return ONLY valid JSON, no explanations

        Expected JSON structure:
        {
          "page_number": 1,
          "statement_period": {
            "start_date": "2025-01-01",  // YYYY-MM-DD format
                                         // PRIMARY: Look for header text: "Statement Period", "From", "Period From"
                                         // FALLBACK: If not found in header, use the EARLIEST transaction date from the statement
            "end_date": "2025-01-31"     // YYYY-MM-DD format
                                         // PRIMARY: Look for header text: "To", "Period To", "Statement Date"
                                         // FALLBACK: If not found in header, use the LATEST transaction date from the statement
                                         // Common header locations:
                                         // - "Statement Period: 01 Feb 2025 - 28 Feb 2025"
                                         // - "From 2025-02-01 To 2025-02-28"
          },
          "account_info": {
            "bank_name": "Maybank",  // REQUIRED: Extract the bank name from statement header/logo/title (Maybank, CIMB, Public Bank, RHB, etc.)
            "account_number": "1234567890",  // Last 4 digits or full if visible
            "account_name": "Maybank Visa Platinum",  // Full account/card name if visible (e.g., "Maybank Visa Platinum")
            "account_type": "Visa Platinum Credit Card",  // IMPORTANT: Extract the ACTUAL type from statement
            // Examples for account_type:
            // - Credit card statements: "Visa Platinum Credit Card", "MasterCard Gold", "American Express", "Credit Card"
            // - Bank statements: "Savings Account-i", "Current Account", "Premier Savings", "Basic Savings Account"
            // - E-wallet: "Touch n Go eWallet", "GrabPay Wallet", "Boost eWallet"
            // DO NOT default to "Savings Account" - read what's on the statement!
            "card_brand": "Visa"  // REQUIRED FOR CREDIT CARDS: Card network brand (Visa, Mastercard, American Express, JCB, UnionPay, etc.)
                                  // Extract from card logo, account name, or account type
                                  // Common brands in Malaysia: Visa, Mastercard, American Express (Amex), JCB, UnionPay
                                  // If not visible or not a credit card, set to null
          },
          "user_info": {
            // Extract user/account holder information if visible on the statement
            "full_name": "Ahmad bin Abdullah",  // Full legal name as shown on statement (e.g., "Ahmad bin Abdullah", "Tan Wei Ming")
            "first_name": "Ahmad",  // First name (extracted from full_name or if shown separately)
            "last_name": "bin Abdullah",  // Last name/surname (extracted from full_name or if shown separately)
            "date_of_birth": "1990-05-15",  // Date of birth in YYYY-MM-DD format (if visible)
            "age": 34,  // Age (calculate from DOB if DOB is found, or extract if shown directly)
            "ic_number": "900515-10-1234",  // Malaysian IC/MyKad number (if visible, may be partially masked)
            "passport_number": "A12345678",  // Passport number (if visible, may be partially masked)
            "address": "123 Jalan Bukit Bintang, 50000 Kuala Lumpur",  // Full address if visible
            "phone_number": "+60123456789",  // Phone number if visible
            "email": "ahmad@example.com"  // Email address if visible
            // If any field is not found on this page, set to null
            // Malaysian statements may show:
            // - Full name in account holder section
            // - IC number (format: YYMMDD-PB-G###)
            // - Address in account details section
          },

            // IMPORTANT FOR CREDIT CARDS:
            // - account_name should be the FULL card name as shown on statement (e.g., "Maybank Visa Platinum")
            // - bank_name should be ONLY the bank/issuer name (e.g., "Maybank", "CIMB", "HSBC")
            // - For Malaysian banks: Maybank, CIMB, Public Bank, RHB, Hong Leong, AmBank, HSBC, Standard Chartered, Citibank, UOB, Alliance Bank, Affin Bank, Bank Islam, Bank Rakyat
          },
          "credit_card_terms": {
            // ONLY include this section if this is a CREDIT CARD statement
            // Extract these values if visible on the statement (usually on first page or summary section)
            // LOOK CAREFULLY in tables, summary boxes, payment information sections
            "credit_limit": 15000.00,           // Total credit limit (if shown)
            "available_credit": 12500.00,       // Available credit (if shown)
            "current_balance": 2500.00,         // Current outstanding balance
            "minimum_payment": 125.00,          // Minimum payment amount due (if shown)
                                                // Look for: "Minimum Payment", "Minimum Payment (RM)", "Min Payment",
                                                // "Bayaran Minimum", "Minimum Amount Due", etc.
                                                // Check in tables, payment summary sections, or payment information boxes
            "total_amount_due": 2500.00,        // Total amount due (if shown)
            "payment_due_date": "2025-02-15",   // Payment due date in YYYY-MM-DD format (if shown)
            "interest_rate": 18.0,              // Annual interest rate percentage (e.g., 18% APR) (if shown)
            "annual_fee": 200.00,               // Annual card fee (if shown)
            "late_payment_fee": 100.00,         // Late payment fee (if shown)
            "overlimit_fee": 50.00,             // Over-limit fee (if shown)
            "cash_advance_rate": 24.0,          // Cash advance interest rate (if shown)
            "rewards_points": 12500             // Rewards/points balance (if shown)
            // If any field is not visible, set to null
            // If this is NOT a credit card statement, omit this entire "credit_card_terms" section

            // EXAMPLE: If you see a table like this on the statement:
            // ┌─────────────────────────┬
            // │ Minimum Payment (RM)    │
            // │ 227.50                  │
            // └─────────────────────────┴
            // You should extract:
            // "minimum_payment": 227.50
          },
          "balances": {
            "opening_balance": 5000.00,  // REQUIRED: Look for "PREVIOUS BALANCE" or "OPENING BALANCE"
                                         // ⚠️ VERY IMPORTANT: Check the FIRST ROW of the transaction table!
                                         // Many statements show previous balance as the FIRST ROW with:
                                         // - Description: "PREVIOUS BALANCE" or "OPENING BALANCE"
                                         // - Empty dates: "()" or "---" or blank
                                         // - Amount in Amount/Debit/Credit column
                                         // Example: | () | () | PREVIOUS BALANCE | 0.0 | → This means opening_balance is 0.0
                                         //
                                         // Also check for separate header sections showing:
                                         // - "Opening Balance", "Previous Statement", "Previous Balance", "Balance B/F", "Balance Brought Forward"
                                         //
                                         // CRITICAL: If you find EITHER format above, you MUST extract it as opening_balance
                                         // Do NOT leave opening_balance as null if you see "PREVIOUS BALANCE" anywhere!
                                         //
                                         // Handle CR/DR notation:
                                         // - "5000.00 CR" or "5000.00CR" → extract as 5000.00 (positive)
                                         // - "500.00 DR" or "500.00DR" → extract as -500.00 (negative)
            "closing_balance": 4250.00   // Look for: "Closing Balance", "Statement Balance", "STATEMENT BALANCE", "Current Balance", "Balance C/F", "Balance Carried Forward", "Ending Balance"
                                         // IMPORTANT: Same rules as opening_balance - check both separate section AND transaction table rows
                                         // Examples:
                                         // - "Statement Balance: 112.01 CR" → extract as 112.01
                                         // - "Closing Balance: 50.00 DR" → extract as -50.00
                                         // - Table row: | () | () | PREVIOUS BALANCE | 0.0 | → extract as 0.0
          },
          "transactions": [
            {
              "date": "2025-01-15",  // YYYY-MM-DD format
              "description": "SALARY DEPOSIT - ABC COMPANY SDN BHD",  // Full description
              "amount": 5000.00,  // POSITIVE for credit/deposit
              "type": "credit",   // "credit" or "debit"
              "balance": 10000.00,  // Running balance if shown
              "category": "Salary"  // Category based on description (see rules below)
            },
            {
              "date": "2025-01-16",
              "description": "GRAB - FOOD DELIVERY",
              "amount": -25.50,  // NEGATIVE for debit/withdrawal
              "type": "debit",
              "balance": 9974.50,
              "category": "Dining"  // Intelligently categorized
            }
          ]
          
          **CREDIT CARD TRANSACTION AMOUNT EXAMPLES**:
          - "179.00" (no CR/DR) → amount: -179.00, type: "debit" (credit card default)
          - "179.00 CR" → amount: 179.00, type: "credit" (payment/credit)
          - "500.00" (no CR/DR) → amount: -500.00, type: "debit" (credit card default)
          - "1000.00 CR" → amount: 1000.00, type: "credit" (payment/credit)
          
          **CREDIT CARD BALANCE EXAMPLES**:
          - "Closing Balance: 179.00" (no CR) → closing_balance: -179.00 (credit card default)
          - "Closing Balance: 179.00 CR" → closing_balance: 179.00 (positive)
          - "Previous Balance: 500.00" (no CR) → opening_balance: -500.00 (credit card default)
        }

        BALANCE EXTRACTION EXAMPLES (Table-Embedded Format):

        Example 1: Balance as first row in transaction table (user's format)
        +--------------+------------------+-------------------------+--------+
        | Posting Date | Transaction Date | Transaction Description | Amount |
        +--------------+------------------+-------------------------+--------+
        | ()           | ()               | PREVIOUS BALANCE        | 0.0    |
        | 31 OCT       | 31 OCT           | PURCHASE                | 10.00CR|
        +--------------+------------------+-------------------------+--------+

        Extract as:
        "opening_balance": 0.0
        DO NOT include "PREVIOUS BALANCE" row in transactions array

        Example 2: Balance with CR notation in table
        | Date     | Description         | Debit   | Credit  |
        |----------|---------------------|---------|---------|
        | -        | OPENING BALANCE     |         | 1250.50CR|
        | 01/02/25 | SALARY CREDIT       |         | 5000.00 |

        Extract as:
        "opening_balance": 1250.50  (CR = positive)
        DO NOT include "OPENING BALANCE" row in transactions array

        Example 3: Closing balance as last row
        | Date     | Description         | Amount  | Balance |
        |----------|---------------------|---------|---------|
        | 15/02/25 | PAYMENT             | -50.00  | 1200.50 |
        | ---      | CLOSING BALANCE     | 1200.50 |         |

        Extract as:
        "closing_balance": 1200.50
        DO NOT include "CLOSING BALANCE" row in transactions array

        CATEGORIZATION RULES:

        For EXPENSE transactions (type: "debit"), choose the MOST appropriate category:

        - "Housing": Rent, rental payments, accommodation, housing costs
          Examples: Rent payment, rental, sewa rumah, apartment rent, housing, lease payment
          CRITICAL: If description contains "rent", "rental", "sewa", "housing" → MUST categorize as "Housing"

        - "Groceries": Supermarkets, convenience stores, wet markets, fresh produce
          Examples: FamilyMart, 7-Eleven, Tesco, Giant, AEON, MyDin, 99 Speedmart, KK Mart, pasar
          Bahasa Malaysia keywords: "pasar", "pasar malam", "pasar pagi", "kedai runcit", "mini market", "sayur", "buah", "basah"
          CRITICAL: If description contains "pasar", "kedai runcit", "sayur", "buah" → MUST categorize as "Groceries"

        - "Food & Dining": Restaurants, cafes, fast food, food delivery, beverages
          Examples: McDonald's, KFC, Starbucks, Grab Food, Old Town, Kopitiam, Secret Recipe, food court
          Bahasa Malaysia keywords: "warung", "kedai", "restoran", "mamak", "kopitiam", "gerai", "nasi", "mee", "roti", "teh", "kopi", "makan", "minum", "ayam", "ikan", "daging"
          CRITICAL: If description contains "warung", "kedai makan", "mamak", "kopitiam", "gerai", "nasi", "mee", "roti", "makan", "minum" → MUST categorize as "Dining"

        - "Transportation": Fuel, parking, ride-sharing, tolls, public transport
          Examples: Shell, Petronas, Grab (rides), taxi, parking, Touch n Go, LRT, MRT

        - "Shopping": Retail stores, fashion, electronics, online shopping, general merchandise
          Examples: Uniqlo, Lazada, Shopee, Mr DIY, Pavilion, H&M, Zara, shopping mall
          CRITICAL: Shopping transactions are ALWAYS discretionary spending (wants, not needs)
          - If description contains: "shop", "shopping", "mall", "purchase", "buy", "retail", "store", "fashion", "clothing", "apparel", "online shopping", "e-commerce", "marketplace" → MUST categorize as "Shopping"
          - Common shopping platforms: Shopee, Lazada, Amazon, Zalora, Fashion Valet, Sephora, etc.
          - Shopping malls: Pavilion, KLCC, Mid Valley, 1 Utama, Sunway Pyramid, The Gardens, etc.

        - "Entertainment": Movies, streaming services, games, theme parks, leisure
          Examples: Netflix, Spotify, GSC Cinema, TGV, Genting, Sunway Lagoon, gym

        - "Healthcare": Clinics, hospitals, pharmacies, medical services, dental, optical
          Examples: Hospital, clinic, Guardian, Watsons, pharmacy, doctor, dental

        - "Bills & Utilities": Electric, water, internet, phone bills, cable TV
          Examples: TNB, Syabas, TM, Unifi, Maxis, Celcom, Digi, Astro, utility bills

        - "Education": Schools, universities, courses, books, tuition, learning materials
          Examples: School fees, university, tuition, bookstore, Popular, Kinokuniya

        - "Travel": Hotels, flights, car rentals, tourism, accommodation
          Examples: Hotel, AirAsia, Malaysia Airlines, Agoda, Booking.com, Airbnb

        - "Insurance": Insurance premiums, takaful, policies
          Examples: Insurance payment, Prudential, AIA, Great Eastern, takaful

        - "Personal Care": Salons, spas, beauty products, wellness, grooming
          Examples: Salon, spa, barbershop, beauty, massage, wellness

        - "Other": If none of the above categories clearly fit

        For INCOME transactions (type: "credit"), choose the MOST appropriate category:

        - "Salary": Salary deposits, payroll, wages
          Examples: Salary credit, payroll, wages, gaji

        - "Freelance": Freelance income, contract work, gig economy
          Examples: Freelance payment, contract work, Upwork, Fiverr, consulting fee

        - "Business": Business income, sales revenue, commission
          Examples: Business income, sales, commission, self-employed income

        - "Investments": Interest, dividends, capital gains, investment returns
          Examples: Interest credit, dividend, investment return

        - "Gifts": Gifts received, ang pow, monetary gifts
          Examples: Gift, ang pow, present, duit raya

        - "Refunds": Refunds, returns, cashback, reimbursements
          Examples: Refund, cashback, reimbursement, return

        - "Transfer": ALL money transfers (both to others AND to own accounts/savings)
          Examples: Transfer from, online transfer, fund transfer, DuitNow transfer, FPX transfer
          CRITICAL: ALL transfers should use category "Transfer" - use the "transfer_type" field to distinguish:
          - "inter_person": Transfer to another person (friend, family, business partner, etc.)
          - "intra_person": Transfer to own account/savings/investment
          - See TRANSFER TYPE CLASSIFICATION section below for detailed rules
          IMPORTANT: Do NOT use "Other" category for transfers - ALL transfers must use "Transfer" category

        - "Other": Other income sources not listed above (EXCLUDING transfers - transfers must use "Transfer" category)

        IMPORTANT:
        - Analyze the FULL description to understand context
        - "GRAB FOOD" → Food & Dining (not Transportation)
        - "SHELL CAR WASH" → Transportation (at gas station) or Personal Care (standalone)
        - Be intelligent about multi-word descriptions
        - Consider Malaysian context (local merchants, Malay terms)
        
        MALAYSIAN CONTEXT & BAHASA MALAYSIA KEYWORDS:
        - Always check for both English AND Bahasa Malaysia terms in transaction descriptions
        - Common dining terms: "warung", "kedai makan", "mamak", "kopitiam", "gerai", "nasi", "mee", "roti", "teh", "kopi", "makan", "minum", "ayam", "ikan", "daging"
        - Common grocery terms: "pasar", "pasar malam", "pasar pagi", "kedai runcit", "sayur", "buah", "basah"
        - Common transportation terms: "minyak", "bensin", "tol", "letak kereta", "teksi", "bas", "lrt", "mrt", "ktm"
        - Common shopping terms: "kedai", "butik", "pasaraya", "beli", "belian"
        - Common healthcare terms: "klinik", "hospital", "farmasi", "doktor", "ubat", "kesihatan"
        - Common utilities terms: "bil", "elektrik", "air", "telefon"
        - Common education terms: "sekolah", "universiti", "kuliah", "buku", "yuran", "tuisyen"
        - When you see mixed language (e.g., "Warung Nasi Lemak", "Kedai Makan ABC", "Mamak Corner"), prioritize the Bahasa Malaysia keyword for accurate categorization
        - Search for place names in both languages - many Malaysian merchants use mixed or full Bahasa Malaysia names
        - Be aware that "kedai" can mean both "shop" (Shopping) and "food shop" (Dining) - use context clues (e.g., "kedai makan" = Dining, "kedai runcit" = Groceries, "kedai pakaian" = Shopping)

        TRANSACTION DESCRIPTION ENHANCEMENT:
        Bank statements often use coded syntax. Parse and create human-readable descriptions:

        Common Malaysian Transaction Patterns:
        1. "GRBPAY*MERCHANT ID LOC" → Parse as GrabPay payment at MERCHANT in LOCATION
        2. "TNG*SERVICE@LOC" → Parse as Touch n Go payment for SERVICE at LOCATION
        3. "SHOPEE*ORDER123" → Parse as Shopee online purchase
        4. "FPX TRANSFER BANK" → Parse as online banking transfer from BANK
        5. "DBT POS MERCHANT LOC" → Parse as debit card payment at MERCHANT in LOCATION
        6. "ATM WDL BANK LOC" → Parse as ATM withdrawal at BANK in LOCATION

        Payment Method Codes to Expand:
        - GRBPAY / GRB / GRAB PAY → GrabPay
        - TNG / TOUCH N GO → Touch n Go
        - FPX → Online Banking (FPX)
        - DBT POS / DEBIT CARD → Debit Card
        - ATM WDL / ATM WITHDRAWAL → ATM Withdrawal
        - CREDIT CARD / CR CARD → Credit Card

        Malaysian Location Abbreviations & Patterns:

        MAJOR CITIES:
        - KULJ / KL → Kuala Lumpur
        - PJ → Petaling Jaya
        - JB → Johor Bahru
        - PG / PNG → Penang
        - SHL ALM / SHA → Shah Alam
        - IPH → Ipoh
        - MLK / MALACCA → Malacca
        - KK → Kota Kinabalu
        - KCH / KUCHING → Kuching
        - SBH / SUBANG → Subang Jaya

        KL DISTRICTS & AREAS:
        - BGSR / BANGSAR → Bangsar
        - DMNS / DAMAN / DAMANSARA → Damansara
        - MK / MONT KIARA → Mont Kiara
        - CHRS / CHERAS → Cheras
        - AMPG / AMPANG → Ampang
        - BKTBINTANG / BB / BKT BINTANG → Bukit Bintang
        - SRI PTG / SRIPET → Sri Petaling
        - TTDI / TTD → Taman Tun Dr Ismail
        - SETIA ALM / SETALAM → Setia Alam
        - SEPTR / SEPANG → Sepang
        - WANGSA MAJU / WM → Wangsa Maju
        - KEPONG / KPG → Kepong
        - PUCHONG / PCH → Puchong
        - SERDANG / SDG → Serdang
        - KAJANG / KJG → Kajang
        - PUTRAJAYA / PTJ → Putrajaya
        - CYBERJAYA / CYB → Cyberjaya
        - SENTUL / STL → Sentul
        - TITIWANGSA / TW → Titiwangsa
        - SETIAWANGSA / SW → Setiawangsa
        - KL CITY / KLCITY → Kuala Lumpur City Centre

        SHOPPING MALLS & LANDMARKS:
        - MID V / MIDVAL / MV → Mid Valley
        - KLCC / SURIA KLCC → KLCC
        - PAV / PAVILION → Pavilion KL
        - 1U / 1 UTAMA → 1 Utama
        - IOI / IOI CM → IOI City Mall
        - SUNWAY PYR / SPY → Sunway Pyramid
        - THE CURVE / CURVE → The Curve
        - NU SENTRAL / NUS → NU Sentral
        - KL SENTRAL / KLSENT → KL Sentral
        - GARDENS / GARDENS MALL → The Gardens Mall
        - AEON → AEON Mall (various locations)

        LOCATION EXTRACTION RULES (CRITICAL - READ CAREFULLY):
        1. Look for location indicators AFTER merchant name in transaction description
        2. Common separators between merchant and location: space, @, -, (, /, comma
        3. Location typically appears at the END of the description string
        4. Location can be a full city name, abbreviation, mall name, or area name
        5. If multiple location indicators are present, extract the MOST SPECIFIC one (e.g., prefer "Mid Valley, KL" over just "KL")
        6. For online-only transactions (Shopee, Lazada, Grab delivery, etc.), location should be null
        7. For delivery services showing restaurant location (e.g., "GRBFOOD*MCDONALD BANGSAR"), extract "Bangsar" as location
        8. Match against the Malaysian location patterns provided above
        9. If no clear location match is found, set to null (DO NOT GUESS or make up locations)
        10. Extract location in its expanded, human-readable form (e.g., "Kuala Lumpur" not "KULJ")

        LOCATION EXTRACTION EXAMPLES:
        - "DBT POS SHELL DAMANSARA" → location: "Damansara"
        - "GRBPAY*7-ELEVEN MID V" → location: "Mid Valley"
        - "PETRONAS STN CHERAS KL" → location: "Cheras, Kuala Lumpur"
        - "SHOPEE*ORDER12345" → location: null (online transaction)
        - "GRAB FOOD DELIVERY" → location: null (delivery service without specific restaurant location)
        - "AEON MALL SHAH ALAM" → location: "Shah Alam"
        - "MCDONALD BANGSAR" → location: "Bangsar"
        - "STARBUCKS KLCC" → location: "KLCC"
        - "GRAB*RIDE" → location: null (ride service - pickup/dropoff not specified)

        Merchant Name Expansions:
        - MCDONALD / MCD → McDonald's
        - 7-ELV / 7ELV / 7-11 → 7-Eleven
        - PETRONAS STN / PDB → Petronas Station
        - SHELL STN → Shell Station
        - TESCO / TSC → Tesco
        - AEON / AON → AEON
        - GRAB FOOD / GRBFOOD → Grab Food Delivery
        - FOODPANDA / FPD → foodpanda
        - SHOPEE / SHP → Shopee
        - LAZADA / LZD → Lazada

        For each transaction, provide:
        {
          "date": "2025-01-15",
          "raw_description": "GRBPAY*MCDONALD 1234 KULJ",  // Exact text from statement
          "description": "McDonald's (Kuala Lumpur) via GrabPay",  // Human-readable version with location in parentheses if available
          "merchant": "McDonald's",  // Extracted merchant name (expanded, without codes)
          "location": "Kuala Lumpur",  // Extracted location in human-readable form (expanded abbreviations), null if not found
          "payment_method": "GrabPay",  // Extracted payment method (if identifiable), null otherwise
          "amount": -15.50,
          "type": "debit",
          "category": "Dining",
          "transfer_type": null  // ONLY set this field if transaction is a transfer. Options: "inter_person" (to another person), "intra_person" (to own account/savings), null (not a transfer)
        }

        TRANSFER TYPE CLASSIFICATION (CRITICAL - READ CAREFULLY):
        
        CRITICAL RULES:
        1. ALL transfers (both to others and to own accounts) MUST have category: "Transfer" (NOT "Other"!)
        2. Transfers to savings/own accounts are NOT expenses and should be classified as "intra_person"
        3. Transfers to others ARE expenses and should be classified as "inter_person"
        
        - If transaction description indicates a transfer, analyze if it's:
          1. "inter_person": Transfer to another person (friend, family, business partner, etc.) - THIS IS AN EXPENSE
             Examples: "Transfer to John", "DuitNow to 0123456789", "FPX transfer to ABC", "Online transfer to friend",
                       "Send money to", "Payment to", "Transfer to [person name]", "DuitNow transfer"
             Category: MUST be "Transfer" (NOT "Other"!)
             transfer_type: MUST be "inter_person"
             
          2. "intra_person": Transfer to own account/savings/investment - THIS IS NOT AN EXPENSE
             Examples: "Transfer to savings", "Self transfer", "Internal transfer", "Transfer to own account",
                       "Tabung Haji", "ASB", "SSPNI", "SSP1M", "Stash", "Goal transfer", "Auto-save", 
                       "Savings account", "tabung", "saving account", "investment account", "duitnow to self",
                       "standing instruction to savings", "auto transfer to savings", "top up savings",
                       "deposit to savings", "rainy day fund", "emergency fund"
             Category: MUST be "Transfer" (NOT "Other"!)
             transfer_type: MUST be "intra_person"
             
        - Key indicators for intra_person transfers (to own savings):
          * Contains: "savings", "saving", "tabung", "asb", "sspni", "ssp1m", "stash", "goal"
          * Contains: "own account", "self transfer", "internal transfer", "auto-save", "autosave"
          * Contains: "standing instruction", "auto transfer", "recurring transfer" (if to savings)
          * Contains: "top up", "top-up", "topup", "deposit to", "cash deposit"
          * Account name matches user's own account names
          * Malaysian savings schemes: "Tabung Haji", "ASB", "SSPNI", "SSP1M", "Amanah Saham", etc.
          
        - Key indicators for inter_person transfers (to others):
          * Contains: "transfer to [name]", "send money", "payment to", "duitnow to [phone number]"
          * Contains: "friend", "family", "colleague", "business partner"
          * No savings/own account keywords present
          
        - If NOT a transfer, set "transfer_type" to null
        - Use AI intelligence to analyze context clues in the description
        - When in doubt about savings transfers, classify as "intra_person" to avoid incorrectly counting as expense
        - REMEMBER: ALL transfers use category "Transfer" - the transfer_type field distinguishes inter_person vs intra_person

        CRITICAL LOCATION EXTRACTION EXAMPLES:
        Example 1: "GRBPAY*MCDONALD 1234 KULJ"
        - merchant: "McDonald's"
        - location: "Kuala Lumpur" (expanded from KULJ)
        - description: "McDonald's (Kuala Lumpur) via GrabPay"

        Example 2: "DBT POS PETRONAS DAMANSARA"
        - merchant: "Petronas"
        - location: "Damansara"
        - description: "Petronas (Damansara)"

        Example 3: "SHOPEE*ORD123456"
        - merchant: "Shopee"
        - location: null (online purchase)
        - description: "Shopee online purchase"

        Example 4: "STARBUCKS MID VALLEY KL"
        - merchant: "Starbucks"
        - location: "Mid Valley, Kuala Lumpur" (combined specific + city)
        - description: "Starbucks (Mid Valley, Kuala Lumpur)"

        Example 5: "7-ELEVEN"
        - merchant: "7-Eleven"
        - location: null (no location indicator in description)
        - description: "7-Eleven"

        If the description is already in natural language, keep it as is but still extract location if present.
        If you cannot parse the coded description, use the raw text for "description" field and attempt to extract location using the patterns above.

        If a field is not visible on this page, use null.
        Extract ALL transactions - don't skip any.
        Be precise with amounts, dates, and categories.

        Return ONLY the JSON object, no markdown formatting.
        """

        # Generate content with image
        response = model.generate_content([
            prompt,
            {'mime_type': 'image/jpeg', 'data': image_bytes}
        ])

        # Parse JSON response
        json_text = response.text.strip()

        # Log raw AI response for debugging
        logger.info(f"Page {page_number}: Raw AI response length: {len(json_text)} characters")
        logger.debug(f"Page {page_number}: Raw AI response: {json_text[:500]}...")  # First 500 chars

        # Remove markdown code blocks if present
        if json_text.startswith('```json'):
            json_text = json_text[7:]
        if json_text.startswith('```'):
            json_text = json_text[3:]
        if json_text.endswith('```'):
            json_text = json_text[:-3]
        json_text = json_text.strip()

        # Parse JSON
        data = json.loads(json_text)

        # Log complete AI response for debugging (helpful for troubleshooting extraction issues)
        logger.info(f"Page {page_number}: ====== COMPLETE AI JSON RESPONSE ======")
        logger.info(json.dumps(data, indent=2))
        logger.info(f"Page {page_number}: ====== END AI RESPONSE ======")

        # Log extracted data structure
        logger.info(f"Page {page_number}: Successfully parsed AI response")
        logger.info(f"Page {page_number}: Extracted {len(data.get('transactions', []))} transactions")

        # Log balance information
        balances = data.get('balances', {})
        if balances:
            opening = balances.get('opening_balance')
            closing = balances.get('closing_balance')
            logger.info(f"Page {page_number}: Balances - Opening: {opening}, Closing: {closing}")
        else:
            logger.warning(f"Page {page_number}: No balance information found in AI response")

        # Log account info
        if data.get('account_info'):
            logger.info(f"Page {page_number}: Account info extracted: {data.get('account_info')}")

        # Log statement period
        if data.get('statement_period'):
            logger.info(f"Page {page_number}: Statement period: {data.get('statement_period')}")

        # Log credit card terms if present
        if data.get('credit_card_terms'):
            logger.info(f"Page {page_number}: Credit card terms extracted")

        return data

    except json.JSONDecodeError as e:
        logger.error(f"Page {page_number}: Failed to parse AI response as JSON: {str(e)}")
        return {
            "page_number": page_number,
            "error": f"Failed to parse AI response: {str(e)}",
            "transactions": []
        }
    except Exception as e:
        logger.error(f"Page {page_number}: AI extraction failed: {str(e)}")
        return {
            "page_number": page_number,
            "error": f"Extraction failed: {str(e)}",
            "transactions": []
        }

def process_statement_pdf(pdf_bytes: bytes) -> Dict[str, Any]:
    """
    Process complete bank statement PDF

    Args:
        pdf_bytes: PDF file as bytes

    Returns:
        Dictionary with all extracted data from all pages
    """
    try:
        logger.info("Starting statement PDF processing")

        # Step 1: Convert PDF to images
        images = convert_pdf_to_images(pdf_bytes)
        total_pages = len(images)

        if total_pages == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="PDF has no pages"
            )

        logger.info(f"Processing {total_pages} pages")

        # Step 2: Process each page
        all_transactions = []
        statement_period = None
        account_info = None
        user_info = None
        opening_balance = None
        closing_balance = None
        credit_card_terms = None
        errors = []

        for i, image in enumerate(images, 1):
            logger.info(f"Processing page {i}/{total_pages}")

            # Convert image to bytes
            image_bytes = image_to_bytes(image)

            # Extract data from page
            page_data = extract_transactions_from_image(image_bytes, i)

            # Collect errors
            if 'error' in page_data:
                errors.append(f"Page {i}: {page_data['error']}")

            # Extract statement period (usually on first page)
            if not statement_period and page_data.get('statement_period'):
                period = page_data['statement_period']
                if period.get('start_date') or period.get('end_date'):
                    statement_period = period

            # Extract account info (usually on first page)
            if not account_info and page_data.get('account_info'):
                account_info = page_data['account_info']

                # Apply card_brand fallback detection if not detected, null, empty, or "Unknown"
                current_brand = account_info.get('card_brand')
                if not current_brand or current_brand == '' or current_brand.lower() == 'unknown':
                    detected_brand = detect_card_brand(
                        account_name=account_info.get('account_name'),
                        account_type=account_info.get('account_type'),
                        card_brand=current_brand if current_brand and current_brand.lower() != 'unknown' else None
                    )
                    account_info['card_brand'] = detected_brand
                    logger.info(f"Applied card_brand fallback detection: {detected_brand} (was: {current_brand})")

            # Extract user info (usually on first page)
            if not user_info and page_data.get('user_info'):
                user_info = page_data['user_info']

            # Extract credit card terms (usually on first page, only for credit card statements)
            if not credit_card_terms and page_data.get('credit_card_terms'):
                credit_card_terms = page_data['credit_card_terms']
                logger.info(f"Extracted credit card terms: {credit_card_terms}")

            # Extract balances
            if page_data.get('balances'):
                page_opening = page_data['balances'].get('opening_balance')
                page_closing = page_data['balances'].get('closing_balance')

                logger.info(f"Page {i}: Found balances in page data - Opening: {page_opening}, Closing: {page_closing}")

                # Use explicit None checks instead of truthiness to allow 0.0 as valid balance
                if page_opening is not None and opening_balance is None:
                    opening_balance = page_opening
                    logger.info(f"Page {i}: Set opening balance to {opening_balance}")

                if page_closing is not None:
                    closing_balance = page_closing
                    logger.info(f"Page {i}: Updated closing balance to {closing_balance}")
            else:
                logger.warning(f"Page {i}: No balances found in page data")

            # Collect transactions
            transactions = page_data.get('transactions', [])
            logger.info(f"Page {i}: Adding {len(transactions)} transactions to collection")
            all_transactions.extend(transactions)

        # Step 3: Post-process transactions
        # AI now provides categories, but fallback to keyword matching if not provided
        for transaction in all_transactions:
            # Only use fallback categorization if AI didn't provide a category
            if 'category' not in transaction or not transaction['category']:
                logger.warning(f"Transaction missing AI category, using fallback: {transaction.get('description', 'Unknown')}")

                # Fallback to keyword-based categorization
                if transaction['type'] == 'debit':
                    transaction['category'] = guess_category(transaction['description'])
                else:
                    # For income, categorize based on description
                    desc = transaction['description'].lower()
                    if 'salary' in desc or 'payroll' in desc:
                        transaction['category'] = 'Salary'
                    elif 'transfer' in desc or 'online transfer' in desc:
                        transaction['category'] = 'Transfer'
                    elif 'interest' in desc or 'dividend' in desc:
                        transaction['category'] = 'Investments'
                    else:
                        transaction['category'] = 'Other'
            else:
                logger.info(f"Using AI category '{transaction['category']}' for: {transaction.get('description', 'Unknown')}")

        # Step 4: Apply fallback for missing statement dates using transaction dates
        if all_transactions:
            # Get earliest and latest transaction dates
            transaction_dates = [txn['date'] for txn in all_transactions if txn.get('date')]

            if transaction_dates:
                earliest_date = min(transaction_dates)
                latest_date = max(transaction_dates)

                # Fallback for missing start_date
                if not statement_period.get('start_date'):
                    statement_period['start_date'] = earliest_date
                    logger.info(f"Applied fallback: Inferred start_date from earliest transaction: {earliest_date}")

                # Fallback for missing end_date
                if not statement_period.get('end_date'):
                    statement_period['end_date'] = latest_date
                    logger.info(f"Applied fallback: Inferred end_date from latest transaction: {latest_date}")

        # Step 5: Build result
        result = {
            "success": True,
            "total_pages": total_pages,
            "total_transactions": len(all_transactions),
            "statement_period": statement_period,
            "account_info": account_info,
            "user_info": user_info,
            "opening_balance": opening_balance,
            "closing_balance": closing_balance,
            "transactions": all_transactions,
            "errors": errors if errors else None,
            "processed_at": datetime.now().isoformat()
        }

        # Add credit card terms if this is a credit card statement
        if credit_card_terms:
            result["credit_card_terms"] = credit_card_terms
            result["credit_card_summary"] = build_credit_card_summary(credit_card_terms, closing_balance)

        # Log final aggregated data
        logger.info("="*80)
        logger.info("FINAL EXTRACTION SUMMARY")
        logger.info("="*80)
        logger.info(f"Total pages processed: {total_pages}")
        logger.info(f"Total transactions extracted: {len(all_transactions)}")
        logger.info(f"Statement period: {statement_period}")
        logger.info(f"Account info: {account_info}")
        logger.info(f"User info: {user_info}")
        logger.info(f"Opening balance: {opening_balance}")
        logger.info(f"Closing balance: {closing_balance}")

        if opening_balance is None and closing_balance is None:
            logger.warning("WARNING: No balance information was extracted from any page!")
        elif opening_balance is None:
            logger.warning("WARNING: Opening balance was not found")
        elif closing_balance is None:
            logger.warning("WARNING: Closing balance was not found")
        else:
            logger.info(f"Balance range: {opening_balance} → {closing_balance}")

        if credit_card_terms:
            logger.info(f"Credit card terms: {credit_card_terms}")
            logger.info(f"Processing complete: Credit card statement with {len(all_transactions)} transactions")
        else:
            logger.info(f"Processing complete: Bank/E-wallet statement with {len(all_transactions)} transactions")

        if errors:
            logger.warning(f"Errors encountered: {errors}")

        logger.info("="*80)

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Statement processing failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process statement: {str(e)}"
        )

def validate_transaction_data(transaction: Dict[str, Any]) -> bool:
    """
    Validate that a transaction has required fields

    Args:
        transaction: Transaction dictionary

    Returns:
        True if valid, False otherwise
    """
    required_fields = ['date', 'description', 'amount', 'type']

    for field in required_fields:
        if field not in transaction or transaction[field] is None:
            return False

    # Validate date format
    try:
        datetime.strptime(transaction['date'], '%Y-%m-%d')
    except:
        return False

    # Validate amount is a number
    try:
        float(transaction['amount'])
    except:
        return False

    # Validate type
    if transaction['type'] not in ['credit', 'debit']:
        return False

    return True

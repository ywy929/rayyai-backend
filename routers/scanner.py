"""
Receipt Scanner API Endpoint
Provides AI-powered receipt scanning functionality
"""

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from PIL import Image
import io
import os
import json
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai

import models
from routers.utils import get_current_user

# Load environment variables
load_dotenv()

# Configure Gemini AI
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# List available models for debugging
def list_available_models():
    """Debug function to list available models"""
    try:
        models_list = genai.list_models()
        return [m.name for m in models_list if 'generateContent' in m.supported_generation_methods]
    except Exception as e:
        return f"Error listing models: {str(e)}"

router = APIRouter()

# Categories for expense categorization
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

def guess_category(merchant_name: str) -> str:
    """Guess expense category based on merchant name"""
    merchant_lower = merchant_name.lower()

    for category, keywords in EXPENSE_CATEGORIES.items():
        if any(keyword in merchant_lower for keyword in keywords):
            return category

    return "Other"

def extract_with_ai(image_bytes: bytes):
    """Extract transaction details using Gemini Vision AI"""
    try:
        if not GEMINI_API_KEY:
            return None, "Gemini API key not configured"

        # Initialize Gemini model with vision capabilities
        # Using gemini-2.0-flash (stable model with vision)
        model = genai.GenerativeModel('gemini-2.0-flash')

        # Create prompt for extraction with AI categorization
        prompt = """
        Analyze this receipt image and extract the following information in JSON format:

        {
          "merchant_name": "The store/merchant name (e.g., FamilyMart, 7-Eleven, Starbucks)",
          "amount": 12.50,
          "date": "2025-10-22",
          "reference_number": "Receipt or invoice number if visible",
          "items": ["List of items purchased if visible"],
          "category": "Groceries",
          "raw_text": "All text visible on the receipt"
        }

        Rules:
        1. Extract EXACT store name as it appears on receipt
        2. **CRITICAL - Amount extraction rules:**
           - Extract the TOTAL PRICE or GRAND TOTAL (the actual purchase amount)
           - DO NOT use "Amount Paid", "Cash", "Tendered", "Payment" amount
           - DO NOT use "Change", "Balance", "Change Due" amount
           - Look for labels like: "Total", "Grand Total", "Amount Due", "Net Total", "Sub Total + Tax"
           - If receipt shows: Amount Paid: 20, Total: 8.50, Change: 11.50 → Use 8.50 (the Total)
           - The amount should be what you actually spent, NOT what you paid or received back
        3. Date should be in YYYY-MM-DD format
        4. Categorize the expense based on merchant and items (see categories below)
        5. Include ALL text you can see in raw_text field
        6. If something is not visible, use null or empty string
        7. Return ONLY valid JSON, no explanations or markdown

        EXPENSE CATEGORIES - Choose the MOST appropriate:

        - "Groceries": Supermarkets, convenience stores, wet markets, fresh produce
          Examples: FamilyMart, 7-Eleven, Tesco, Giant, AEON, MyDin, 99 Speedmart, KK Mart

        - "Dining": Restaurants, cafes, fast food, food delivery, beverages
          Examples: McDonald's, KFC, Starbucks, Old Town, Kopitiam, Secret Recipe, food court

        - "Transportation": Fuel, parking, ride-sharing, tolls
          Examples: Shell, Petronas, Grab (rides), parking, Touch n Go

        - "Shopping": Retail stores, fashion, electronics, online shopping
          Examples: Uniqlo, Lazada, Mr DIY, H&M, Zara, shopping mall

        - "Entertainment": Movies, streaming, games, theme parks, leisure
          Examples: Cinema, Netflix, Spotify, gym, theme park

        - "Healthcare": Pharmacies, clinics, medical supplies
          Examples: Guardian, Watsons, pharmacy, clinic

        - "Bills & Utilities": Electric, water, internet, phone bills
          Examples: Bill payments, utilities

        - "Education": Books, courses, tuition, learning materials
          Examples: Bookstore, Popular, tuition center

        - "Travel": Hotels, flights, tourism, accommodation
          Examples: Hotel receipts, flight tickets, car rental

        - "Insurance": Insurance receipts, policy payments
          Examples: Insurance payment receipts

        - "Personal Care": Salons, spas, beauty products, wellness
          Examples: Salon, spa, barbershop, beauty store

        - "Other": If none of the above clearly fit

        IMPORTANT:
        - Look at both merchant name AND items to determine category
        - Starbucks receipt with food items → "Dining"
        - 7-Eleven with snacks → "Groceries" or "Dining" (use judgment)
        - Shell with car wash → "Transportation"
        - Consider Malaysian context

        Return ONLY valid JSON.
        """

        # Generate content with image
        response = model.generate_content([
            prompt,
            {'mime_type': 'image/jpeg', 'data': image_bytes}
        ])

        # Parse JSON response
        json_text = response.text.strip()

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

        return data, None

    except json.JSONDecodeError as e:
        return None, f"Failed to parse AI response as JSON: {str(e)}"
    except Exception as e:
        return None, f"AI extraction failed: {str(e)}"

@router.get("/available-models")
async def get_available_models(
    current_user: models.User = Depends(get_current_user)
):
    """
    Debug endpoint to list available Gemini models
    """
    if not GEMINI_API_KEY:
        return {"error": "Gemini API key not configured"}

    return {"models": list_available_models()}

@router.post("/scan-receipt")
async def scan_receipt(
    file: UploadFile = File(...),
    current_user: models.User = Depends(get_current_user)
):
    """
    Scan a receipt image and extract transaction details using AI
    """

    # Check if AI is configured
    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI scanning service is not configured"
        )

    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be an image (PNG, JPG, JPEG)"
        )

    try:
        # Read image file
        image_bytes = await file.read()

        # Validate image can be opened
        try:
            image = Image.open(io.BytesIO(image_bytes))
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Convert back to bytes
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='JPEG')
            image_bytes = img_byte_arr.getvalue()
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid image file: {str(e)}"
            )

        # Extract with AI
        ai_data, ai_error = extract_with_ai(image_bytes)

        if ai_error:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=ai_error
            )

        if not ai_data:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to extract data from receipt"
            )

        # Process extracted data
        merchant_name = ai_data.get('merchant_name', 'Unknown')
        amount = ai_data.get('amount', 0)
        date_str = ai_data.get('date', datetime.now().strftime('%Y-%m-%d'))
        reference = ai_data.get('reference_number', '')
        items = ai_data.get('items', [])
        raw_text = ai_data.get('raw_text', '')

        # Validate and parse amount
        try:
            amount = float(amount)
            if amount <= 0:
                amount = 1.0
        except (ValueError, TypeError):
            amount = 1.0

        # Validate date format
        try:
            parsed_date = datetime.strptime(date_str, '%Y-%m-%d')
            date_str = parsed_date.strftime('%Y-%m-%d')
        except:
            date_str = datetime.now().strftime('%Y-%m-%d')

        # Use AI-provided category, fallback to keyword matching if not provided
        ai_category = ai_data.get('category', '')
        if ai_category:
            category = ai_category
            print(f"Using AI category '{category}' for merchant: {merchant_name}")
        else:
            # Fallback to keyword-based categorization
            category = guess_category(merchant_name)
            print(f"AI category missing, using fallback '{category}' for merchant: {merchant_name}")

        # Build response
        response = {
            "merchant": merchant_name,
            "amount": amount,
            "date": date_str,
            "reference": reference,
            "category": category,
            "description": f"Purchase from {merchant_name}",
            "items": items,
            "raw_text": raw_text,
            "extraction_method": "AI Vision (Gemini)",
            "ai_categorized": bool(ai_category)  # Flag to indicate if AI provided category
        }

        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process receipt: {str(e)}"
        )

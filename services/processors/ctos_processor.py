"""
CTOS Credit Report Processor
Extracts credit score and information from CTOS credit reports using Gemini Vision AI
"""

import os
import json
from dotenv import load_dotenv
import google.generativeai as genai
from typing import Dict, Any
import logging

# Import shared utilities from statement_processor
from services.processors.statement_processor import convert_pdf_to_images, image_to_bytes

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configure Gemini AI
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


def process_ctos_pdf(pdf_bytes: bytes) -> Dict[str, Any]:
    """
    Extract comprehensive credit information from CTOS credit report PDF using Gemini Vision AI
    
    Args:
        pdf_bytes: CTOS PDF file as bytes
        
    Returns:
        Dictionary containing all extracted CTOS data including:
        - Personal identification details
        - CTOS Score & Risk Factors
        - Bankruptcy, Legal & Special Attention Records
        - Credit Facility Summary
        - Full CCRIS Loan Details
        - Conduct of Payment
        - Credit Utilisation Metrics
        - Loan Applications
        - Employment/Business Information
        - PTPTN Status
    """
    try:
        if not GEMINI_API_KEY:
            return {
                "success": False,
                "error": "Gemini API key not configured"
            }
        
        logger.info("Processing CTOS credit report with Gemini Vision AI")
        
        # Convert PDF to images
        images = convert_pdf_to_images(pdf_bytes, dpi=200)
        
        if not images:
            return {
                "success": False,
                "error": "Failed to convert PDF to images"
            }
        
        # Initialize Gemini model
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Process all pages (CTOS reports can have detailed info across many pages)
        pages_to_process = min(10, len(images))  # Process up to 10 pages
        all_extracted_data = []
        
        for page_num in range(pages_to_process):
            logger.info(f"Processing CTOS page {page_num + 1}/{pages_to_process}")
            
            # Convert image to bytes
            image_bytes = image_to_bytes(images[page_num])
            
            # Comprehensive prompt for detailed CTOS credit report extraction
            prompt = """
            Analyze this CTOS credit report page and extract ALL available information in JSON format.
            
            IMPORTANT: Extract as much detail as possible from this page. Look for:
            
            1. PERSONAL IDENTIFICATION DETAILS:
               - Full Name (e.g., "MUHAMMAD FARIS AL-HELMI BIN MAD KAMAL")
               - IC/NRIC number (Malaysian ID, may be partially masked)
               - Date of Birth (format: YYYY-MM-DD)
               - Nationality (usually "Malaysia")
               - Address Line 1 and Address Line 2
            
            2. CTOS SCORE & RISK FACTORS:
               - CTOS Score (number between 300-850)
               - Score text/rating ("Excellent", "Very Good", "Good", "Fair", "Poor")
               - Risk factors (array of strings like ["Too many recent credit applications", "High loan utilisation", "Last enquiry is too recent", "Short account history"])
            
            3. BANKRUPTCY, LEGAL & SPECIAL ATTENTION RECORDS:
               - Bankruptcy status (true/false)
               - Legal records (personal) in last 24 months (count)
               - Legal records (non-personal) in last 24 months (count)
               - Special Attention Accounts (true/false)
               - Trade Referee Listing (true/false)
            
            4. CREDIT FACILITY SUMMARY (CCRIS Overview):
               - Total outstanding balance (RM amount)
               - Total credit limit (RM amount)
               - Credit applications in last 12 months: total, approved, pending (counts)
            
            5. FULL CCRIS LOAN DETAILS (for each facility):
               - Facility number (#1, #2, etc.)
               - Facility type (e.g., "CRDTCARD", "OTLNFNCE", "PCPASCAR", "PELNFNCE")
               - Facility name (e.g., "Credit Card", "Term Financing", "Car Loan", "Personal Financing")
               - Bank name (e.g., "Maybank Islamic")
               - Credit limit (RM amount)
               - Outstanding balance (RM amount)
               - Collateral type (e.g., "Clean (00)", "Unit Trust (23)", "Motor Vehicle (JPJ) (30)")
               - Collateral code (e.g., "00", "23", "30")
               - Conduct of payment for last 12 months (array of 12 numbers, 0 = good payment)
            
            6. CREDIT UTILISATION METRICS:
               - Earliest known facility date (YYYY-MM-DD)
               - Total outstanding (RM amount)
               - Outstanding as percentage of limit (e.g., 90.0 for 90%)
               - Number of unsecured facilities (count)
               - Number of secured facilities (count)
               - Average credit card utilisation over last 6 months (percentage)
               - Average revolving credit utilisation over last 6 months (percentage)
            
            7. LOAN APPLICATIONS (recent applications in last 12 months):
               - Application date (YYYY-MM-DD)
               - Application type (e.g., "credit_card", "personal_loan")
               - Amount (RM)
               - Status ("Approved", "Pending", "Rejected")
               - Lender name
            
            8. EMPLOYMENT / BUSINESS INFORMATION:
               - Has directorships (true/false)
               - Number of directorships (count)
               - Has business interests (true/false)
               - Number of business interests (count)
            
            9. PTPTN STATUS:
               - Number of PTPTN loans (count)
               - Local lenders count (count)
               - Foreign lenders count (count)
            
            10. REPORT METADATA:
                - Report date (when report was generated, YYYY-MM-DD)
                - Period start date (YYYY-MM-DD)
                - Period end date (YYYY-MM-DD)
            
            Return ONLY valid JSON in this structure:
            {
              "page_number": 1,
              "personal_info": {
                "full_name": "MUHAMMAD FARIS AL-HELMI BIN MAD KAMAL",
                "ic_nric": "971226105799",
                "date_of_birth": "1997-12-26",
                "nationality": "Malaysia",
                "address_line1": "...",
                "address_line2": "..."
              },
              "ctos_score": {
                "score": 713,
                "score_text": "Good",
                "risk_factors": ["Too many recent credit applications", "High loan utilisation"]
              },
              "legal_records": {
                "is_bankrupt": false,
                "legal_records_personal_24m": 0,
                "legal_records_non_personal_24m": 0,
                "has_special_attention_accounts": false,
                "has_trade_referee_listing": false
              },
              "credit_facility_summary": {
                "total_outstanding_balance": 146098.00,
                "total_credit_limit": 165466.00,
                "credit_applications_12m_total": 1,
                "credit_applications_12m_approved": 1,
                "credit_applications_12m_pending": 0
              },
              "credit_facilities": [
                {
                  "facility_number": 1,
                  "facility_type": "CRDTCARD",
                  "facility_name": "Credit Card",
                  "bank_name": "Maybank Islamic",
                  "credit_limit": 6000.00,
                  "outstanding_balance": 1104.00,
                  "collateral_type": "Clean (00)",
                  "collateral_code": "00",
                  "conduct_12m": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                }
              ],
              "credit_utilisation": {
                "earliest_known_facility_date": "2022-09-15",
                "total_outstanding": 126105.00,
                "outstanding_percentage_of_limit": 90.0,
                "number_of_unsecured_facilities": 2,
                "number_of_secured_facilities": 2,
                "avg_utilisation_credit_card_6m": 32.14,
                "avg_utilisation_revolving_6m": 0.0
              },
              "loan_applications": [
                {
                  "application_date": "2024-01-15",
                  "application_type": "credit_card",
                  "amount": 20000.00,
                  "status": "Approved",
                  "lender_name": "..."
                }
              ],
              "employment_info": {
                "has_directorships": false,
                "directorships_count": 0,
                "has_business_interests": false,
                "business_interests_count": 0
              },
              "ptptn_status": {
                "number_of_ptptn_loans": 0,
                "local_lenders_count": 4,
                "foreign_lenders_count": 0
              },
              "report_metadata": {
                "report_date": "2025-01-15",
                "period_start": "2024-01-01",
                "period_end": "2024-12-31"
              }
            }
            
            If information is not found on this page, use null for missing fields.
            For arrays (like credit_facilities, loan_applications), return empty array [] if none found.
            """
            
            try:
                # Call Gemini Vision API
                response = model.generate_content(
                    [prompt, {"mime_type": "image/jpeg", "data": image_bytes}]
                )
                
                # Extract JSON from response
                response_text = response.text.strip()
                
                # Remove markdown code blocks if present
                if response_text.startswith("```json"):
                    response_text = response_text[7:]
                if response_text.startswith("```"):
                    response_text = response_text[3:]
                if response_text.endswith("```"):
                    response_text = response_text[:-3]
                response_text = response_text.strip()
                
                # Parse JSON
                page_data = json.loads(response_text)
                all_extracted_data.append(page_data)
                
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON from page {page_num + 1}: {str(e)}")
                logger.warning(f"Response text: {response_text[:500]}")
                continue
            except Exception as e:
                logger.warning(f"Error processing page {page_num + 1}: {str(e)}")
                continue
        
        if not all_extracted_data:
            return {
                "success": False,
                "error": "Failed to extract data from any page"
            }
        
        # Merge data from all pages
        merged_data = {
            "personal_info": {},
            "ctos_score": {},
            "legal_records": {},
            "credit_facility_summary": {},
            "credit_facilities": [],
            "credit_utilisation": {},
            "loan_applications": [],
            "employment_info": {},
            "ptptn_status": {},
            "report_metadata": {}
        }
        
        # Track seen facilities and applications to avoid duplicates
        seen_facilities = set()
        seen_applications = set()
        
        for page_data in all_extracted_data:
            # Merge personal_info
            if page_data.get("personal_info"):
                for key, value in page_data.get("personal_info", {}).items():
                    if value is not None and key not in merged_data["personal_info"]:
                        merged_data["personal_info"][key] = value
            
            # Merge ctos_score
            if page_data.get("ctos_score"):
                for key, value in page_data.get("ctos_score", {}).items():
                    if value is not None and key not in merged_data["ctos_score"]:
                        merged_data["ctos_score"][key] = value
            
            # Merge legal_records
            if page_data.get("legal_records"):
                for key, value in page_data.get("legal_records", {}).items():
                    if value is not None and key not in merged_data["legal_records"]:
                        merged_data["legal_records"][key] = value
            
            # Merge credit_facility_summary
            if page_data.get("credit_facility_summary"):
                for key, value in page_data.get("credit_facility_summary", {}).items():
                    if value is not None and key not in merged_data["credit_facility_summary"]:
                        merged_data["credit_facility_summary"][key] = value
            
            # Merge credit_facilities (avoid duplicates by facility_number + bank_name)
            if page_data.get("credit_facilities"):
                for facility in page_data.get("credit_facilities", []):
                    facility_key = (facility.get("facility_number"), facility.get("bank_name"))
                    if facility_key not in seen_facilities:
                        seen_facilities.add(facility_key)
                        merged_data["credit_facilities"].append(facility)
            
            # Merge credit_utilisation
            if page_data.get("credit_utilisation"):
                for key, value in page_data.get("credit_utilisation", {}).items():
                    if value is not None and key not in merged_data["credit_utilisation"]:
                        merged_data["credit_utilisation"][key] = value
            
            # Merge loan_applications (avoid duplicates by date + type + amount)
            if page_data.get("loan_applications"):
                for app in page_data.get("loan_applications", []):
                    app_key = (app.get("application_date"), app.get("application_type"), app.get("amount"))
                    if app_key not in seen_applications:
                        seen_applications.add(app_key)
                        merged_data["loan_applications"].append(app)
            
            # Merge employment_info
            if page_data.get("employment_info"):
                for key, value in page_data.get("employment_info", {}).items():
                    if value is not None and key not in merged_data["employment_info"]:
                        merged_data["employment_info"][key] = value
            
            # Merge ptptn_status
            if page_data.get("ptptn_status"):
                for key, value in page_data.get("ptptn_status", {}).items():
                    if value is not None and key not in merged_data["ptptn_status"]:
                        merged_data["ptptn_status"][key] = value
            
            # Merge report_metadata
            if page_data.get("report_metadata"):
                for key, value in page_data.get("report_metadata", {}).items():
                    if value is not None and key not in merged_data["report_metadata"]:
                        merged_data["report_metadata"][key] = value
        
        # Validate and set defaults
        if merged_data["ctos_score"].get("score") is not None:
            score = merged_data["ctos_score"]["score"]
            if not isinstance(score, int) or score < 300 or score > 850:
                logger.warning(f"Invalid credit score {score}, setting to None")
                merged_data["ctos_score"]["score"] = None
            elif not merged_data["ctos_score"].get("score_text"):
                # Auto-determine score text if not provided
                if score >= 750:
                    merged_data["ctos_score"]["score_text"] = "Excellent"
                elif score >= 700:
                    merged_data["ctos_score"]["score_text"] = "Very Good"
                elif score >= 650:
                    merged_data["ctos_score"]["score_text"] = "Good"
                elif score >= 600:
                    merged_data["ctos_score"]["score_text"] = "Fair"
                else:
                    merged_data["ctos_score"]["score_text"] = "Poor"
        
        # Extract legacy fields for backward compatibility
        credit_score = merged_data["ctos_score"].get("score")
        score_text = merged_data["ctos_score"].get("score_text")
        report_date = merged_data["report_metadata"].get("report_date")
        period_start = merged_data["report_metadata"].get("period_start")
        period_end = merged_data["report_metadata"].get("period_end")
        
        logger.info(f"Successfully extracted CTOS data: score={credit_score}, period={period_start} to {period_end}")
        logger.info(f"Found {len(merged_data['credit_facilities'])} credit facilities")
        logger.info(f"Found {len(merged_data['loan_applications'])} loan applications")
        
        return {
            "success": True,
            # Legacy fields for backward compatibility
            "credit_score": credit_score,
            "score_text": score_text,
            "report_date": report_date,
            "period_start": period_start,
            "period_end": period_end,
            # Detailed structured data
            "personal_info": merged_data["personal_info"] if merged_data["personal_info"] else None,
            "ctos_score": merged_data["ctos_score"] if merged_data["ctos_score"] else None,
            "legal_records": merged_data["legal_records"] if merged_data["legal_records"] else None,
            "credit_facility_summary": merged_data["credit_facility_summary"] if merged_data["credit_facility_summary"] else None,
            "credit_facilities": merged_data["credit_facilities"] if merged_data["credit_facilities"] else [],
            "credit_utilisation": merged_data["credit_utilisation"] if merged_data["credit_utilisation"] else None,
            "loan_applications": merged_data["loan_applications"] if merged_data["loan_applications"] else [],
            "employment_info": merged_data["employment_info"] if merged_data["employment_info"] else None,
            "ptptn_status": merged_data["ptptn_status"] if merged_data["ptptn_status"] else None,
        }
        
    except Exception as e:
        logger.error(f"CTOS processing failed: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": f"Failed to process CTOS report: {str(e)}"
        }


"""
AI-powered financial insights endpoints.

Generates culturally-aware smart analysis using Gemini for the selected period.
"""
from __future__ import annotations

import calendar
import json
import logging
from collections import defaultdict, Counter
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

import models
import schemas
from database import get_db
from routers.utils import get_current_user
from services.gemini_service import GeminiService
from services.rag_service import RAGService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/insights", tags=["Insights"])

_gemini_service: Optional[GeminiService] = None


def get_gemini_service() -> GeminiService:
    """Return a singleton Gemini service instance."""
    global _gemini_service
    if _gemini_service is None:
        _gemini_service = GeminiService()
    return _gemini_service


def _period_bounds(view_mode: str, selected: date) -> Tuple[date, date, str]:
    """Calculate inclusive start/end dates and a friendly label for the selected period."""
    today = date.today()
    if view_mode == "yearly":
        start = date(selected.year, 1, 1)
        end = date(selected.year, 12, 31)
        if selected.year == today.year and today < end:
            end = today
        label = str(selected.year)
    else:
        start = date(selected.year, selected.month, 1)
        last_day = calendar.monthrange(selected.year, selected.month)[1]
        end = date(selected.year, selected.month, last_day)
        if selected.year == today.year and selected.month == today.month and today < end:
            end = today
        label = start.strftime("%B %Y")
    return start, end, label


def _previous_period(view_mode: str, start: date, end: date) -> Tuple[date, date]:
    """Return the previous period matching the selected window."""
    if view_mode == "yearly":
        prev_start = date(start.year - 1, 1, 1)
        prev_end = date(start.year - 1, 12, 31)
    else:
        prev_month = start.month - 1
        prev_year = start.year
        if prev_month == 0:
            prev_month = 12
            prev_year -= 1
        prev_start = date(prev_year, prev_month, 1)
        last_day = calendar.monthrange(prev_year, prev_month)[1]
        prev_end = date(prev_year, prev_month, last_day)
    return prev_start, prev_end


def _safe_ratio(numerator: float, denominator: float) -> Optional[float]:
    """Protect against division by zero."""
    if denominator == 0:
        return None
    return numerator / denominator


def _analyze_location_patterns(top_locations: List[Dict[str, Any]], user_location: str, country: str) -> Dict[str, Any]:
    """
    Analyze location-based spending patterns and provide regional insights.
    Uses API logic for structured data that Gemini can synthesize.
    """
    insights = {
        "primary_locations": [],
        "location_diversity": 0,
        "regional_patterns": [],
        "cost_indicators": [],
    }
    
    if not top_locations:
        return insights
    
    # Extract primary spending locations
    insights["primary_locations"] = top_locations[:5]  # Top 5 locations
    
    # Calculate location diversity (number of unique locations)
    insights["location_diversity"] = len(top_locations)
    
    # Malaysian city/region mapping for cost-of-living context
    malaysian_cities = {
        "kuala lumpur": {"region": "Central", "cost_level": "high", "avg_monthly": 3500},
        "kl": {"region": "Central", "cost_level": "high", "avg_monthly": 3500},
        "klcc": {"region": "Central", "cost_level": "very_high", "avg_monthly": 4500},
        "petaling jaya": {"region": "Central", "cost_level": "high", "avg_monthly": 3200},
        "pj": {"region": "Central", "cost_level": "high", "avg_monthly": 3200},
        "subang jaya": {"region": "Central", "cost_level": "medium_high", "avg_monthly": 3000},
        "shah alam": {"region": "Central", "cost_level": "medium", "avg_monthly": 2800},
        "damansara": {"region": "Central", "cost_level": "high", "avg_monthly": 3300},
        "cheras": {"region": "Central", "cost_level": "medium", "avg_monthly": 2700},
        "bangsar": {"region": "Central", "cost_level": "very_high", "avg_monthly": 4000},
        "mid valley": {"region": "Central", "cost_level": "high", "avg_monthly": 3500},
        "penang": {"region": "Northern", "cost_level": "medium", "avg_monthly": 2500},
        "george town": {"region": "Northern", "cost_level": "medium", "avg_monthly": 2500},
        "johor bahru": {"region": "Southern", "cost_level": "medium", "avg_monthly": 2600},
        "jb": {"region": "Southern", "cost_level": "medium", "avg_monthly": 2600},
        "melaka": {"region": "Southern", "cost_level": "medium", "avg_monthly": 2400},
        "malacca": {"region": "Southern", "cost_level": "medium", "avg_monthly": 2400},
        "ipoh": {"region": "Northern", "cost_level": "low_medium", "avg_monthly": 2200},
        "kota kinabalu": {"region": "Sabah", "cost_level": "medium", "avg_monthly": 2500},
        "kuching": {"region": "Sarawak", "cost_level": "medium", "avg_monthly": 2400},
    }
    
    # Analyze regional patterns
    if country.lower() in {"malaysia", "my", ""}:
        location_regions = {}
        for loc_data in top_locations:
            loc_name = loc_data.get("name", "").lower()
            for city_key, city_info in malaysian_cities.items():
                if city_key in loc_name:
                    region = city_info["region"]
                    if region not in location_regions:
                        location_regions[region] = {"total": 0, "locations": []}
                    location_regions[region]["total"] += loc_data.get("amount", 0)
                    location_regions[region]["locations"].append(loc_data.get("name", ""))
                    break
        
        insights["regional_patterns"] = [
            {
                "region": region,
                "total_spend": round(data["total"], 2),
                "locations": data["locations"][:3],  # Top 3 locations per region
            }
            for region, data in location_regions.items()
        ]
        
        # Cost indicators based on locations
        high_cost_locations = []
        for loc_data in top_locations:
            loc_name = loc_data.get("name", "").lower()
            for city_key, city_info in malaysian_cities.items():
                if city_key in loc_name and city_info["cost_level"] in {"high", "very_high"}:
                    high_cost_locations.append({
                        "location": loc_data.get("name", ""),
                        "cost_level": city_info["cost_level"],
                        "spend": loc_data.get("amount", 0),
                    })
                    break
        
        insights["cost_indicators"] = high_cost_locations[:3]  # Top 3 high-cost locations
    
    return insights


def _seasonal_context(view_mode: str, start: date, end: date, religion: Optional[str], country: Optional[str]) -> List[str]:
    """Provide contextual notes about Malaysian seasonality and cultural events."""
    markers: List[str] = []
    religion_lower = (religion or "").lower()
    country_lower = (country or "").lower()

    months = {start.month} if view_mode == "monthly" else set(range(1, 13))

    if country_lower in {"malaysia", "my", ""}:
        if 12 in months or 1 in months:
            markers.append("Year-end Mega Sales and school reopening demand higher retail and education spend.")
        if 3 in months or 4 in months:
            markers.append("Ramadan and Hari Raya drive gifting, bazaars, balik kampung travel, and zakat payments.")
        if 5 in months or 6 in months:
            markers.append("Post-Raya recovery period ideal for resetting budgets and topping up Tabung Haji/ASB.")
        if 8 in months or 9 in months:
            markers.append("Merdeka & Malaysia Day promos encourage patriotic spending—watch impulse buys.")
        if 10 in months or 11 in months:
            markers.append("Tax season planning and year-end bonuses enable retirement top-ups and charitable giving.")

    if religion_lower in {"islam", "muslim"}:
        markers.append("Ensure zakat fitrah and zakat pendapatan are budgeted alongside recurring commitments.")
        markers.append("Prioritise Shariah-compliant financing (e.g., Murabahah, Musharakah) for major purchases.")

    return markers


def _collect_period_data(
    db: Session,
    user: models.User,
    view_mode: str,
    start: date,
    end: date
) -> Dict[str, Any]:
    """Aggregate income, expense, budget, and credit data for the selected window."""
    # Expenses & incomes
    expense_rows: List[models.Expense] = (
        db.query(models.Expense)
        .filter(
            models.Expense.user_id == user.user_id,
            models.Expense.is_deleted.is_(False),
            models.Expense.date_spent >= start,
            models.Expense.date_spent <= end,
        )
        .all()
    )

    income_rows: List[models.Income] = (
        db.query(models.Income)
        .filter(
            models.Income.user_id == user.user_id,
            models.Income.is_deleted.is_(False),
            models.Income.date_received >= start,
            models.Income.date_received <= end,
        )
        .all()
    )

    total_income = sum(row.amount or 0 for row in income_rows)
    total_expenses = sum(row.amount or 0 for row in expense_rows)
    net_cash_flow = total_income - total_expenses

    needs_total = sum(row.amount or 0 for row in expense_rows if row.expense_type == "needs")
    wants_total = sum(row.amount or 0 for row in expense_rows if row.expense_type == "wants")
    needs_pct = _safe_ratio(needs_total, total_expenses)
    wants_pct = _safe_ratio(wants_total, total_expenses)

    category_totals: Dict[str, float] = defaultdict(float)
    merchant_totals: Dict[str, float] = defaultdict(float)

    timeseries_map: Dict[str, Dict[str, float]] = defaultdict(lambda: {"needs": 0.0, "wants": 0.0, "all": 0.0})

    for expense in expense_rows:
        category = expense.category or "Uncategorised"
        merchant = expense.seller or "General"
        category_totals[category] += expense.amount or 0.0
        merchant_totals[merchant] += expense.amount or 0.0

        bucket_label: str
        if view_mode == "yearly":
            bucket_label = expense.date_spent.strftime("%Y-%m")
        else:
            bucket_label = expense.date_spent.strftime("%Y-%m-%d")

        bucket = timeseries_map[bucket_label]
        bucket["all"] += expense.amount or 0.0
        if expense.expense_type == "needs":
            bucket["needs"] += expense.amount or 0.0
        elif expense.expense_type == "wants":
            bucket["wants"] += expense.amount or 0.0

    # Sort time series chronologically
    timeseries = [
        {
            "label": label,
            "needs": round(values["needs"], 2),
            "wants": round(values["wants"], 2),
            "total": round(values["all"], 2),
        }
        for label, values in sorted(timeseries_map.items(), key=lambda item: item[0])
    ]

    days_tracked = max((end - start).days + 1, 1)
    average_daily_spend = total_expenses / days_tracked if days_tracked else 0.0

    # Previous-period comparison
    prev_start, prev_end = _previous_period(view_mode, start, end)

    prev_expense_rows: List[models.Expense] = (
        db.query(models.Expense)
        .filter(
            models.Expense.user_id == user.user_id,
            models.Expense.is_deleted.is_(False),
            models.Expense.date_spent >= prev_start,
            models.Expense.date_spent <= prev_end,
        )
        .all()
    )
    prev_income_rows: List[models.Income] = (
        db.query(models.Income)
        .filter(
            models.Income.user_id == user.user_id,
            models.Income.is_deleted.is_(False),
            models.Income.date_received >= prev_start,
            models.Income.date_received <= prev_end,
        )
        .all()
    )

    prev_expenses_total = sum(row.amount or 0 for row in prev_expense_rows)
    prev_income_total = sum(row.amount or 0 for row in prev_income_rows)

    spend_trend_pct = None
    income_trend_pct = None

    if prev_expenses_total > 0:
        spend_trend_pct = ((total_expenses - prev_expenses_total) / prev_expenses_total) * 100
    if prev_income_total > 0:
        income_trend_pct = ((total_income - prev_income_total) / prev_income_total) * 100

    # Budgets overlapping the period
    budget_rows: List[models.Budget] = (
        db.query(models.Budget)
        .filter(
            models.Budget.user_id == user.user_id,
            models.Budget.is_deleted.is_(False),
            models.Budget.period_start <= end,
            models.Budget.period_end >= start,
        )
        .all()
    )

    budget_summaries: List[Dict[str, Any]] = []
    for budget in budget_rows:
        spend_for_budget = sum(
            expense.amount or 0.0
            for expense in expense_rows
            if expense.category == budget.category
            and budget.period_start <= expense.date_spent <= budget.period_end
        )
        utilisation_ratio = _safe_ratio(spend_for_budget, budget.limit_amount)
        budget_summaries.append(
            {
                "budget_id": budget.budget_id,
                "name": budget.name,
                "category": budget.category,
                "limit": budget.limit_amount,
                "spent": round(spend_for_budget, 2),
                "remaining": round(max(budget.limit_amount - spend_for_budget, 0.0), 2),
                "utilisation_pct": round((utilisation_ratio or 0.0) * 100, 1),
                "period_start": budget.period_start.isoformat(),
                "period_end": budget.period_end.isoformat(),
            }
        )

    active_budget = None
    if budget_summaries:
        # Active budget is the one whose window covers the start date (for monthly view) or overlaps most
        def _overlap_days(b: Dict[str, Any]) -> int:
            start_dt = datetime.fromisoformat(b["period_start"]).date()
            end_dt = datetime.fromisoformat(b["period_end"]).date()
            overlap_start = max(start_dt, start)
            overlap_end = min(end_dt, end)
            return max((overlap_end - overlap_start).days + 1, 0)

        active_budget = max(budget_summaries, key=_overlap_days, default=None)

    # Credit card snapshot
    credit_cards: List[models.UserCreditCard] = (
        db.query(models.UserCreditCard)
        .filter(
            models.UserCreditCard.user_id == user.user_id,
            models.UserCreditCard.is_deleted.is_(False),
        )
        .all()
    )
    total_card_limit = sum(card.credit_limit or 0 for card in credit_cards)
    total_card_balance = sum(card.current_balance or 0 for card in credit_cards)
    utilisation_pct = _safe_ratio(total_card_balance, total_card_limit)

    upcoming_payments = [
        {
            "card_name": card.card_name,
            "bank_name": card.bank_name,
            "amount": card.next_payment_amount,
            "due_date": card.next_payment_date.isoformat() if card.next_payment_date else None,
        }
        for card in credit_cards
        if card.next_payment_amount and card.next_payment_date
    ]
    upcoming_payments.sort(key=lambda item: item["due_date"] or "")

    # Goals snapshot (helps recommendations)
    goal_rows: List[models.Goal] = (
        db.query(models.Goal)
        .filter(
            models.Goal.user_id == user.user_id,
            models.Goal.is_deleted.is_(False),
        )
        .all()
    )

    goal_summaries = [
        {
            "goal_id": goal.goal_id,
            "name": goal.goal_name,
            "category": goal.category,
            "target_amount": goal.target_amount,
            "current_amount": goal.current_amount,
            "progress_pct": round(_safe_ratio(goal.current_amount, goal.target_amount) * 100, 2)
            if goal.target_amount
            else None,
            "target_date": goal.target_date.isoformat() if goal.target_date else None,
        }
        for goal in goal_rows
    ]

    top_categories = [
        {"name": name, "amount": round(amount, 2)}
        for name, amount in Counter(category_totals).most_common(6)
    ]
    top_merchants = [
        {"name": name, "amount": round(amount, 2)}
        for name, amount in Counter(merchant_totals).most_common(6)
    ]

    # Location analysis: extract spending patterns by location
    location_totals: Dict[str, float] = defaultdict(float)
    location_counts: Dict[str, int] = defaultdict(int)
    for expense in expense_rows:
        if expense.location:
            location = expense.location.strip()
            if location:
                location_totals[location] += expense.amount or 0.0
                location_counts[location] += 1

    top_locations = [
        {"name": name, "amount": round(amount, 2), "transaction_count": location_counts.get(name, 0)}
        for name, amount in Counter(location_totals).most_common(10)
    ]

    return {
        "income": {
            "total": round(total_income, 2),
            "previous_total": round(prev_income_total, 2),
            "trend_pct": round(income_trend_pct, 2) if income_trend_pct is not None else None,
        },
        "expenses": {
            "total": round(total_expenses, 2),
            "previous_total": round(prev_expenses_total, 2),
            "trend_pct": round(spend_trend_pct, 2) if spend_trend_pct is not None else None,
            "needs_total": round(needs_total, 2),
            "wants_total": round(wants_total, 2),
            "needs_pct": round((needs_pct or 0.0) * 100, 2),
            "wants_pct": round((wants_pct or 0.0) * 100, 2),
            "average_daily_spend": round(average_daily_spend, 2),
            "time_series": timeseries,
            "top_categories": top_categories,
            "top_merchants": top_merchants,
            "top_locations": top_locations,
            "transaction_count": len(expense_rows),
        },
        "cash_flow": {
            "net_cash_flow": round(net_cash_flow, 2),
            "days_tracked": days_tracked,
        },
        "budgets": {
            "active": active_budget,
            "summaries": budget_summaries,
        },
        "credit": {
            "total_limit": round(total_card_limit, 2),
            "total_balance": round(total_card_balance, 2),
            "utilisation_pct": round((utilisation_pct or 0.0) * 100, 2),
            "cards": [
                {
                    "name": card.card_name,
                    "bank": card.bank_name,
                    "limit": card.credit_limit,
                    "balance": card.current_balance,
                    "utilisation_pct": round(
                        (_safe_ratio(card.current_balance, card.credit_limit) or 0.0) * 100, 2
                    )
                    if card.credit_limit
                    else None,
                }
                for card in credit_cards
            ],
            "upcoming_payments": upcoming_payments[:3],
        },
        "goals": goal_summaries,
    }


SMART_ANALYSIS_SYSTEM_PROMPT = """
You are RayyAI's Smart Financial Analyst for Malaysian users. Produce empathetic, culturally-aware insights that respect Islamic finance preferences when applicable.

Output MUST be valid JSON compliant with this schema (no markdown, no backticks, no commentary):
{
  "summary_title": string,
  "analysis_points": [string, ...],
  "recommendations": [string, ...],
  "seasonal_signals": [string, ...],
  "savings_opportunities": [string, ...],
  "risk_alerts": [string, ...],
  "cultural_notes": [string, ...],
  "tone": string
}

CRITICAL REQUIREMENTS:
1. PERIOD REFERENCE: ALWAYS explicitly mention the selected period (from period.label) in your analysis. For example:
   - "In {period_label}, your spending patterns show..."
   - "During {period_label}, you managed to..."
   - "For {period_label}, here's what stands out..."
   - Reference specific dates when relevant (e.g., "In the first week of {period_label}")

2. DATA AVAILABILITY HANDLING:
   - If transaction_count is 0 or very low (< 5 for monthly, < 20 for yearly):
     * Acknowledge limited data: "With limited transaction data for {period_label}..."
     * Focus on setup guidance: "To get better insights, start tracking your expenses..."
     * Provide general best practices rather than specific analysis
     * Still reference the period explicitly
   
   - If total_expenses is 0 or very low (< 100):
     * Note: "Early days in {period_label} - minimal spending tracked so far"
     * Suggest: "Consider uploading bank statements or manually adding transactions"
     * Provide encouragement: "As you track more, insights will become more personalized"
   
   - If data is sparse but present:
     * Use qualifiers: "Based on the available data for {period_label}..."
     * Acknowledge limitations: "With more transactions, we can provide deeper insights"
     * Still provide what insights are possible from the limited data

3. CONTEXT-SPECIFIC GUIDANCE:
- Tailor guidance to Malaysian context: EPF/PRS, ASB, Tabung Haji, BR1M/BKM, e-wallet trends, government incentives.
- Use the provided seasonal context to highlight upcoming events (Ramadan, Hari Raya, school reopenings, tax season, year-end sales).

4. CULTURAL NOTES PERSONALIZATION (CRITICAL):
- The "cultural_notes" section MUST be personalized based on the user_profile data provided in the payload.
- Check user_profile.religion: 
  * If "Islam" or "Muslim": Include zakat fitrah/penghasilan reminders, halal budgeting guidance, Shariah-compliant financing options (Murabahah, Musharakah), Tabung Haji/ASB recommendations, and Islamic financial principles.
  * If other religions (e.g., "Christian", "Buddhist", "Hindu"): Include relevant religious financial practices, festivals, and cultural spending patterns for that faith.
  * If not specified: Provide general Malaysian cultural financial guidance.
- Check user_profile.country:
  * If "Malaysia" or "MY": Focus on Malaysian-specific financial products, government incentives, and local cultural practices.
  * If other countries: Adapt to that country's financial culture and products.
- Check user_profile.location: Reference specific regional financial opportunities, cost-of-living considerations, and local cultural events if available.
- DO NOT provide generic cultural notes that apply to all users. Each user's cultural_notes should reflect their specific religion, country, and location from user_profile.

5. LOCATION-BASED INSIGHTS:
- Analyze location_insights data to provide regional spending patterns and cost-of-living context.
- If user spends heavily in high-cost areas (KLCC, Bangsar, Mid Valley), suggest cost-saving alternatives or budget adjustments.
- Highlight location diversity: if spending is concentrated in few locations, suggest exploring alternatives for better deals.
- Regional patterns: if spending spans multiple regions, acknowledge travel/commuting costs and suggest optimization.
- Compare spending in user's primary locations against typical cost levels for those areas.
- Suggest location-specific opportunities: cheaper alternatives nearby, regional promotions, or local financial products.
- If location data shows high concentration in premium areas, provide actionable tips to reduce costs without sacrificing convenience.

6. OUTPUT QUALITY:
- Every list must contain at least 2 items; recommendations should be actionable, quantified where possible, and sensitive (no shaming).
- Keep each bullet under 180 characters and maintain an encouraging, respectful tone.
- If data is missing (e.g., no budgets, no credit cards), suggest culturally-relevant next steps rather than generic statements.
- Do not invent data that is not present in the payload.
- Always ground insights in the actual period being analyzed.
"""


@router.post(
    "/smart-analysis",
    response_model=schemas.SmartAnalysisResponse,
    status_code=status.HTTP_200_OK,
)
async def generate_smart_analysis(
    request: schemas.SmartAnalysisRequest,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
) -> schemas.SmartAnalysisResponse:
    """
    Generate an AI-powered smart analysis for the selected month or year.
    """
    gemini_service = get_gemini_service()
    rag_service = RAGService(db)

    start_date, end_date, period_label = _period_bounds(request.view_mode, request.selected_date)

    period_payload = _collect_period_data(
        db=db,
        user=current_user,
        view_mode=request.view_mode,
        start=start_date,
        end=end_date,
    )

    # Additional context for the prompt
    user_location = getattr(current_user, "location", None) or ""
    user_country = getattr(current_user, "country", "Malaysia") if hasattr(current_user, "country") else "Malaysia"
    user_religion = getattr(current_user, "religion", "Islam") if hasattr(current_user, "religion") else "Islam"
    
    user_profile = {
        "first_name": getattr(current_user, "first_name", ""),
        "last_name": getattr(current_user, "last_name", ""),
        "gender": getattr(current_user, "gender", None),
        "dob": getattr(current_user, "dob", None).isoformat() if getattr(current_user, "dob", None) else None,
        "country": user_country,
        "religion": user_religion,
        "location": user_location,
    }
    
    # Extract location insights from transaction data
    location_insights = _analyze_location_patterns(period_payload.get("expenses", {}).get("top_locations", []), user_location, user_country)

    seasonal_notes = _seasonal_context(
        request.view_mode,
        start_date,
        end_date,
        user_profile.get("religion"),
        user_profile.get("country"),
    )

    # Broader financial snapshot (leveraging existing service)
    try:
        financial_summary = rag_service.get_financial_summary(current_user.user_id)
    except Exception:
        financial_summary = None

    prompt_payload: Dict[str, Any] = {
        "period": {
            "view_mode": request.view_mode,
            "label": period_label,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
        },
        "user_profile": user_profile,
        "seasonal_context": seasonal_notes,
        "location_insights": location_insights,
        "metrics": period_payload,
        "financial_snapshot": financial_summary,
    }

    serialized_payload = json.dumps(prompt_payload, indent=2)

    messages = [
        {
            "role": "user",
            "content": (
                "Analyse the following Malaysian user's financial data and respond with JSON only.\n"
                "```\n"
                f"{serialized_payload}\n"
                "```"
            ),
        }
    ]

    try:
        ai_response = await gemini_service.generate_response(
            system_instruction=SMART_ANALYSIS_SYSTEM_PROMPT,
            messages=messages,
            temperature=0.45,
            max_output_tokens=1800,
        )
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Unable to generate smart analysis: {err}",
        ) from err

    raw_content = ai_response.get("content", "") or ""

    def _extract_json(text: str) -> Dict[str, Any]:
        """Extract JSON object from raw model text."""
        text = text.strip()
        if not text:
            raise ValueError("Empty response from Gemini")

        # Attempt direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Look for JSON object within text (e.g., inside fences)
        import re

        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            raise ValueError("Gemini response did not contain JSON object")
        return json.loads(match.group(0))

    try:
        parsed = _extract_json(raw_content)
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to parse smart analysis response: {err}",
        ) from err

    def _as_list(key: str) -> List[str]:
        values = parsed.get(key)
        if isinstance(values, list):
            return [str(item).strip() for item in values if str(item).strip()]
        if isinstance(values, str) and values.strip():
            return [values.strip()]
        return []

    summary_title = parsed.get("summary_title") or f"Smart Analysis – {period_label}"

    model_usage = ai_response.get("usage_metadata")

    return schemas.SmartAnalysisResponse(
        generated_at=datetime.utcnow(),
        period_label=period_label,
        summary_title=summary_title,
        analysis_points=_as_list("analysis_points"),
        recommendations=_as_list("recommendations"),
        seasonal_signals=_as_list("seasonal_signals") or seasonal_notes,
        savings_opportunities=_as_list("savings_opportunities"),
        risk_alerts=_as_list("risk_alerts"),
        cultural_notes=_as_list("cultural_notes"),
        tone=parsed.get("tone"),
        model_usage=model_usage,
    )


NEEDS_VS_WANTS_SYSTEM_PROMPT = """
You are RayyAI's Financial Spending Analyst for Malaysian users. Generate concise, actionable insights about needs vs wants spending patterns.

Output MUST be valid JSON compliant with this schema (no markdown, no backticks, no commentary):
{
  "summary": string,
  "localized_guidance": [string, ...],
  "spend_optimization": [string, ...]
}

REQUIREMENTS:
1. PERIOD REFERENCE: ALWAYS explicitly mention the selected period (from period.label) in your insights.
2. SUMMARY: Provide a brief 2-3 sentence summary of the spending patterns for the period. Keep it under 150 words.
3. LOCALIZED_GUIDANCE: 
   - Check user_profile.religion: If "Islam" or "Muslim", include zakat, Tabung Haji, ASB, Shariah-compliant financing tips.
   - Check user_profile.country: If "Malaysia", reference EPF, PRS, Malaysian tax reliefs, local financial products.
   - TAX REBATE SUGGESTIONS: Analyze spending categories and suggest relevant Malaysian tax reliefs/rebates:
     * Medical expenses (self, spouse, children, parents) - up to RM10,000
     * Education fees (self, spouse, children) - up to RM7,000
     * Lifestyle expenses (gym, sports equipment) - up to RM500
     * Books, computers, internet subscriptions - up to RM2,500
     * EPF/PRS contributions - up to RM4,000
     * Life insurance/PRS - up to RM3,000
     * SOCSO contributions
     * Medical insurance - up to RM3,000
     * Electric vehicle charging expenses
     * Reference specific spending categories from top_categories that qualify for tax reliefs.
   - Keep each item under 120 characters.
   - Provide 2-4 culturally relevant tips including tax rebate opportunities.
4. SPEND_OPTIMIZATION:
   - Analyze the actual spending data provided (needs vs wants breakdown, top categories, trends).
   - Reference specific categories or amounts when relevant.
   - Provide actionable, specific tips based on the data.
   - Keep each item under 120 characters.
   - Provide 3-4 optimization tips.
5. Keep all content concise and focused. Do not be verbose.
"""


@router.post(
    "/needs-vs-wants-insights",
    response_model=schemas.NeedsVsWantsInsightsResponse,
    status_code=status.HTTP_200_OK,
)
async def generate_needs_vs_wants_insights(
    request: schemas.NeedsVsWantsInsightsRequest,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
) -> schemas.NeedsVsWantsInsightsResponse:
    """
    Generate AI-powered insights for needs vs wants spending patterns.
    """
    try:
        logger.info(f"Starting needs vs wants insights for user {current_user.user_id}, view_mode={request.view_mode}, date={request.selected_date}")

        gemini_service = get_gemini_service()

        start_date, end_date, period_label = _period_bounds(request.view_mode, request.selected_date)
        logger.info(f"Period bounds: {start_date} to {end_date}, label={period_label}")
    except Exception as e:
        logger.error(f"Error in initial setup: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error in initial setup: {str(e)}",
        ) from e

    # Get expenses for the period
    try:
        logger.info(f"Querying expenses for user {current_user.user_id}")
        expense_rows: List[models.Expense] = (
            db.query(models.Expense)
            .filter(
                models.Expense.user_id == current_user.user_id,
                models.Expense.is_deleted.is_(False),
                models.Expense.date_spent >= start_date,
                models.Expense.date_spent <= end_date,
            )
            .all()
        )
        logger.info(f"Found {len(expense_rows)} expense rows")

        total_needs = sum(row.amount or 0 for row in expense_rows if row.expense_type == "needs")
        total_wants = sum(row.amount or 0 for row in expense_rows if row.expense_type == "wants")
        total_spending = total_needs + total_wants
        logger.info(f"Spending totals - needs: {total_needs}, wants: {total_wants}, total: {total_spending}")

        needs_pct = _safe_ratio(total_needs, total_spending) or 0.0
        wants_pct = _safe_ratio(total_wants, total_spending) or 0.0
    except Exception as e:
        logger.error(f"Error querying expenses: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error querying expenses: {str(e)}",
        ) from e

    # Category breakdown
    category_totals: Dict[str, Dict[str, float]] = defaultdict(lambda: {"needs": 0.0, "wants": 0.0})
    for expense in expense_rows:
        category = expense.category or "Uncategorised"
        amount = expense.amount or 0.0
        expense_type = expense.expense_type or "needs"
        category_totals[category][expense_type] += amount

    top_categories = [
        {
            "name": name,
            "needs": round(data["needs"], 2),
            "wants": round(data["wants"], 2),
            "total": round(data["needs"] + data["wants"], 2),
        }
        for name, data in sorted(
            category_totals.items(),
            key=lambda x: x[1]["needs"] + x[1]["wants"],
            reverse=True
        )[:10]
    ]

    # User profile
    user_country = getattr(current_user, "country", "Malaysia") if hasattr(current_user, "country") else "Malaysia"
    user_religion = getattr(current_user, "religion", "Islam") if hasattr(current_user, "religion") else "Islam"
    user_location = getattr(current_user, "location", None) or ""

    user_profile = {
        "country": user_country,
        "religion": user_religion,
        "location": user_location,
    }

    # Previous period comparison
    prev_start, prev_end = _previous_period(request.view_mode, start_date, end_date)
    prev_expense_rows: List[models.Expense] = (
        db.query(models.Expense)
        .filter(
            models.Expense.user_id == current_user.user_id,
            models.Expense.is_deleted.is_(False),
            models.Expense.date_spent >= prev_start,
            models.Expense.date_spent <= prev_end,
        )
        .all()
    )

    prev_needs = sum(row.amount or 0 for row in prev_expense_rows if row.expense_type == "needs")
    prev_wants = sum(row.amount or 0 for row in prev_expense_rows if row.expense_type == "wants")
    prev_total = prev_needs + prev_wants

    needs_change = total_needs - prev_needs if prev_needs > 0 else None
    wants_change = total_wants - prev_wants if prev_wants > 0 else None
    needs_change_pct = ((total_needs - prev_needs) / prev_needs * 100) if prev_needs > 0 else None
    wants_change_pct = ((total_wants - prev_wants) / prev_wants * 100) if prev_wants > 0 else None

    prompt_payload: Dict[str, Any] = {
        "period": {
            "view_mode": request.view_mode,
            "label": period_label,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
        },
        "user_profile": user_profile,
        "spending": {
            "needs_total": round(total_needs, 2),
            "wants_total": round(total_wants, 2),
            "total_spending": round(total_spending, 2),
            "needs_percent": round(needs_pct * 100, 1) if needs_pct else 0.0,
            "wants_percent": round(wants_pct * 100, 1) if wants_pct else 0.0,
        },
        "previous_period": {
            "needs_total": round(prev_needs, 2),
            "wants_total": round(prev_wants, 2),
            "total_spending": round(prev_total, 2),
        },
        "trends": {
            "needs_change": round(needs_change, 2) if needs_change is not None else None,
            "wants_change": round(wants_change, 2) if wants_change is not None else None,
            "needs_change_pct": round(needs_change_pct, 2) if needs_change_pct is not None else None,
            "wants_change_pct": round(wants_change_pct, 2) if wants_change_pct is not None else None,
        },
        "top_categories": top_categories,
    }

    serialized_payload = json.dumps(prompt_payload, indent=2)

    messages = [
        {
            "role": "user",
            "content": (
                "Analyse the following Malaysian user's needs vs wants spending data and respond with JSON only.\n"
                "```\n"
                f"{serialized_payload}\n"
                "```"
            ),
        }
    ]

    try:
        logger.info("Calling Gemini API for needs vs wants insights")
        logger.debug(f"Payload being sent to Gemini: {serialized_payload[:500]}...")  # Log first 500 chars

        ai_response = await gemini_service.generate_response(
            system_instruction=NEEDS_VS_WANTS_SYSTEM_PROMPT,
            messages=messages,
            temperature=0.45,
            max_output_tokens=2048,  # Increased from 1200 to allow for longer responses
        )
        logger.info("Gemini API call successful")
        logger.info(f"Token usage: {ai_response.get('usage_metadata')}")
    except Exception as err:
        logger.error(f"Gemini API error: {str(err)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Unable to generate needs vs wants insights: {err}",
        ) from err

    raw_content = ai_response.get("content", "") or ""

    def _extract_json(text: str) -> Dict[str, Any]:
        """Extract JSON object from raw model text."""
        text = text.strip()
        if not text:
            raise ValueError("Empty response from Gemini")

        # Attempt direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning(f"Direct JSON parse failed: {str(e)}")

        # Look for JSON object within text (e.g., inside fences)
        import re

        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            raise ValueError(f"Gemini response did not contain JSON object. Response: {text[:300]}...")

        json_str = match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            # If JSON is incomplete/malformed, provide helpful error
            raise ValueError(f"Gemini response contained incomplete or malformed JSON: {str(e)}. Content length: {len(text)}, JSON excerpt: {json_str[:500]}...")

    try:
        logger.info(f"Parsing Gemini response, content length: {len(raw_content)}")
        logger.debug(f"Raw content from Gemini: {raw_content[:800]}...")  # Log first 800 chars
        parsed = _extract_json(raw_content)
        logger.info(f"Successfully parsed JSON with keys: {list(parsed.keys())}")
    except Exception as err:
        logger.error(f"JSON parsing error: {str(err)}", exc_info=True)
        logger.error(f"Failed to parse content: {raw_content}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to parse needs vs wants insights response: {err}",
        ) from err

    def _as_list(key: str) -> List[str]:
        values = parsed.get(key)
        if isinstance(values, list):
            return [str(item).strip() for item in values if str(item).strip()]
        if isinstance(values, str) and values.strip():
            return [values.strip()]
        return []

    model_usage = ai_response.get("usage_metadata")

    try:
        logger.info("Building response")
        response = schemas.NeedsVsWantsInsightsResponse(
            generated_at=datetime.utcnow(),
            period_label=period_label,
            summary=parsed.get("summary", ""),
            localized_guidance=_as_list("localized_guidance"),
            spend_optimization=_as_list("spend_optimization"),
            model_usage=model_usage,
        )
        logger.info("Successfully created needs vs wants insights response")
        return response
    except Exception as err:
        logger.error(f"Error building response: {str(err)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error building response: {err}",
        ) from err


@router.post(
    "/analyze-suspicious-transactions",
    response_model=schemas.SuspiciousTransactionsResponse,
    status_code=status.HTTP_200_OK,
)
async def analyze_suspicious_transactions(
    request: schemas.SuspiciousTransactionsRequest,
    current_user: models.User = Depends(get_current_user),
) -> schemas.SuspiciousTransactionsResponse:
    """
    Analyze transactions for suspicious or potentially fraudulent activity using Gemini AI.
    """
    try:
        logger.info(f"Starting suspicious transaction analysis for user {current_user.user_id}")
        gemini_service = get_gemini_service()

        # Prepare transactions for analysis - convert Pydantic models to dicts
        transactions_to_analyze = [tx.model_dump() for tx in request.transactions[:100]]  # Limit to 100 for API efficiency

        prompt = f"""You are a financial fraud detection expert. Analyze these transactions and identify any that appear dubious, suspicious, or potentially fraudulent.

Consider these factors:
- Malaysian-specific scam patterns (Macau scams, love scams, investment scams)
- Duplicate or near-duplicate transactions
- Unusually large amounts compared to typical spending
- Vague or suspicious descriptions
- Rapid repeated transactions to the same merchant
- Transfers to unknown parties
- Suspicious keywords (police, court, urgent help, guaranteed returns, crypto, gift cards, etc.)

Transactions to analyze:
{json.dumps(transactions_to_analyze, indent=2)}

Return ONLY a JSON array of suspicious transaction objects with this exact format:
[
  {{
    "id": "transaction_id",
    "reason": "Brief reason (e.g., 'Potential Macau Scam', 'Duplicate Transaction', 'Unusually Large Amount')",
    "severity": "high" or "medium",
    "details": "Detailed explanation of why this is suspicious"
  }}
]

If no suspicious transactions are found, return an empty array: []

Return ONLY valid JSON, no other text."""

        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        logger.info("Calling Gemini API for suspicious transaction analysis")
        ai_response = await gemini_service.generate_response(
            system_instruction="You are a fraud detection expert. Return only valid JSON arrays, no markdown or commentary.",
            messages=messages,
            temperature=0.3,
            max_output_tokens=2048,
        )
        logger.info("Gemini API call successful")

        raw_content = ai_response.get("content", "") or ""

        # Extract JSON from response
        import re
        json_match = re.search(r"\[[\s\S]*\]", raw_content)
        if not json_match:
            # Try to find JSON in markdown code block
            json_match = re.search(r"```(?:json)?\s*(\[[\s\S]*\])\s*```", raw_content)
            if json_match:
                json_match = [json_match.group(1)]

        if json_match:
            suspicious_results = json.loads(json_match[0] if isinstance(json_match, list) else json_match.group(0))
        else:
            suspicious_results = []

        logger.info(f"Found {len(suspicious_results)} suspicious transactions")

        return schemas.SuspiciousTransactionsResponse(
            suspicious_transactions=suspicious_results,
            analyzed_count=len(transactions_to_analyze),
            model_usage=ai_response.get("usage_metadata"),
        )

    except Exception as err:
        logger.error(f"Error analyzing suspicious transactions: {str(err)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to analyze suspicious transactions: {err}",
        ) from err


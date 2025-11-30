"""
Context Summarization Service
Generates and caches financial summaries for efficient context management
"""
from typing import Dict, Any, Optional, List
from datetime import datetime, date, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_
import os
import models
from services.gemini_service import GeminiService
from services.rag_service import RAGService
import json
import logging

logger = logging.getLogger(__name__)

class ContextSummarizer:
    """Service for summarizing financial context and conversation history"""
    
    def __init__(self, db: Session, gemini_service: GeminiService):
        """
        Initialize context summarizer.
        
        Args:
            db: Database session
            gemini_service: Gemini service instance
        """
        self.db = db
        self.gemini_service = gemini_service
        self.summary_expiry_hours = int(
            os.getenv("CONTEXT_SUMMARY_EXPIRY_HOURS", "24")
        )
    
    def get_cached_summary(
        self,
        user_id: int,
        summary_type: str = "financial_snapshot"
    ) -> Optional[models.ContextSummary]:
        """
        Get cached summary if it exists and hasn't expired.
        
        Args:
            user_id: User ID
            summary_type: Type of summary to retrieve
            
        Returns:
            ContextSummary if found and valid, None otherwise
        """
        now = datetime.now()
        
        summary = self.db.query(models.ContextSummary).filter(
            models.ContextSummary.user_id == user_id,
            models.ContextSummary.summary_type == summary_type,
            models.ContextSummary.expires_at > now
        ).order_by(models.ContextSummary.created_at.desc()).first()
        
        return summary
    
    async def generate_financial_summary(self, user_id: int) -> str:
        """
        Generate comprehensive financial summary using AI.
        
        Args:
            user_id: User ID
            
        Returns:
            Summary text
        """
        # Get financial data
        rag_service = RAGService(self.db)
        financial_data = rag_service.get_financial_summary(user_id)
        
        # Format as text
        context_text = rag_service.format_context_for_llm(financial_data)
        
        # Create prompt for summarization
        summary_prompt = f"""Summarize the following financial data in a concise, structured format that captures key insights, trends, and important information. Maintain a humorous, slightly sarcastic yet helpful and professional tone. Focus on:
1. Overall financial health
2. Spending patterns and trends
3. Budget performance
4. Goal progress
5. Credit card utilization
6. Areas of concern or opportunity

Response format requirements (very important):
- Always respond in Markdown and bullets.
- Start with a clear title line: `# <Concise Title>` with an appropriate emoji.
- Use `##` section headings (e.g., "Summary", "Key Insights", "Recommendations", "Next Steps").
- Use bullet lists with `• ` and indent sub-bullets by two spaces.
- Insert a blank line before and after every title and subtitles.
- Keep each bullet to a single line (no hard wraps inside bullets).
- Keep paragraphs short and concise; use bold for key terms and numbers.
- Tables are allowed when listing comparable items.

Financial Data:
{context_text}

Provide a clear, concise summary that would be useful for financial decision-making:"""
        
        try:
            response = await self.gemini_service.generate_response(
                system_instruction="You are a humorous, slightly sarcastic, yet helpful and professional financial analysis assistant specializing in summarizing financial data. Inject wit and gentle humor while remaining constructive and trustworthy. Keep summaries engaging but informative.",
                messages=[{"role": "user", "content": summary_prompt}],
                temperature=0.5,  # Slightly higher temperature to allow personality
                max_output_tokens=2000
            )
            
            return response.get("content", "")
        except Exception as e:
            logger.error(f"Error generating financial summary: {e}")
            # Fallback to simple summary
            return self._generate_fallback_summary(financial_data)
    
    def _generate_fallback_summary(self, financial_data: Dict[str, Any]) -> str:
        """Generate a simple text summary without AI."""
        parts = []
        
        accounts = financial_data.get("accounts", {})
        parts.append(f"Total Balance: RM{accounts.get('total_balance', 0):,.2f}")
        
        transactions = financial_data.get("transactions", {})
        parts.append(
            f"Last 90 Days: Income RM{transactions.get('total_income_90d', 0):,.2f}, "
            f"Expenses RM{transactions.get('total_expenses_90d', 0):,.2f}"
        )
        
        spending = financial_data.get("spending_summary", {})
        parts.append(f"Last 30 Days Spending: RM{spending.get('total_spending', 0):,.2f}")
        
        budgets = financial_data.get("budgets", {})
        if budgets.get("over_budget_count", 0) > 0:
            parts.append(f"Warning: {budgets['over_budget_count']} budgets over limit")
        
        goals = financial_data.get("goals", {})
        parts.append(f"Goals: {goals.get('completed_count', 0)}/{goals.get('total_count', 0)} completed")
        
        return "\n".join(parts)
    
    async def get_or_generate_summary(
        self,
        user_id: int,
        summary_type: str = "financial_snapshot",
        force_refresh: bool = False
    ) -> models.ContextSummary:
        """
        Get cached summary or generate new one.
        
        Args:
            user_id: User ID
            summary_type: Type of summary
            force_refresh: Force generation of new summary
            
        Returns:
            ContextSummary instance
        """
        if not force_refresh:
            cached = self.get_cached_summary(user_id, summary_type)
            if cached:
                return cached
        
        # Generate new summary
        summary_content = await self.generate_financial_summary(user_id)
        
        # Calculate expiry
        expires_at = datetime.now() + timedelta(hours=self.summary_expiry_hours)
        
        # Create new summary record
        summary = models.ContextSummary(
            user_id=user_id,
            summary_type=summary_type,
            summary_content=summary_content,
            data_snapshot_date=date.today(),
            expires_at=expires_at
        )
        
        self.db.add(summary)
        self.db.commit()
        self.db.refresh(summary)
        
        return summary
    
    async def summarize_conversation(
        self,
        messages: List[Dict[str, str]],
        max_messages_to_summarize: int = 20
    ) -> str:
        """
        Summarize conversation history when it gets too long.
        
        Args:
            messages: List of conversation messages
            max_messages_to_summarize: Maximum messages to include in summarization
            
        Returns:
            Summary text
        """
        if len(messages) <= max_messages_to_summarize:
            return ""
        
        # Get older messages to summarize
        messages_to_summarize = messages[:-max_messages_to_summarize]
        
        # Format messages for summarization
        conversation_text = "\n".join(
            f"{msg.get('role', 'user')}: {msg.get('content', '')}"
            for msg in messages_to_summarize
        )
        
        summary_prompt = f"""Summarize the following conversation history, preserving important context, 
decisions made, and key information that would be needed for continuing the conversation:

{conversation_text}

Provide a concise summary:"""
        
        try:
            response = await self.gemini_service.generate_response(
                system_instruction="You are a humorous, slightly sarcastic, yet helpful and professional conversation summarizer. Create concise summaries that preserve key context while maintaining a light, engaging tone. Keep it witty but informative.",
                messages=[{"role": "user", "content": summary_prompt}],
                temperature=0.5,  # Slightly higher temperature to allow personality
                max_output_tokens=1000
            )
            
            return response.get("content", "")
        except Exception as e:
            logger.error(f"Error summarizing conversation: {e}")
            return f"[Previous conversation with {len(messages_to_summarize)} messages summarized]"
    
    def should_summarize_conversation(
        self,
        messages: List[Dict[str, str]],
        token_count: int
    ) -> bool:
        """
        Check if conversation should be summarized.
        
        Args:
            messages: Conversation messages
            token_count: Current token count
            
        Returns:
            True if should summarize
        """
        # Check token threshold
        if self.gemini_service.should_summarize(token_count):
            return True
        
        # Also summarize if too many messages
        if len(messages) > 50:
            return True
        
        return False


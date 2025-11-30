"""
Conversation Manager Service
Handles conversation lifecycle, message history, and context window management
"""
from typing import List, Dict, Any, Optional
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import func
import models
from services.gemini_service import GeminiService
from services.context_summarizer import ContextSummarizer
import logging

logger = logging.getLogger(__name__)

class ConversationManager:
    """Service for managing chat conversations"""
    
    def __init__(
        self,
        db: Session,
        gemini_service: GeminiService,
        context_summarizer: ContextSummarizer
    ):
        """
        Initialize conversation manager.
        
        Args:
            db: Database session
            gemini_service: Gemini service instance
            context_summarizer: Context summarizer instance
        """
        self.db = db
        self.gemini_service = gemini_service
        self.context_summarizer = context_summarizer
    
    def create_conversation(
        self,
        user_id: int,
        title: Optional[str] = None
    ) -> models.ChatConversation:
        """
        Create a new conversation.
        
        Args:
            user_id: User ID
            title: Optional conversation title
            
        Returns:
            Created ChatConversation
        """
        conversation = models.ChatConversation(
            user_id=user_id,
            title=title
        )
        
        self.db.add(conversation)
        self.db.commit()
        self.db.refresh(conversation)
        
        return conversation
    
    def get_conversation(
        self,
        conversation_id: int,
        user_id: int
    ) -> Optional[models.ChatConversation]:
        """
        Get conversation by ID (with user permission check).
        
        Args:
            conversation_id: Conversation ID
            user_id: User ID
            
        Returns:
            ChatConversation if found and accessible, None otherwise
        """
        conversation = self.db.query(models.ChatConversation).filter(
            models.ChatConversation.conversation_id == conversation_id,
            models.ChatConversation.user_id == user_id,
            models.ChatConversation.is_deleted == False
        ).first()
        
        return conversation
    
    def get_user_conversations(
        self,
        user_id: int,
        limit: int = 50,
        skip: int = 0
    ) -> List[models.ChatConversation]:
        """
        Get all conversations for a user.
        
        Args:
            user_id: User ID
            limit: Maximum number of conversations
            skip: Number to skip
            
        Returns:
            List of conversations
        """
        conversations = self.db.query(models.ChatConversation).filter(
            models.ChatConversation.user_id == user_id,
            models.ChatConversation.is_deleted == False
        ).order_by(
            models.ChatConversation.updated_at.desc()
        ).offset(skip).limit(limit).all()
        
        return conversations
    
    def get_conversation_messages(
        self,
        conversation_id: int,
        user_id: int,
        limit: Optional[int] = None
    ) -> List[models.ChatMessage]:
        """
        Get messages for a conversation.
        
        Args:
            conversation_id: Conversation ID
            user_id: User ID (for permission check)
            limit: Maximum number of messages (None for all)
            
        Returns:
            List of messages
        """
        # Verify conversation belongs to user
        conversation = self.get_conversation(conversation_id, user_id)
        if not conversation:
            return []
        
        query = self.db.query(models.ChatMessage).filter(
            models.ChatMessage.conversation_id == conversation_id
        ).order_by(models.ChatMessage.created_at.asc())
        
        if limit:
            query = query.limit(limit)
        
        return query.all()
    
    def add_message(
        self,
        conversation_id: int,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        token_count: Optional[int] = None
    ) -> models.ChatMessage:
        """
        Add a message to a conversation.
        
        Args:
            conversation_id: Conversation ID
            role: Message role ('user' or 'assistant')
            content: Message content
            metadata: Optional metadata
            token_count: Optional token count
            
        Returns:
            Created ChatMessage
        """
        message = models.ChatMessage(
            conversation_id=conversation_id,
            role=role,
            content=content,
            metadata_json=metadata,
            token_count=token_count
        )
        
        self.db.add(message)
        
        # Update conversation timestamp
        conversation = self.db.query(models.ChatConversation).filter(
            models.ChatConversation.conversation_id == conversation_id
        ).first()
        if conversation:
            conversation.updated_at = datetime.now()
            # Auto-generate title from first user message if not set
            if not conversation.title and role == "user":
                # Use first 50 chars of message as title
                conversation.title = content[:50] + "..." if len(content) > 50 else content
        
        self.db.commit()
        self.db.refresh(message)
        
        return message
    
    async def prepare_conversation_context(
        self,
        conversation_id: int,
        user_id: int,
        include_recent_messages: int = 20
    ) -> Dict[str, Any]:
        """
        Prepare conversation context for LLM, handling summarization if needed.
        
        Args:
            conversation_id: Conversation ID
            user_id: User ID
            include_recent_messages: Number of recent messages to include
            
        Returns:
            Dictionary with 'messages', 'summary', 'token_count'
        """
        # Get all messages
        all_messages = self.get_conversation_messages(conversation_id, user_id)
        
        # Convert to dict format
        message_dicts = [
            {
                "role": msg.role,
                "content": msg.content,
                "metadata": msg.metadata_json
            }
            for msg in all_messages
        ]
        
        # Count tokens
        token_count = self.gemini_service.count_message_tokens(message_dicts)
        
        # Check if summarization is needed
        conversation_summary = ""
        messages_to_use = message_dicts
        
        if self.context_summarizer.should_summarize_conversation(message_dicts, token_count):
            # Summarize older messages
            conversation_summary = await self.context_summarizer.summarize_conversation(
                message_dicts,
                max_messages_to_summarize=include_recent_messages
            )
            
            # Use only recent messages + summary
            messages_to_use = message_dicts[-include_recent_messages:] if len(message_dicts) > include_recent_messages else message_dicts
            
            # Recalculate token count
            token_count = self.gemini_service.count_message_tokens(messages_to_use)
            if conversation_summary:
                token_count += self.gemini_service.count_tokens(conversation_summary)
        
        return {
            "messages": messages_to_use,
            "summary": conversation_summary,
            "token_count": token_count,
            "total_message_count": len(all_messages)
        }
    
    def delete_conversation(
        self,
        conversation_id: int,
        user_id: int
    ) -> bool:
        """
        Soft delete a conversation.
        
        Args:
            conversation_id: Conversation ID
            user_id: User ID
            
        Returns:
            True if deleted, False if not found
        """
        conversation = self.get_conversation(conversation_id, user_id)
        if not conversation:
            return False
        
        conversation.is_deleted = True
        self.db.commit()
        
        return True


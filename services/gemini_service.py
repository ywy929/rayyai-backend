"""
Gemini 2.5 Pro Integration Service
Handles communication with Google Gemini API
"""
import os
from typing import List, Dict, Any, Optional, AsyncGenerator
from datetime import datetime
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from dotenv import load_dotenv
import logging

load_dotenv()

logger = logging.getLogger(__name__)

class GeminiService:
    """Service for interacting with Google Gemini 2.5 Pro API"""
    
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        
        genai.configure(api_key=api_key)
        
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        self.context_token_limit = int(os.getenv("CONTEXT_TOKEN_LIMIT", "1000000"))  # Gemini 2.0 has 1M context
        self.token_threshold = int(self.context_token_limit * 0.8)  # Summarize at 80%
        
        # Safety settings
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        
        try:
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                safety_settings=self.safety_settings
            )
            logger.info(f"Initialized Gemini model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini model: {e}")
            raise
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using Gemini's token counting.
        
        Args:
            text: Input text to count tokens for
            
        Returns:
            Number of tokens
        """
        try:
            # Use the model's count_tokens method
            response = self.model.count_tokens(text)
            return response.total_tokens
        except Exception as e:
            logger.warning(f"Token counting failed, estimating: {e}")
            # Fallback: rough estimation (4 chars per token average)
            return len(text) // 4
    
    def count_message_tokens(self, messages: List[Dict[str, str]]) -> int:
        """
        Count tokens for a list of messages.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            
        Returns:
            Total token count
        """
        total = 0
        for msg in messages:
            # Count role + content
            text = f"{msg.get('role', '')} {msg.get('content', '')}"
            total += self.count_tokens(text)
        return total
    
    def format_messages_for_gemini(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Format messages for Gemini API.
        Gemini expects alternating user/assistant messages.
        
        Args:
            messages: List of messages with 'role' and 'content'
            
        Returns:
            Formatted messages compatible with Gemini
        """
        formatted = []
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            
            # Gemini uses 'user' and 'model' instead of 'assistant'
            if role == 'assistant':
                formatted.append({'role': 'model', 'parts': [content]})
            else:
                formatted.append({'role': 'user', 'parts': [content]})
        
        return formatted
    
    async def generate_response(
        self,
        system_instruction: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_output_tokens: Optional[int] = None,
        stream: bool = False,
        model_override: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate response from Gemini.

        Args:
            system_instruction: System prompt/instructions
            messages: Conversation history
            temperature: Sampling temperature (0.0-1.0)
            max_output_tokens: Maximum tokens in response
            stream: Whether to stream the response
            model_override: Optional model name to override the default model

        Returns:
            Dictionary with 'content', 'token_count', 'finish_reason', etc.
        """
        try:
            # Format messages for Gemini
            formatted_messages = self.format_messages_for_gemini(messages)

            # Create generation config
            generation_config = {
                "temperature": temperature,
            }
            if max_output_tokens:
                generation_config["max_output_tokens"] = max_output_tokens

            # Use override model if provided, otherwise use default
            model_to_use = model_override if model_override else self.model_name

            # Create a model instance with system instruction (SDK does not accept
            # system_instruction in start_chat)
            model_with_sys = genai.GenerativeModel(
                model_name=model_to_use,
                safety_settings=self.safety_settings,
                system_instruction=system_instruction
            )
            # Start chat with history
            chat = model_with_sys.start_chat(
                history=formatted_messages[:-1] if len(formatted_messages) > 1 else []
            )
            
            # Get the last message (user's current message)
            last_message = formatted_messages[-1] if formatted_messages else {'role': 'user', 'parts': ['']}
            user_content = last_message.get('parts', [''])[0]
            
            if stream:
                # Return generator for streaming
                response = chat.send_message(
                    user_content,
                    generation_config=generation_config,
                    stream=True
                )
                return response
            else:
                # Get complete response
                response = chat.send_message(
                    user_content,
                    generation_config=generation_config
                )
                
                # Extract response content
                content = response.text
                token_count = self.count_tokens(content)

                # Build usage metadata defensively (SDK field names vary by version)
                usage_payload = None
                usage = getattr(response, "usage_metadata", None)
                if usage is not None:
                    usage_payload = {}
                    # Common field names across SDK versions
                    if hasattr(usage, "prompt_token_count"):
                        usage_payload["prompt_tokens"] = usage.prompt_token_count
                    if hasattr(usage, "candidates_token_count"):
                        usage_payload["candidates_tokens"] = usage.candidates_token_count
                    if hasattr(usage, "output_token_count"):
                        usage_payload["output_tokens"] = usage.output_token_count
                    if hasattr(usage, "total_token_count"):
                        usage_payload["total_tokens"] = usage.total_token_count
                
                return {
                    "content": content,
                    "token_count": token_count,
                    "finish_reason": response.candidates[0].finish_reason if response.candidates else "STOP",
                    "safety_ratings": response.candidates[0].safety_ratings if response.candidates else [],
                    "usage_metadata": usage_payload,
                }
        
        except Exception as e:
            logger.error(f"Error generating Gemini response: {e}")
            raise Exception(f"Failed to generate response: {str(e)}")
    
    async def generate_streaming_response(
        self,
        system_instruction: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        model_override: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        Generate streaming response from Gemini.

        Args:
            system_instruction: System prompt/instructions
            messages: Conversation history
            temperature: Sampling temperature
            model_override: Optional model name to override the default model

        Yields:
            Text chunks as they are generated
        """
        try:
            formatted_messages = self.format_messages_for_gemini(messages)

            generation_config = {"temperature": temperature}

            # Use override model if provided, otherwise use default
            model_to_use = model_override if model_override else self.model_name

            model_with_sys = genai.GenerativeModel(
                model_name=model_to_use,
                safety_settings=self.safety_settings,
                system_instruction=system_instruction
            )
            chat = model_with_sys.start_chat(
                history=formatted_messages[:-1] if len(formatted_messages) > 1 else []
            )
            
            last_message = formatted_messages[-1] if formatted_messages else {'role': 'user', 'parts': ['']}
            user_content = last_message.get('parts', [''])[0]
            
            response = chat.send_message(
                user_content,
                generation_config=generation_config,
                stream=True
            )
            
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        
        except Exception as e:
            logger.error(f"Error in streaming response: {e}")
            raise Exception(f"Failed to stream response: {str(e)}")
    
    def should_summarize(self, token_count: int) -> bool:
        """
        Check if conversation should be summarized based on token count.
        
        Args:
            token_count: Current token count
            
        Returns:
            True if should summarize
        """
        return token_count >= self.token_threshold
    
    def get_token_limit(self) -> int:
        """Get the context token limit for this model."""
        return self.context_token_limit

    def generate_content_sync(self, prompt: str, temperature: float = 0.7) -> str:
        """
        Generate content synchronously without conversation history.
        Useful for one-off tasks like card recommendations.

        Args:
            prompt: The prompt to send to Gemini
            temperature: Sampling temperature (0.0-1.0)

        Returns:
            Generated text content
        """
        try:
            generation_config = {"temperature": temperature}

            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )

            return response.text

        except Exception as e:
            logger.error(f"Error generating sync content: {e}")
            raise Exception(f"Failed to generate content: {str(e)}")


# Global instance (singleton pattern)
_gemini_service_instance = None

def get_gemini_service() -> GeminiService:
    """
    Get or create the global GeminiService instance.

    Returns:
        Shared GeminiService instance
    """
    global _gemini_service_instance
    if _gemini_service_instance is None:
        _gemini_service_instance = GeminiService()
    return _gemini_service_instance


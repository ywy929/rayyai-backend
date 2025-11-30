"""
PII (Personally Identifiable Information) Masking Service
Masks sensitive data before sending to LLM
"""
import re
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class PIIMaskingService:
    """Service for masking PII in user data before sending to LLM"""
    
    # Patterns for detecting sensitive information
    CREDIT_CARD_PATTERN = re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b')
    ACCOUNT_NUMBER_PATTERN = re.compile(r'\b\d{8,}\b')  # Account numbers typically 8+ digits
    EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    PHONE_PATTERN = re.compile(r'\b\d{3}[-\s]?\d{3}[-\s]?\d{4}\b|\b\d{10}\b|\+\d{1,3}[\s-]?\d{3,4}[\s-]?\d{3,4}[\s-]?\d{4}\b')
    SSN_PATTERN = re.compile(r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b')
    
    def __init__(self, user_first_name: Optional[str] = None, user_last_name: Optional[str] = None):
        """
        Initialize PII masking service.
        
        Args:
            user_first_name: User's first name (to preserve in masking)
            user_last_name: User's last name (to preserve in masking)
        """
        self.user_first_name = user_first_name or ""
        self.user_last_name = user_last_name or ""
        self.allowed_names = {name.lower() for name in [user_first_name, user_last_name] if name}
    
    def mask_credit_card(self, text: str) -> str:
        """
        Mask credit card numbers, keeping last 4 digits.
        
        Args:
            text: Text containing credit card numbers
            
        Returns:
            Text with masked credit card numbers
        """
        def mask_card(match):
            card = re.sub(r'[-\s]', '', match.group())
            if len(card) >= 4:
                return f"****-****-****-{card[-4:]}"
            return "****-****-****-****"
        
        return self.CREDIT_CARD_PATTERN.sub(mask_card, text)
    
    def mask_account_number(self, text: str) -> str:
        """
        Mask account numbers.
        
        Args:
            text: Text containing account numbers
            
        Returns:
            Text with masked account numbers
        """
        def mask_account(match):
            account = match.group()
            # Keep last 4 digits if long enough, otherwise fully mask
            if len(account) >= 4:
                return f"****{account[-4:]}"
            return "****"
        
        return self.ACCOUNT_NUMBER_PATTERN.sub(mask_account, text)
    
    def mask_email(self, text: str) -> str:
        """
        Mask email addresses.
        
        Args:
            text: Text containing email addresses
            
        Returns:
            Text with masked emails
        """
        def mask_email_addr(match):
            email = match.group()
            local, domain = email.split('@')
            # Show first char and last char of local part, mask domain
            if len(local) > 1:
                masked_local = f"{local[0]}***{local[-1]}"
            else:
                masked_local = "***"
            return f"{masked_local}@***.{domain.split('.')[-1] if '.' in domain else 'com'}"
        
        return self.EMAIL_PATTERN.sub(mask_email_addr, text)
    
    def mask_phone(self, text: str) -> str:
        """
        Mask phone numbers.
        
        Args:
            text: Text containing phone numbers
            
        Returns:
            Text with masked phone numbers
        """
        return self.PHONE_PATTERN.sub("***-***-****", text)
    
    def mask_ssn(self, text: str) -> str:
        """
        Mask Social Security Numbers.
        
        Args:
            text: Text containing SSNs
            
        Returns:
            Text with masked SSNs
        """
        return self.SSN_PATTERN.sub("***-**-****", text)
    
    def mask_personal_names(self, text: str, preserve_user_name: bool = True) -> str:
        """
        Mask personal names, optionally preserving user's own name.
        This is a simple implementation - in production, use NER models.
        
        Args:
            text: Text containing names
            preserve_user_name: Whether to preserve user's name
            
        Returns:
            Text with masked names
        """
        # Simple approach: Replace common name patterns
        # Note: This is basic - production should use proper NER
        if not preserve_user_name:
            # Mask all potential names (very basic)
            # Replace capitalized words that look like names (heuristic)
            words = text.split()
            masked_words = []
            for word in words:
                # If it's a capitalized word and not in allowed names, mask it
                if word[0].isupper() and len(word) > 2 and word.lower() not in self.allowed_names:
                    masked_words.append("***")
                else:
                    masked_words.append(word)
            return " ".join(masked_words)
        
        return text  # Preserve all names if preserving user name
    
    def mask_text(self, text: str, preserve_user_name: bool = True) -> str:
        """
        Apply all PII masking to text.
        
        Args:
            text: Text to mask
            preserve_user_name: Whether to preserve user's own name
            
        Returns:
            Fully masked text
        """
        if not text:
            return text
        
        # Apply all masking functions
        masked = text
        masked = self.mask_credit_card(masked)
        masked = self.mask_account_number(masked)
        masked = self.mask_email(masked)
        masked = self.mask_phone(masked)
        masked = self.mask_ssn(masked)
        masked = self.mask_personal_names(masked, preserve_user_name)
        
        return masked
    
    def mask_dict(self, data: Dict[str, Any], fields_to_mask: Optional[list] = None) -> Dict[str, Any]:
        """
        Mask PII in dictionary data structure.
        
        Args:
            data: Dictionary containing potentially sensitive data
            fields_to_mask: List of field names to mask (if None, masks all string values)
            
        Returns:
            Dictionary with masked values
        """
        if not isinstance(data, dict):
            return data
        
        masked_data = {}
        
        # Fields that should always be masked
        sensitive_fields = {
            'card_number', 'account_no', 'account_number', 'email', 'phone',
            'phone_number', 'ssn', 'social_security_number', 'password',
            'credit_card', 'card_no'
        }
        
        for key, value in data.items():
            key_lower = key.lower()
            
            # Check if this field should be masked
            should_mask = (
                fields_to_mask and key in fields_to_mask
            ) or (
                not fields_to_mask and (
                    any(sensitive in key_lower for sensitive in sensitive_fields) or
                    isinstance(value, str)
                )
            )
            
            if should_mask and isinstance(value, str):
                masked_data[key] = self.mask_text(value)
            elif isinstance(value, dict):
                masked_data[key] = self.mask_dict(value, fields_to_mask)
            elif isinstance(value, list):
                masked_data[key] = [
                    self.mask_dict(item, fields_to_mask) if isinstance(item, dict)
                    else self.mask_text(item) if isinstance(item, str)
                    else item
                    for item in value
                ]
            else:
                masked_data[key] = value
        
        return masked_data
    
    def mask_financial_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Mask PII in financial context data.
        Specifically handles financial data structures.
        
        Args:
            context: Financial context dictionary
            
        Returns:
            Masked financial context
        """
        # Fields to always mask in financial data
        fields_to_mask = [
            'card_number', 'account_no', 'account_number', 'email',
            'phone', 'ssn', 'reference_no'
        ]
        
        return self.mask_dict(context, fields_to_mask)


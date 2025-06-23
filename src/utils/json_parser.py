# flake8: noqa: E501
"""
Robust JSON parsing utilities for model responses.
"""

import json
import re
from typing import Dict, Any, Optional, List


def extract_json_from_response(response_text: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON from a model response that may contain additional text.
    
    This function handles various response formats:
    - Pure JSON
    - JSON wrapped in markdown code blocks (```json ... ```)
    - JSON preceded by thinking sections (<think>...</think>)
    - JSON mixed with explanatory text
    
    Args:
        response_text: The raw response text from the model
        
    Returns:
        Parsed JSON dictionary or None if no valid JSON found
    """
    if not response_text:
        return None
    
    # Clean the response text
    cleaned_text = response_text.strip()
    
    # Try to parse as pure JSON first
    try:
        return json.loads(cleaned_text)
    except json.JSONDecodeError:
        pass
    
    # Look for JSON in markdown code blocks
    json_blocks = extract_json_from_markdown(cleaned_text)
    for json_str in json_blocks:
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            continue
    
    # Look for JSON objects in the text using regex
    json_objects = extract_json_objects(cleaned_text)
    for json_str in json_objects:
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            continue
    
    # If no valid JSON found, try to extract sentiment and classification from text
    return extract_sentiment_from_text(cleaned_text)


def extract_json_from_markdown(text: str) -> List[str]:
    """
    Extract JSON strings from markdown code blocks.
    
    Args:
        text: Text that may contain markdown code blocks
        
    Returns:
        List of JSON strings found in code blocks
    """
    # Pattern to match ```json ... ``` blocks
    pattern = r'```(?:json)?\s*\n(.*?)\n```'
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    
    # Also look for ``` ... ``` blocks without json specifier
    pattern2 = r'```\s*\n(.*?)\n```'
    matches2 = re.findall(pattern2, text, re.DOTALL)
    
    # Combine and deduplicate
    all_matches = matches + matches2
    return [match.strip() for match in all_matches if match.strip()]


def extract_json_objects(text: str) -> List[str]:
    """
    Extract JSON objects from text using regex patterns.
    
    Args:
        text: Text that may contain JSON objects
        
    Returns:
        List of potential JSON strings
    """
    # Remove thinking sections first
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Look for JSON objects with the expected structure
    # This pattern looks for objects containing sentiment, classification, confidence, and tags
    pattern = r'\{[^{}]*"sentiment"[^{}]*"classification"[^{}]*"confidence"[^{}]*"tags"[^{}]*\}'
    matches = re.findall(pattern, text, re.DOTALL)
    
    # Also look for any JSON object that might be valid
    # This is a more general pattern
    general_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    general_matches = re.findall(general_pattern, text, re.DOTALL)
    
    # Combine and deduplicate
    all_matches = matches + general_matches
    return [match.strip() for match in all_matches if match.strip()]


def extract_sentiment_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract sentiment and classification from text when JSON parsing fails.
    
    Args:
        text: Text to analyze for sentiment indicators
        
    Returns:
        Dictionary with extracted sentiment information or None
    """
    text_lower = text.lower()
    
    # Extract sentiment
    sentiment = "Neutral"  # Default
    if any(word in text_lower for word in ["positive", "satisfaction", "gratitude", "thank", "great", "good"]):
        sentiment = "Positive"
    elif any(word in text_lower for word in ["negative", "dissatisfaction", "complaint", "issue", "problem", "broken", "urgent"]):
        sentiment = "Negative"
    
    # Extract classification based on keywords
    classification = "General Follow-ups"  # Default
    if any(word in text_lower for word in ["product", "stock", "carry", "available", "inquiry"]):
        classification = "Product/Stocking Requests"
    elif any(word in text_lower for word in ["admin", "coordination", "schedule", "meeting", "access"]):
        classification = "Admin/Coordination"
    elif any(word in text_lower for word in ["feedback", "complaint", "suggestion", "opinion"]):
        classification = "Feedback/Complaints"
    elif any(word in text_lower for word in ["maintenance", "repair", "broken", "fix", "technician"]):
        classification = "Maintenance/Repairs"
    elif any(word in text_lower for word in ["billing", "invoice", "charge", "payment", "bill"]):
        classification = "Billing/Invoices"
    elif any(word in text_lower for word in ["logistics", "delivery", "pickup", "installation", "removal"]):
        classification = "Operational Logistics"
    
    # Extract confidence based on text clarity
    confidence = 50  # Default
    if any(word in text_lower for word in ["clear", "straightforward", "obvious", "definitely"]):
        confidence = 85
    elif any(word in text_lower for word in ["ambiguous", "unclear", "vague", "might"]):
        confidence = 30
    
    # Extract tags (simple keyword extraction)
    tags = []
    tag_keywords = ["machine", "coffee", "delivery", "repair", "billing", "product", "schedule", "urgent"]
    for keyword in tag_keywords:
        if keyword in text_lower:
            tags.append(keyword)
    
    return {
        "sentiment": sentiment,
        "classification": classification,
        "confidence": confidence,
        "tags": tags[:5],  # Limit to 5 tags
        "raw_response": text
    }


def validate_json_structure(data: Dict[str, Any]) -> bool:
    """
    Validate that the parsed JSON has the expected structure.
    
    Args:
        data: Parsed JSON data
        
    Returns:
        True if the structure is valid, False otherwise
    """
    required_fields = ["sentiment", "classification", "confidence", "tags"]
    
    # Check if all required fields are present
    if not all(field in data for field in required_fields):
        return False
    
    # Validate sentiment
    valid_sentiments = ["Positive", "Neutral", "Negative"]
    if data.get("sentiment") not in valid_sentiments:
        return False
    
    # Validate classification
    valid_classifications = [
        "Product/Stocking Requests",
        "Admin/Coordination", 
        "Feedback/Complaints",
        "Maintenance/Repairs",
        "Billing/Invoices",
        "General Follow-ups",
        "Operational Logistics"
    ]
    if data.get("classification") not in valid_classifications:
        return False
    
    # Validate confidence (should be 0-100)
    confidence = data.get("confidence")
    if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 100:
        return False
    
    # Validate tags (should be a list)
    if not isinstance(data.get("tags"), list):
        return False
    
    return True


def parse_model_response(response_text: str) -> Dict[str, Any]:
    """
    Parse a model response and return a validated result.
    
    Args:
        response_text: Raw response text from the model
        
    Returns:
        Parsed and validated result dictionary
    """
    # Try to extract JSON
    result = extract_json_from_response(response_text)
    
    # If no result found, create a default
    if result is None:
        result = {
            "sentiment": "Neutral",
            "classification": "General Follow-ups", 
            "confidence": 50,
            "tags": [],
            "raw_response": response_text
        }
    
    # Validate the structure
    if not validate_json_structure(result):
        # If validation fails, create a corrected version
        result = {
            "sentiment": result.get("sentiment", "Neutral"),
            "classification": result.get("classification", "General Follow-ups"),
            "confidence": result.get("confidence", 50),
            "tags": result.get("tags", []) if isinstance(result.get("tags"), list) else [],
            "raw_response": response_text
        }
    
    return result 
# flake8: noqa: E501
"""
Combined analyzer for sentiment and classification of emails.
"""

import json
import time
from typing import Dict, Any, List, Optional
from data.ollama_client import OllamaClient
from core.email_processor import EmailProcessor
from utils.json_parser import parse_model_response


class CombinedAnalyzer:
    """Analyze sentiment and classify emails using Ollama models."""

    def __init__(
        self,
        model_name: str = "deepseek-r1:1.5b",
        ollama_url: str = "http://localhost:11434",
        confidence_threshold: float = 0.6,
        max_retries: int = 2,
    ):
        """
        Initialize the combined analyzer.

        Args:
            model_name: Name of the Ollama model to use
            ollama_url: URL of the Ollama API
            confidence_threshold: Minimum confidence threshold for classification
            max_retries: Maximum number of retries for low confidence results
        """
        self.model_name = model_name
        self.client = OllamaClient(base_url=ollama_url)
        self.email_processor = EmailProcessor()
        self.confidence_threshold = confidence_threshold
        self.max_retries = max_retries

    def analyze_email(self, email_text: str) -> Dict[str, Any]:
        """
        Analyze the sentiment and classify an email.

        Args:
            email_text: Email text to analyze

        Returns:
            Analysis result with sentiment, classification, confidence, and tags
        """
        # Clean the email text
        cleaned_text = self.email_processor._clean_text(email_text)

        # Create the system prompt with examples
        system_prompt = self._create_system_prompt()

        # Analyze the email
        result = self._analyze_with_retry(cleaned_text, system_prompt)

        return result

    def _analyze_with_retry(self, text: str, system_prompt: str) -> Dict[str, Any]:
        """
        Analyze text with retry mechanism for low confidence results.

        Args:
            text: Text to analyze
            system_prompt: System prompt for the model

        Returns:
            Analysis result
        """
        retries = 0
        best_result = None
        best_confidence = 0.0

        while retries <= self.max_retries:
            # Analyze the text
            prompt = f"Analyze the following email text:\n\n{text}"
            response = self.client.generate(self.model_name, prompt, system_prompt)

            # Use the robust JSON parser to extract the result
            result = parse_model_response(response.get("response", ""))

            # Check if the result has the expected structure
            if "classification" in result and "confidence" in result:
                confidence = float(result["confidence"]) / 100.0  # Convert percentage to decimal

                # If confidence is above threshold, return the result
                if confidence >= self.confidence_threshold:
                    return result

                # Otherwise, keep track of the best result so far
                if best_result is None or confidence > best_confidence:
                    best_result = result
                    best_confidence = confidence

            # Increment retry counter
            retries += 1

            # If we're going to retry, wait a bit
            if retries <= self.max_retries:
                time.sleep(1)

        # Return the best result we found
        return best_result or {
            "sentiment": "Neutral",
            "classification": "General Follow-ups",
            "confidence": 50,
            "tags": [],
        }

    def _create_system_prompt(self) -> str:
        """
        Create a system prompt for the model with examples.

        Returns:
            System prompt string
        """
        return """
You are a meticulous and expert email analysis AI. Your primary task is to accurately analyze the core sentiment and classify the primary purpose of incoming emails based on predefined categories. You must adhere strictly to the specified output format.

**Analysis Steps (Internal Thought Process):**

1.  **Read the entire email carefully** to understand the context and sender's main goal.
2.  **Determine the overall sentiment:** Is the tone Positive, Neutral, or Negative?
3.  **Identify the primary purpose:** Based on the definitions below, which category best represents the core reason for the email? If multiple topics are mentioned, focus on the main action requested or information conveyed.
4.  **Assign a confidence score:** How clearly does the email fit into the chosen category? Use the guidance provided.
5.  **Extract relevant tags:** Identify key entities, actions, or topics as concise tags.
6.  **Format the output:** Construct the JSON object precisely as specified.

**Sentiment Categories:**

* `Positive`: Expresses satisfaction, agreement, confirmation, or gratitude.
* `Neutral`: Informative, asking questions, making arrangements without strong emotion.
* `Negative`: Expresses dissatisfaction, complaints, urgency due to problems, or issues.

**Classification Categories (Choose ONE):**

* `Product/Stocking Requests`: Inquiries about product availability, requests for new products, specific product details, or restocking existing items. (e.g., "Do you carry brand X?", "Can we get more coffee?", "Need specs for model Y.")
* `Admin/Coordination`: Internal or external communication regarding account setup, user access, contact information changes, scheduling meetings (unrelated to logistics/repairs), or general administrative tasks. (e.g., "Update our billing contact.", "Need access for a new employee.", "Can we schedule a call?")
* `Feedback/Complaints`: Expressing opinions about service, products, or experiences; suggestions for improvement; formal complaints not solely related to a broken item or incorrect bill. (e.g., "The delivery driver was rude.", "Love the new selection!", "Suggestion for your website.")
* `Maintenance/Repairs`: Reporting malfunctioning equipment, requesting repairs, scheduling maintenance visits, or following up on existing repair tickets. (e.g., "The coffee machine is broken.", "Need someone to fix the cooler.", "When is the technician coming?")
* `Billing/Invoices`: Questions about charges, requests for invoices, disputes over bills, notifications of payment. (e.g., "Received an incorrect invoice.", "Where can I find my last bill?", "Payment has been sent.")
* `General Follow-ups`: Emails checking in on previous requests or communications where the original topic isn't being reiterated in detail, or simple status checks. (e.g., "Just checking in on my previous email.", "Any update on this?", "Following up on our conversation.")
* `Operational Logistics`: Coordination related to deliveries, pickups, installations, removals, specific event support, or on-site service timing *not* primarily about billing or maintenance. (e.g., "Confirming delivery for Tuesday.", "Need to reschedule the pickup.", "Kegerator removal request.")

**Confidence Score Guidance:**

* `85-100`: High confidence. The email clearly fits one category with minimal ambiguity.
* `60-84`: Medium confidence. The email primarily fits one category, but has elements that could touch on another, or the intent is slightly unclear.
* `0-59`: Low confidence. The email is ambiguous, vague, or could reasonably fit into multiple categories. Requires careful review.

**Tag Generation Guidance:**

* Extract 2-5 concise tags.
* Focus on key nouns (products, locations, specific issues), key verbs/actions (request, repair, schedule, remove), and important modifiers.
* Keep tags 1-3 words long.

**Output Format (Strictly JSON):**

Respond *only* with a JSON object adhering to this structure:
```json
{
    "sentiment": "Positive|Neutral|Negative",
    "classification": "One of the 7 categories listed above",
    "confidence": 0-100,
    "tags": ["tag1", "tag2", "tag3"]
}

Here are some examples to guide your analysis:

Example 1:
Email: You guys have been so quick with helping us to get this fixed, thank you!! We have a few events happening in office that day (Monday, July 8th), but things should be in more of a lull around 10 AM. Would we be able to schedule the removal/delivery for 10:00 AM instead of 9:00 AM?
Analysis: {
    "sentiment": "Positive",
    "classification": "Operational Logistics",
    "confidence": 95,
    "tags": ["scheduling", "delivery time change", "removal"]
}

Example 2:
Email: I hope everyone is doing well. Hate to be a squeaky wheel here but is there an update on the Pepsi machine? It appears the machine is still out of order and is not cooling.
Analysis: {
     "sentiment": "Negative",
    "classification": "Maintenance/Repairs",
    "confidence": 100,
    "tags": ["Pepsi machine", "cooling issue", "out of order", "update request"]
}

Example 3:
Email: I received another bill for a minimum charge. As the contract says below, we aren't held accountable for paying a minimum monthly sales amount.
Analysis: {
    "sentiment": "Negative",
    "classification": "Billing/Invoices",
    "confidence": 100,
    "tags": ["minimum charge", "contract", "billing dispute"]
}

Example 4:
Email: The kegerator on the 7th floor is still onsite. We must get this removed by tomorrow morning. I've told our upper management this was gone, because over 2 weeks ago it was supposed to be but I've walked through the 7th floor and it's still there. Please let me know the plans to get this removed by tomorrow morning.
Analysis: {
     "sentiment": "Negative",
    "classification": "Operational Logistics",
    "confidence": 90,
    "tags": ["kegerator", "removal", "urgent", "7th floor"]
}

Example 5:
Email: I hope all is well with you, I have a question about cold brew coffee. Do you carry Wander Bear Straight Black Organic Cold Brew Coffee?
Analysis: {
    "sentiment": "Neutral",
    "classification": "Product/Stocking Requests",
    "confidence": 100,
    "tags": ["cold brew", "coffee", "organic", "product inquiry"]
}

Example 6:
Email: We are working on adding a hard wire connection to the kiosk. We already ran the wires and just need our IT department to reconfigure the network requirements. We will have this work completed before July 22nd and will let you know once it's completed.
Analysis: {
    "sentiment": "Neutral",
    "classification": "Operational Logistics",
    "confidence": 75,
    "tags": ["kiosk", "network configuration", "installation update", "IT department"]
}

Only respond with the JSON object, nothing else.
"""

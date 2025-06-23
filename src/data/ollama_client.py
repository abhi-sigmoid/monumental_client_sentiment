# flake8: noqa: E501
"""
Ollama Client for interacting with local Ollama models.
"""

import requests
import json
import time
from typing import Dict, Any, Optional, List
from utils.json_parser import parse_model_response


class OllamaClient:
    """Client for interacting with Ollama API."""

    def __init__(self, base_url: str = "http://localhost:11434"):
        """
        Initialize the Ollama client.

        Args:
            base_url: Base URL for the Ollama API
        """
        self.base_url = base_url
        self.api_url = f"{base_url}/api"

    def list_models(self) -> List[Dict[str, Any]]:
        """
        List all available models.

        Returns:
            List of model information dictionaries
        """
        response = requests.get(f"{self.api_url}/tags")
        response.raise_for_status()
        return response.json().get("models", [])

    def generate(
        self,
        model: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate a response from the model.

        Args:
            model: Name of the model to use
            prompt: The prompt to send to the model
            system_prompt: Optional system prompt to guide the model
            temperature: Sampling temperature (higher = more creative)
            max_tokens: Maximum number of tokens to generate

        Returns:
            Response from the model
        """
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "temperature": temperature,
        }

        if system_prompt:
            payload["system"] = system_prompt

        if max_tokens:
            payload["max_tokens"] = max_tokens

        response = requests.post(f"{self.api_url}/generate", json=payload)
        response.raise_for_status()
        return response.json()

    def analyze_sentiment(
        self, model: str, text: str, system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze sentiment of the given text.

        Args:
            model: Name of the model to use
            text: Text to analyze
            system_prompt: Optional system prompt to guide the model

        Returns:
            Sentiment analysis result
        """
        if not system_prompt:
            system_prompt = """
            You are a sentiment analysis expert. Analyze the sentiment of the text and
            respond with a JSON object with the following structure:
            {
                "sentiment": "positive|negative|neutral",
                "confidence": 0.0-1.0,
                "explanation": "brief explanation of the sentiment"
            }
            Only respond with the JSON object, nothing else.
            """

        prompt = f"Analyze the sentiment of the following text:\n\n{text}"
        response = self.generate(model, prompt, system_prompt)

        # Use the robust JSON parser to extract the result
        result = parse_model_response(response.get("response", ""))
        
        # Convert the result to the expected format for sentiment analysis
        sentiment_result = {
            "sentiment": result.get("sentiment", "neutral").lower(),
            "confidence": result.get("confidence", 50) / 100.0,  # Convert to 0-1 scale
            "explanation": result.get("explanation", ""),
            "raw_response": result.get("raw_response", "")
        }
        
        return sentiment_result

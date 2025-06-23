# flake8: noqa: E501
"""
Test script to verify Ollama connection and model availability.
"""

import unittest
import sys
import os
from pathlib import Path

# Add src to path for imports
# sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.data.ollama_client import OllamaClient


def test_ollama_connection():
    """Test the Ollama connection and list available models."""
    try:
        print("Testing Ollama connection...")

        # Initialize client
        client = OllamaClient()

        # List available models
        print("Available models:")
        models = client.list_models()

        if not models:
            print(
                "No models found. Please ensure Ollama is running and models are installed."
            )
            return False

        for model in models:
            print(f"  - {model.get('name', 'Unknown')}")

        # Check if deepseek model is available
        model_names = [model.get("name", "") for model in models]
        if "deepseek-r1:1.5b" in model_names:
            print("\n✓ deepseek-r1:1.5b model is available!")
            return True
        else:
            print("\n✗ deepseek-r1:1.5b model not found.")
            print("Please install the deepseek model using:")
            print("  ollama pull deepseek-r1:1.5b")
            return False

    except Exception as e:
        print(f"Error connecting to Ollama: {str(e)}")
        print("Please ensure Ollama is running on http://localhost:11434")
        return False


def test_model_generation():
    """Test a simple generation with the deepseek model."""
    try:
        print("\nTesting model generation...")

        client = OllamaClient()

        # Simple test prompt
        test_prompt = "Hello, this is a test. Please respond with 'Test successful'."

        response = client.generate("deepseek-r1:1.5b", test_prompt)

        if response and "response" in response:
            print("✓ Model generation successful!")
            print(f"Response: {response['response'][:100]}...")
            return True
        else:
            print("✗ Model generation failed.")
            return False

    except Exception as e:
        print(f"Error testing model generation: {str(e)}")
        return False


def main():
    """Main test function."""
    print("=" * 50)
    print("OLLAMA CONNECTION TEST")
    print("=" * 50)

    # Test connection
    connection_ok = test_ollama_connection()

    if connection_ok:
        # Test generation
        generation_ok = test_model_generation()

        if generation_ok:
            print("\n" + "=" * 50)
            print("✓ ALL TESTS PASSED")
            print("Ollama is ready for email analysis!")
            print("=" * 50)
        else:
            print("\n" + "=" * 50)
            print("✗ MODEL GENERATION FAILED")
            print("Please check your Ollama setup.")
            print("=" * 50)
            sys.exit(1)
    else:
        print("\n" + "=" * 50)
        print("✗ CONNECTION FAILED")
        print("Please start Ollama and install the deepseek model.")
        print("=" * 50)
        sys.exit(1)


if __name__ == "__main__":
    main()

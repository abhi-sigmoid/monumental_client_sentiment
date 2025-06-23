# flake8: noqa: E501
"""
Text preprocessing utilities for email analysis.
"""

import re
import string
from typing import List, Optional


def clean_text(text: str) -> str:
    """
    Clean text by removing special characters, extra whitespace, etc.

    Args:
        text: Text to clean

    Returns:
        Cleaned text
    """
    # Remove email signatures (simple heuristic)
    text = re.sub(r"--+[\s\S]+", "", text)

    # Remove URLs
    text = re.sub(r"https?://\S+", "", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def remove_html_tags(html_text: str) -> str:
    """
    Remove HTML tags from text.

    Args:
        html_text: HTML text to clean

    Returns:
        Plain text without HTML tags
    """
    # Simple HTML tag removal
    text = re.sub(r"<[^>]+>", " ", html_text)
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    """
    Tokenize text into words.

    Args:
        text: Text to tokenize

    Returns:
        List of tokens
    """
    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Split into words
    tokens = text.split()

    return tokens


def remove_stopwords(
    tokens: List[str], stopwords: Optional[List[str]] = None
) -> List[str]:
    """
    Remove stopwords from a list of tokens.

    Args:
        tokens: List of tokens
        stopwords: Optional list of stopwords to remove

    Returns:
        List of tokens without stopwords
    """
    # Default English stopwords if none provided
    if stopwords is None:
        stopwords = [
            "a",
            "an",
            "the",
            "and",
            "or",
            "but",
            "if",
            "because",
            "as",
            "what",
            "i",
            "me",
            "my",
            "myself",
            "we",
            "our",
            "ours",
            "ourselves",
            "you",
            "your",
            "yours",
            "yourself",
            "yourselves",
            "he",
            "him",
            "his",
            "himself",
            "she",
            "her",
            "hers",
            "herself",
            "it",
            "its",
            "itself",
            "they",
            "them",
            "their",
            "theirs",
            "themselves",
            "this",
            "that",
            "these",
            "those",
            "am",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "having",
            "do",
            "does",
            "did",
            "doing",
            "would",
            "should",
            "could",
            "ought",
            "to",
            "at",
            "by",
            "for",
            "with",
            "about",
            "against",
            "between",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "from",
            "up",
            "down",
            "in",
            "out",
            "on",
            "off",
            "over",
            "under",
            "again",
            "further",
            "then",
            "once",
            "here",
            "there",
            "when",
            "where",
            "why",
            "how",
            "all",
            "any",
            "both",
            "each",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "no",
            "nor",
            "not",
            "only",
            "own",
            "same",
            "so",
            "than",
            "too",
            "very",
        ]

    return [token for token in tokens if token not in stopwords]


def extract_keywords(text: str, num_keywords: int = 10) -> List[str]:
    """
    Extract keywords from text based on frequency.

    Args:
        text: Text to extract keywords from
        num_keywords: Number of keywords to extract

    Returns:
        List of keywords
    """
    # Clean the text
    clean = clean_text(text)

    # Tokenize
    tokens = tokenize(clean)

    # Remove stopwords
    filtered_tokens = remove_stopwords(tokens)

    # Count frequency
    word_freq = {}
    for token in filtered_tokens:
        if token in word_freq:
            word_freq[token] += 1
        else:
            word_freq[token] = 1

    # Sort by frequency
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)

    # Return top keywords
    return [word for word, _ in sorted_words[:num_keywords]]

# flake8: noqa: E501
"""
Email processor for handling and preparing email data.
"""

import re
import email
from email import policy
from email.parser import BytesParser
from typing import Dict, Any, List, Optional, Union
from pathlib import Path


class EmailProcessor:
    """Process and prepare emails for sentiment analysis and classification."""

    def __init__(self):
        """Initialize the email processor."""
        pass

    def parse_email_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Parse an email file and extract relevant information.

        Args:
            file_path: Path to the email file

        Returns:
            Dictionary containing email information
        """
        with open(file_path, "rb") as fp:
            msg = BytesParser(policy=policy.default).parse(fp)

        # Extract basic information
        email_data = {
            "subject": msg.get("subject", ""),
            "from": msg.get("from", ""),
            "to": msg.get("to", ""),
            "date": msg.get("date", ""),
            "body": self._get_email_body(msg),
            "attachments": self._get_attachments(msg),
        }

        return email_data

    def _get_email_body(self, msg: email.message.EmailMessage) -> str:
        """
        Extract the body text from an email message.

        Args:
            msg: Email message object

        Returns:
            Body text of the email
        """
        body = ""

        if msg.is_multipart():
            for part in msg.iter_parts():
                content_type = part.get_content_type()
                if content_type == "text/plain":
                    body += part.get_content()
                elif content_type == "text/html":
                    # If we only have HTML, use it but we'll need to clean it
                    if not body:
                        body = self._clean_html(part.get_content())
        else:
            content_type = msg.get_content_type()
            if content_type == "text/plain":
                body = msg.get_content()
            elif content_type == "text/html":
                body = self._clean_html(msg.get_content())

        return body

    def _clean_html(self, html_content: str) -> str:
        """
        Clean HTML content to extract plain text.

        Args:
            html_content: HTML content to clean

        Returns:
            Cleaned plain text
        """
        # Simple HTML tag removal - for production use consider using a library like BeautifulSoup
        text = re.sub(r"<[^>]+>", " ", html_content)
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _get_attachments(self, msg: email.message.EmailMessage) -> List[Dict[str, Any]]:
        """
        Extract attachment information from an email message.

        Args:
            msg: Email message object

        Returns:
            List of attachment information dictionaries
        """
        attachments = []

        if msg.is_multipart():
            for part in msg.iter_parts():
                if part.get_content_disposition() == "attachment":
                    attachments.append(
                        {
                            "filename": part.get_filename(),
                            "content_type": part.get_content_type(),
                            "size": len(part.get_content()),
                        }
                    )

        return attachments

    def preprocess_for_sentiment(self, email_data: Dict[str, Any]) -> str:
        """
        Preprocess email data for sentiment analysis.

        Args:
            email_data: Email data dictionary

        Returns:
            Preprocessed text for sentiment analysis
        """
        # For sentiment analysis, we'll focus on the subject and body
        subject = email_data.get("subject", "")
        body = email_data.get("body", "")

        # Combine subject and body with appropriate weighting
        # Subject often carries significant sentiment information
        text = f"{subject}\n\n{body}"

        # Clean the text
        text = self._clean_text(text)

        return text

    def _clean_text(self, text: str) -> str:
        """
        Clean text for analysis.

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
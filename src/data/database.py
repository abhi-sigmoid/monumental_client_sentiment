# flake8: noqa: E501
"""
Database module for storing email analysis results.
"""

import os
import json
import sqlite3
from typing import Dict, Any, List, Optional
from datetime import datetime


class Database:
    """Database for storing email analysis results."""

    # Valid classification categories
    VALID_CLASSIFICATIONS = {
        'Product/Stocking Requests',
        'Admin/Coordination', 
        'Feedback/Complaints',
        'Maintenance/Repairs',
        'Billing/Invoices',
        'General Follow-ups',
        'Operational Logistics'
    }
    
    # Valid sentiment values
    VALID_SENTIMENTS = {
        'Positive',
        'Neutral', 
        'Negative'
    }

    def __init__(self, db_path: str = "data/email_analysis.db"):
        """
        Initialize the database.

        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path

        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        # Initialize the database
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create the email_analysis table
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS email_analysis (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email_date TEXT,
            email_text TEXT NOT NULL,
            sentiment TEXT NOT NULL,
            classification TEXT NOT NULL,
            confidence REAL NOT NULL,
            tags TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        )

        conn.commit()
        conn.close()

    def _validate_analysis_result(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and normalize analysis result data.
        
        Args:
            analysis_result: Analysis result dictionary
            
        Returns:
            Validated and normalized analysis result
        """
        validated_result = analysis_result.copy()
        
        # Validate and normalize sentiment
        sentiment = analysis_result.get("sentiment", "Neutral")
        if sentiment not in self.VALID_SENTIMENTS:
            # Try to normalize common variations
            sentiment_lower = sentiment.lower()
            if 'positive' in sentiment_lower:
                sentiment = 'Positive'
            elif 'negative' in sentiment_lower:
                sentiment = 'Negative'
            else:
                sentiment = 'Neutral'  # Default to neutral for unknown values
        validated_result["sentiment"] = sentiment
        
        # Validate and normalize classification
        classification = analysis_result.get("classification", "General Follow-ups")
        if classification not in self.VALID_CLASSIFICATIONS:
            # Try to normalize common variations
            classification_lower = classification.lower()
            if 'product' in classification_lower or 'stock' in classification_lower:
                classification = 'Product/Stocking Requests'
            elif 'admin' in classification_lower or 'coordination' in classification_lower:
                classification = 'Admin/Coordination'
            elif 'feedback' in classification_lower or 'complaint' in classification_lower:
                classification = 'Feedback/Complaints'
            elif 'maintenance' in classification_lower or 'repair' in classification_lower:
                classification = 'Maintenance/Repairs'
            elif 'billing' in classification_lower or 'invoice' in classification_lower:
                classification = 'Billing/Invoices'
            elif 'logistics' in classification_lower or 'operational' in classification_lower:
                classification = 'Operational Logistics'
            else:
                classification = 'General Follow-ups'  # Default for unknown values
        validated_result["classification"] = classification
        
        # Validate confidence
        confidence = analysis_result.get("confidence", 0)
        try:
            confidence = float(confidence)
            confidence = max(0, min(100, confidence))  # Clamp between 0 and 100
        except (ValueError, TypeError):
            confidence = 50.0  # Default confidence
        validated_result["confidence"] = confidence
        
        # Validate tags
        tags = analysis_result.get("tags", [])
        if not isinstance(tags, list):
            tags = []
        validated_result["tags"] = tags
        
        return validated_result

    def save_analysis(self, email_text: str, analysis_result: Dict[str, Any], email_date: Optional[str] = None) -> int:
        """
        Save an analysis result to the database.

        Args:
            email_text: Original email text
            analysis_result: Analysis result dictionary
            email_date: Date of the email (optional)

        Returns:
            ID of the inserted record
        """
        # Validate and normalize the analysis result
        validated_result = self._validate_analysis_result(analysis_result)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Extract the validated analysis data
        sentiment = validated_result.get("sentiment", "Neutral")
        classification = validated_result.get("classification", "General Follow-ups")
        confidence = float(validated_result.get("confidence", 0)) / 100.0  # Convert percentage to decimal
        tags = json.dumps(validated_result.get("tags", []))

        # Insert the record
        cursor.execute(
            """
            INSERT INTO email_analysis
            (email_date, email_text, sentiment, classification, confidence, tags, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (email_date, email_text, sentiment, classification, confidence, tags, datetime.now()),
        )

        # Get the ID of the inserted record
        record_id = cursor.lastrowid

        conn.commit()
        conn.close()

        return record_id

    def get_analysis(self, record_id: int) -> Optional[Dict[str, Any]]:
        """
        Get an analysis result from the database.

        Args:
            record_id: ID of the record to retrieve

        Returns:
            Analysis result dictionary or None if not found
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Get the record
        cursor.execute("SELECT * FROM email_analysis WHERE id = ?", (record_id,))

        row = cursor.fetchone()
        conn.close()

        if row is None:
            return None

        # Convert the row to a dictionary
        result = dict(row)

        # Parse the tags JSON
        result["tags"] = json.loads(result["tags"])

        # Convert confidence to percentage
        result["confidence"] = int(result["confidence"] * 100)

        return result

    def get_all_analyses(self) -> List[Dict[str, Any]]:
        """
        Get all analysis results from the database.

        Returns:
            List of analysis result dictionaries
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Get all records
        cursor.execute("SELECT * FROM email_analysis ORDER BY created_at DESC")

        rows = cursor.fetchall()
        conn.close()

        # Convert the rows to dictionaries
        results = []
        for row in rows:
            result = dict(row)

            # Parse the tags JSON
            result["tags"] = json.loads(result["tags"])

            # Convert confidence to percentage
            result["confidence"] = int(result["confidence"] * 100)

            results.append(result)

        return results

    def delete_analysis(self, record_id: int) -> bool:
        """
        Delete an analysis result from the database.

        Args:
            record_id: ID of the record to delete

        Returns:
            True if the record was deleted, False otherwise
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Delete the record
        cursor.execute("DELETE FROM email_analysis WHERE id = ?", (record_id,))

        deleted = cursor.rowcount > 0

        conn.commit()
        conn.close()

        return deleted

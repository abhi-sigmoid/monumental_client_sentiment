#!/usr/bin/env python3
# flake8: noqa: E501
"""
Database sanitization script for email analysis data.
Normalizes classification and sentiment values to ensure data quality.
"""

import sqlite3
import pandas as pd
from typing import Dict, List, Tuple
import sys
import os
import re
from datetime import datetime

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from data.database import Database


class DatabaseSanitizer:
    """Class for sanitizing email analysis database data."""

    # Expected classification categories
    EXPECTED_CLASSIFICATIONS = {
        "Product/Stocking Requests",
        "Admin/Coordination",
        "Feedback/Complaints",
        "Maintenance/Repairs",
        "Billing/Invoices",
        "General Follow-ups",
        "Operational Logistics",
    }

    # Expected sentiment values
    EXPECTED_SENTIMENTS = {"Positive", "Neutral", "Negative"}

    # Classification mapping for normalization
    CLASSIFICATION_MAPPING = {
        # Product/Stocking Requests variations
        "Product/Stockings Requests": "Product/Stocking Requests",
        "Product/StockING REQUESTS": "Product/Stocking Requests",
        "Product/Stockting Requests": "Product/Stocking Requests",
        # Admin/Coordination variations
        "Administrative/Coordination": "Admin/Coordination",
        "Administration/Coordination": "Admin/Coordination",
        # Operational Logistics variations
        "Operations/Logistics": "Operational Logistics",
        "Maintaining Operations Logistics": "Operational Logistics",
        "Logistical Operations": "Operational Logistics",
        # Maintenance variations
        "Maintenance/Coordination": "Maintenance/Repairs",
        # Communication variations
        "Internal/External Communication": "Admin/Coordination",
        "Communication/Feedback": "Feedback/Complaints",
        # Incorrect sentiment values in classification column
        "Neutral": "General Follow-ups",
        "Negative": "Feedback/Complaints",
    }

    # Sentiment mapping for normalization
    SENTIMENT_MAPPING = {
        "Positive|Neutral|Negative": "Neutral",  # Default to neutral for ambiguous cases
        "Neutral|Slight Positive": "Positive",  # Slight positive -> positive
    }

    def __init__(self, db_path: str = "data/email_analysis.db"):
        """
        Initialize the sanitizer.

        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self.db = Database(db_path)

    def get_data_quality_report(self) -> Dict:
        """Generate a report of current data quality issues."""
        conn = sqlite3.connect(self.db_path)

        # Get current classifications
        classification_df = pd.read_sql_query(
            "SELECT classification, COUNT(*) as count FROM email_analysis GROUP BY classification ORDER BY count DESC",
            conn,
        )

        # Get current sentiments
        sentiment_df = pd.read_sql_query(
            "SELECT sentiment, COUNT(*) as count FROM email_analysis GROUP BY sentiment ORDER BY count DESC",
            conn,
        )

        conn.close()

        # Identify issues
        invalid_classifications = []
        invalid_sentiments = []

        for classification in classification_df["classification"]:
            if classification not in self.EXPECTED_CLASSIFICATIONS:
                invalid_classifications.append(classification)

        for sentiment in sentiment_df["sentiment"]:
            if sentiment not in self.EXPECTED_SENTIMENTS:
                invalid_sentiments.append(sentiment)

        return {
            "total_records": classification_df["count"].sum(),
            "valid_classifications": len(classification_df)
            - len(invalid_classifications),
            "invalid_classifications": len(invalid_classifications),
            "valid_sentiments": len(sentiment_df) - len(invalid_sentiments),
            "invalid_sentiments": len(invalid_sentiments),
            "classification_details": classification_df.to_dict("records"),
            "sentiment_details": sentiment_df.to_dict("records"),
            "invalid_classifications_list": invalid_classifications,
            "invalid_sentiments_list": invalid_sentiments,
        }

    def sanitize_classifications(self) -> Tuple[int, int]:
        """
        Sanitize classification values.

        Returns:
            Tuple of (records_updated, records_skipped)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        records_updated = 0
        records_skipped = 0

        # Get all records with their current classifications
        cursor.execute("SELECT id, classification FROM email_analysis")
        records = cursor.fetchall()

        for record_id, current_classification in records:
            if current_classification in self.CLASSIFICATION_MAPPING:
                new_classification = self.CLASSIFICATION_MAPPING[current_classification]
                cursor.execute(
                    "UPDATE email_analysis SET classification = ? WHERE id = ?",
                    (new_classification, record_id),
                )
                records_updated += 1
            elif current_classification not in self.EXPECTED_CLASSIFICATIONS:
                # For unmapped invalid classifications, default to General Follow-ups
                cursor.execute(
                    "UPDATE email_analysis SET classification = ? WHERE id = ?",
                    ("General Follow-ups", record_id),
                )
                records_updated += 1
            else:
                records_skipped += 1

        conn.commit()
        conn.close()

        return records_updated, records_skipped

    def sanitize_sentiments(self) -> Tuple[int, int]:
        """
        Sanitize sentiment values.

        Returns:
            Tuple of (records_updated, records_skipped)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        records_updated = 0
        records_skipped = 0

        # Get all records with their current sentiments
        cursor.execute("SELECT id, sentiment FROM email_analysis")
        records = cursor.fetchall()

        for record_id, current_sentiment in records:
            if current_sentiment in self.SENTIMENT_MAPPING:
                new_sentiment = self.SENTIMENT_MAPPING[current_sentiment]
                cursor.execute(
                    "UPDATE email_analysis SET sentiment = ? WHERE id = ?",
                    (new_sentiment, record_id),
                )
                records_updated += 1
            elif current_sentiment not in self.EXPECTED_SENTIMENTS:
                # For unmapped invalid sentiments, default to Neutral
                cursor.execute(
                    "UPDATE email_analysis SET sentiment = ? WHERE id = ?",
                    ("Neutral", record_id),
                )
                records_updated += 1
            else:
                records_skipped += 1

        conn.commit()
        conn.close()

        return records_updated, records_skipped

    def sanitize_email_dates(self) -> int:
        """
        Sanitize email_date values to ensure they are in YYYY-MM-DD format.
        Attempts to parse malformed dates and falls back to created_at if parsing fails.
        Returns:
            Number of records updated
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        records_updated = 0
        # Get all records with their current email_date and created_at
        cursor.execute("SELECT id, email_date, created_at FROM email_analysis")
        records = cursor.fetchall()
        for record_id, email_date, created_at in records:
            new_date = None
            # Try to parse email_date
            try:
                # Remove any percent-encoding or weird chars
                cleaned = re.sub(r"[^0-9\-]", "-", str(email_date))
                # Try to parse as date
                dt = pd.to_datetime(cleaned, errors='coerce')
                if pd.notnull(dt):
                    new_date = dt.strftime("%Y-%m-%d")
            except Exception:
                new_date = None
            # If parsing failed, fallback to created_at
            if not new_date or new_date.startswith('NaT'):
                try:
                    dt = pd.to_datetime(created_at, errors='coerce')
                    if pd.notnull(dt):
                        new_date = dt.strftime("%Y-%m-%d")
                except Exception:
                    new_date = None
            # Only update if new_date is valid and different
            if new_date and new_date != email_date:
                cursor.execute(
                    "UPDATE email_analysis SET email_date = ? WHERE id = ?",
                    (new_date, record_id),
                )
                records_updated += 1
        conn.commit()
        conn.close()
        return records_updated

    def sanitize_all(self) -> Dict:
        """
        Sanitize all data quality issues.

        Returns:
            Dictionary with sanitization results
        """
        print("🔍 Generating data quality report...")
        before_report = self.get_data_quality_report()

        print("🧹 Sanitizing classifications...")
        class_updated, class_skipped = self.sanitize_classifications()

        print("🧹 Sanitizing sentiments...")
        sent_updated, sent_skipped = self.sanitize_sentiments()

        print("🧹 Sanitizing email dates...")
        date_updated = self.sanitize_email_dates()

        print("🔍 Generating post-sanitization report...")
        after_report = self.get_data_quality_report()

        return {
            "before": before_report,
            "after": after_report,
            "classifications_updated": class_updated,
            "classifications_skipped": class_skipped,
            "sentiments_updated": sent_updated,
            "sentiments_skipped": sent_skipped,
            "email_dates_updated": date_updated,
            "total_updates": class_updated + sent_updated + date_updated,
        }

    def print_sanitization_report(self, results: Dict):
        """Print a detailed sanitization report."""
        print("\n" + "=" * 60)
        print("📊 DATABASE SANITIZATION REPORT")
        print("=" * 60)

        print(f"\n📈 SUMMARY:")
        print(f"   Total records: {results['before']['total_records']}")
        print(f"   Classifications updated: {results['classifications_updated']}")
        print(f"   Sentiments updated: {results['sentiments_updated']}")
        print(f"   Email dates updated: {results['email_dates_updated']}")
        print(f"   Total updates: {results['total_updates']}")

        print(f"\n🔍 BEFORE SANITIZATION:")
        print(
            f"   Invalid classifications: {results['before']['invalid_classifications']}"
        )
        print(f"   Invalid sentiments: {results['before']['invalid_sentiments']}")

        if results["before"]["invalid_classifications_list"]:
            print(f"   Invalid classifications found:")
            for invalid in results["before"]["invalid_classifications_list"]:
                print(f"     - {invalid}")

        if results["before"]["invalid_sentiments_list"]:
            print(f"   Invalid sentiments found:")
            for invalid in results["before"]["invalid_sentiments_list"]:
                print(f"     - {invalid}")

        print(f"\n✅ AFTER SANITIZATION:")
        print(
            f"   Invalid classifications: {results['after']['invalid_classifications']}"
        )
        print(f"   Invalid sentiments: {results['after']['invalid_sentiments']}")

        print(f"\n📋 CURRENT CLASSIFICATIONS:")
        for item in results["after"]["classification_details"]:
            print(f"   {item['classification']}: {item['count']} records")

        print(f"\n📋 CURRENT SENTIMENTS:")
        for item in results["after"]["sentiment_details"]:
            print(f"   {item['sentiment']}: {item['count']} records")

        print("\n" + "=" * 60)


def main():
    """Main function to run database sanitization."""
    print("🧹 Email Analysis Database Sanitization")
    print("=" * 50)

    # Initialize sanitizer
    sanitizer = DatabaseSanitizer()

    # Generate initial report
    print("\n🔍 Current data quality report:")
    initial_report = sanitizer.get_data_quality_report()
    print(f"   Total records: {initial_report['total_records']}")
    print(f"   Invalid classifications: {initial_report['invalid_classifications']}")
    print(f"   Invalid sentiments: {initial_report['invalid_sentiments']}")

    if (
        initial_report["invalid_classifications"] == 0
        and initial_report["invalid_sentiments"] == 0
    ):
        print("\n✅ Database is already clean! No sanitization needed.")
        return

    # Ask for confirmation
    print(
        f"\n⚠️  Found {initial_report['invalid_classifications']} invalid classifications and {initial_report['invalid_sentiments']} invalid sentiments."
    )
    response = (
        input("Do you want to proceed with sanitization? (y/N): ").strip().lower()
    )

    if response not in ["y", "yes"]:
        print("Sanitization cancelled.")
        return

    # Perform sanitization
    print("\n🧹 Starting sanitization...")
    results = sanitizer.sanitize_all()

    # Print detailed report
    sanitizer.print_sanitization_report(results)

    print("\n✅ Sanitization completed successfully!")
    print("The database now contains only valid classification and sentiment values.")


if __name__ == "__main__":
    main()

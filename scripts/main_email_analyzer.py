# flake8: noqa: E501
"""
Main orchestration script for email sentiment analysis and classification.
Processes emails from CSV data and stores results in the database.
"""

import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import pandas as pd

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from core.combined_analyzer import CombinedAnalyzer
from core.email_processor import EmailProcessor
from data.database import Database


class EmailAnalysisOrchestrator:
    """Orchestrates the email analysis pipeline."""

    def __init__(
        self, csv_path: str = "data/output/synthetic_emails_jan2025_jun2025.csv"
    ):
        """
        Initialize the orchestrator.

        Args:
            csv_path: Path to the CSV file containing email data
        """
        self.csv_path = csv_path
        self.analyzer = CombinedAnalyzer(model_name="deepseek-r1:1.5b")
        self.email_processor = EmailProcessor()
        self.database = Database()

    def load_email_data(self) -> pd.DataFrame:
        """
        Load email data from CSV file.

        Returns:
            DataFrame containing email data
        """
        print(f"Loading email data from: {self.csv_path}")

        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        df = pd.read_csv(self.csv_path)

        # Verify required columns exist
        required_columns = ["date", "body"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        print(f"Loaded {len(df)} emails from CSV")
        return df

    def process_single_email(self, email_date: str, email_body: str) -> Dict[str, Any]:
        """
        Process a single email through the analysis pipeline.

        Args:
            email_date: Date of the email
            email_body: Body text of the email

        Returns:
            Analysis result dictionary
        """
        try:
            # Clean the email text using the email processor
            cleaned_text = self.email_processor._clean_text(email_body)

            # Analyze the email using the combined analyzer
            analysis_result = self.analyzer.analyze_email(cleaned_text)

            # Add the email date and original email text to the result
            analysis_result["email_date"] = email_date
            analysis_result["email_text"] = email_body  # Store the original email text

            return analysis_result

        except Exception as e:
            print(f"Error processing email from {email_date}: {str(e)}")
            # Return a default result for failed analysis
            return {
                "sentiment": "Neutral",
                "classification": "General Follow-ups",
                "confidence": 50,
                "tags": [],
                "email_date": email_date,
                "email_text": email_body,  # Store the original email text even for failed analyses
                "error": str(e),
            }

    def process_emails_batch(self, df: pd.DataFrame, max_emails: int = 600, max_workers: int = 8) -> List[Dict[str, Any]]:
        """
        Process a batch of emails in parallel and store results in database.
        
        Args:
            df: DataFrame containing email data
            max_emails: Maximum number of emails to process
            max_workers: Number of parallel threads
        Returns:
            List of analysis results
        """
        df_subset = df.head(max_emails)
        total_emails = len(df_subset)
        print(f"Processing {total_emails} emails in parallel with {max_workers} workers...")
        results = [None] * total_emails
        futures = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for idx, row in df_subset.iterrows():
                email_date = row['date']
                email_body = row['body']
                futures.append(
                    executor.submit(self.process_single_email, email_date, email_body)
                )
            for i, future in enumerate(tqdm(as_completed(futures), total=total_emails, desc="Processing emails")):
                result = future.result()
                # Store in database
                try:
                    record_id = self.database.save_analysis(
                        email_text=result.get('email_text', ''),
                        analysis_result=result,
                        email_date=result.get('email_date', None)
                    )
                    result['record_id'] = record_id
                except Exception as e:
                    result['error'] = str(e)
                results[i] = result
        print(f"Completed processing {total_emails} emails")
        return results

    def run_analysis(self, max_emails: int = 600) -> Dict[str, Any]:
        """
        Run the complete email analysis pipeline.

        Args:
            max_emails: Maximum number of emails to process

        Returns:
            Summary of the analysis run
        """
        print("=" * 60)
        print("EMAIL ANALYSIS PIPELINE")
        print("=" * 60)

        start_time = time.time()

        try:
            # Load email data
            df = self.load_email_data()

            # Process emails
            results = self.process_emails_batch(df, max_emails)

            # Calculate summary statistics
            end_time = time.time()
            processing_time = end_time - start_time

            # Count successful vs failed analyses
            successful = len([r for r in results if "error" not in r])
            failed = len([r for r in results if "error" in r])

            # Count classifications
            classifications = {}
            sentiments = {}
            for result in results:
                if "error" not in result:
                    classification = result.get("classification", "Unknown")
                    sentiment = result.get("sentiment", "Unknown")

                    classifications[classification] = (
                        classifications.get(classification, 0) + 1
                    )
                    sentiments[sentiment] = sentiments.get(sentiment, 0) + 1

            # Calculate average confidence
            confidences = [r.get("confidence", 0) for r in results if "error" not in r]
            avg_confidence = sum(confidences) / max(len(confidences), 1)

            summary = {
                "total_emails": len(results),
                "successful_analyses": successful,
                "failed_analyses": failed,
                "processing_time_seconds": processing_time,
                "classifications": classifications,
                "sentiments": sentiments,
                "average_confidence": avg_confidence,
            }

            # Print summary
            print("\n" + "=" * 60)
            print("ANALYSIS SUMMARY")
            print("=" * 60)
            print(f"Total emails processed: {summary['total_emails']}")
            print(f"Successful analyses: {summary['successful_analyses']}")
            print(f"Failed analyses: {summary['failed_analyses']}")
            print(f"Processing time: {processing_time:.2f} seconds")
            print(f"Average confidence: {summary['average_confidence']:.1f}%")

            print("\nClassification Distribution:")
            for classification, count in classifications.items():
                percentage = (count / successful) * 100
                print(f"  {classification}: {count} ({percentage:.1f}%)")

            print("\nSentiment Distribution:")
            for sentiment, count in sentiments.items():
                percentage = (count / successful) * 100
                print(f"  {sentiment}: {count} ({percentage:.1f}%)")

            return summary

        except Exception as e:
            print(f"Error in analysis pipeline: {str(e)}")
            raise


def main():
    """Main entry point for the email analysis pipeline."""
    try:
        # Initialize orchestrator
        orchestrator = EmailAnalysisOrchestrator()

        # Run analysis for 600 emails
        summary = orchestrator.run_analysis(max_emails=600)

        print("\nAnalysis completed successfully!")
        print(f"Results stored in database: {orchestrator.database.db_path}")

    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()

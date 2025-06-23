import sys
from pathlib import Path
import csv
import random
from datetime import datetime, timedelta
from data.ollama_client import OllamaClient
import pandas as pd

# Categories and prompt templates
CATEGORIES = [
    "Product/Stocking Requests",
    "Admin/Coordination",
    "Feedback/Complaints",
    "Maintenance/Repairs",
    "Billing/Invoices",
    "General Follow-ups",
    "Operational Logistics",
]

PROMPT_TEMPLATES = {
    "Product/Stocking Requests": (
        """Write a realistic business email requesting information about a product,
        its availability, or a restocking request.
        Include a plausible subject line.
        Example:
        Subject: Inquiry about cold brew coffee
        Body: I hope all is well with you, I have a question about cold brew
        coffee. Do you carry Wander Bear Straight Black Organic Cold Brew
        Coffee? """
    ),
    "Admin/Coordination": (
        """Write a realistic business email about an administrative or
        coordination matter (e.g., scheduling, access, contact info).
        Include a plausible subject line.
        Example:
        Subject: Request for access to the new system
        Body: I'm trying to access the new system, but I'm getting an error.
        Can you help me? """
    ),
    "Feedback/Complaints": (
        """Write a realistic business email providing feedback or a complaint
        about a service or product. Include a plausible subject line.
        Example:
        Subject: Complaint about the service
        Body: I'm very disappointed with the service I received. I've been
        waiting for a response for over a week now. """
    ),
    "Maintenance/Repairs": (
        """Write a realistic business email reporting a maintenance or repair
        issue, or requesting a repair. Include a plausible subject line.
        Example:
        Subject: Update on Pepsi machine
        Body: I hope everyone is doing well. Hate to be a squeaky wheel here
        but is there an update on the Pepsi machine? It appears the machine is
        still out of order and is not cooling. """
    ),
    "Billing/Invoices": (
        """Write a realistic business email about a billing or invoice issue,
        question, or request. Include a plausible subject line.
        Example:
        Subject: Minimum charge on bill
        Body: I received another bill for a minimum charge. As the contract
        says below, we aren't held accountable for paying a minimum monthly
        sales amount. """
    ),
    "General Follow-ups": (
        """Write a realistic business email that is a follow-up to a previous
        conversation or request. Include a plausible subject line.
        Example:
        Subject: Follow-up on the project
        Body: I'm following up on the project. We need to get this done by
        the end of the month. """
    ),
    "Operational Logistics": (
        """Write a realistic business email about operational logistics
        (e.g., delivery, pickup, installation, event support).
        Include a plausible subject line.
        Example:
        Subject: Kiosk network installation update
        Body: We are working on adding a hard wire connection to the kiosk. We
        already ran the wires and just need our IT department to reconfigure
        the network requirements. We will have this work completed before July
        22nd and will let you know once it's completed. """
    ),
}

# Output file
OUTPUT_DIR = Path("data/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "synthetic_emails_jan2025_jun2025.csv"

# Date range
START_DATE = datetime(2025, 1, 1)
END_DATE = datetime(2025, 6, 30)

# Number of emails per month
EMAILS_PER_MONTH = 100

# Ollama model
OLLAMA_MODEL = "llama3.2:1b"

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))


def random_date_in_month(year, month):
    start = datetime(year, month, 1)
    if month == 12:
        end = datetime(year + 1, 1, 1) - timedelta(days=1)
    else:
        end = datetime(year, month + 1, 1) - timedelta(days=1)
    delta = end - start
    random_day = random.randint(0, delta.days)
    return (start + timedelta(days=random_day)).strftime("%Y-%m-%d")


def main():
    client = OllamaClient()
    all_rows = []
    for year in [2025]:
        for month in range(1, 7):  # Jan to June
            for _ in range(EMAILS_PER_MONTH):
                category = random.choice(CATEGORIES)
                prompt = PROMPT_TEMPLATES[category]
                # Add some randomization to the prompt
                prompt_with_variation = prompt + (
                    "\nMake the scenario unique and plausible."
                )
                try:
                    response = client.generate(
                        model=OLLAMA_MODEL,
                        prompt=prompt_with_variation,
                        temperature=0.8,
                        max_tokens=400,
                    )
                    text = response.get("response", "")
                    # Try to split subject and body
                    if "Subject:" in text:
                        parts = text.split("Subject:", 1)[1].split("\n", 1)
                        subject = parts[0].strip()
                        body = parts[1].strip() if len(parts) > 1 else ""
                    else:
                        # Fallback: first line as subject
                        lines = text.strip().split("\n", 1)
                        subject = lines[0][:80]
                        body = lines[1] if len(lines) > 1 else ""
                    date = random_date_in_month(year, month)
                    all_rows.append(
                        {
                            "date": date,
                            "category": category,
                            "subject": subject,
                            "body": body,
                        }
                    )
                except Exception as e:
                    print(f"Error generating email: {e}")

    # Write to CSV
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["date", "category", "subject", "body"])
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)
    print(f"Generated {len(all_rows)} synthetic emails in {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

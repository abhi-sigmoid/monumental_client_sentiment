# Monumental Client Sentiment Analysis

A comprehensive email sentiment analysis and classification system using Ollama models for processing client emails.

## Overview

This system processes email data to:
- Analyze sentiment (Positive, Neutral, Negative)
- Classify emails into predefined categories
- Store results in a SQLite database
- Provide detailed analytics and reporting

## Features

- **Email Processing**: Clean and prepare email text for analysis
- **Sentiment Analysis**: Determine emotional tone of emails
- **Classification**: Categorize emails into 7 predefined categories
- **Database Storage**: Persistent storage of analysis results
- **Batch Processing**: Process large volumes of emails efficiently
- **Progress Tracking**: Real-time progress monitoring
- **Error Handling**: Robust error handling and recovery

## Email Classification Categories

1. **Product/Stocking Requests**: Inquiries about product availability, requests for new products
2. **Admin/Coordination**: Internal/external communication for account setup, scheduling
3. **Feedback/Complaints**: Opinions about service, products, or experiences
4. **Maintenance/Repairs**: Reporting malfunctioning equipment, repair requests
5. **Billing/Invoices**: Questions about charges, invoice requests, payment disputes
6. **General Follow-ups**: Checking in on previous requests or communications
7. **Operational Logistics**: Coordination for deliveries, pickups, installations

## Prerequisites

- Python 3.8+
- Ollama installed and running locally
- deepseek model installed in Ollama

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd monumental_client_sentiment
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install and start Ollama**:
   - Download from [https://ollama.ai](https://ollama.ai)
   - Start Ollama service
   - Install the deepseek model:
     ```bash
     ollama pull deepseek
     ```

## Setup Verification

Run the connection test to verify everything is working:

```bash
python scripts/test_ollama_connection.py
```

This will:
- Test connection to Ollama
- Verify deepseek model availability
- Test basic model generation

## Data Preparation

The system expects email data in CSV format with the following columns:
- `date`: Email date (YYYY-MM-DD format)
- `body`: Email body text
- `category`: Original category (optional, for comparison)
- `subject`: Email subject (optional)

Example CSV structure:
```csv
date,category,subject,body
2025-01-12,Admin/Coordination,Request for Support,"Dear Team, I need help with..."
```

## Usage

### Running the Email Analysis Pipeline

1. **Ensure your data is in the correct location**:
   - Place your CSV file in `data/output/`
   - Default filename: `synthetic_emails_jan2025_jun2025.csv`

2. **Run the main analysis script**:
   ```bash
   python main_email_analyzer.py
   ```

This will:
- Load email data from the CSV file
- Process up to 600 emails (configurable)
- Analyze sentiment and classification for each email
- Store results in the database
- Display progress and summary statistics

### Configuration Options

You can modify the following parameters in `main_email_analyzer.py`:

- `max_emails`: Number of emails to process (default: 600)
- `csv_path`: Path to your CSV file
- `model_name`: Ollama model to use (default: "deepseek")

### Database Access

Results are stored in `data/email_analysis.db`. You can query the database directly:

```python
from src.database import Database

db = Database()
results = db.get_all_analyses()
for result in results:
    print(f"Date: {result['email_date']}")
    print(f"Sentiment: {result['sentiment']}")
    print(f"Classification: {result['classification']}")
    print(f"Confidence: {result['confidence']}%")
    print("---")
```

## Project Structure

```
monumental_client_sentiment/
├── data/
│   └── output/
│       └── synthetic_emails_jan2025_jun2025.csv
├── models/
├── notebooks/
│   └── datasanity_check.ipynb
├── scripts/
│   ├── check_ollama.py
│   ├── generate_synthetic_emails.py
│   └── test_ollama_connection.py
├── src/
│   ├── __init__.py
│   ├── combined_analyzer.py      # Main analysis orchestrator
│   ├── database.py              # Database operations
│   ├── email_processor.py       # Email text preprocessing
│   ├── ollama_client.py         # Ollama API client
│   └── utils/
│       ├── __init__.py
│       └── text_preprocessing.py
├── main_email_analyzer.py       # Main execution script
├── requirements.txt
└── README.md
```

## Key Components

### CombinedAnalyzer (`src/combined_analyzer.py`)
- Orchestrates sentiment analysis and classification
- Uses retry mechanism for low-confidence results
- Configurable confidence thresholds

### EmailProcessor (`src/email_processor.py`)
- Cleans and preprocesses email text
- Handles HTML content extraction
- Removes signatures, URLs, and formatting

### Database (`src/database.py`)
- SQLite database operations
- Stores analysis results with metadata
- Provides query and retrieval methods

### OllamaClient (`src/ollama_client.py`)
- Interfaces with Ollama API
- Handles model generation requests
- Error handling and response parsing

## Performance Considerations

- **Batch Processing**: The system processes emails in batches with progress tracking
- **API Rate Limiting**: Includes delays between requests to avoid overwhelming Ollama
- **Error Recovery**: Failed analyses are logged and the pipeline continues
- **Memory Management**: Processes emails one at a time to manage memory usage

## Troubleshooting

### Common Issues

1. **Ollama Connection Failed**:
   - Ensure Ollama is running: `ollama serve`
   - Check if the service is accessible at `http://localhost:11434`

2. **Model Not Found**:
   - Install the deepseek model: `ollama pull deepseek`
   - Verify with: `ollama list`

3. **CSV File Not Found**:
   - Ensure your CSV file is in `data/output/`
   - Check the filename matches the expected pattern

4. **Database Errors**:
   - Ensure the `data/` directory exists
   - Check file permissions for database creation

### Debug Mode

For detailed debugging, you can modify the logging in the main script:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Output and Results

The analysis produces:
- **Database Records**: Each email analysis stored with metadata
- **Console Output**: Real-time progress and summary statistics
- **Classification Distribution**: Breakdown of email categories
- **Sentiment Distribution**: Breakdown of emotional tones
- **Confidence Metrics**: Average confidence scores

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Add your license information here]

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the logs for error messages
3. Ensure all prerequisites are met
4. Create an issue with detailed error information
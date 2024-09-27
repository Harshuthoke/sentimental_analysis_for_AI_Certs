# Sentiment Analysis API with Groq LLM Integration

## Objective:
This project develops a Python-based API that processes customer reviews from XLSX or CSV files, performs sentiment analysis using the Groq Large Language Model (LLM), and returns a structured JSON response with sentiment scores (positive, negative, neutral).

## Features:
- Accepts customer reviews in `.xlsx` or `.csv` file format.
- Uses the Groq API to classify sentiments as positive, negative, or neutral.
- Returns a structured JSON response with the sentiment analysis results.
- Handles error cases like invalid file format and API errors.

## Requirements:
- Python 3.x
- Flask
- pandas
- openpyxl (for XLSX support)
- Groq API integration (`langchain_groq`)
- Gradio (optional, if building a UI)

## Installation:
1. Clone this repository:
   ```bash
   git clone <repo-url>
   cd <project-directory>
Install the required dependencies:

bash

pip install -r requirements.txt
Set up your Groq API key by exporting it as an environment variable:

bash

export GROQ_API_KEY='your-groq-api-key'
Usage:
Run the API server:

bash

python app.py
Submit customer reviews for analysis: You can use curl to send a file for analysis:

bash

curl -X POST -F "file=@path_to_file/customer_reviews.xlsx" http://127.0.0.1:5000/analyze
Response format: The API will return a JSON response like:

json

{
    "positive": 28,
    "negative": 23,
    "neutral": 0
}
API Endpoint:
/analyze (POST)
Accepts .xlsx or .csv files containing customer reviews.
Returns a JSON object with the sentiment analysis results.
Example:
Submit a .xlsx file containing 50 customer reviews:

bash

curl -X POST -F "file=@customer_reviews.xlsx" http://127.0.0.1:5000/analyze
Limitations:
Sentiment classification accuracy depends on the Groq LLM.
Does not handle multilingual input.
Large files may take longer to process.
Improvements:
Add support for multilingual sentiment analysis.
Optimize file processing for larger datasets.
Provide more granular sentiment scores.
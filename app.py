from flask import Flask, request, jsonify
import pandas as pd
from werkzeug.utils import secure_filename
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

# Initialize Flask app
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "./uploads"

# Load Groq API
groq_api_key = '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'  # Replace with your actual API key

# Define a function to perform sentiment analysis
def analyze_sentiment(review_text):
    chat = ChatGroq(
        api_key=groq_api_key,
        model_name="mixtral-8x7b-32768"
    )
    
    system = "You are a helpful assistant that performs sentiment analysis."
    human = "Analyze the sentiment of this text: {text}"
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", human)
    ])
    
    chain = prompt | chat | StrOutputParser()
    output = chain.invoke({"text": review_text})
    return output

# Route to process the uploaded file and return sentiment analysis
@app.route('/analyze', methods=['POST'])
def analyze_reviews():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    # Ensure the file is secure and save it
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    
    # Load the file as a DataFrame
    try:
        if filename.endswith('.csv'):
            reviews_df = pd.read_csv(file_path)
        elif filename.endswith('.xlsx'):
            reviews_df = pd.read_excel(file_path)
        else:
            return jsonify({"error": "Unsupported file format"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
    # Ensure the file contains the necessary 'Review' column
    if 'Review' not in reviews_df.columns:
        return jsonify({"error": "CSV/XLSX file must contain a 'Review' column"}), 400
    
    # Perform sentiment analysis on each review
    results = {
        "positive": 0,
        "negative": 0,
        "neutral": 0
    }
    
    for review in reviews_df['Review']:
        sentiment_result = analyze_sentiment(review)
        if "positive" in sentiment_result.lower():
            results['positive'] += 1
        elif "negative" in sentiment_result.lower():
            results['negative'] += 1
        else:
            results['neutral'] += 1
    
    # Return the aggregated result in JSON format
    return jsonify(results)

# Route for basic health check
@app.route('/', methods=['GET'])
def home():
    return "Sentiment Analysis API is running"

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)

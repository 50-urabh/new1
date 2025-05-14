from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from bs4 import BeautifulSoup
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import os
import json

app = Flask(__name__)
CORS(app)

# Initialize or load the fake news detection model
MODEL_PATH = "fake_news_model.pkl"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"

if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
else:
    # Simple fallback model (in a real app, you'd want to train properly)
    vectorizer = TfidfVectorizer(max_features=1000)
    # Dummy training data
    texts = [
        "This is clearly fake news designed to mislead people",
        "The government announced new policies today",
        "Celebrity spotted with alien in downtown LA",
        "Scientists confirm climate change findings",
        "The moon landing was faked by Hollywood",
        "New study shows benefits of exercise"
    ]
    labels = [1, 0, 1, 0, 1, 0]  # 1 = fake, 0 = real
    X = vectorizer.fit_transform(texts)
    model = LogisticRegression()
    model.fit(X, labels)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

# Mock fact-checking database
FACT_CHECKS = {
    "moon landing": [
        {
            "claim": "The moon landing was faked",
            "rating": "False",
            "url": "https://example.com/moon-landing-fact-check"
        }
    ],
    "climate change": [
        {
            "claim": "Climate change is a hoax",
            "rating": "False",
            "url": "https://example.com/climate-change-fact-check"
        }
    ]
}

def extract_text_from_url(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer"]):
            script.decompose()
            
        # Get text from paragraphs
        paragraphs = soup.find_all('p')
        text = ' '.join([p.get_text() for p in paragraphs])
        return text
    except Exception as e:
        print(f"Error extracting text from URL: {e}")
        return None

def analyze_text(text):
    # Vectorize the text
    X = vectorizer.transform([text])
    
    # Get prediction probabilities
    proba = model.predict_proba(X)[0]
    fake_prob = proba[1]  # Probability it's fake
    score = int((1 - fake_prob) * 100)  # Convert to reliability score
    
    # Generate details based on analysis
    details = []
    
    # Check for sensational words
    sensational_words = ['shocking', 'unbelievable', 'amazing', 'secret', 'exposed']
    found_words = [word for word in sensational_words if word in text.lower()]
    if found_words:
        details.append({
            "type": "warning",
            "message": f"Sensational language detected: {', '.join(found_words)}"
        })
    
    # Check for all caps
    if re.search(r'\b[A-Z]{3,}\b', text):
        details.append({
            "type": "warning",
            "message": "Excessive capitalization detected (often used in misleading content)"
        })
    
    # Check for exclamation points
    if text.count('!') > 3:
        details.append({
            "type": "warning",
            "message": "Excessive exclamation points (may indicate sensationalism)"
        })
    
    # Add positive indicators for longer text
    if len(text.split()) > 200:
        details.append({
            "type": "positive",
            "message": "Detailed content (often a sign of more reliable information)"
        })
    
    # Check against fact-check database (simple keyword matching)
    fact_checks = []
    for keyword in FACT_CHECKS:
        if keyword.lower() in text.lower():
            fact_checks.extend(FACT_CHECKS[keyword])
    
    return {
        "score": score,
        "details": details,
        "fact_checks": fact_checks
    }

@app.route('/api/analyze-text', methods=['POST'])
def analyze_text_endpoint():
    data = request.json
    text = data.get('text', '')
    
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    results = analyze_text(text)
    return jsonify(results)

@app.route('/api/analyze-url', methods=['POST'])
def analyze_url_endpoint():
    data = request.json
    url = data.get('url', '')
    
    if not url:
        return jsonify({"error": "No URL provided"}), 400
    
    text = extract_text_from_url(url)
    if not text:
        return jsonify({"error": "Could not extract text from URL"}), 400
    
    results = analyze_text(text)
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
# flask_app.py
from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import joblib
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables for model and vectorizer
model = None
vectorizer = None

def load_model():
    """Load the trained model and vectorizer"""
    global model, vectorizer
    
    try:
        # Load the tuned Random Forest model
        model = joblib.load('Dectection_System/models/original_random_forest_model.pkl')
        logger.info("✅ Tuned Random Forest model loaded successfully")
        
        # Extract vectorizer from the pipeline
        vectorizer = model.named_steps['tfidf']
        logger.info("✅ TF-IDF vectorizer loaded successfully")
        
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        raise

def clean_text(text):
    """Clean and preprocess input text"""
    if pd.isna(text):
        return ""
    
    # Convert to string and lowercase
    text = str(text).lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove social media elements
    text = re.sub(r'@\w+|#\w+', '', text)
    
    # Remove special characters and numbers but keep basic punctuation
    text = re.sub(r'[^\w\s\.\,\!\?]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def analyze_text_features(text):
    """Analyze various text features for additional insights"""
    features = {}
    
    # Basic statistics
    features['text_length'] = len(text)
    features['word_count'] = len(text.split())
    features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text.split() else 0
    
    # Linguistic features
    features['has_exclamation'] = 1 if '!' in text else 0
    features['has_question'] = 1 if '?' in text else 0
    features['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / max(1, len(text))
    features['sentence_count'] = text.count('.') + text.count('!') + text.count('?')
    
    return features

@app.route('/')
def home():
    """Home page with API documentation"""
    return render_template('index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint for fake news prediction"""
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        # Extract text from request
        text = data.get('text', '')
        title = data.get('title', '')
        
        if not text and not title:
            return jsonify({'error': 'Either text or title must be provided'}), 400
        
        # Combine title and text
        combined_text = f"{title} {text}".strip()
        
        if len(combined_text) < 10:
            return jsonify({'error': 'Text is too short for analysis'}), 400
        
        # Clean the text
        cleaned_text = clean_text(combined_text)
        
        # Make prediction
        prediction = model.predict([cleaned_text])
        prediction_proba = model.predict_proba([cleaned_text])
        
        # Analyze text features
        text_features = analyze_text_features(cleaned_text)
        
        # Prepare response
        response = {
            'prediction': 'fake' if prediction[0] == 0 else 'true',
            'confidence': {
                'fake': float(prediction_proba[0][0]),
                'true': float(prediction_proba[0][1])
            },
            'text_analysis': text_features,
            'timestamp': datetime.now().isoformat(),
            'text_preview': cleaned_text[:100] + '...' if len(cleaned_text) > 100 else cleaned_text
        }
        
        logger.info(f"Prediction made: {response['prediction']} with confidence {max(response['confidence'].values()):.3f}")
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500

@app.route('/api/batch_predict', methods=['POST'])
def batch_predict():
    """API endpoint for batch predictions"""
    try:
        data = request.get_json()
        
        if not data or 'articles' not in data:
            return jsonify({'error': 'No articles array provided'}), 400
        
        articles = data['articles']
        
        if not isinstance(articles, list):
            return jsonify({'error': 'Articles must be an array'}), 400
        
        results = []
        
        for i, article in enumerate(articles):
            try:
                text = article.get('text', '')
                title = article.get('title', '')
                article_id = article.get('id', i)
                
                combined_text = f"{title} {text}".strip()
                cleaned_text = clean_text(combined_text)
                
                if len(cleaned_text) < 10:
                    results.append({
                        'id': article_id,
                        'error': 'Text too short',
                        'prediction': None,
                        'confidence': None
                    })
                    continue
                
                prediction = model.predict([cleaned_text])
                prediction_proba = model.predict_proba([cleaned_text])
                
                results.append({
                    'id': article_id,
                    'prediction': 'fake' if prediction[0] == 0 else 'true',
                    'confidence': {
                        'fake': float(prediction_proba[0][0]),
                        'true': float(prediction_proba[0][1])
                    },
                    'text_preview': cleaned_text[:50] + '...' if len(cleaned_text) > 50 else cleaned_text
                })
                
            except Exception as e:
                results.append({
                    'id': article.get('id', i),
                    'error': str(e),
                    'prediction': None,
                    'confidence': None
                })
        
        return jsonify({
            'results': results,
            'total_processed': len(results),
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500

@app.route('/api/model_info', methods=['GET'])
def model_info():
    """Get information about the loaded model"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    model_info = {
        'model_type': type(model.named_steps['rf']).__name__,
        'model_name': 'Random Forest Classifier',
        'feature_engineer': 'TF-IDF Vectorizer',
        'max_features': vectorizer.max_features,
        'ngram_range': vectorizer.ngram_range,
        'loaded_at': app.config.get('model_loaded_at', 'Unknown')
    }
    
    return jsonify(model_info)

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({'error': 'Method not allowed'}), 405

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Load model when starting the app
    load_model()
    app.config['model_loaded_at'] = datetime.now().isoformat()
    
    # Run the Flask app
    print("🚀 Fake News Detection API Starting...")
    print("✅ Model loaded successfully")
    print("📊 Available endpoints:")
    print("   GET  /              - API Documentation")
    print("   GET  /api/health    - Health check")
    print("   POST /api/predict   - Single prediction")
    print("   POST /api/batch_predict - Batch predictions")
    print("   GET  /api/model_info - Model information")
    print("\n🌐 Server running on http://localhost:5000")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
# app.py
from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import joblib
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import logging
from datetime import datetime
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables for model and vectorizer
model = None
vectorizer = None

def create_fallback_model():
    """Create a simple fallback model for testing"""
    global model, vectorizer
    logger.info("🔄 Creating fallback model...")
    
    try:
        # Create a simple pipeline
        vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        rf = RandomForestClassifier(n_estimators=10, random_state=42)
        model = Pipeline([
            ('tfidf', vectorizer),
            ('rf', rf)
        ])
        
        # Train on dummy data
        dummy_texts = [
            "this is real news about politics and government",
            "fake news spreading misinformation false claim",
            "official statement from government authorities",
            "false claim debunked by experts and fact checkers",
            "breaking news real event confirmed by sources",
            "viral hoax fake information circulating online",
            "verified information from trusted news sources",
            "conspiracy theory baseless claim without evidence",
            "scientific study published in journal research",
            "misleading headline clickbait fake content"
        ]
        dummy_labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1=real, 0=fake
        
        model.fit(dummy_texts, dummy_labels)
        logger.info("✅ Fallback model created successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to create fallback model: {e}")
        return False

def load_model():
    """Load the trained model and vectorizer"""
    global model, vectorizer
    
    try:
        # Try different possible paths
        possible_paths = [
            'models/original_random_forest_model.pkl',
            'model/original_random_forest_model.pkl',
            'original_random_forest_model.pkl',
            './models/original_random_forest_model.pkl',
            './model/original_random_forest_model.pkl',
            'Detection_System/models/original_random_forest_model.pkl',
            'Dectection_System/models/original_random_forest_model.pkl'
        ]
        
        model_loaded = False
        loaded_path = None
        
        for model_path in possible_paths:
            try:
                if os.path.exists(model_path):
                    model = joblib.load(model_path)
                    logger.info(f"✅ Model loaded successfully from: {model_path}")
                    model_loaded = True
                    loaded_path = model_path
                    break
                else:
                    logger.info(f"📁 Path not found: {model_path}")
            except Exception as e:
                logger.warning(f"❌ Failed to load from {model_path}: {e}")
                continue
        
        if not model_loaded:
            logger.error("❌ Could not load model from any path - using fallback")
            if create_fallback_model():
                app.config['model_type'] = 'fallback'
            return
        
        # Extract vectorizer from the pipeline
        if hasattr(model, 'named_steps') and 'tfidf' in model.named_steps:
            vectorizer = model.named_steps['tfidf']
            logger.info("✅ TF-IDF vectorizer loaded successfully")
            app.config['model_type'] = 'production'
        elif hasattr(model, 'named_steps') and 'vectorizer' in model.named_steps:
            vectorizer = model.named_steps['vectorizer']
            logger.info("✅ Vectorizer loaded successfully")
            app.config['model_type'] = 'production'
        else:
            logger.warning("⚠️ Could not extract vectorizer from pipeline")
            # Create a vectorizer for text processing
            vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
            app.config['model_type'] = 'production_modified'
            
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        # Create fallback model
        if create_fallback_model():
            app.config['model_type'] = 'fallback'

def clean_text(text):
    """Clean and preprocess input text"""
    if pd.isna(text) or text is None:
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
    
    try:
        # Basic statistics
        features['text_length'] = len(text)
        features['word_count'] = len(text.split())
        features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text.split() else 0
        
        # Linguistic features
        features['has_exclamation'] = 1 if '!' in text else 0
        features['has_question'] = 1 if '?' in text else 0
        features['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / max(1, len(text))
        features['sentence_count'] = text.count('.') + text.count('!') + text.count('?')
        features['unique_words'] = len(set(text.split()))
        features['unique_ratio'] = features['unique_words'] / max(1, features['word_count'])
        
    except Exception as e:
        logger.error(f"Error analyzing text features: {e}")
        # Set default values
        features = {
            'text_length': len(text),
            'word_count': 0,
            'avg_word_length': 0,
            'has_exclamation': 0,
            'has_question': 0,
            'uppercase_ratio': 0,
            'sentence_count': 0,
            'unique_words': 0,
            'unique_ratio': 0
        }
    
    return features

@app.route('/')
def home():
    """Home page with API documentation"""
    return render_template('index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    model_status = "loaded" if model is not None else "failed"
    vectorizer_status = "loaded" if vectorizer is not None else "failed"
    model_type = app.config.get('model_type', 'unknown')
    
    health_data = {
        'status': 'healthy' if model is not None else 'degraded',
        'model_loaded': model is not None,
        'vectorizer_loaded': vectorizer is not None,
        'model_status': model_status,
        'vectorizer_status': vectorizer_status,
        'model_type': model_type,
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    }
    
    return jsonify(health_data)

@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint for fake news prediction"""
    # Check if model is loaded
    if model is None:
        logger.error("Model not loaded - attempting to reload")
        load_model()
        if model is None:
            return jsonify({
                'error': 'Model not available', 
                'message': 'Service temporarily unavailable. Please try again shortly.',
                'timestamp': datetime.now().isoformat()
            }), 503
    
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
            return jsonify({'error': 'Text is too short for analysis (minimum 10 characters)'}), 400
        
        # Clean the text
        cleaned_text = clean_text(combined_text)
        
        if len(cleaned_text) < 5:
            return jsonify({'error': 'Text too short after cleaning'}), 400
        
        # Make prediction
        prediction = model.predict([cleaned_text])
        prediction_proba = model.predict_proba([cleaned_text])
        
        # Analyze text features
        text_features = analyze_text_features(cleaned_text)
        
        # Determine prediction label
        prediction_label = 'fake' if prediction[0] == 0 else 'true'
        confidence_fake = float(prediction_proba[0][0])
        confidence_true = float(prediction_proba[0][1])
        
        # Get highest confidence
        highest_confidence = max(confidence_fake, confidence_true)
        
        # Prepare response
        response = {
            'prediction': prediction_label,
            'confidence': {
                'fake': confidence_fake,
                'true': confidence_true
            },
            'confidence_score': highest_confidence,
            'text_analysis': text_features,
            'timestamp': datetime.now().isoformat(),
            'text_preview': cleaned_text[:100] + '...' if len(cleaned_text) > 100 else cleaned_text,
            'model_type': app.config.get('model_type', 'unknown')
        }
        
        logger.info(f"Prediction made: {response['prediction']} with confidence {highest_confidence:.3f}")
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({
            'error': 'Internal server error', 
            'details': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/batch_predict', methods=['POST'])
def batch_predict():
    """API endpoint for batch predictions"""
    # Check if model is loaded
    if model is None:
        logger.error("Model not loaded for batch prediction")
        return jsonify({
            'error': 'Model not available',
            'timestamp': datetime.now().isoformat()
        }), 503
    
    try:
        data = request.get_json()
        
        if not data or 'articles' not in data:
            return jsonify({'error': 'No articles array provided'}), 400
        
        articles = data['articles']
        
        if not isinstance(articles, list):
            return jsonify({'error': 'Articles must be an array'}), 400
        
        if len(articles) > 100:
            return jsonify({'error': 'Too many articles (maximum 100 per request)'}), 400
        
        results = []
        successful_predictions = 0
        
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
                        'confidence': None,
                        'text_preview': cleaned_text[:50] + '...' if len(cleaned_text) > 50 else cleaned_text
                    })
                    continue
                
                prediction = model.predict([cleaned_text])
                prediction_proba = model.predict_proba([cleaned_text])
                
                prediction_label = 'fake' if prediction[0] == 0 else 'true'
                confidence_fake = float(prediction_proba[0][0])
                confidence_true = float(prediction_proba[0][1])
                
                results.append({
                    'id': article_id,
                    'prediction': prediction_label,
                    'confidence': {
                        'fake': confidence_fake,
                        'true': confidence_true
                    },
                    'confidence_score': max(confidence_fake, confidence_true),
                    'text_preview': cleaned_text[:50] + '...' if len(cleaned_text) > 50 else cleaned_text
                })
                
                successful_predictions += 1
                
            except Exception as e:
                logger.error(f"Error processing article {i}: {e}")
                results.append({
                    'id': article.get('id', i),
                    'error': str(e),
                    'prediction': None,
                    'confidence': None,
                    'text_preview': None
                })
        
        return jsonify({
            'results': results,
            'total_processed': len(results),
            'successful_predictions': successful_predictions,
            'failed_predictions': len(results) - successful_predictions,
            'timestamp': datetime.now().isoformat(),
            'model_type': app.config.get('model_type', 'unknown')
        })
    
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({
            'error': 'Internal server error', 
            'details': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/model_info', methods=['GET'])
def model_info():
    """Get information about the loaded model"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        model_type = app.config.get('model_type', 'unknown')
        
        if model_type == 'fallback':
            model_info = {
                'model_type': 'Fallback Random Forest',
                'model_name': 'Emergency Fallback Model',
                'feature_engineer': 'TF-IDF Vectorizer',
                'max_features': 1000,
                'ngram_range': '(1, 2)',
                'status': 'fallback_mode',
                'loaded_at': app.config.get('model_loaded_at', 'Unknown'),
                'note': 'Using fallback model - main model failed to load'
            }
        else:
            model_info = {
                'model_type': type(model.named_steps.get('rf', model.named_steps.get('classifier', model))).__name__,
                'model_name': 'Random Forest Classifier',
                'feature_engineer': 'TF-IDF Vectorizer',
                'status': 'production_mode',
                'loaded_at': app.config.get('model_loaded_at', 'Unknown')
            }
            
            # Try to get vectorizer info
            if hasattr(vectorizer, 'max_features'):
                model_info['max_features'] = vectorizer.max_features
            if hasattr(vectorizer, 'ngram_range'):
                model_info['ngram_range'] = str(vectorizer.ngram_range)
        
        return jsonify(model_info)
    
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        return jsonify({
            'model_type': 'Unknown',
            'status': 'error',
            'error': str(e)
        })

@app.route('/api/reload_model', methods=['POST'])
def reload_model():
    """Reload the model (admin endpoint)"""
    try:
        logger.info("Reloading model...")
        load_model()
        
        if model is not None:
            return jsonify({
                'message': 'Model reloaded successfully',
                'model_type': app.config.get('model_type', 'unknown'),
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'error': 'Failed to reload model',
                'timestamp': datetime.now().isoformat()
            }), 500
            
    except Exception as e:
        logger.error(f"Error reloading model: {e}")
        return jsonify({
            'error': f'Failed to reload model: {str(e)}',
            'timestamp': datetime.now().isoformat()
        }), 500

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

@app.errorhandler(503)
def service_unavailable(error):
    return jsonify({'error': 'Service temporarily unavailable'}), 503

if __name__ == '__main__':
    # Load model when starting the app
    print("🚀 Fake News Detection API Starting...")
    print("📦 Loading model...")
    
    load_model()
    
    if model is not None:
        app.config['model_loaded_at'] = datetime.now().isoformat()
        model_type = app.config.get('model_type', 'unknown')
        print(f"✅ Model loaded successfully ({model_type} mode)")
    else:
        print("❌ Model loading failed completely")
        app.config['model_loaded_at'] = 'Failed'
    
    # Print available endpoints
    print("📊 Available endpoints:")
    print("   GET  /                  - API Documentation")
    print("   GET  /api/health        - Health check")
    print("   POST /api/predict       - Single prediction")
    print("   POST /api/batch_predict - Batch predictions")
    print("   GET  /api/model_info    - Model information")
    print("   POST /api/reload_model  - Reload model")
    print("\n🌐 Server running on http://0.0.0.0:5000")
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)
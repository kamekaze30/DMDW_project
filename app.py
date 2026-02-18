"""
MedPredict AI - Diabetes Risk Assessment Application
Flask Web Application Version

This application provides a web interface for diabetes risk prediction
using a trained Logistic Regression model.
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime

app = Flask(__name__)

# Load model and scaler
def load_model():
    """Load the trained model and scaler from pickle files."""
    try:
        with open('diabetes_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except FileNotFoundError:
        return None, None

# Initialize model
model, scaler = load_model()

def categorize_glucose(glucose):
    """Categorize glucose levels based on medical standards."""
    if glucose < 100:
        return "Healthy", "normal", "#22c55e"
    elif 100 <= glucose < 126:
        return "Pre-diabetic", "warning", "#eab308"
    else:
        return "Diabetic", "critical", "#ef4444"

def get_recommendations(category):
    """Get recommendations based on risk category."""
    recommendations = {
        "normal": {
            "title": "✅ Healthy - Maintain Your Lifestyle",
            "color": "#22c55e",
            "steps": [
                "Continue with annual health check-ups",
                "Maintain balanced diet with vegetables, lean proteins, whole grains",
                "Aim for 150 minutes of moderate exercise weekly",
                "Keep BMI between 18.5-24.9",
                "Stay hydrated - drink 8 glasses of water daily",
                "Get 7-9 hours of quality sleep nightly",
                "Manage stress through meditation or yoga",
                "Limit processed foods and sugar"
            ],
            "message": "Great news! Your glucose levels are in the normal range."
        },
        "warning": {
            "title": "⚡ Pre-Diabetic - Take Action Now",
            "color": "#eab308",
            "steps": [
                "Schedule doctor appointment within 2-4 weeks",
                "Monitor fasting glucose weekly",
                "Cut sugar intake by 50% - eliminate sugary drinks",
                "Reduce refined carbs - switch to whole grains",
                "Exercise 150 minutes/week (30 min x 5 days)",
                "Target 5-7% weight loss if overweight",
                "Increase fiber intake to 25-30g daily",
                "Limit alcohol consumption"
            ],
            "message": "Your glucose is elevated. This is your chance to prevent diabetes!"
        },
        "critical": {
            "title": "⚠️ Diabetic - Immediate Action Required",
            "color": "#ef4444",
            "steps": [
                "Schedule doctor appointment within 1 week - URGENT",
                "Begin daily blood glucose monitoring (2-4 times/day)",
                "Strict diet: 45-60g carbs per meal maximum",
                "Exercise 30-45 minutes daily",
                "Target 5-10% weight reduction if overweight",
                "Watch for symptoms: thirst, frequent urination, fatigue",
                "Prepare for possible medication (Metformin)",
                "Get HbA1c test and full diabetes workup"
            ],
            "message": "Your glucose level indicates diabetes. Medical attention needed."
        }
    }
    return recommendations.get(category, recommendations["normal"])

@app.route('/')
def index():
    """Render the main page with input form."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request."""
    try:
        # Get form data
        data = {
            'Pregnancies': float(request.form['pregnancies']),
            'Glucose': float(request.form['glucose']),
            'BloodPressure': float(request.form['blood_pressure']),
            'SkinThickness': float(request.form['skin_thickness']),
            'Insulin': float(request.form['insulin']),
            'BMI': float(request.form['bmi']),
            'DiabetesPedigreeFunction': float(request.form['diabetes_pedigree']),
            'Age': float(request.form['age'])
        }
        
        if model is None or scaler is None:
            return jsonify({'error': 'Model not loaded. Please train the model first.'}), 500
        
        # Create DataFrame
        input_df = pd.DataFrame([data])
        
        # Scale features
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0]
        
        # Get risk category based on glucose
        glucose_category, category_type, color = categorize_glucose(data['Glucose'])
        
        # Get recommendations
        recommendations = get_recommendations(category_type)
        
        # Calculate risk probability
        diabetes_probability = prediction_proba[1] * 100
        healthy_probability = prediction_proba[0] * 100
        
        result = {
            'prediction': int(prediction),
            'diabetes_probability': round(diabetes_probability, 2),
            'healthy_probability': round(healthy_probability, 2),
            'glucose_category': glucose_category,
            'category_type': category_type,
            'color': color,
            'input_data': data,
            'recommendations': recommendations,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return render_template('result.html', result=result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for programmatic predictions."""
    try:
        data = request.get_json()
        
        required_fields = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                          'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        
        # Validate input
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        if model is None or scaler is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Create DataFrame
        input_df = pd.DataFrame([data])
        
        # Scale and predict
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0]
        
        # Get category
        glucose_category, category_type, color = categorize_glucose(data['Glucose'])
        
        response = {
            'prediction': int(prediction),
            'diabetes_probability': round(prediction_proba[1] * 100, 2),
            'healthy_probability': round(prediction_proba[0] * 100, 2),
            'glucose_category': glucose_category,
            'risk_level': category_type,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/health')
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'timestamp': datetime.now().isoformat()
    }), 200

@app.route('/about')
def about():
    """Render about page."""
    return render_template('about.html')

if __name__ == '__main__':
    print("=" * 60)
    print("MedPredict AI - Diabetes Risk Assessment")
    print("=" * 60)
    
    if model is None or scaler is None:
        print("⚠️  Warning: Model files not found!")
        print("Please run: python train_model.py")
    else:
        print("✓ Model loaded successfully")
        print("✓ Scaler loaded successfully")
    
    print("\nStarting Flask server...")
    print("Open your browser and navigate to: http://localhost:5000")
    print("\nPress CTRL+C to stop the server")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)

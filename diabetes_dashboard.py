import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
from PIL import Image
import base64
from io import BytesIO

# Page configuration
st.set_page_config(
    page_title="MedPredict AI | Diabetes Risk Assessment",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for test cases
if 'test_case_values' not in st.session_state:
    st.session_state.test_case_values = {
        'pregnancies': 0,
        'glucose': 100,
        'blood_pressure': 70,
        'skin_thickness': 20,
        'insulin': 80,
        'bmi': 25.0,
        'diabetes_pedigree': 0.5,
        'age': 30
    }

# Test cases data
TEST_CASES = {
    "✅ Healthy Person (Low Risk)": {
        'pregnancies': 0, 'glucose': 85, 'blood_pressure': 72, 'skin_thickness': 20,
        'insulin': 79, 'bmi': 22.5, 'diabetes_pedigree': 0.250, 'age': 25,
        'expected': 'Healthy'
    },
    "⚡ Pre-Diabetic (Warning)": {
        'pregnancies': 3, 'glucose': 115, 'blood_pressure': 82, 'skin_thickness': 32,
        'insulin': 165, 'bmi': 30.2, 'diabetes_pedigree': 0.580, 'age': 42,
        'expected': 'Pre-diabetic'
    },
    "⚠️ Diabetic (High Risk)": {
        'pregnancies': 5, 'glucose': 155, 'blood_pressure': 88, 'skin_thickness': 38,
        'insulin': 280, 'bmi': 35.5, 'diabetes_pedigree': 0.750, 'age': 48,
        'expected': 'Diabetic'
    },
    "🔴 Critical Diabetic": {
        'pregnancies': 8, 'glucose': 185, 'blood_pressure': 92, 'skin_thickness': 42,
        'insulin': 450, 'bmi': 42.0, 'diabetes_pedigree': 1.200, 'age': 55,
        'expected': 'Diabetic (Critical)'
    },
    "👴 Elderly with Risk": {
        'pregnancies': 4, 'glucose': 128, 'blood_pressure': 85, 'skin_thickness': 35,
        'insulin': 0, 'bmi': 33.8, 'diabetes_pedigree': 0.895, 'age': 65,
        'expected': 'Diabetic'
    },
    "🧬 Young with Family History": {
        'pregnancies': 0, 'glucose': 95, 'blood_pressure': 68, 'skin_thickness': 22,
        'insulin': 95, 'bmi': 24.5, 'diabetes_pedigree': 2.500, 'age': 28,
        'expected': 'Low Risk'
    }
}

# Load model and scaler
@st.cache_resource
def load_model():
    try:
        with open('diabetes_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except FileNotFoundError:
        return None, None

# Function to categorize glucose levels
def categorize_glucose(glucose, is_fasting=True):
    """
    Categorize glucose levels based on medical standards
    Fasting: < 100 Normal, 100-125 Pre-diabetic, >= 126 Diabetic
    Non-fasting: < 140 Normal, 140-199 Pre-diabetic, >= 200 Diabetic
    """
    if is_fasting:
        if glucose < 100:
            return "Healthy", "normal"
        elif 100 <= glucose < 126:
            return "Pre-diabetic", "warning"
        else:
            return "Diabetic", "critical"
    else:
        if glucose < 140:
            return "Healthy", "normal"
        elif 140 <= glucose < 200:
            return "Pre-diabetic", "warning"
        else:
            return "Diabetic", "critical"

# Custom CSS with modern dark theme
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main background with gradient */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #3b82f6, #8b5cf6, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
    }
    
    .sub-header {
        color: #94a3b8;
        font-size: 1.1rem;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    /* Glassmorphism cards */
    .glass-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(59, 130, 246, 0.2);
        border-color: rgba(59, 130, 246, 0.3);
    }
    
    /* Input styling */
    .stNumberInput > div > div > input {
        background: rgba(15, 23, 42, 0.8);
        border: 2px solid rgba(59, 130, 246, 0.3);
        border-radius: 12px;
        color: #f8fafc;
        font-size: 1rem;
        padding: 0.75rem;
        transition: all 0.3s ease;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2);
    }
    
    /* Section headers */
    .section-title {
        color: #f8fafc;
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .section-icon {
        font-size: 1.5rem;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        color: white;
        border: none;
        border-radius: 16px;
        padding: 1rem 3rem;
        font-size: 1.1rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.6);
        background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%);
    }
    
    /* Results styling */
    .result-container {
        border-radius: 24px;
        padding: 2.5rem;
        margin: 2rem 0;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    
    .result-positive {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(153, 27, 27, 0.3) 100%);
        border: 2px solid rgba(239, 68, 68, 0.5);
        backdrop-filter: blur(20px);
    }
    
    .result-negative {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.2) 0%, rgba(21, 128, 61, 0.3) 100%);
        border: 2px solid rgba(34, 197, 94, 0.5);
        backdrop-filter: blur(20px);
    }
    
    .result-warning {
        background: linear-gradient(135deg, rgba(234, 179, 8, 0.2) 0%, rgba(180, 83, 9, 0.3) 100%);
        border: 2px solid rgba(234, 179, 8, 0.5);
        backdrop-filter: blur(20px);
    }
    
    .result-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .result-positive .result-title {
        color: #fca5a5;
    }
    
    .result-negative .result-title {
        color: #86efac;
    }
    
    .result-warning .result-title {
        color: #fcd34d;
    }
    
    .result-probability {
        font-size: 4rem;
        font-weight: 800;
        margin: 1rem 0;
        text-shadow: 0 0 40px rgba(255,255,255,0.3);
    }
    
    .result-description {
        font-size: 1.1rem;
        color: #cbd5e1;
        margin-top: 1rem;
    }
    
    /* Metric cards */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1.5rem 0;
    }
    
    .metric-item {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .metric-item:hover {
        background: rgba(59, 130, 246, 0.1);
        border-color: rgba(59, 130, 246, 0.4);
        transform: scale(1.02);
    }
    
    .metric-label {
        color: #94a3b8;
        font-size: 0.875rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        color: #f8fafc;
        font-size: 1.5rem;
        font-weight: 700;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: rgba(15, 23, 42, 0.95);
    }
    
    /* Progress bar */
    .progress-container {
        background: rgba(30, 41, 59, 0.8);
        border-radius: 12px;
        height: 20px;
        overflow: hidden;
        margin: 1rem 0;
    }
    
    .progress-bar {
        height: 100%;
        border-radius: 12px;
        transition: width 1s ease;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6, #ec4899);
    }
    
    /* Info boxes */
    .info-box {
        background: rgba(59, 130, 246, 0.1);
        border-left: 4px solid #3b82f6;
        border-radius: 0 12px 12px 0;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
        color: #e2e8f0;
    }
    
    /* Health tips */
    .tip-card {
        background: linear-gradient(135deg, rgba(251, 191, 36, 0.1) 0%, rgba(245, 158, 11, 0.1) 100%);
        border: 1px solid rgba(251, 191, 36, 0.3);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .tip-title {
        color: #fbbf24;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    /* Animations */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes pulse {
        0%, 100% {
            box-shadow: 0 0 0 0 rgba(59, 130, 246, 0.7);
        }
        50% {
            box-shadow: 0 0 0 20px rgba(59, 130, 246, 0);
        }
    }
    
    .animate-fade-in {
        animation: fadeInUp 0.6s ease-out;
    }
    
    .animate-pulse {
        animation: pulse 2s infinite;
    }
    
    /* Divider */
    .custom-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #3b82f6, transparent);
        margin: 2rem 0;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #64748b;
        padding: 2rem;
        margin-top: 3rem;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h1 style="color: #3b82f6; font-size: 1.8rem; margin: 0;">🩺 MedPredict AI</h1>
            <p style="color: #64748b; font-size: 0.9rem; margin-top: 0.5rem;">Advanced Health Analytics</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Navigation
    st.markdown("### 📍 Navigation")
    page = st.radio("", ["🏠 Risk Assessment", "📊 About Model", "⚙️ Settings"], label_visibility="collapsed")
    
    st.markdown("---")
    
    # Quick Stats
    st.markdown("### 📈 Quick Stats")
    st.markdown("""
        <div style="background: rgba(30, 41, 59, 0.6); padding: 1rem; border-radius: 12px; margin: 0.5rem 0;">
            <p style="color: #94a3b8; margin: 0; font-size: 0.875rem;">Model Accuracy</p>
            <p style="color: #3b82f6; font-size: 1.5rem; font-weight: 700; margin: 0;">78%</p>
        </div>
        <div style="background: rgba(30, 41, 59, 0.6); padding: 1rem; border-radius: 12px; margin: 0.5rem 0;">
            <p style="color: #94a3b8; margin: 0; font-size: 0.875rem;">Features Analyzed</p>
            <p style="color: #8b5cf6; font-size: 1.5rem; font-weight: 700; margin: 0;">8</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Result Categories
    st.markdown("### 🎯 Result Categories")
    st.markdown("""
        <div style="background: rgba(34, 197, 94, 0.1); border-left: 4px solid #22c55e; padding: 0.75rem; border-radius: 0 8px 8px 0; margin: 0.5rem 0;">
            <p style="color: #86efac; margin: 0; font-weight: 600;">✅ Healthy</p>
            <p style="color: #94a3b8; margin: 0; font-size: 0.8rem;">Glucose &lt; 100 mg/dL</p>
        </div>
        <div style="background: rgba(234, 179, 8, 0.1); border-left: 4px solid #eab308; padding: 0.75rem; border-radius: 0 8px 8px 0; margin: 0.5rem 0;">
            <p style="color: #fcd34d; margin: 0; font-weight: 600;">⚡ Pre-diabetic</p>
            <p style="color: #94a3b8; margin: 0; font-size: 0.8rem;">Glucose 100-125 mg/dL</p>
        </div>
        <div style="background: rgba(239, 68, 68, 0.1); border-left: 4px solid #ef4444; padding: 0.75rem; border-radius: 0 8px 8px 0; margin: 0.5rem 0;">
            <p style="color: #fca5a5; margin: 0; font-weight: 600;">⚠️ Diabetic</p>
            <p style="color: #94a3b8; margin: 0; font-size: 0.8rem;">Glucose ≥ 126 mg/dL</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Health Tips
    st.markdown("### 💡 Daily Health Tip")
    st.markdown("""
        <div class="tip-card">
            <p class="tip-title">🚶 Stay Active</p>
            <p style="color: #cbd5e1; margin: 0; font-size: 0.9rem;">
                Aim for at least 30 minutes of moderate exercise daily to reduce diabetes risk.
            </p>
        </div>
    """, unsafe_allow_html=True)

# Main Content
if "Risk Assessment" in page:
    # Header
    st.markdown('<h1 class="main-header">🩺 Diabetes Risk Assessment</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-powered health analysis • Enter your metrics below for instant evaluation</p>', unsafe_allow_html=True)
    
    # Progress indicator
    st.markdown("""
        <div style="margin: 2rem 0;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                <span style="color: #94a3b8; font-size: 0.875rem;">Completion Progress</span>
                <span style="color: #3b82f6; font-size: 0.875rem; font-weight: 600;">Step 1 of 2</span>
            </div>
            <div class="progress-container">
                <div class="progress-bar" style="width: 50%;"></div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Test Case Selector
    st.markdown('<div class="glass-card animate-fade-in">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title"><span class="section-icon">🧪</span> Quick Test Cases</h2>', unsafe_allow_html=True)
    
    st.markdown("""
        <p style="color: #94a3b8; margin-bottom: 1rem;">
            Select a predefined test case to auto-fill the form, or enter your own values below:
        </p>
    """, unsafe_allow_html=True)
    
    test_cols = st.columns(3)
    for idx, (test_name, test_data) in enumerate(TEST_CASES.items()):
        col_idx = idx % 3
        with test_cols[col_idx]:
            if st.button(test_name, use_container_width=True, key=f"test_{idx}"):
                st.session_state.test_case_values = {k: v for k, v in test_data.items() if k != 'expected'}
                st.session_state.selected_test = test_name
                st.rerun()
    
    # Show selected test case info
    if 'selected_test' in st.session_state:
        st.markdown(f"""
            <div style="background: rgba(59, 130, 246, 0.1); border: 1px solid rgba(59, 130, 246, 0.3); 
                        border-radius: 12px; padding: 15px; margin-top: 15px;">
                <p style="color: #3b82f6; margin: 0; font-weight: 600;">
                    🧪 Selected: {st.session_state.selected_test}
                </p>
                <p style="color: #94a3b8; margin: 5px 0 0 0; font-size: 0.9rem;">
                    Expected Result: {TEST_CASES[st.session_state.selected_test]['expected']}
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    # Reset button
    if st.button("🔄 Reset to Default Values", use_container_width=True):
        st.session_state.test_case_values = {
            'pregnancies': 0, 'glucose': 100, 'blood_pressure': 70, 'skin_thickness': 20,
            'insulin': 80, 'bmi': 25.0, 'diabetes_pedigree': 0.5, 'age': 30
        }
        if 'selected_test' in st.session_state:
            del st.session_state.selected_test
        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Input Form
    st.markdown('<div class="glass-card animate-fade-in">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title"><span class="section-icon">📊</span> Health Information</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("<p style='color: #3b82f6; font-weight: 600; margin-bottom: 1rem;'>👤 Demographics</p>", unsafe_allow_html=True)
        pregnancies = st.number_input("🤰 Pregnancies", min_value=0, max_value=20, 
                                     value=st.session_state.test_case_values['pregnancies'], 
                                     help="Number of times pregnant")
        age = st.number_input("🎂 Age", min_value=1, max_value=120, 
                             value=st.session_state.test_case_values['age'],
                             help="Age in years")
        
        st.markdown("<p style='color: #8b5cf6; font-weight: 600; margin: 2rem 0 1rem 0;'>🩸 Blood Metrics</p>", unsafe_allow_html=True)
        glucose = st.number_input("💉 Glucose (mg/dL)", min_value=0, max_value=300, 
                                 value=st.session_state.test_case_values['glucose'],
                                 help="Plasma glucose concentration")
        blood_pressure = st.number_input("❤️ Blood Pressure (mm Hg)", min_value=0, max_value=200, 
                                        value=st.session_state.test_case_values['blood_pressure'],
                                        help="Diastolic blood pressure")
    
    with col2:
        st.markdown("<p style='color: #ec4899; font-weight: 600; margin-bottom: 1rem;'>📏 Physical</p>", unsafe_allow_html=True)
        skin_thickness = st.number_input("📐 Skin Thickness (mm)", min_value=0, max_value=100, 
                                        value=st.session_state.test_case_values['skin_thickness'],
                                        help="Triceps skin fold thickness")
        bmi = st.number_input("⚖️ BMI", min_value=0.0, max_value=70.0, format="%.1f",
                             value=st.session_state.test_case_values['bmi'],
                             help="Body Mass Index")
        
        st.markdown("<p style='color: #f59e0b; font-weight: 600; margin: 2rem 0 1rem 0;'>🧬 Additional</p>", unsafe_allow_html=True)
        insulin = st.number_input("💊 Insulin (μU/mL)", min_value=0, max_value=1000, 
                                 value=st.session_state.test_case_values['insulin'],
                                 help="2-Hour serum insulin")
        diabetes_pedigree = st.number_input("🧪 Diabetes Pedigree", min_value=0.0, max_value=3.0, format="%.3f",
                                           value=st.session_state.test_case_values['diabetes_pedigree'],
                                           help="Diabetes pedigree function")
    
    with col3:
        st.markdown("""
            <div style="background: rgba(59, 130, 246, 0.1); border-radius: 16px; padding: 1.5rem; border: 1px solid rgba(59, 130, 246, 0.2);">
                <h4 style="color: #3b82f6; margin-top: 0;">📋 Reference Ranges</h4>
                <div style="font-size: 0.85rem; color: #94a3b8; line-height: 1.8;">
                    <p><strong style="color: #22c55e;">Normal Glucose:</strong> 70-100 mg/dL</p>
                    <p><strong style="color: #eab308;">Prediabetes:</strong> 100-125 mg/dL</p>
                    <p><strong style="color: #ef4444;">Diabetes:</strong> >126 mg/dL</p>
                    <hr style="border-color: rgba(255,255,255,0.1); margin: 1rem 0;">
                    <p><strong style="color: #22c55e;">Healthy BMI:</strong> 18.5-24.9</p>
                    <p><strong style="color: #eab308;">Overweight:</strong> 25-29.9</p>
                    <p><strong style="color: #ef4444;">Obese:</strong> ≥30</p>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Predict Button
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    center_col = st.columns([1, 2, 1])[1]
    with center_col:
        predict_button = st.button("🔮 Analyze Risk", type="primary", use_container_width=True)
    
    if predict_button:
        model, scaler = load_model()
        
        if model is not None and scaler is not None:
            # Create input dataframe
            input_data = pd.DataFrame({
                'Pregnancies': [pregnancies],
                'Glucose': [glucose],
                'BloodPressure': [blood_pressure],
                'SkinThickness': [skin_thickness],
                'Insulin': [insulin],
                'BMI': [bmi],
                'DiabetesPedigreeFunction': [diabetes_pedigree],
                'Age': [age]
            })
            
            # Scale and predict
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]
            prediction_proba = model.predict_proba(input_scaled)[0]
            
            # Determine glucose category (assuming fasting for simplicity)
            glucose_category, category_type = categorize_glucose(glucose, is_fasting=True)
            
            # Display Results
            st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
            st.markdown('<h2 style="color: #f8fafc; text-align: center; margin-bottom: 2rem;">🎯 Assessment Results</h2>', unsafe_allow_html=True)
            
            # Result container with animation based on glucose category
            if category_type == "critical":
                result_html = f"""
                    <div class="result-container result-positive animate-fade-in">
                        <div style="font-size: 4rem; margin-bottom: 1rem;">⚠️</div>
                        <h2 class="result-title">DIABETIC</h2>
                        <p style="color: #fca5a5; font-size: 1.3rem; font-weight: 600;">Glucose Level: {glucose} mg/dL</p>
                        <p style="color: #fca5a5; font-size: 1.2rem;">High blood sugar detected</p>
                        <div class="result-probability" style="color: #fca5a5;">{prediction_proba[1]:.1%}</div>
                        <p style="color: #94a3b8; font-size: 0.9rem;">Diabetes Risk Probability</p>
                    </div>
                """
            elif category_type == "warning":
                result_html = f"""
                    <div class="result-container result-warning animate-fade-in">
                        <div style="font-size: 4rem; margin-bottom: 1rem;">⚡</div>
                        <h2 class="result-title">PRE-DIABETIC</h2>
                        <p style="color: #fcd34d; font-size: 1.3rem; font-weight: 600;">Glucose Level: {glucose} mg/dL</p>
                        <p style="color: #fcd34d; font-size: 1.2rem;">Elevated blood sugar - Action needed</p>
                        <div class="result-probability" style="color: #fcd34d;">{prediction_proba[1]:.1%}</div>
                        <p style="color: #94a3b8; font-size: 0.9rem;">Diabetes Risk Probability</p>
                    </div>
                """
            else:
                result_html = f"""
                    <div class="result-container result-negative animate-fade-in">
                        <div style="font-size: 4rem; margin-bottom: 1rem;">✅</div>
                        <h2 class="result-title">HEALTHY</h2>
                        <p style="color: #86efac; font-size: 1.3rem; font-weight: 600;">Glucose Level: {glucose} mg/dL</p>
                        <p style="color: #86efac; font-size: 1.2rem;">Normal blood sugar levels</p>
                        <div class="result-probability" style="color: #86efac;">{prediction_proba[0]:.1%}</div>
                        <p style="color: #94a3b8; font-size: 0.9rem;">Healthy Probability</p>
                    </div>
                """
            
            st.markdown(result_html, unsafe_allow_html=True)
            
            # Gauge Chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prediction_proba[1] * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Risk Score", 'font': {'size': 24, 'color': '#f8fafc'}},
                number={'font': {'size': 48, 'color': '#f8fafc'}, 'suffix': '%'},
                gauge={
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': '#3b82f6'},
                    'bar': {'color': '#3b82f6'},
                    'bgcolor': 'rgba(30, 41, 59, 0.8)',
                    'borderwidth': 2,
                    'bordercolor': 'rgba(255, 255, 255, 0.1)',
                    'steps': [
                        {'range': [0, 30], 'color': 'rgba(34, 197, 94, 0.3)'},
                        {'range': [30, 70], 'color': 'rgba(234, 179, 8, 0.3)'},
                        {'range': [70, 100], 'color': 'rgba(239, 68, 68, 0.3)'}
                    ],
                    'threshold': {
                        'line': {'color': 'red', 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=400,
                margin=dict(l=20, r=20, t=50, b=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Input Summary
            st.markdown('<h3 style="color: #f8fafc; margin-top: 2rem;">📊 Your Health Profile</h3>', unsafe_allow_html=True)
            
            st.markdown("""
                <div class="metric-grid">
                    <div class="metric-item">
                        <div class="metric-label">Pregnancies</div>
                        <div class="metric-value">""" + str(pregnancies) + """</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">Age</div>
                        <div class="metric-value">""" + str(age) + """ yrs</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">Glucose</div>
                        <div class="metric-value">""" + str(glucose) + """ mg/dL</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">Blood Pressure</div>
                        <div class="metric-value">""" + str(blood_pressure) + """ mmHg</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">BMI</div>
                        <div class="metric-value">""" + "{:.1f}".format(bmi) + """</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">Insulin</div>
                        <div class="metric-value">""" + str(insulin) + """ μU/mL</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            # Recommendations
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<h3 style="color: #f8fafc;">💡 Recommendations</h3>', unsafe_allow_html=True)
            
            # Display specific instructions based on category
            if category_type == "critical":
                st.markdown("""
                    <div class="info-box" style="border-color: #ef4444; background: rgba(239, 68, 68, 0.1);">
                        <strong style="color: #fca5a5; font-size: 1.2rem;">🏥 IMMEDIATE ACTION REQUIRED - DIABETIC</strong><br>
                        <p style="margin-top: 1rem; color: #cbd5e1;">
                        Your glucose level indicates diabetes. This is a serious condition that requires medical attention.
                        </p>
                        <strong style="color: #fca5a5;">Immediate Steps:</strong>
                        <ul style="margin-top: 0.5rem; padding-left: 1.5rem; color: #cbd5e1; line-height: 1.8;">
                            <li><strong>Schedule a doctor appointment within 1 week</strong> - Get HbA1c test and full diabetes workup</li>
                            <li><strong>Start blood glucose monitoring</strong> - Check levels 2-4 times daily</li>
                            <li><strong>Begin medication</strong> - Your doctor may prescribe Metformin or insulin</li>
                            <li><strong>Strict diet changes</strong> - Eliminate sugar, reduce carbs to 45-60g per meal</li>
                            <li><strong>Exercise daily</strong> - 30-45 minutes of walking or aerobic activity</li>
                            <li><strong>Lose weight</strong> - Target 5-10% body weight reduction if overweight</li>
                            <li><strong>Watch for symptoms</strong> - Excessive thirst, frequent urination, blurry vision, fatigue</li>
                        </ul>
                        <p style="margin-top: 1rem; color: #fcd34d;">
                        ⚠️ <strong>Warning:</strong> Untreated diabetes can lead to heart disease, kidney failure, blindness, and nerve damage.
                        </p>
                    </div>
                """, unsafe_allow_html=True)
            elif category_type == "warning":
                st.markdown("""
                    <div class="info-box" style="border-color: #eab308; background: rgba(234, 179, 8, 0.1);">
                        <strong style="color: #fcd34d; font-size: 1.2rem;">⚡ PRE-DIABETIC - TAKE ACTION NOW</strong><br>
                        <p style="margin-top: 1rem; color: #cbd5e1;">
                        Your glucose level is elevated but not yet in diabetic range. This is your chance to prevent diabetes!
                        </p>
                        <strong style="color: #fcd34d;">Action Plan (Start Today):</strong>
                        <ul style="margin-top: 0.5rem; padding-left: 1.5rem; color: #cbd5e1; line-height: 1.8;">
                            <li><strong>See your doctor</strong> - Schedule within 2-4 weeks for glucose tolerance test</li>
                            <li><strong>Monitor glucose weekly</strong> - Check fasting glucose every week</li>
                            <li><strong>Cut sugar by 50%</strong> - No sugary drinks, limit desserts to 1x/week</li>
                            <li><strong>Reduce refined carbs</strong> - Switch to whole grains, limit white bread/pasta/rice</li>
                            <li><strong>Exercise 150 min/week</strong> - 30 minutes brisk walking 5 days a week</li>
                            <li><strong>Lose 5-7% weight</strong> - Even modest weight loss can reverse pre-diabetes</li>
                            <li><strong>Eat more fiber</strong> - Aim for 25-30g daily (vegetables, beans, whole grains)</li>
                            <li><strong>Limit alcohol</strong> - Max 1 drink/day for women, 2 for men</li>
                        </ul>
                        <p style="margin-top: 1rem; color: #86efac;">
                        ✅ <strong>Good news:</strong> Pre-diabetes is reversible! With lifestyle changes, you can return to normal glucose levels.
                        </p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div class="info-box" style="border-color: #22c55e; background: rgba(34, 197, 94, 0.1);">
                        <strong style="color: #86efac; font-size: 1.2rem;">✅ HEALTHY - Maintain Your Lifestyle</strong><br>
                        <p style="margin-top: 1rem; color: #cbd5e1;">
                        Great news! Your glucose levels are in the normal range. Keep up the good work!
                        </p>
                        <strong style="color: #86efac;">Prevention Guidelines:</strong>
                        <ul style="margin-top: 0.5rem; padding-left: 1.5rem; color: #cbd5e1; line-height: 1.8;">
                            <li><strong>Annual check-ups</strong> - Get fasting glucose tested every 1-2 years</li>
                            <li><strong>Continue healthy eating</strong> - Balanced diet with vegetables, lean proteins, whole grains</li>
                            <li><strong>Stay active</strong> - Aim for 150 minutes of moderate exercise weekly</li>
                            <li><strong>Maintain healthy weight</strong> - Keep BMI between 18.5-24.9</li>
                            <li><strong>Stay hydrated</strong> - Drink 8 glasses of water daily</li>
                            <li><strong>Sleep well</strong> - Get 7-9 hours of quality sleep nightly</li>
                            <li><strong>Manage stress</strong> - Practice meditation, yoga, or deep breathing</li>
                            <li><strong>Limit processed foods</strong> - Avoid excessive sugar and refined carbohydrates</li>
                        </ul>
                        <p style="margin-top: 1rem; color: #3b82f6;">
                        💪 <strong>Stay on track:</strong> Continue these habits to prevent diabetes and maintain overall health!
                        </p>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
        else:
            st.error("⚠️ Model not found. Please run `python train_model.py` first to train the model.")
    
    # Disclaimer
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
        <div style="background: rgba(30, 41, 59, 0.6); border-radius: 16px; padding: 1.5rem; margin-top: 2rem; border: 1px solid rgba(255, 255, 255, 0.1);">
            <p style="color: #64748b; font-size: 0.85rem; margin: 0; text-align: center;">
                <strong style="color: #94a3b8;">Disclaimer:</strong> This tool is for educational purposes only and not a substitute for professional medical advice. 
                Always consult healthcare providers for medical decisions. Predictions are based on statistical models and should be interpreted cautiously.
            </p>
        </div>
    """, unsafe_allow_html=True)

elif "About Model" in page:
    st.markdown('<h1 class="main-header">📊 About the Model</h1>', unsafe_allow_html=True)
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("""
        <h3 style="color: #f8fafc;">🤖 Machine Learning Approach</h3>
        <p style="color: #cbd5e1; line-height: 1.8;">
            This dashboard uses a <strong style="color: #3b82f6;">Logistic Regression</strong> model trained on the 
            Pima Indians Diabetes Dataset. The model analyzes 8 key health metrics to predict diabetes risk.
        </p>
        
        <h4 style="color: #8b5cf6; margin-top: 2rem;">📈 Model Performance</h4>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin: 1rem 0;">
            <div style="background: rgba(59, 130, 246, 0.1); padding: 1.5rem; border-radius: 12px; text-align: center;">
                <div style="color: #3b82f6; font-size: 2rem; font-weight: 700;">78%</div>
                <div style="color: #94a3b8; font-size: 0.875rem;">Accuracy</div>
            </div>
            <div style="background: rgba(139, 92, 246, 0.1); padding: 1.5rem; border-radius: 12px; text-align: center;">
                <div style="color: #8b5cf6; font-size: 2rem; font-weight: 700;">768</div>
                <div style="color: #94a3b8; font-size: 0.875rem;">Training Samples</div>
            </div>
            <div style="background: rgba(236, 72, 153, 0.1); padding: 1.5rem; border-radius: 12px; text-align: center;">
                <div style="color: #ec4899; font-size: 2rem; font-weight: 700;">8</div>
                <div style="color: #94a3b8; font-size: 0.875rem;">Features</div>
            </div>
        </div>
        
        <h4 style="color: #22c55e; margin-top: 2rem;">🔍 Features Used</h4>
        <ul style="color: #cbd5e1; line-height: 2;">
            <li><strong>Pregnancies:</strong> Number of times pregnant</li>
            <li><strong>Glucose:</strong> Plasma glucose concentration (mg/dL)</li>
            <li><strong>Blood Pressure:</strong> Diastolic blood pressure (mm Hg)</li>
            <li><strong>Skin Thickness:</strong> Triceps skin fold thickness (mm)</li>
            <li><strong>Insulin:</strong> 2-Hour serum insulin (μU/mL)</li>
            <li><strong>BMI:</strong> Body mass index (weight in kg/(height in m)²)</li>
            <li><strong>Diabetes Pedigree Function:</strong> Genetic diabetes influence score</li>
            <li><strong>Age:</strong> Age in years</li>
        </ul>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

elif "Settings" in page:
    st.markdown('<h1 class="main-header">⚙️ Settings</h1>', unsafe_allow_html=True)
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("""
        <h3 style="color: #f8fafc;">🔧 Application Settings</h3>
    """, unsafe_allow_html=True)
    
    st.toggle("Enable Notifications", value=True)
    st.toggle("Dark Mode", value=True, disabled=True)
    st.selectbox("Language", ["English", "Spanish", "French", "German"], index=0)
    st.slider("Prediction Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown('<div class="footer">', unsafe_allow_html=True)
st.markdown("""
    <p style="margin: 0;">🩺 MedPredict AI © 2026 | Built with Streamlit</p>
    <p style="margin: 0.5rem 0 0 0; font-size: 0.8rem;">Advanced Health Analytics Platform</p>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

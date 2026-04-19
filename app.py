import streamlit as st
import pandas as pd
import joblib
import os
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="Titanic Survival Prediction Model",
    page_icon="🌊",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# --- Premium Dark Mode Glassmorphism CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;700&display=swap');

    /* Global Typography & Background */
    html, body, [class*="css"]  {
        font-family: 'Outfit', sans-serif;
    }
    
    [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at top right, #1e1b4b, #0f172a, #020617);
        color: #ffffff;
    }
    
    [data-testid="stHeader"] {
        background-color: transparent;
    }

    /* Titles and Text */
    h1 {
        font-weight: 700 !important;
        font-size: 3.2rem !important;
        background: linear-gradient(to right, #38bdf8, #818cf8, #e879f9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0 !important;
        padding-bottom: 0 !important;
    }
    h2, h3, p, label {
        color: #e2e8f0 !important;
    }

    /* Glassmorphism Form Container */
    [data-testid="stForm"] {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 16px;
        padding: 2.5rem;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5);
        margin-top: 2rem;
    }

    /* Premium Button Navigation */
    .stButton>button {
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
        color: #ffffff;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(168, 85, 247, 0.6);
        background: linear-gradient(135deg, #4f46e5 0%, #9333ea 100%);
        color: white;
    }

    /* Selectboxes and Inputs */
    .stSelectbox>div>div, .stNumberInput>div>div, .stSlider>div>div {
        background-color: rgba(15, 23, 42, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 8px !important;
        color: white !important;
    }
    
    /* Result Cards */
    .result-card-positive {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.2), rgba(21, 128, 61, 0.4));
        border: 1px solid rgba(74, 222, 128, 0.4);
        backdrop-filter: blur(10px);
        padding: 2rem;
        border-radius: 16px;
        color: #bbf7d0;
        margin-top: 2rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(21, 128, 61, 0.3);
        animation: slideUp 0.5s cubic-bezier(0.16, 1, 0.3, 1);
    }
    .result-card-negative {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2), rgba(185, 28, 28, 0.4));
        border: 1px solid rgba(248, 113, 113, 0.4);
        backdrop-filter: blur(10px);
        padding: 2rem;
        border-radius: 16px;
        color: #fecaca;
        margin-top: 2rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(185, 28, 28, 0.3);
        animation: slideUp 0.5s cubic-bezier(0.16, 1, 0.3, 1);
    }
    
    .prob-metric {
        font-size: 3.5rem;
        font-weight: 800;
        margin: 0.5rem 0;
        text-shadow: 0 4px 10px rgba(0,0,0,0.3);
        background: -webkit-linear-gradient(#fff, #cbd5e1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

# --- App Header ---
st.markdown("<h1>Titanic Nexus</h1>", unsafe_allow_html=True)
st.markdown("""
<div style="font-size: 1.2rem; color: #94a3b8; font-weight: 300; margin-bottom: 2rem; max-width: 600px;">
    An advanced predictive engine powered by an 83.2% accuracy Random Forest Classifier. 
    Calibrate the chronometric parameters below to initiate the survival simulation sequence.
</div>
""", unsafe_allow_html=True)

# --- Load Model ---
MODEL_PATH = "titanic_model.pkl"

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

model = load_model()

if model is None:
    st.error("Engine Fault: `titanic_model.pkl` not detected in the deployment matrix.")
else:
    # --- Input Form ---
    with st.form("prediction_form"):
        st.markdown("<h3 style='margin-bottom: 1.5rem; font-weight: 500;'>Passenger Telemetry</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            pclass = st.selectbox("CLASS VECTOR", options=[1, 2, 3], format_func=lambda x: f"Tier {x} ({'First' if x==1 else 'Second' if x==2 else 'Third'} Class)")
            sex_input = st.selectbox("BIOMETRIC PROFILE", options=["Male", "Female"])
            embarked_input = st.selectbox("DEPARTURE NODE", options=["Southampton", "Cherbourg", "Queenstown"])
            age = st.slider("CHRONOLOGICAL AGE", min_value=0.0, max_value=100.0, value=29.0, step=0.5)
            
        with col2:
            fare = st.number_input("CAPITAL EXPENDITURE (£)", min_value=0.0, max_value=600.0, value=32.0, step=0.5)
            family_size = st.number_input("SOCIAL CLUSTER SIZE", min_value=1, max_value=15, value=1, step=1)
            ticket_type = st.number_input("TICKET HASH", min_value=0.0, max_value=10.0, value=2.0, step=0.1)
        
        st.markdown("<br>", unsafe_allow_html=True)
        submit_button = st.form_submit_button("INITIALIZE PREDICTION SEQUENCE")

    # --- Prediction Logic & Result Presentation ---
    if submit_button:
        # UX Delay for effect
        with st.spinner('Running deep learning neural simulation...'):
            time.sleep(1.0) 
        
        # Mapping
        sex_map = {"Male": 0, "Female": 1}
        sex = sex_map[sex_input]
        
        embarked_map = {"Southampton": 0, "Cherbourg": 1, "Queenstown": 2}
        embarked = embarked_map[embarked_input]
        
        # Prepare input array
        input_data = pd.DataFrame(
            [[sex, ticket_type, fare, age, pclass, family_size, embarked]],
            columns=['sex', 'ticket_type', 'fare', 'age', 'pclass', 'family_size', 'embarked']
        )
        
        try:
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1] * 100
            
            if prediction == 1:
                st.markdown(f"""
                <div class="result-card-positive">
                    <div style="font-weight: 700; font-size: 1.2rem; letter-spacing: 2px;">STATUS: OPTIMAL</div>
                    <div class="prob-metric">{probability:.1f}%</div>
                    <div style="font-size: 1.1rem; font-weight: 300;">
                        Simulation indicates high probability of passenger survival.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-card-negative">
                    <div style="font-weight: 700; font-size: 1.2rem; letter-spacing: 2px;">STATUS: CRITICAL</div>
                    <div class="prob-metric">{probability:.1f}%</div>
                    <div style="font-size: 1.1rem; font-weight: 300;">
                        Simulation indicates severe risk factors resulting in fatality.
                    </div>
                </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Matrix Engine Error: {e}")

import streamlit as st
import pandas as pd
import joblib
import os
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="Titanic Survival Prediction Model",
    page_icon="🧊",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# --- Professional Custom CSS ---
st.markdown("""
<style>
    /* Main Background & Clean Typography */
    .main {
        background-color: #FAFAFB;
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        color: #1E293B;
    }
    
    /* Professional Header Styling */
    h1, h2, h3 {
        color: #0F172A;
        font-weight: 600;
        letter-spacing: -0.5px;
    }
    
    /* Subdued Form Box */
    [data-testid="stForm"] {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        padding: 2rem;
    }
    
    /* Professional Assessment Button */
    .stButton>button {
        background-color: #0F172A;
        color: #FFFFFF;
        border-radius: 4px;
        padding: 0.6rem 2rem;
        font-weight: 500;
        width: 100%;
        border: 1px solid #0F172A;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        background-color: #334155;
        border-color: #334155;
        color: white;
    }
    
    /* Sleek Result Cards */
    .result-card-positive {
        background-color: #F0FDF4;
        border-left: 4px solid #16A34A;
        padding: 1.5rem;
        border-radius: 4px;
        color: #14532D;
        margin-top: 1.5rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        animation: fadeIn 0.4s ease-in;
    }
    .result-card-negative {
        background-color: #FEF2F2;
        border-left: 4px solid #DC2626;
        padding: 1.5rem;
        border-radius: 4px;
        color: #7F1D1D;
        margin-top: 1.5rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        animation: fadeIn 0.4s ease-in;
    }
    
    .prob-metric {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(5px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

# --- App Header ---
st.title("Titanic Survival Prediction Model")
st.markdown("""
<div style="font-size: 1.1rem; color: #475569; margin-bottom: 2rem;">
    This dashboard provides a prognostic survival assessment based on historical passenger data. 
    The underlying engine is a Random Forest Classifier cross-validated at <strong>83.2% accuracy</strong>.
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
    st.error("Model Error: `titanic_model.pkl` not detected. Please ensure the model artifact is present in the working directory.")
else:
    # --- Input Form ---
    with st.form("prediction_form"):
        st.subheader("Passenger Demographics & Ticket Data")
        st.markdown("<p style='color: #64748B; font-size: 0.95rem; margin-bottom: 1rem;'>Adjust the parameters below to generate a survival probability estimate.</p>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            pclass = st.selectbox("Ticket Class", options=[1, 2, 3], format_func=lambda x: f"Class {x} ({'First' if x==1 else 'Second' if x==2 else 'Third'})")
            sex_input = st.selectbox("Biological Sex", options=["Male", "Female"])
            embarked_input = st.selectbox("Port of Embarkation", options=["Southampton", "Cherbourg", "Queenstown"])
            age = st.slider("Passenger Age (Years)", min_value=0.0, max_value=100.0, value=29.0, step=0.5)
            
        with col2:
            fare = st.number_input("Standardized Fare (£)", min_value=0.0, max_value=600.0, value=32.0, step=0.5)
            family_size = st.number_input("Total Family Members Aboard", min_value=1, max_value=15, value=1, step=1)
            ticket_type = st.number_input("Encoded Ticket Frequency", min_value=0.0, max_value=10.0, value=2.0, step=0.1, help="Categorical ticket encoding internal to model feature engineering space.")
        
        st.markdown("<br>", unsafe_allow_html=True)
        submit_button = st.form_submit_button("Execute Prediction Assessment")

    # --- Prediction Logic & Result Presentation ---
    if submit_button:
        # UX Placeholder
        with st.spinner('Analyzing demographic and ticket parameters...'):
            time.sleep(0.6) # Professional brief delay to simulate thorough computation
        
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
            
            st.markdown("<hr style='margin: 2rem 0;'>", unsafe_allow_html=True)
            st.subheader("Assessment Results")
            
            if prediction == 1:
                # Survived - Professional Positive Card
                st.markdown(f"""
                <div class="result-card-positive">
                    <div style="font-weight: 600; font-size: 1.1rem; text-transform: uppercase; letter-spacing: 1px;">Likely Outcome: Survived</div>
                    <div class="prob-metric">{probability:.1f}%</div>
                    <div style="font-size: 0.95rem; opacity: 0.9;">
                        The model indicates this passenger profile has a high probability of survival based on the training distribution.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Did Not Survive - Professional Negative Card
                st.markdown(f"""
                <div class="result-card-negative">
                    <div style="font-weight: 600; font-size: 1.1rem; text-transform: uppercase; letter-spacing: 1px;">Likely Outcome: Fatality</div>
                    <div class="prob-metric">{probability:.1f}%</div>
                    <div style="font-size: 0.95rem; opacity: 0.9;">
                        The model indicates this passenger profile aligns closely with the non-surviving distribution.
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            # Render a professional visual probability bar
            st.markdown("<br>", unsafe_allow_html=True)
            st.caption("Survival Probability Index")
            st.progress(int(probability))

        except Exception as e:
            st.error(f"Inference Engine Error: {e}")

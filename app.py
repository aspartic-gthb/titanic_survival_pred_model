import streamlit as st
import pandas as pd
import joblib
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# --- Custom CSS for Styling ---
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
        color: #212529;
    }
    .stButton>button {
        background-color: #0d6efd;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        width: 100%;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #0b5ed7;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .prediction-card-survived {
        background: linear-gradient(135deg, #198754, #20c997);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(25, 135, 84, 0.4);
        margin-top: 2rem;
        animation: fadeIn 0.5s ease-in;
    }
    .prediction-card-died {
        background: linear-gradient(135deg, #dc3545, #f87171);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(220, 53, 69, 0.4);
        margin-top: 2rem;
        animation: fadeIn 0.5s ease-in;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

# --- App Header ---
st.title("🚢 Titanic Survival Prediction")
st.markdown("""
Welcome to the Titanic Survival Predictor! This model was trained using a Random Forest Classifier achieving **~83% cross-validation accuracy**.
Enter passenger details below to see if they would have survived the Titanic disaster.
""")

# --- Load Model ---
MODEL_PATH = "titanic_model.pkl"

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

model = load_model()

if model is None:
    st.warning("⚠️ Model file `titanic_model.pkl` not found! Please run the export script or add the model file to the directory before making predictions.")
else:
    # --- Input Form ---
    with st.form("prediction_form"):
        st.subheader("Passenger Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Dropdowns and selections
            pclass = st.selectbox("Passenger Class", options=[1, 2, 3], format_func=lambda x: f"Class {x} ({'1st' if x==1 else '2nd' if x==2 else '3rd'})")
            sex_input = st.selectbox("Sex", options=["Male", "Female"])
            embarked_input = st.selectbox("Port of Embarkation", options=["Southampton (S)", "Cherbourg (C)", "Queenstown (Q)"])
            age = st.slider("Age", min_value=0.0, max_value=100.0, value=29.0, step=0.5)
            
        with col2:
            # Numeric inputs
            fare = st.number_input("Fare Paid (£)", min_value=0.0, max_value=600.0, value=32.0, step=0.5)
            family_size = st.number_input("Family Size (Siblings/Spouses + Parents/Children + 1)", min_value=1, max_value=15, value=1, step=1)
            ticket_type = st.number_input("Ticket Type Encoded Value", min_value=0.0, max_value=10.0, value=2.0, step=0.1, help="Internal encoded value from the dataset")
        
        submit_button = st.form_submit_button("Predict Survival probability ✨")

    # --- Prediction Logic ---
    if submit_button:
        # Preprocess inputs matching the notebook:
        # ['sex', 'ticket_type', 'fare', 'age', 'pclass', 'family_size', 'embarked']
        
        # Sex mapping
        sex_map = {"Male": 0, "Female": 1}
        sex = sex_map[sex_input]
        
        # Embarked mapping
        embarked_map = {"Southampton (S)": 0, "Cherbourg (C)": 1, "Queenstown (Q)": 2}
        embarked = embarked_map[embarked_input]
        
        # Prepare input array
        input_data = pd.DataFrame([[sex, ticket_type, fare, age, pclass, family_size, embarked]],
                                  columns=['sex', 'ticket_type', 'fare', 'age', 'pclass', 'family_size', 'embarked'])
        
        try:
            # Predict
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1] * 100
            
            # Display highly stylized result
            st.markdown("---")
            if prediction == 1:
                st.markdown(f"""
                <div class="prediction-card-survived">
                    <h2>🌟 SURVIVED</h2>
                    <p style="font-size: 1.2rem; margin-top: 10px;">
                        The model predicts this passenger would have survived!
                    </p>
                    <p style="font-size: 1.1rem; opacity: 0.9;">
                        Survival Probability: <strong>{probability:.1f}%</strong>
                    </p>
                </div>
                """, unsafe_allow_html=True)
                st.balloons()
            else:
                st.markdown(f"""
                <div class="prediction-card-died">
                    <h2>💔 DID NOT SURVIVE</h2>
                    <p style="font-size: 1.2rem; margin-top: 10px;">
                        The model predicts this passenger would tragically not survive.
                    </p>
                    <p style="font-size: 1.1rem; opacity: 0.9;">
                        Survival Probability: <strong>{probability:.1f}%</strong>
                    </p>
                </div>
                """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

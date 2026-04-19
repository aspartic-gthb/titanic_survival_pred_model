# 🚢 Titanic Survival Prediction Web App

A machine learning web application that predicts passenger survival on the Titanic. Built with Python, Scikit-Learn, and Streamlit, this project provides an interactive and highly stylized user interface designed for hackathons and fast deployments.

The model is powered by a **Random Forest Classifier**, tuned with GridSearchCV, achieving approximately **83.2% cross-validation accuracy** on the Titanic dataset.

## ✨ Features
- **Interactive UI**: Users can input passenger details like Age, Fare, Sex, Ticket Class, and Embarkation Port.
- **Micro-Animations**: Custom styled prediction cards returning survival probabilities.
- **Fast Deployment**: Architected perfectly to deploy on Streamlit Community Cloud or Hugging Face Spaces instantly.

## 🧠 Model Optimization Approach
To achieve a high cross-validation accuracy of **83.16%** beyond the baseline, we implemented the following strategies:
1. **Intelligent Imputation**: We avoided global averages for missing data. For example, missing `Fare` values were imputed based on the median of the passenger's specific `pclass` to accurately reflect economic disparities.
2. **Feature Engineering**: Constructed an aggregated `family_size` feature (combining `sibsp` and `parch`) to capture the survival dynamics of group travelers versus solo passengers.
3. **Data Standardization**: Consolidated messy text inputs (various capitalizations of gender and ports) into clean, model-ready numerical matrices.
4. **Exhaustive Hyperparameter Tuning**: Rather than relying on base settings, we ran `GridSearchCV` on our Random Forest across 135 distinct parameter combinations (tuning `max_depth`, `min_samples_split`, `min_samples_leaf`, and `n_estimators`). With 5-fold cross-validation, this thoroughly optimized the bias-variance tradeoff and generalized the model perfectly.

## 🛠 Tech Stack
- **Frontend/Backend**: Streamlit
- **Machine Learning**: Scikit-Learn (Random Forest)
- **Data Manipulation**: Pandas
- **Model Serialization**: Joblib

## 📁 Repository Structure
Note: For deployment efficiency, the raw CSV datasets are ignored via `.gitignore`.
```text
├── app.py                      # Main Streamlit web application
├── titanic_model.pkl           # Pre-trained Random Forest model
├── requirements.txt            # Python dependencies for deployment
├── train_and_export_model.py   # Training script to reproduce model
└── README.md                   # Project documentation
```

## 🚀 Running Locally

To run this application on your own machine:

1. **Clone the repository:**
   ```bash
   git clone <your-repository-url>
   cd titanic-survival-prediction-model
   ```

2. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit application:**
   ```bash
   python -m streamlit run app.py
   ```
   *The app will automatically open in your default web browser.*


---
*Built for Hackathon Submission.*

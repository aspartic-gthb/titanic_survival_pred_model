# 🚢 Titanic Survival Prediction Web App

A machine learning web application that predicts passenger survival on the Titanic. Built with Python, Scikit-Learn, and Streamlit, this project provides an interactive and highly stylized user interface designed for hackathons and fast deployments.

The model is powered by a **Random Forest Classifier**, tuned with GridSearchCV, achieving approximately **83.2% cross-validation accuracy** on the Titanic dataset.

## ✨ Features
- **Interactive UI**: Users can input passenger details like Age, Fare, Sex, Ticket Class, and Embarkation Port.
- **Micro-Animations**: Custom styled prediction cards returning survival probabilities.
- **Fast Deployment**: Architected perfectly to deploy on Streamlit Community Cloud or Hugging Face Spaces instantly.

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

## 🌍 Hackathon Deployment Instructions

Deploying this app for the world to see is completely free and takes 2 minutes:

1. Push this repository to your GitHub account.
2. Sign into [Streamlit Community Cloud](https://share.streamlit.io/) with your GitHub account.
3. Click **New app**, select your repository, and set the main file path to `app.py`.
4. Click **Deploy!**

---
*Built for Hackathon Submission.*
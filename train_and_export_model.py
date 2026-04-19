import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import joblib
import os

print("=" * 40)
print("TITANIC MODEL EXPORTER")
print("=" * 40)

# Check if data exists
train_path = 'titanic_train.csv'
test_path = 'titanic_test.csv'

if not os.path.exists(train_path) or not os.path.exists(test_path):
    print(f"Error: Could not find '{train_path}' or '{test_path}' in the current directory.")
    print("Please make sure you have the CSV files available locally to export the model.")
    print("Alternatively, you can run the following code directly inside your Google Colab Notebook before saving the .pkl file:")
    print('''
    # ADD THIS TO YOUR COLAB NOTEBOOK AFTER TRAINING:
    import joblib
    joblib.dump(best_model, 'titanic_model.pkl')
    # Then download titanic_model.pkl from the colab files menu!
    ''')
    exit(1)

print("Loading data...")
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# --- PREPROCESSING MATCHING THE NOTEBOOK ---
print("Preprocessing data...")
# Clean Sex
train_df['sex'] = train_df['sex'].str.lower().str.strip()
sex_map = {'male': 0, 'm': 0, 'female': 1, 'f': 1}
train_df['sex'] = train_df['sex'].map(sex_map)

# Missing Age
median_age = train_df['age'].median()
train_df['age'] = train_df['age'].fillna(median_age)

# Missing Fare
train_df['fare'] = train_df.groupby('pclass')['fare'].transform(lambda x: x.fillna(x.median()))

# Missing Embarked
train_df['embarked'] = train_df['embarked'].fillna('S')
embarked_map = {'S': 0, 'C': 1, 'Q': 2}
train_df['embarked'] = train_df['embarked'].map(embarked_map)

# Feature Engineering
train_df['family_size'] = train_df['sibsp'] + train_df['parch'] + 1

# Define Features
features = ['sex', 'ticket_type', 'fare', 'age', 'pclass', 'family_size', 'embarked']
X_train = train_df[features]
y_train = train_df['Survived']

# --- TRAINING ---
print("Training model using GridSearchCV...")
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [4, 5, 6, 7, 8],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid, cv=5,
    scoring='accuracy',
    n_jobs=-1, verbose=1
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print(f"Model trained successfully! Best CV Accuracy: {grid_search.best_score_*100:.2f}%")

# --- SAVING MODEL ---
model_filename = 'titanic_model.pkl'
joblib.dump(best_model, model_filename)
print(f"Model saved to '{model_filename}'")
print("You can now run 'streamlit run app.py' to start the web app.")

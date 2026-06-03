# 🚢 Titanic Survival Prediction
### ML Track — Random Forest Classifier | 83.16% CV Accuracy

[![Live App](https://img.shields.io/badge/Live%20App-Streamlit-ff4b4b?style=for-the-badge&logo=streamlit)](https://titanicsurvivalpredmodel-77phk2fwakgzxbjzk5zumm.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?style=for-the-badge&logo=scikit-learn)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Deployed-Streamlit%20Cloud-ff4b4b?style=for-the-badge&logo=streamlit)](https://streamlit.io)

---

## 🔗 Live Demo
**Try the app here →** [titanicsurvivalpredmodel-77phk2fwakgzxbjzk5zumm.streamlit.app](https://titanicsurvivalpredmodel-77phk2fwakgzxbjzk5zumm.streamlit.app/)

Enter any passenger's details and get an instant survival prediction from our trained model.

---

## 📌 Project Overview

This project was built as part of an **ML Track competition**. The task was to predict which passengers survived the Titanic disaster using structured tabular data.

A baseline model was intentionally provided with weak performance — using only 5 trees, depth of 2, and no proper preprocessing. Our goal was to beat it through smart decision-making at every step:

- Thorough data exploration
- Proper missing value handling
- Fixing inconsistent data
- Feature engineering
- Hyperparameter tuning
- Live deployment as a web app

---

## 📊 Final Results

| Metric | Baseline | Our Model |
|---|---|---|
| Test Accuracy | ~72% | **77.65%** |
| CV Accuracy (5-fold) | Not measured | **83.16%** |
| Trees (n_estimators) | 5 | 200 |
| Max Depth | 2 | 7 |
| Feature Engineering | ❌ None | ✅ Yes |
| Smart Missing Value Handling | ❌ No | ✅ Yes |

> **Improvement: +11 percentage points over baseline**

---

## 📁 Repository Structure

```
titanic-survival-prediction-model/
│
├── titanicfinal.ipynb                 # Main notebook — full pipeline
├── titanic_train.csv                  # Training data (712 passengers)
├── titanic_test.csv                   # Test data (179 passengers)
└── README.md                          # This file
```

---

## 📂 Dataset Overview

The dataset contains records of Titanic passengers with the following columns:

| Column | Type | Description |
|---|---|---|
| Survived | Integer (0/1) | Target — 1 = survived, 0 = died |
| pclass | Integer (1/2/3) | Ticket class |
| sex | Text | Passenger gender |
| age | Float | Age in years |
| sibsp | Integer | Siblings / spouses aboard |
| parch | Integer | Parents / children aboard |
| fare | Float | Ticket price paid |
| embarked | Text | Port of boarding (S / C / Q) |
| ticket_type | Float | Encoded ticket category |
| has_cabin_id | Integer (0/1) | Whether a cabin was recorded |

**Train file:** 712 passengers | **Test file:** 179 passengers

---

## 🔍 Step 1 — Data Exploration

Before touching anything, we explored the data to understand exactly what was broken:

```python
train_df.info()          # column types and missing value counts
train_df.head()          # first 5 rows
train_df['sex'].unique() # spot inconsistent values
train_df.isnull().sum()  # count missing values per column
```

**Problems found:**

| Column | Problem | Scale |
|---|---|---|
| sex | 6 inconsistent variants: male, M, Male, female, F, Female | All rows |
| age | Missing values | 216 in train, 56 in test |
| fare | Missing values | 89 in train, 22 in test |
| embarked | Missing values + stored as text | 86 in train, 28 in test |

---

## 🧹 Step 2 — Data Cleaning

### 2.1 Fixing the Sex Column
The baseline only mapped `male` and `female` correctly. `M`, `Male`, `F`, `Female` all became `-1` — which is wrong.

```python
# Standardize all to lowercase first
train_df['sex'] = train_df['sex'].str.lower().str.strip()
test_df['sex']  = test_df['sex'].str.lower().str.strip()

# Map all variants to 0 or 1
sex_map = {'male': 0, 'm': 0, 'female': 1, 'f': 1}
train_df['sex'] = train_df['sex'].map(sex_map)
test_df['sex']  = test_df['sex'].map(sex_map)
```

### 2.2 Missing Age — Median Imputation
Median is used instead of mean because age data has outliers (very young and very old passengers) that skew the average. The median is always calculated from training data only to avoid data leakage.

```python
median_age = train_df['age'].median()  # = 29.0
train_df['age'] = train_df['age'].fillna(median_age)
test_df['age']  = test_df['age'].fillna(median_age)
```

### 2.3 Missing Fare — Class-Based Median
This was our smartest cleaning decision. Filling all missing fares with the overall median is wrong because fares vary enormously by class:

| Ticket Class | Median Fare |
|---|---|
| 1st class | 56.92 |
| 2nd class | 15.04 |
| 3rd class | 8.05 |

So we filled each passenger's missing fare using their own class median:

```python
train_df['fare'] = train_df.groupby('pclass')['fare'].transform(
    lambda x: x.fillna(x.median()))
test_df['fare'] = test_df.groupby('pclass')['fare'].transform(
    lambda x: x.fillna(x.median()))
```

### 2.4 Missing Embarked — Mode Imputation
Port S (Southampton) appeared 541 times — the most common. Missing ports were filled with S, then all ports were converted to numbers:

```python
train_df['embarked'] = train_df['embarked'].fillna('S')
embarked_map = {'S': 0, 'C': 1, 'Q': 2}
train_df['embarked'] = train_df['embarked'].map(embarked_map)
```

---

## ⚙️ Step 3 — Feature Engineering

Feature engineering means creating new, more informative columns from existing ones — giving the model stronger signals to learn from.

### 3.1 family_size
The dataset had two separate columns for family: `sibsp` (siblings/spouses) and `parch` (parents/children). We combined them into one meaningful feature:

```python
train_df['family_size'] = train_df['sibsp'] + train_df['parch'] + 1
test_df['family_size']  = test_df['sibsp']  + test_df['parch']  + 1
```

The `+1` includes the passenger themselves. A person with 2 siblings and 1 parent has `family_size = 4`.

**Why it matters:** Passengers travelling with family had very different survival odds than solo travellers.

### 3.2 is_alone
A binary flag indicating whether the passenger was travelling completely alone:

```python
train_df['is_alone'] = (train_df['family_size'] == 1).astype(int)
test_df['is_alone']  = (test_df['family_size']  == 1).astype(int)
```

`1` = travelling alone | `0` = with at least one family member

### 3.3 Feature Selection
After training, we checked which columns the model found useful and removed the weakest ones:

```python
# Removed — too weak to help
# has_cabin_id  → 0.78% importance
# is_alone      → 1.2% importance  (already captured by family_size)

features = ['sex', 'ticket_type', 'fare', 'age', 'pclass', 'family_size', 'embarked']
```

---

## 🌲 Step 4 — The Algorithm: Random Forest

### What is a Decision Tree?
A decision tree is a flowchart of yes/no questions:

```
Is the passenger female?
├── YES → Did they travel in 1st or 2nd class?
│         ├── YES → SURVIVED ✅
│         └── NO  → Did they have family onboard?
│                   ├── YES → SURVIVED ✅
│                   └── NO  → DIED ❌
└── NO  → Is their age under 10?
          ├── YES → SURVIVED ✅
          └── NO  → DIED ❌
```

### What is a Random Forest?
A Random Forest builds hundreds of decision trees, each trained on a slightly different random sample of the data. The final prediction is a **majority vote** across all trees.

```
Tree 1  → SURVIVED
Tree 2  → SURVIVED
Tree 3  → DIED
Tree 4  → SURVIVED
Tree 5  → SURVIVED
...
200 trees vote → majority says SURVIVED ✅
```

**Why better than one tree?**
A single tree can memorize training data (overfitting). Many trees voting together cancel out each other's mistakes and give more reliable predictions on new data.

### Key Hyperparameters

| Parameter | What it controls | Baseline | Ours |
|---|---|---|---|
| n_estimators | Number of trees voting | 5 | 200 |
| max_depth | Questions per tree | 2 | 7 |
| min_samples_split | Min passengers before splitting | default | 5 |
| min_samples_leaf | Min passengers at final node | default | 2 |
| random_state | Fixes randomness for reproducibility | 42 | 42 |

---

## 🔁 Step 5 — Cross Validation

Instead of testing accuracy once, we used **5-fold cross validation** for a more honest and reliable score:

```
712 training passengers split into 5 groups of ~142:

Round 1: Train on groups 2,3,4,5 → Test on group 1 → 78%
Round 2: Train on groups 1,3,4,5 → Test on group 2 → 85%
Round 3: Train on groups 1,2,4,5 → Test on group 3 → 83%
Round 4: Train on groups 1,2,3,5 → Test on group 4 → 82%
Round 5: Train on groups 1,2,3,4 → Test on group 5 → 84%
                                        Average → 83.16%
```

```python
scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
print(f"Average CV Accuracy: {scores.mean()*100:.2f}%")
```

---

## 🎛️ Step 6 — Hyperparameter Tuning (GridSearchCV)

GridSearchCV automatically tests every combination of settings and finds the best one:

```python
param_grid = {
    'n_estimators':      [100, 200, 300],   # 3 options
    'max_depth':         [4, 5, 6, 7, 8],   # 5 options
    'min_samples_split': [2, 5, 10],        # 3 options
    'min_samples_leaf':  [1, 2, 4]          # 3 options
}
# 3 × 5 × 3 × 3 = 135 combinations
# 135 × 5 folds = 675 models trained automatically
```

**Best settings found:**
```python
RandomForestClassifier(
    n_estimators=200,
    max_depth=7,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
# Best CV Accuracy: 83.16%
```

---

## 📈 Feature Importance

After training, the Random Forest scores every column by how much it contributed to correct predictions:

| Feature | Importance | Insight |
|---|---|---|
| sex | 43.5% | Gender was the single biggest survival factor |
| ticket_type | 15.6% | Ticket category carries survival signal |
| fare | 13.0% | How much paid — proxy for wealth and class |
| age | 10.2% | Children were prioritized in evacuation |
| pclass | 7.7% | 1st class passengers had better access to lifeboats |
| family_size | 5.5% | Engineered feature — solo vs with family |
| embarked | 2.4% | Port of boarding — weak but kept |

---

## 📉 Accuracy Journey

| Stage | What we did | Accuracy |
|---|---|---|
| Baseline | 5 trees, depth 2, no cleaning | ~72% |
| After cleaning | Fixed sex, age, fare, embarked | 77.09% |
| After cross validation | Honest 5-fold measurement | 81.89% |
| After dropping weak features | Removed noise columns | 82.31% |
| After GridSearchCV | Best hyperparameters found | **83.16%** |

---

## 🌐 Deployment

The trained model was deployed as a live interactive web app using **Streamlit**:

**Live URL:** https://titanicsurvivalpredmodel-77phk2fwakgzxbjzk5zumm.streamlit.app/

### How the app works:
1. User enters passenger details — sex, age, class, fare, family size
2. App applies the same cleaning and feature engineering as training
3. Trained Random Forest model predicts survival
4. Result displayed instantly — **Survived ✅** or **Did Not Survive ❌**

### Tech stack:
- Python + scikit-learn — model
- Streamlit — web app framework
- Streamlit Cloud — free hosting
- GitHub — code repository linked to deployment

---

## 🚀 How to Run Locally

```bash
# Clone the repository
git clone https://github.com/Tanvibiswas/titanic-survival-prediction-model.git
cd titanic-survival-prediction-model

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook titanicfinal.ipynb

# Or run the Streamlit app locally
streamlit run app.py
```

---

## 🧰 Tech Stack

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.10 | Programming language |
| pandas | Latest | Data loading and cleaning |
| scikit-learn | Latest | Model training and evaluation |
| matplotlib | Latest | Data visualisation |
| Streamlit | Latest | Web app and deployment |
| Google Colab | — | Development environment |
| GitHub | — | Version control |

---

## 💡 Key Concepts Covered

- Exploratory Data Analysis (EDA)
- Missing value imputation — median, mode, group-based
- Label encoding for categorical variables
- Feature engineering — creating new columns from existing ones
- Feature selection — removing weak columns
- Random Forest algorithm — how ensemble methods work
- Overfitting vs underfitting — why max_depth matters
- Cross validation — why one accuracy measurement is not enough
- Hyperparameter tuning with GridSearchCV
- Feature importance analysis
- Model deployment with Streamlit

---

## 👤 Author

**Tanvi Biswas**
GitHub: [@Tanvibiswas](https://github.com/Tanvibiswas)

---


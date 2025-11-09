# 🩺 Diabetes Prediction using Machine Learning

## 📌 Overview
This project predicts whether a person is diabetic or not using different Machine Learning models — **Logistic Regression**, **Random Forest**, and **XGBoost** — trained on the **PIMA Indian Diabetes dataset** from Kaggle.

The project demonstrates the complete ML workflow: data preprocessing, model training, evaluation, and real-time prediction.

---

## 📊 Dataset
- **Dataset Name:** PIMA Indians Diabetes Database  
- **Source:** [Kaggle - Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)  
- **Description:**  
  The dataset contains diagnostic measurements for female patients of Pima Indian heritage, aged 21 or older.

| Column Name              | Description |
|--------------------------|--------------|
| Pregnancies              | Number of times pregnant |
| Glucose                  | Plasma glucose concentration |
| BloodPressure            | Diastolic blood pressure (mm Hg) |
| SkinThickness            | Triceps skin fold thickness (mm) |
| Insulin                  | 2-Hour serum insulin (mu U/ml) |
| BMI                      | Body mass index (weight in kg/(height in m)^2) |
| DiabetesPedigreeFunction | Diabetes pedigree function |
| Age                      | Age (years) |
| Outcome                  | 1 = Diabetic, 0 = Non-diabetic |

---

## 🧠 Models Used
Three ML models were implemented and compared:
1. **Logistic Regression**
2. **Random Forest Classifier**
3. **XGBoost Classifier**

Each model was trained on the same preprocessed dataset and evaluated based on accuracy and F1-score.

---

## ⚙️ Technologies Used
- Python
- NumPy
- Pandas
- Scikit-learn
- XGBoost
- Google Colab / Kaggle Notebooks

---

## 🧩 Steps Involved

### 1️⃣ Data Preprocessing
- Handling missing values  
- Standardizing the data using `StandardScaler`  
- Splitting data into training and testing sets  

### 2️⃣ Model Training
```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

log_model = LogisticRegression()
rf_model = RandomForestClassifier()
xgb_model = XGBClassifier()

log_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)

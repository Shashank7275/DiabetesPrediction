import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('diabetes.csv')

# Separate features and target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Streamlit app
st.title('Diabetes Prediction')

st.sidebar.header('User Input Features')

# Function to get user input
def get_user_input():
    pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('Glucose', 0, 199, 117)
    blood_pressure = st.sidebar.slider('Blood Pressure', 0, 122, 72)
    skin_thickness = st.sidebar.slider('Skin Thickness', 0, 99, 23)
    insulin = st.sidebar.slider('Insulin', 0, 846, 30)
    bmi = st.sidebar.slider('BMI', 0.0, 67.1, 32.0)
    dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.078, 2.42, 0.3725)
    age = st.sidebar.slider('Age', 21, 81, 29)

    user_data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }
    features = pd.DataFrame(user_data, index=[0])
    return features

# Get user input
user_input = get_user_input()

# Display user input
st.subheader('User Input features')
st.write(user_input)

# Make predictions
prediction = model.predict(user_input)
prediction_proba = model.predict_proba(user_input)

# Display prediction
st.subheader('Prediction')
st.write('Diabetic' if prediction[0] == 1 else 'Not Diabetic')

st.subheader('Prediction Probability')
st.write(prediction_proba)

# Display model accuracy
st.subheader('Model Accuracy')
accuracy = accuracy_score(y_test, model.predict(X_test))
st.write(f'Accuracy: {accuracy * 100:.2f}%')

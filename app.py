import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model and feature columns
model = joblib.load("titanic_model.pkl")
model_columns = joblib.load("model_columns.pkl")

st.title("🚢 Titanic Survival Prediction App")
st.markdown("Enter passenger details below to check survival probability.")

# UI inputs
Pclass = st.selectbox("Passenger Class", [1, 2, 3])
Sex = st.radio("Sex", ["male", "female"])
Age = st.slider("Age", 0, 80, 25)
SibSp = st.slider("Number of Siblings/Spouses aboard", 0, 5, 0)
Parch = st.slider("Number of Parents/Children aboard", 0, 5, 0)
Fare = st.slider("Fare Paid", 0, 500, 50)
Embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

# Process inputs
data = {
    'Pclass': Pclass,
    'Sex': 0 if Sex == 'male' else 1,
    'Age': Age,
    'SibSp': SibSp,
    'Parch': Parch,
    'Fare': Fare,
    'Embarked_Q': 1 if Embarked == 'Q' else 0,
    'Embarked_S': 1 if Embarked == 'S' else 0
}

input_df = pd.DataFrame([data])

# Predict
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    st.success("Survived! 🎉" if prediction == 1 else "Did not survive 😔")

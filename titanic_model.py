import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv(r"C:\Users\ajaz6\OneDrive\Desktop\Titanic_Survival_App\titanic.csv")
# print("Shape:", df.shape)
# print(df.head())
# print(df.info())
# print(df.describe())
# print(df.isnull().sum())
# print(df.columns)
# Drop PassengerId, Ticket, and Cabin
df.drop(['PassengerId', 'Ticket', 'Cabin','Name'], axis=1, inplace=True)
# Fill missing Age with median (less sensitive to outliers)
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill missing Embarked with mode (most common port)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
# print("Remaining null values:\n", df.isnull().sum())
# Encode Sex (binary column)
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
# One-hot encode Embarked (C = Cherbourg, Q = Queenstown, S = Southampton)
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
# print(df.head())
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# Define input and target variables
X = df.drop('Survived', axis=1)
y = df['Survived']

# Split into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Initialize and train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
# Predict on test set
y_pred = model.predict(X_test)

# Evaluate
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
# print("\nClassification Report:\n", classification_report(y_test, y_pred))
import joblib

# Save the model to a file
joblib.dump(model, 'titanic_model.pkl')

# Save the columns used for prediction
joblib.dump(X.columns.tolist(), 'model_columns.pkl')

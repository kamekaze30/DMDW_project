import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pickle
import os

# Load the dataset
df = pd.read_csv('diabetes.csv')

# Handle missing values (zero imputation for certain columns)
columns_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in columns_with_zeros:
    median_val = df[col][df[col] != 0].median()
    df[col] = df[col].replace(0, median_val)

# Prepare features and target
X = df.drop(columns=['Outcome'])
y = df['Outcome']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = LogisticRegression(random_state=42, solver='liblinear')
model.fit(X_train_scaled, y_train)

# Save the model and scaler
with open('diabetes_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Test accuracy
accuracy = model.score(X_test_scaled, y_test)
print(f"Model trained successfully!")
print(f"Test Accuracy: {accuracy:.2%}")
print(f"Model saved as 'diabetes_model.pkl'")
print(f"Scaler saved as 'scaler.pkl'")

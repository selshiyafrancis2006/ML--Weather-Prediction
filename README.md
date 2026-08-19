# Implementation of Random Forest Algorithm for Weather Prediction
## AIM:
To write a program to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data using Random Forest Algorithm.

## Problem Statement and Dataset

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries such as Pandas, train_test_split, and RandomForestRegressor from Scikit-learn.
2.Load and preprocess the environmental sensor dataset, separating the sensor features from the target variables: daily temperature, PM2.5 3.pollution level, and energy.
4Split the dataset into training and testing sets, then train the Random Forest model using the training data.
5.Predict the target values using the trained Random Forest model and evaluate the prediction performance using suitable regression metrics.

## Program:
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data = pd.read_csv("weather-station-eee-block_2024_07_13.csv")

print(data.head())

# Remove rows where temperature is missing
data = data.dropna(subset=["tem"])

# Select input features
X = data[[
    "hum",
    "co2",
    "illumination",
    "pressure",
    "pm2_5",
    "pm10",
    "wind_direction_angle",
    "wind_speed",
    "wind_speed_level",
    "tsr"
]]

# Target variable
y = data["tem"]

# Fill missing values in input features
X = X.fillna(X.mean())

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Create Random Forest Regressor
temp_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train model
temp_model.fit(X_train, y_train)

# Predict temperature
temp_pred = temp_model.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, temp_pred)
r2 = r2_score(y_test, temp_pred)

print("Mean Squared Error:", mse)
print("R2 Score:", r2)


/*
Program to implement the Random Forest Algorithm to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data.
Developed by: Selshiya F
RegisterNumber: 212224060241
*/
```

## Output:

<img width="907" height="534" alt="image" src="https://github.com/user-attachments/assets/fb534281-9908-4319-9592-60c20cda06d9" />

## Result:

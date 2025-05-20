import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Example dataset
data = pd.DataFrame({
    'rainfall_mm': [100, 150, 200, 250, 300, 350],
    'river_level_m': [1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
    'risk': [0, 0, 1, 1, 1, 1]  # 0 = Low Risk, 1 = High Risk
})

# Feature and label selection
X = data[['rainfall_mm', 'river_level_m']]
y = data['risk']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluation
print(X_test)
print("\nPredicted Risk:", y_pred)
print("Actual Risk:", list(y_test))
print("\nAccuracy:", accuracy_score(y_test, y_pred))

# Predict new data
new_data = pd.DataFrame({'rainfall_mm': [250], 'river_level_m': [3.6]})
prediction = model.predict(new_data)

print("\nNew Prediction for rainfall 250mm & river level 3.6m:",
      "High Risk" if prediction[0] == 1 else "Low Risk")
import random
import time
import json

# Simulated IoT sensor data generator
def read_sensor_data():
    data = {
        "temperature": round(random.uniform(20, 50), 2),  # Celsius
        "humidity": round(random.uniform(10, 80), 2),     # Percentage
        "location": {"latitude": 37.7749, "longitude": -122.4194},  # Example: San Francisco
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    return data

# Simple threshold-based alert system for wildfire risk
def evaluate_risk(data):
    if data["temperature"] > 40 and data["humidity"] < 20:
        return "🔥 High wildfire risk!"
    elif data["temperature"] > 35 and data["humidity"] < 30:
        return "⚠️ Moderate wildfire risk."
    else:
        return "✅ Low risk."

# Main simulation loop
def main():
    print("🔍 Starting IoT Disaster Prediction System...\n")
    for _ in range(5):  # simulate 5 readings
        sensor_data = read_sensor_data()
        risk_level = evaluate_risk(sensor_data)

        print(json.dumps(sensor_data, indent=2))
        print("ALERT:", risk_level)
        print("-" * 50)
        time.sleep(2)

if __name__ == "__main__":
    main()
# Source code:
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Sample dataset (synthetic)
data = {
    'rainfall': [100, 200, 150, 80, 90, 300, 400, 50, 600, 220],
    'river_level': [4, 6, 5.5, 3, 3.5, 7, 8, 2.5, 3, 6.5],
    'humidity': [90, 95, 85, 70, 65, 98, 99, 55, 60, 96],
    'flood': [1, 1, 1, 0, 0, 1, 1, 0, 1, 1]  # 1: Flood, 0: No Flood
}

# Create DataFrame
df = pd.DataFrame(data)

# Features and Labels
X = df[['rainfall', 'river_level', 'humidity']]
y = df['flood']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Output
print("Predicted flood labels:", y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))

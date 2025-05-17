
# combined_system.py

# ========== IMPORTS ==========
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from flask import Flask, request, jsonify
import joblib
import threading
import requests
import random
import time
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential

# ========== SECTION 1: Flood Risk AI Model ==========
def train_flood_model():
    data = pd.read_csv('flood_data.csv')
    features = data[['rainfall', 'river_level', 'soil_moisture']]
    labels = data['flood_risk']
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print("Flood Risk Model Evaluation:\n", classification_report(y_test, predictions))

# ========== SECTION 2: Healthcare AI Model ==========
def train_health_model():
    # Mock dataset loading logic
    dataset = tf.keras.utils.get_file('health_data.csv', 'https://example.com/health_data.csv')  # Placeholder
    # Simulate train-test split
    train_data = tf.data.Dataset.range(1000).batch(32)
    test_data = tf.data.Dataset.range(200).batch(32)

    model = Sequential([
        Input(shape=(100,)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_data, epochs=20, validation_data=test_data)
    model.save('phase4_healthcare_model.h5')
    print("Healthcare model trained and saved.")

# ========== SECTION 3: Disaster Prediction AI Model ==========
def train_disaster_model():
    data = pd.read_csv('disaster_data.csv')
    X = data[['temperature', 'humidity', 'seismic_activity']]
    y = data['disaster_risk']
    model = RandomForestClassifier()
    model.fit(X, y)
    joblib.dump(model, 'disaster_model.pkl')
    print("Disaster prediction model trained and saved.")

# ========== SECTION 4: Flask Alert API ==========
app = Flask(__name__)
model = None

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = [[data['temperature'], data['humidity'], data['seismic_activity']]]
    risk = model.predict(features)[0]
    return jsonify({'risk': risk})

def run_flask_api():
    global model
    model = joblib.load('disaster_model.pkl')
    app.run(debug=False, port=5000)

# ========== SECTION 5: IoT Simulator ==========
def simulate_iot():
    url = "http://localhost:5000/predict"
    while True:
        payload = {
            'temperature': random.uniform(30, 45),
            'humidity': random.uniform(60, 90),
            'seismic_activity': random.uniform(3, 6)
        }
        try:
            response = requests.post(url, json=payload)
            print("IoT Simulator:", response.json())
        except:
            print("Waiting for Flask API...")
        time.sleep(10)

# ========== MAIN EXECUTION ==========
if __name__ == '__main__':
    print("Training models...")
    train_flood_model()
    train_disaster_model()
    # train_health_model()  # Optional, commented due to dataset URL placeholder

    print("Starting Flask API and IoT Simulator...")

    # Start Flask API in a separate thread
    threading.Thread(target=run_flask_api).start()

    # Wait a few seconds for Flask to start
    time.sleep(5)

    # Start IoT simulation
    simulate_iot()

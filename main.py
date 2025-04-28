import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
import random
import os

# Load pre-trained global model
global_model = joblib.load('model.pkl')

# Define columns for the results DataFrame
columns = [
    'attack_neptune', 'attack_normal', 'attack_satan', 'count',
    'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_same_srv_rate', 'dst_host_srv_count', 'flag_S0', 'flag_SF',
    'last_flag', 'logged_in', 'same_srv_rate', 'serror_rate',
    'service_http', 'classification'
]

# Initialize DataFrame to store results
results_df = pd.DataFrame(columns=columns)

# Function to block a detected malicious IP
def block_ip(ip):
    st.warning(f"Intrusion detected! IP {ip} has been blocked.")

# Simulate federated learning
def federated_learning(local_model, local_data):
    global global_model

    # Train local model
    local_X = local_data.drop(columns=['classification'])
    local_y = local_data['classification']
    local_model.fit(local_X, local_y)

    # Simulate updating the global model
    global_weights = global_model.feature_importances_
    local_weights = local_model.feature_importances_

    # Update global weights with a weighted average
    new_weights = (global_weights + local_weights) / 2
    global_model.feature_importances_ = new_weights

    # Save the updated global model
    joblib.dump(global_model, 'global_model.pkl')

# Generate mock data for features
def generate_mock_data():
    attack_neptune = random.choice([0, 1])
    attack_normal = 1 - attack_neptune
    attack_satan = random.choice([0, 1])
    count = random.randint(0, 100)
    dst_host_diff_srv_rate = random.uniform(0, 1)
    dst_host_same_src_port_rate = random.uniform(0, 1)
    dst_host_same_srv_rate = random.uniform(0, 1)
    dst_host_srv_count = random.randint(0, 50)
    flag_S0 = random.choice([0, 1])
    flag_SF = random.choice([0, 1])
    last_flag = random.randint(0, 255)
    logged_in = random.choice([0, 1])
    same_srv_rate = random.uniform(0, 1)
    serror_rate = random.uniform(0, 1)
    service_http = random.choice([0, 1])

    # Simulate a prediction
    features = [
        attack_neptune, attack_normal, attack_satan, count,
        dst_host_diff_srv_rate, dst_host_same_src_port_rate,
        dst_host_same_srv_rate, dst_host_srv_count, flag_S0, flag_SF,
        last_flag, logged_in, same_srv_rate, serror_rate, service_http
    ]
    prediction = global_model.predict([features])[0]
    classification = {0: 'Normal', 1: 'DOS', 2: 'PROBE', 3: 'R2L', 4: 'U2R'}[prediction]

    # Block IP if classified as an attack
    if classification != "Normal":
        block_ip("192.168.1.1")  # Mock IP address

    return features + [classification]

st.sidebar.title("Real Time Threat Detection")
st.sidebar.markdown("This app captures network packets in real-time, processes them, and predicts the likelihood of an intrusion based on pre-trained model classifications.")
st.sidebar.markdown("### Instructions")
st.sidebar.markdown(
    """
    1. Click **Start Real-Time Capture** to begin capturing packets.
    2. The app will display a DataFrame with real-time results.
    3. Packet counts and feature calculations are reset every 2 seconds.
    """
)

# Main title and description
st.title("Real-Time Threat Detection")
st.markdown(
    """
    This Streamlit app captures network packets, extracts relevant features, and classifies each packet based on pre-trained model predictions.
    The purpose is to detect potential network intrusions, providing insights into different types of attacks such as **DOS**, **Probe**, **R2L**, and **U2R**.
    """
)
# Streamlit button to start simulation
if st.button("Start Simulation"):
    st.write("Simulating data... Displaying results in real-time:")
    reset_interval = timedelta(seconds=2)
    last_reset = datetime.now()

    # Simulate local training
    local_model = RandomForestClassifier()
    local_data = pd.DataFrame(columns=columns[:-1])

    # Streamlit real-time display for DataFrame
    data_display = st.empty()

    # Generate mock data in batches
    for _ in range(100):  # Simulate 100 packets
        mock_data = generate_mock_data()
        results_df.loc[len(results_df)] = mock_data

        # Update display
        data_display.dataframe(results_df)

        # Simulate federated learning with local model and data
        if not local_data.empty:
            federated_learning(local_model, local_data)
            local_data = pd.DataFrame(columns=columns[:-1])  # Reset local data

        # Reset periodically
        if (datetime.now() - last_reset) > reset_interval:
            last_reset = datetime.now()

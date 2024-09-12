import streamlit as st
import pickle
import numpy as np

# Load the model and scaler
model = pickle.load(open('placement_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.title("Placement Prediction App")

# Input fields
cgpa = st.number_input("Enter CGPA:", min_value=0.0, max_value=10.0, step=0.1)
iq = st.number_input("Enter IQ:", min_value=50, max_value=250, step=1)

# Predict button
if st.button("Predict Placement"):
    # Scale the inputs
    input_data = np.array([[cgpa, iq]])
    input_data_scaled = scaler.transform(input_data)
    
    # Prediction
    prediction = model.predict(input_data_scaled)
    
    if prediction[0] == 1:
        st.success("The student is likely to be placed.")
    else:
        st.warning("The student is unlikely to be placed.")

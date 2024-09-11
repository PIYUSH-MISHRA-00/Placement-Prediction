import streamlit as st
import pickle
import numpy as np

# Load the improved model
model = pickle.load(open('placement_model_improved.pkl', 'rb'))

st.title("Student Placement Predictor")

# Input fields
cgpa = st.number_input("Enter your CGPA (0-10)", min_value=0.0, max_value=10.0, step=0.1)
iq = st.number_input("Enter your IQ", min_value=70, max_value=160, step=1)

# Predict button
if st.button("Predict"):
    cgpa_iq = cgpa * iq
    input_data = np.array([[cgpa, iq, cgpa_iq]])
    prediction = model.predict(input_data)
    
    if prediction[0] == 1:
        st.success("You will get placed!")
    else:
        st.error("You will not get placed.")

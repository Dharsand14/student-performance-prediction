import streamlit as st
import numpy as np
import joblib 
import warnings
warnings.filterwarnings("ignore")

model = joblib.load("best_model.pkl")

st.title("Student Performance Prediction")

attendance = st.slider("Attendance percentage", 0.0, 100.0, 75.0)
study_hours = st.slider("Study Hours", 0.0, 6.0, 2.0)
mental_health = st.slider("Mental Health Rating(1-10)", 1, 10, 5)
sleep_hours = st.slider("Sleep Hours", 0.0, 9.0, 6.0)
exam_scores = st.slider("Previous Exam Scores", 0.0, 100.0, 60.0)

if st.button("Predict Performance"):
    
    input_data = np.array([[study_hours, attendance, mental_health, sleep_hours, exam_scores]])
    prediction = model.predict(input_data)[0]

    prediction =max(0, min(100, prediction))
                    
    st.success(f"Predicted Performance: {prediction:.2f}")
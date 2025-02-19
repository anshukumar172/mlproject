import streamlit as st
import pandas as pd

# Dummy function for prediction (replace with your actual prediction pipeline)
def predict_math_score(gender, ethnicity, parental_level_of_education, lunch, test_preparation_course, reading_score, writing_score):
    # For demonstration, let's create a simple dummy prediction:
    # Math Score = average of reading and writing scores plus 10.
    math_score = (reading_score + writing_score) / 2 + 10
    return math_score

def main():
    st.title("Student Exam Performance Predictor")

    st.header("Enter Student Details")

    # Form Inputs
    gender = st.selectbox("Gender", options=["male", "female"])
    ethnicity = st.selectbox("Race/Ethnicity", options=["group A", "group B", "group C", "group D", "group E"])
    parental_level_of_education = st.selectbox("Parental Level of Education", 
                                                 options=["associate's degree", "bachelor's degree", "high school", "master's degree", "some college", "some high school"])
    lunch = st.selectbox("Lunch Type", options=["free/reduced", "standard"])
    test_preparation_course = st.selectbox("Test Preparation Course", options=["none", "completed"])
    reading_score = st.number_input("Reading Score (out of 100)", min_value=0, max_value=100, value=50)
    writing_score = st.number_input("Writing Score (out of 100)", min_value=0, max_value=100, value=50)

    # When the user clicks the button, perform prediction.
    if st.button("Predict Math Score"):
        # You would normally create a DataFrame and pass it through your model pipeline here.
        # For now, we'll use a dummy function to simulate a prediction.
        prediction = predict_math_score(gender, ethnicity, parental_level_of_education, lunch, test_preparation_course, reading_score, writing_score)
        st.success(f"The predicted Math Score is: {prediction:.2f}")

if __name__ == "__main__":
    main()

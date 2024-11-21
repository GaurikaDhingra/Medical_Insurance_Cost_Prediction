import streamlit as st
import numpy as np
import joblib 

model = joblib.load("D:\Gaurika Dhingra\Gaurika_CS\Medical Insurance Cost Prediction\insurance_cost_model.pkl")


st.title("Medical Insurance Predictor")
st.write("Get cost predictions for your next Medical Insurance")
#st.write("Fill in the below details to get your predictions")

user_name = st.text_input("Enter your name:", key="user_name_input")

if user_name:
    st.write(f"Welcome, {user_name}! Please provide the following details for prediction:")
    
    # User input for prediction
    age = st.slider("Age", 18, 100)
    sex = st.selectbox("Gender", ("Male", "Female"))
    bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=50.0, step=0.1)
    children = st.slider("Number of Children", 0, 10)
    smoker = st.selectbox("Do you smoke?", ("Yes", "No"))
    
    
    if st.button("Predict"):
        # Convert inputs to model-friendly format
        sex_binary = 1 if sex == "Male" else 0
        smoker_binary = 1 if smoker == "Yes" else 0
        
        input_data = np.array([[age, sex_binary, bmi, children, smoker_binary]])
        
        # Make the prediction using the trained model
        prediction_usd = model.predict(input_data)[0]  # Extract the scalar value
        
        # Conversion rate from USD to INR (hardcoded as 83 INR per 1 USD)
        conversion_rate = 83
        
        # Convert the prediction to INR
        prediction_inr = prediction_usd * conversion_rate
        
        # Display the predicted insurance cost in INR
        st.write(f"{user_name}, your estimated insurance cost is: ₹{prediction_inr:.2f} INR")
else:
    st.write("Please enter your name to proceed.")



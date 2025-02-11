import numpy as np
import streamlit as st
import pandas as pd
import joblib as jb

model = jb.load('dragon.joblib')

st.title("Dragon Real estate House Price Prediction")
st.write("Enter the details below to predict the house price")

inputs = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
       'PTRATIO', 'B', 'LSTAT']

input_data = {}
for feature in inputs:
    if feature == 'CHAS':  
        input_data[feature] = st.selectbox(f"{feature}", options=[0, 1], index=0)
    else:
        input_data[feature] = st.number_input(f"{feature}", value=0.0, step=0.1)

if st.button("Predict Price"):
    input_df = pd.DataFrame([input_data], columns=inputs)
    predictions = model.predict(input_df)
    st.success(f"Predicted House Price: ${predictions[0]:,.2f}")
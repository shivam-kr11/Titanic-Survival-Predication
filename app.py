# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 13:18:46 2026

@author: HP
"""

import streamlit as st
import pickle
import base64


# Load Model
model = pickle.load(open("titanic_model.pkl", "rb"))


# Background Image
def set_background(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}

        /* Main title */
        h1 {{
            color: white !important;
        }}

        /* Headers */
        h2, h3 {{
            color: white !important;
        }}

        /* Normal text */
        p, label {{
            color: white !important;
        }}

        /* Button */
        button {{
            color: white !important;
            background-color: rgba(0, 0, 0, 0.6) !important;
            border: 1px solid white !important;
            border-radius: 10px !important;
        }}

        /* Button hover */
        button:hover {{
            background-color: rgba(255, 255, 255, 0.2) !important;
            border: 1px solid white !important;
        }}

        /* Prediction card */
        .result-card {{
            padding: 25px;
            margin-top: 25px;
            border-radius: 15px;
            text-align: center;
            backdrop-filter: blur(10px);
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
        }}

        /* Survive */
        .survive {{
            background: rgba(0, 120, 70, 0.75);
            border: 2px solid #00ff9d;
        }}

        /* Not survive */
        .danger {{
            background: rgba(150, 20, 20, 0.75);
            border: 2px solid #ff6b6b;
        }}

        /* Result icon */
        .result-icon {{
            font-size: 45px;
            font-weight: bold;
            color: white;
        }}

        /* Result title */
        .result-title {{
            font-size: 26px;
            font-weight: bold;
            color: white;
        }}

        /* Result text */
        .result-text {{
            font-size: 16px;
            color: white;
            margin-top: 8px;
        }}

        </style>
        """,
        unsafe_allow_html=True
    )


set_background("bg.png")


# Title
st.title("🚢 Titanic Survival Prediction")

st.write(
    "Welcome! This app predicts whether a passenger "
    "would survive the Titanic disaster."
)


# Passenger Details
st.header("Enter Passenger Details")


pclass = st.selectbox(
    "Passenger Class",
    [1, 2, 3]
)


sex = st.selectbox(
    "Gender",
    ["Male", "Female"]
)


age = st.number_input(
    "Age",
    min_value=0,
    max_value=100,
    value=25
)


sibsp = st.number_input(
    "Siblings/Spouses",
    min_value=0,
    value=0
)


parch = st.number_input(
    "Parents/Children",
    min_value=0,
    value=0
)


fare = st.number_input(
    "Fare",
    min_value=0.0,
    value=50.0
)


embarked = st.selectbox(
    "Embarked",
    ["C", "Q", "S"]
)


# Encode Gender
if sex == "Male":
    sex = 1
else:
    sex = 0


# One-Hot Encode Embarked
embarked_Q = 0
embarked_S = 0

if embarked == "Q":
    embarked_Q = 1
elif embarked == "S":
    embarked_S = 1


# Prediction
if st.button("Predict"):

    input_data = [[
        pclass,
        sex,
        age,
        sibsp,
        parch,
        fare,
        embarked_Q,
        embarked_S
    ]]

    prediction = model.predict(input_data)


    # Result
    if prediction[0] == 1:

        st.markdown(
            """
<div class="result-card survive">
    <div class="result-icon">✓</div>
    <div class="result-title">Passenger Likely to Survive</div>
    
</div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            """
<div class="result-card danger">
    <div class="result-icon">!</div>
    <div class="result-title">Passenger Unlikely to Survive</div>
    
</div>
            """,
            unsafe_allow_html=True
        )
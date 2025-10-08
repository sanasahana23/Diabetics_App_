# ==============================
# 🩺 Interactive Diabetes Predictor
# ==============================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image
import plotly.express as px

# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config(
    page_title="Diabetes Predictor",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ------------------------------
# LOAD MODEL
# ------------------------------
try:
    with open("logisticRegr.pkl", "rb") as pickle_in:
        classifier = pickle.load(pickle_in)
except Exception as e:
    st.error(f"⚠️ Model file not found or corrupted: {e}")
    st.stop()

# ------------------------------
# CUSTOM STYLES
# ------------------------------
st.markdown("""
    <style>
        body {
            background: linear-gradient(135deg, #E0F7FA 0%, #80DEEA 100%);
            color: #00363A;
        }
        .main {
            background-color: #f8fcfd;
            padding: 2rem;
            border-radius: 1rem;
            box-shadow: 0px 4px 20px rgba(0,0,0,0.1);
        }
        .stButton>button {
            background-color: #00796B;
            color: white;
            border-radius: 10px;
            padding: 0.6rem 1.2rem;
            font-size: 1.1rem;
            font-weight: 600;
        }
        .stButton>button:hover {
            background-color: #004D40;
            color: white;
            transform: scale(1.05);
            transition: 0.3s;
        }
        .result-box {
            padding: 1.5rem;
            border-radius: 1rem;
            text-align: center;
            font-size: 1.2rem;
            font-weight: 600;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# ------------------------------
# SIDEBAR
# ------------------------------
st.sidebar.title("🧮 Diabetes Prediction")
st.sidebar.markdown("This app predicts whether a person is diabetic based on medical inputs.")
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2947/2947355.png", width=120)

# Option to show/hide form
show_form = st.sidebar.checkbox("Show Prediction Form", True)

# ------------------------------
# MAIN CONTENT
# ------------------------------
st.title("🩺 Diabetes Risk Prediction App")
st.markdown("### Check your health risk level using Machine Learning")

if show_form:
    with st.container():
        st.markdown("#### Please enter your health details:")
        name = st.text_input("👤 Name", placeholder="Enter your name")

        col1, col2 = st.columns(2)
        with col1:
            pregnancy = st.number_input("🤰 Number of Pregnancies", min_value=0, max_value=20, value=1)
            glucose = st.number_input("🩸 Plasma Glucose Concentration", min_value=0, max_value=300, value=120)
            bp = st.number_input("💓 Diastolic Blood Pressure (mm Hg)", min_value=0, max_value=150, value=70)
            skin = st.number_input("💪 Triceps Skin Fold Thickness (mm)", min_value=0, max_value=100, value=20)
        with col2:
            insulin = st.number_input("💉 2-Hour Serum Insulin (mu U/ml)", min_value=0, max_value=900, value=80)
            bmi = st.number_input("⚖️ Body Mass Index", min_value=0.0, max_value=70.0, value=25.0)
            dpf = st.number_input("🧬 Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
            age = st.number_input("🎂 Age", min_value=21, max_value=100, value=30)

        submit = st.button("🔍 Predict")

        if submit:
            input_data = np.array([[pregnancy, glucose, bp, skin, insulin, bmi, dpf, age]])
            prediction = classifier.predict(input_data)[0]

            if prediction == 0:
                st.markdown(
                    f"<div class='result-box' style='background-color:#43A047;'>✅ Congratulations {name}! You are **not diabetic.**</div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<div class='result-box' style='background-color:#E53935;'>⚠️ Sorry {name}, it seems you are **diabetic.** Please consult a doctor.</div>",
                    unsafe_allow_html=True
                )

# ------------------------------
# OPTIONAL VISUALIZATION
# ------------------------------
st.markdown("---")
st.markdown("### 📊 Visual Insight")
data = {
    'Category': ['Normal BMI', 'Your BMI', 'Normal Glucose', 'Your Glucose'],
    'Value': [24, bmi, 120, glucose]
}
df = pd.DataFrame(data)
fig = px.bar(
    df, x='Category', y='Value', color='Category',
    color_discrete_sequence=['#26C6DA', '#00796B', '#26C6DA', '#00796B'],
    title="Comparison of Your Values with Normal Ranges"
)
st.plotly_chart(fig, use_container_width=True)

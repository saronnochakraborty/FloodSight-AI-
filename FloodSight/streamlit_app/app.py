import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import io
import time
import sys
import os

# Add tensorflow_model path dynamically
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "tensorflow_model")))
from model_predict import predict_weather, FEATURE_COLS

# =========================
# Streamlit Setup
# =========================
st.set_page_config(page_title="FloodSight — Live Flood & Rainfall Predictor", layout="centered")

st.markdown("""
    <style>
        .main {
            background: linear-gradient(135deg, #e0f3ff, #f9fbff);
        }
        .stButton>button {
            background-color: #0078D7;
            color: white;
            border-radius: 10px;
            font-weight: bold;
        }
        .stButton>button:hover {
            background-color: #005a9e;
        }
        .title {
            text-align: center;
            font-size: 2em;
            font-weight: bold;
            color: #003366;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<p class='title'> FloodSight — Live Flood & Rainfall Predictor</p>", unsafe_allow_html=True)
st.markdown("Predicts **rainfall (mm)** and **flood risk (%)** with live updating chart.")
st.markdown("---")

# =========================
# Initialize session state
# =========================
if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=["Run", "Rainfall_mm", "FloodRisk_pct"])

# =========================
# User Inputs
# =========================
user_input = {}
col1, col2 = st.columns(2)
for i, feature in enumerate(FEATURE_COLS):
    target_col = col1 if i % 2 == 0 else col2
    with target_col:
        user_input[feature] = st.number_input(f"{feature.replace('_',' ').title()}", value=0.0, step=0.1)

st.markdown("---")

# =========================
# Predict Button
# =========================
if st.button("🔮 Predict"):
    with st.spinner("Running TensorFlow model..."):
        time.sleep(0.5)
        try:
            prediction = predict_weather(user_input)
            rainfall = prediction["predicted_rainfall_mm"]
            flood = prediction["predicted_flood_risk_pct"]

            st.success("✅ Prediction successful!")
            st.metric("🌧️ Predicted Rainfall (mm)", f"{rainfall:.2f}")
            st.metric("🌊 Predicted Flood Risk (%)", f"{flood:.2f}")

            # Add to history
            new_row = pd.DataFrame({
                "Run": [len(st.session_state.history)+1],
                "Rainfall_mm": [rainfall],
                "FloodRisk_pct": [flood]
            })
            st.session_state.history = pd.concat([st.session_state.history, new_row], ignore_index=True)

            # =========================
            # Live chart using Pillow
            # =========================
            fig, ax1 = plt.subplots()
            ax2 = ax1.twinx()
            ax1.plot(st.session_state.history["Run"], st.session_state.history["Rainfall_mm"], 'b-o', label="Rainfall (mm)")
            ax2.plot(st.session_state.history["Run"], st.session_state.history["FloodRisk_pct"], 'r-s', label="Flood Risk (%)")
            ax1.set_xlabel("Run Number")
            ax1.set_ylabel("Rainfall (mm)", color='b')
            ax2.set_ylabel("Flood Risk (%)", color='r')
            ax1.set_title("🌧️ Live Prediction Chart")
            ax1.grid(True)

            # Save plot to PIL image
            buf = io.BytesIO()
            fig.tight_layout()
            fig.savefig(buf, format='png')
            buf.seek(0)
            img = Image.open(buf)
            st.image(img)
            plt.close(fig)

            # Risk messages
            if flood > 70:
                st.error("⚠️ High Flood Risk! Immediate action advised.")
            elif flood > 40:
                st.warning("⚠️ Moderate Flood Risk. Stay alert.")
            else:
                st.info("✅ Low Flood Risk. Conditions stable.")

        except Exception as e:
            st.error(f"❌ Error: {e}")

# Display full history
if len(st.session_state.history) > 0:
    st.markdown("---")
    st.markdown("### 📊 Prediction History")
    st.dataframe(st.session_state.history)

st.markdown("---")
st.caption("Made Saronno Chakraborty | Powered by TensorFlow & Streamlit")

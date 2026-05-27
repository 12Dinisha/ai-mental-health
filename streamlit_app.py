import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open("models/ensemble_v3.pkl", "rb"))

st.set_page_config(page_title="AI Mental Health", layout="centered")

st.title("🧠 AI Mental Health Prediction System")

st.write("Enter feature values to predict mental health risk.")

features = []

# FIXED → 14 features
for i in range(14):
    value = st.number_input(
        f"Feature {i+1}",
        value=0.0,
        step=0.1
    )
    features.append(value)

if st.button("Predict Risk"):

    X = np.array(features).reshape(1, -1)

    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]

    st.subheader("Prediction Result")

    if pred == 1:
        st.error("⚠️ HIGH RISK")
    else:
        st.success("✅ LOW RISK")

    st.write(f"### Probability Score: {prob:.2f}")

    confidence = (
        "HIGH"
        if prob > 0.7
        else "MEDIUM"
        if prob > 0.3
        else "LOW"
    )

    st.write(f"### Confidence: {confidence}")
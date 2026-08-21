import streamlit as st
import pickle

# Page configuration
st.set_page_config(
    page_title="Cloud Bug Detector",
    page_icon="🐞",
    layout="centered"
)

# Title
st.title("🐞 Cloud Bug Detector")
st.write("Enter a bug description to detect whether it is a cloud-related bug.")

# Load trained model
try:
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("model.pkl not found. Please run train_model.py first.")
    st.stop()

# User input
st.subheader("Bug Description")

description = st.text_area(
    "Describe the bug:",
    placeholder="Example: Server becomes unavailable when multiple users access the cloud application."
)

# Prediction button
if st.button("🔍 Detect Bug"):

    if description.strip() == "":
        st.warning("Please enter a bug description.")
    else:
        prediction = model.predict([description])[0]

        st.subheader("Prediction")

        if prediction == 1 or str(prediction).lower() in ["1", "yes", "true", "bug"]:
            st.error("🚨 Cloud-related bug detected!")
        else:
            st.success("✅ This does not appear to be a cloud-related bug.")

        st.write("Prediction:", prediction)
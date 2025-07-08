import streamlit as st
import joblib
from spellchecker import SpellChecker

# Initialize spell checker
spell = SpellChecker()

def correct_text(text):
    words = text.split()
    corrected_words = [spell.correction(word) for word in words]
    return " ".join(corrected_words)

# Load saved model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Set up Streamlit app
st.set_page_config(page_title="Symptoms Diagnosis Classifier", page_icon="🩺")
st.title("🩺 Symptoms Diagnosis Classifier")
st.write("Enter your symptoms or health text and get a predicted diagnosis!")

# User input
user_input = st.text_area("Enter symptoms here:")

# Predict button
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        corrected_input = correct_text(user_input)
        input_vec = vectorizer.transform([corrected_input])
        prediction = model.predict(input_vec)[0]
        st.success(f"✅ Predicted Diagnosis: **{prediction}**")
        st.info(f"Corrected input used for prediction: {corrected_input}")

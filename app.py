import streamlit as st
import joblib

# Load saved model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Set up Streamlit app
st.set_page_config(page_title="Text Classifier", page_icon="🧠")
st.title("🧠 Text Classification App")
st.write("Enter some text and get a predicted label!")

# User input
user_input = st.text_area("Enter text here:")

# Predict button
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        input_vec = vectorizer.transform([user_input])
        prediction = model.predict(input_vec)[0]
        st.success(f"✅ Predicted Label: **{prediction}**")

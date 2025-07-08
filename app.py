import streamlit as st
import joblib
import numpy as np

# Load the trained model and TF-IDF vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Your symptom list (you can keep it as your symptom columns or a fixed list)
symptom_cols = [
    'itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering',
    'chills', 'joint_pain', 'stomach_pain', 'acidity', 'ulcers_on_tongue',
    # ... add all your symptoms here ...
]

st.set_page_config(page_title="Symptoms Diagnosis Classifier", page_icon="🩺")
st.title("🩺 Symptoms Diagnosis Classifier")

selected_symptoms = st.multiselect(
    "Select up to 10 symptoms (start typing to search):",
    options=symptom_cols,
    max_selections=10
)

if st.button("Predict Disease"):
    if not selected_symptoms:
        st.warning("Please select at least one symptom.")
    else:
        # Convert selected symptoms list to a single string
        symptom_text = " ".join(selected_symptoms)

        # Vectorize input text using the loaded vectorizer
        input_vector = vectorizer.transform([symptom_text])

        # Predict disease using the trained model
        prediction = model.predict(input_vector)[0]

        st.success(f"✅ Predicted Disease: {prediction}")

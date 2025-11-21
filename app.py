import streamlit as st
import pandas as pd
from joblib import load

# Page setup
st.set_page_config(page_title="Disease Prediction", page_icon="🩺", layout="centered")
st.title("🩺 Disease Prediction System")
st.markdown("Select the symptoms you are experiencing and get a predicted disease!")

# Load symptom names
df = pd.read_csv("./dataset/training_data.csv")
symptoms = df.columns[:-2]

# 🧠 Model selection option
model_choice = st.selectbox(
    "Select Prediction Model",
    ["Decision Tree", "Random Forest", "Naive Bayes", "Gradient Boost"]
)

# Load selected model based on choice
if model_choice == "Decision Tree":
    model = load("./saved_model/decision_tree.joblib")
elif model_choice == "Random Forest":
    model = load("./saved_model/random_forest.joblib")
elif model_choice == "Naive Bayes":
    model = load("./saved_model/mnb.joblib")
else:
    model = load("./saved_model/gradient_boost.joblib")

# Symptom selection
selected_symptoms = st.multiselect("Select Symptoms", symptoms)

# Prediction
if st.button("🔍 Predict Disease"):
    if len(selected_symptoms) == 0:
        st.warning("⚠️ Please select at least one symptom before predicting.")
    else:
        # Create input vector
        input_data = [0] * len(symptoms)
        for symptom in selected_symptoms:
            index = list(symptoms).index(symptom)
            input_data[index] = 1

        # Convert to DataFrame
        input_df = pd.DataFrame([input_data], columns=symptoms)

        # Predict
        prediction = model.predict(input_df)[0]
        probs = model.predict_proba(input_df)[0]
        confidence = max(probs) * 100  # convert to %

        # Display
        st.success(f"🧠 **Predicted Disease:** {prediction}")
        st.info(f"✅ **Confidence Level:** {confidence:.2f}%")

st.markdown("---")
st.caption("© 2025 Disease Prediction System | Built with Streamlit 🧬") 
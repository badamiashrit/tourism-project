import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st

# 1. Page Configuration & Setup
st.set_page_config(
    page_title="Tourism Demand Predictor",
    page_icon="✈️",
    layout="centered"
)

# 2. Model Loading Utility
@st.cache_resource
def load_trained_model():
    """Loads the model artifact committed by the pipeline."""
    model_path = os.path.join("tourism_project", "deployment", "best_model.pkl")

    if not os.path.exists(model_path):
        st.error(f"⚠️ Model file not found at `{model_path}`. Please ensure the pipeline has successfully committed the model to the repo.")
        return None

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

# Main Execution Flow
def main():
    st.title("✈️ Tourism Demand Forecasting App")
    st.write("Enter the parameters below to predict tourism volume and demand metrics using our latest production model.")

    model = load_trained_model()
    if model is None:
        return

    st.divider()
    st.subheader("📋 Input parameters")

    # 3. Collect User Inputs
    col1, col2 = st.columns(2)

    with col1:
        season = st.selectbox("Season", ["Spring", "Summer", "Autumn", "Winter"])
        avg_temperature = st.slider("Average Temperature (°C)", min_value=-10.0, max_value=45.0, value=22.0)
        marketing_spend = st.number_input("Marketing Spend ($)", min_value=0.0, value=5000.0, step=500.0)

    with col2:
        is_holiday = st.selectbox("Is Holiday Period?", ["Yes", "No"])
        economic_index = st.slider("Local Economic Index", min_value=50.0, max_value=150.0, value=100.0)
        prev_month_visitors = st.number_input("Previous Month Visitors", min_value=0, value=15000, step=1000)

    # 4. Convert inputs to a DataFrame matching expected training layout
    input_data = {
        'avg_temperature': [avg_temperature],
        'marketing_spend': [marketing_spend],
        'economic_index': [economic_index],
        'prev_month_visitors': [prev_month_visitors],
        'is_holiday_Yes': [1 if is_holiday == "Yes" else 0],
        'season_Summer': [1 if season == "Summer" else 0],
        'season_Autumn': [1 if season == "Autumn" else 0],
        'season_Winter': [1 if season == "Winter" else 0],
    }

    input_df = pd.DataFrame(input_data)

    # 5. Prediction Engine
    st.divider()
    if st.button("🔮 Predict Tourism Demand", type="primary"):
        with st.spinner("Calculating forecast..."):
            try:
                prediction = model.predict(input_df)

                st.success("🎉 Prediction Generated Successfully!")
                st.metric(
                    label="Estimated Visitors / Demand Score",
                    value=f"{int(prediction[0]):,}" if prediction[0] >= 0 else f"{prediction[0]:.2f}"
                )
            except Exception as e:
                st.error("An error occurred during prediction. Please verify that your input dataframe columns perfectly align with the model's training expectations.")
                st.exception(e)

if __name__ == "__main__":
    main()

# streamlit_app.py

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Load saved model and preprocessor
# -------------------------
model = joblib.load('car_price_xgb_model.pkl')
preprocessor = joblib.load('xgb_preprocessor.pkl')

# -------------------------
# Streamlit App
# -------------------------
st.title("Car Price Prediction App 🚗💰")
st.write("Upload your test data CSV to get price predictions and visualize results.")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Load test data
    test_df = pd.read_csv(uploaded_file)
    
    if 'Actual Price' not in test_df.columns:
        st.error("CSV must contain 'Actual Price' column along with features.")
    else:
        # Separate features and actual price
        X_test = test_df.drop('Actual Price', axis=1)
        y_actual = test_df['Actual Price']
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Add predictions to dataframe
        results_df = X_test.copy()
        results_df['Actual Price'] = y_actual
        results_df['Predicted Price'] = y_pred
        
        st.subheader("Prediction Results")
        st.dataframe(results_df.head(20))

        # -------------------------
        # Matplotlib Bar Chart
        # -------------------------
        st.subheader("Actual vs Predicted Price Bar Chart")

        fig, ax = plt.subplots(figsize=(12, 6))

        indices = np.arange(len(y_actual))

        ax.bar(indices - 0.2, y_actual, width=0.4, label='Actual Price')
        ax.bar(indices + 0.2, y_pred, width=0.4, label='Predicted Price')

        ax.set_xlabel("Car Index")
        ax.set_ylabel("Price")
        ax.set_title("Actual vs Predicted Prices")
        ax.legend()

        st.pyplot(fig)

        # -------------------------
        # Metrics
        # -------------------------
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        mae = mean_absolute_error(y_actual, y_pred)
        rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
        r2 = r2_score(y_actual, y_pred)
        
        st.subheader("Evaluation Metrics")
        st.write(f"**MAE:** ${mae:.2f}")
        st.write(f"**RMSE:** ${rmse:.2f}")
        st.write(f"**R² Score:** {r2:.4f}")

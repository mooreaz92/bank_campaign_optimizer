import pickle
import pandas as pd
import streamlit as st
from functions import *

# Load the pickle file containing the pipeline
with open("App/pipeline.pkl", "rb") as f:
    pipeline = pickle.load(f)

# Create a Streamlit app that allows the user to upload a CSV file
st.title("Model Predictions")
st.write("Upload a CSV file to make predictions with the model.")
uploaded_file = st.file_uploader("Choose a file", type="csv")

if uploaded_file is not None:
    # Use pandas to read the uploaded CSV file into a DataFrame
    df = pd.read_csv(uploaded_file)

    # Apply the pipeline to the DataFrame using the predict and predict_proba methods
    predictions = pipeline.predict(df)
    probabilities = pipeline.predict_proba(df)

    # Add the predictions and probabilities to the DataFrame
    df["Prediction"] = predictions
    df["Probability"] = probabilities[:, 1]

    # Return the updated DataFrame as a CSV file for download
    st.download_button(
        label="Download predictions",
        data=df.to_csv(index=False),
        file_name="predictions.csv",
        mime="text/csv"
    )


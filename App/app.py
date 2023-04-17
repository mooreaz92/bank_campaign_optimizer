import pickle
import pandas as pd
import streamlit as st
from functions import *

# Create a Streamlit app that allows the user to upload a CSV file
st.title("Model Predictions")
st.write("Upload your list of leads to get the probability of a lead subscribing to a term deposit.")

# Load the model from the pickle file
model = pickle.load(open('App/final_model.pkl', 'rb'))

### CSV Upload Button
uploaded_file = st.file_uploader("Choose a file", type="csv")

if uploaded_file is not None:
    # Use pandas to read the uploaded CSV file into a DataFrame
    df_input_raw = pd.read_csv(uploaded_file)

    # Apply the functions found in App/functions.py to the DataFrame
    df_cleaned = cast_as_columns(df_input_raw)
    df_cleaned = ordinal_encode_education(df_cleaned)
    df_cleaned = set_pdays_to_zero(df_cleaned)
    df_cleaned = drop_features(df_cleaned)
    df_cleaned = drop_features_corr(df_cleaned)
    df_cleaned = one_hot_encode(df_cleaned)
    df_cleaned = standard_scale(df_cleaned)

    # Apply the model to the DataFrame and append the predictions to a new dataframe

    predictions = model.predict(df_cleaned)
    df_copy = df_cleaned.copy()
    df_copy['predictions'] = predictions
    df_copy['probabilities'] = model.predict_proba(df_cleaned)[:, 1]

    # Display the probabilities of the predictions across three labeled bins

    st.write("Probabilities of predictions:")
    st.write(pd.cut(df_copy['probabilities'], bins=3, labels=['Low', 'Medium', 'High']).value_counts())

    # Display some useful metrics about the predictions

    st.write("Mean probability of predictions:", df_copy['probabilities'].mean())


    # Allow the user to download the DataFrame with the predictions as a CSV file

    st.download_button(label='Download Predictions', data=df_copy.to_csv(), file_name='predictions.csv', mime='text/csv')

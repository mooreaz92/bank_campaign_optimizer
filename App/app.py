import pickle
import pandas as pd
import streamlit as st
from functions import *

st.set_page_config(page_title='Call Center Lead Predictor', page_icon=':telephone_receiver:', layout='centered', initial_sidebar_state='auto')

# Create a Streamlit app that allows the user to upload a CSV file
st.title("Call Center Lead Predictor")
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

    # Display the count of the predictions across three bins, and label the column 'Count of Leads', and suppress the warnings

    prob_df = pd.cut(df_copy['probabilities'], bins=3, labels=['Low', 'Medium', 'High']).value_counts().to_frame(name='Count of Leads')

    st.write("Leads by probability:")
    st.write(prob_df)

    # Supressing warnings in the Streamlit app

    st.set_option('deprecation.showPyplotGlobalUse', False)

    # Allow the user to download the DataFrame with the predictions as a CSV file

    st.download_button(label='Download Predictions', data=df_copy.to_csv(), file_name='predictions.csv', mime='text/csv')

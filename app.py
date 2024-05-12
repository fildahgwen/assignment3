import streamlit as st
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn import metrics
import pickle

# Load the saved models
with open('scikit_learn_sgd.pickle', 'rb') as f:
    sgd_pipe = pickle.load(f)
    
with open('scikit_learn_svm.pickle', 'rb') as f:
    svc_pipe = pickle.load(f)

# Function to predict clusters and URLs


def display_instructions():
    instructions = """
    Instructions:
    1. Open the app in your web browser.
    2. Enter your news text in the text area provided.
    3. Click on the "Cluster News" button.
    4. The app will predict the cluster number and URL for the entered news text.
    5. The predicted cluster number and URL will be displayed below the button.
    6. If the predicted cluster number is not recognized, the URL will be displayed as "Unknown".
    
    Please note that the app may take some time to process the news text and display the results.
    
    If you encounter any issues or have specific questions about the app, please let me know, and I'll be happy to assist you further.
    """
    print(instructions)

# Call the function to display the instructions
display_instructions()
def predict_clusters_and_urls(text):
    # Predict cluster numbers
    sgd_cluster = sgd_pipe.predict([text])[0]
    svc_cluster = svc_pipe.predict([text])[0]
    
    # Define URLs based on cluster categories
    url_mapping = {
        '0': 'https://www.bbc.com/sport',
        '1': 'https://www.bbc.com/news/world',
        '2': 'https://www.bbc.com/business',
        '3': 'https://www.bbc.com/culture/entertainment-news'
    }
    
    # Get the cluster categories
    sgd_category = 'Unknown' if str(sgd_cluster) not in url_mapping else str(sgd_cluster)
    svc_category = 'Unknown' if str(svc_cluster) not in url_mapping else str(svc_cluster)
    
    return {
        'Top cluster number (SGD)': sgd_cluster,
        'Top cluster number (SVC)': svc_cluster,
        'URL (SGD)': url_mapping.get(sgd_category, None),
        'URL (SVC)': url_mapping.get(svc_category, None)
    }

# Streamlit app
st.title('News Clustering App')

text_input = st.text_area("Enter your news text here:", "")

if st.button("Cluster News"):
    if text_input:
        result = predict_clusters_and_urls(text_input)
        st.write(result)
    else:
        st.write("Please enter some text to cluster the news.")
